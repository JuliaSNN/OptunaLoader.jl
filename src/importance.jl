
"""
    retrieve_PedAnova_importance(db_path, studies; kwargs...) -> Dict{String, Vector{Dict}}

Compute PedAnova parameter importance for each study in `studies`.

Returns a `Dict` mapping study name → `Vector` of per-objective importance `Dict`s.
"""
function retrieve_PedAnova_importance(db_path::String, studies; kwargs...)
    @assert isfile(db_path) "The database file does not exist. Please check the path."
    storage = journal_storage(db_path; lock = JournalFileOpenLock)
    return retrieve_PedAnova_importance(storage, studies; kwargs...)
end

# PedAnova (via optuna.search_space.IntersectionSearchSpace) intersects the
# recorded param distributions across ALL trials in the study. A single trial
# with empty `distributions` (e.g. an enqueued/seeded trial completed without
# its params attached) collapses that intersection to the empty set and makes
# get_param_importances silently return {} for every objective. Importance is
# only ever computed here (not from the real db elsewhere), so we compute it
# against an in-memory copy with those trials dropped instead of touching the
# stored study.
function _dense_trials_only(opt_study)
    good = [t for t in opt_study.trials if pyconvert(Int, length(t.distributions)) > 0]
    n_dropped = pyconvert(Int, length(opt_study.trials)) - length(good)
    if n_dropped > 0
        @warn "Dropping $(n_dropped) trial(s) with empty distributions before computing importance" study=opt_study.study_name
    end
    filtered = optuna[].create_study(directions = opt_study.directions)
    filtered.add_trials(good)
    return filtered
end

function retrieve_PedAnova_importance(storage, studies;
            target_quantile = 0.8,
            evaluate_on_local = true,)
    param_importance = Dict{String,Any}()
    for study_name in studies
        opt_study = optuna[].load_study(study_name = study_name, storage = storage)
        targets = pyconvert(Vector, opt_study.directions)
        imp_study = _dense_trials_only(opt_study)
        try
            importance = map(eachindex(targets)) do n
                pyconvert(Dict, Importance[].get_param_importances(imp_study,
                                target = t -> t.values[n - 1],
                                evaluator = ImportanceEvaluator[](
                                    target_quantile = target_quantile,
                                    evaluate_on_local = evaluate_on_local
                                ))
                                )
            end
            push!(param_importance, study_name => importance)
            @info "Study $(study_name) loaded"
        catch e
            @error e
            @warn "Study $(study_name) not found in $(storage)"
            push!(param_importance, study_name => map(eachindex(targets)) do n
                Dict{String,Float64}()
            end)
        end
    end
    return param_importance
end


"""
    aggregate_PDE_importance(importance, experiment_info) -> Vector{Matrix}

Aggregate raw PedAnova importance into a (n_param_groups × n_value_groups) matrix per study.

Returns a `Vector` of matrices, one per study in `experiment_info.all_studies`.
"""
function aggregate_PDE_importance(importance, experiment_info)
    map(experiment_info.all_studies) do study_name
        aggregate_PDE_importance(importance, experiment_info, study_name)
    end
end

"""
    aggregate_PDE_importance(importance, experiment_info, study_name) -> Matrix

Aggregate raw PedAnova importance for a single study into a
(n_param_groups × n_value_groups) matrix.
"""
function aggregate_PDE_importance(importance, experiment_info, study_name::AbstractString)
    study_importance  = importance[study_name]
    parameters_info   = experiment_info.studies[study_name].parameters

    valid_params = []
    M = []
    for param in parameters_info.order
        imp_values = map(eachindex(study_importance)) do n
            target_importance = dict2ntuple(study_importance[n])
            get(target_importance, param, NaN)
        end 
        if any(isnan.(imp_values)) 
            @warn "Parameter $(param) has missing importance values in study $(study_name)"
            continue
        else
            push!(M, imp_values)
            push!(valid_params, param)
        end
    end 
    M = hcat(M... )

    return (params=M, names=collect(valid_params))


end


"""
    compute_study_importance(db_path, experiment_info) -> Dict{String, NamedTuple}

Compute PedAnova importance for each study independently.

Runs at three target quantiles (0.1 / 0.5 / 0.8) in both global and local modes.

Returns `Dict{String, NamedTuple}` — one entry per study. Each value is:
```
(;
    averages = (; glo, loc),   # n_param_groups × n_value_groups × 3 (quantile levels)
    full     = (; glo, loc),   # raw per-objective importance Dict at quantile 0.1
)
```
"""
function compute_study_importance(db_path, experiment_info)
    @info "Computing Optuna Parameter Importance, it may take a while..."
    storage = journal_storage(db_path; lock = JournalFileOpenLock)
    studies = experiment_info.all_studies

    quantile_levels = [0.1, 0.5, 0.8]
    imp_glo = map(q -> retrieve_PedAnova_importance(storage, studies;
                        target_quantile = q, evaluate_on_local = false), quantile_levels)
    imp_loc = map(q -> retrieve_PedAnova_importance(storage, studies;
                        target_quantile = q, evaluate_on_local = true),  quantile_levels)

    Dict(study_name => begin
        agg(imps) = cat([aggregate_PDE_importance(imp, experiment_info, study_name).params
                         for imp in imps]...; dims = 3)
        (;
            averages = (; glo = agg(imp_glo), loc = agg(imp_loc)),
            full     = (; glo = imp_glo[1][study_name], loc = imp_loc[1][study_name]),
            names = aggregate_PDE_importance(imp_glo[1], experiment_info, study_name).names
        )
    end for study_name in studies)
end
