
get_artitfact_store(config) = FileSystemArtifactStore[](base_path = config.artifact_path)

"""
    remove_non_best_trials_artifacts(data_path, db, study)

Delete large (>160 MB) artifact files that belong to non-Pareto trials.
"""
function remove_non_best_trials_artifacts(data_path, db, study)
    artifact_folder = joinpath(data_path, "artifacts", db[1:end-3])
    @info "Cleaning artifacts in $(artifact_folder), db in $(db)"
    best_trials = retrieve_trials(data_path, db, study; best = true)
    all_trials  = retrieve_trials(data_path, db, study; best = false)
    best_ids    = [pyconvert(Int, trial.id) for trial in best_trials]
    some_trial  = false
    @info "Retrieved: $(length(best_trials)) best trials and $(length(all_trials)) total trials for study $(study)"
    for trial in all_trials
        trial_id = pyconvert(Int, trial.id)
        (trial_id in best_ids) && continue
        this_trial = false
        @debug "Removing artifacts for trial $(trial_id)"
        for attr in keys(trial.user_attrs) |> collect
            occursin("artifact", attr) || continue
            user_attrs = pyconvert(Dict{String,Any}, trial.user_attrs)
            file = joinpath(artifact_folder, user_attrs[attr])
            !isfile(file) && continue
            if stat(file).size > 160MB
                @warn "Large file detected for trial $(trial.id): $attr, size: $(stat(file).size/MB) MB"
                some_trial = true
                this_trial = true
                rm(file)
            end
        end
        this_trial && @info "Deleted artifacts of trial $(trial.id)"
    end
    !some_trial && @info "Study $(study) has no artifacts to remove"
    return some_trial
end

"""
    move_large_artifact_to_scratch(data_path, db, study)

Move artifact files larger than 1 GB to the scratch folder.
"""
function move_large_artifact_to_scratch(data_path, db, study)
    scratch_folder  = "/pasteur/appa/scratch/aquaresi/optuna_artifacts/"
    artifact_folder = joinpath(data_path, "artifacts", db[1:end-3])
    @info "Cleaning artifacts in $(artifact_folder), db in $(db)"
    all_trials  = retrieve_trials(data_path, db, study; best = false)
    some_trial  = false
    @info "Retrieved $(length(all_trials)) total trials for study $(study)"
    for trial in all_trials
        trial_id   = pyconvert(Int, trial.id)
        this_trial = false
        @debug "Moving artifacts for trial $(trial_id)"
        for attr in keys(trial.user_attrs) |> collect
            occursin("artifact", attr) || continue
            user_attrs = pyconvert(Dict{String,Any}, trial.user_attrs)
            path = user_attrs[attr]
            file = joinpath(artifact_folder, path)
            !isfile(file) && continue
            if stat(file).size > 1000MB
                @warn "Large file detected for trial $(trial.id): $attr, size: $(stat(file).size/MB) MB"
                some_trial = true
                this_trial = true
                mv(file, joinpath(scratch_folder, path))
            end
        end
        this_trial && @info "Moved artifacts of trial $(trial.id)"
    end
    !some_trial && @info "Study $(study) has no artifacts to move"
    return some_trial
end

"""
    remove_recordings_from_best_trials_artifacts(data_path, db, study)

Strip the `recordings` key from JLD2 model files in best-trial artifacts to save disk space.
"""
function remove_recordings_from_best_trials_artifacts(data_path, db, study)
    artifact_folder = joinpath(data_path, "artifacts", db[1:end-3])
    best_trials = retrieve_trials(data_path, db, study; best = true)
    all_trials  = retrieve_trials(data_path, db, study; best = false)
    best_ids    = [trial.id for trial in best_trials]
    some_trial  = false
    @info "Retrieved: $(length(best_trials)) best trials and $(length(all_trials)) total trials for study $(study)"
    for trial in all_trials
        (trial.id in best_ids) || continue
        path = trial.user_attrs["artifact_model"]
        file = joinpath(artifact_folder, path)
        if !isfile(file)
            @warn "No model file found for best trial $(trial.id), skipping"
            continue
        end
        try
            model = DrWatson.load(DrWatson.FileIO.File{DrWatson.FileIO.format"JLD2"}(file))
            haskey(model, "recordings") || continue
            temp_model = Dict{String,Any}(k => v for (k, v) in pairs(model) if k != "recordings")
            rm(file)
            DrWatson.save(DrWatson.FileIO.File{DrWatson.FileIO.format"JLD2"}(file), temp_model)
            some_trial = true
            @info "Stripped recordings from best trial $(trial.id): $path"
        catch e
            @error "Skipping deletion of recordings for $file: $e"
        end
    end
    !some_trial && @info "Study $(study) has no recordings artifacts to remove"
    return some_trial
end

"""
    save_figure_artifact(p, name, trial, config)

Save a figure as an artifact.

# Arguments
- `p`: Plot object
- `name`: Name of the figure
- `trial`: Trial object
- `config`: Configuration object

# Returns
- `Any`: The plot object
"""
# `empty!(fig)` below is load-bearing, not cleanup: each optimization trial
# builds ~14 Figures here (raster, firing rate, response, histograms, on/off,
# tonotopy, confmat, correlation, representation). Without emptying them
# after save, Makie's global Scene/Observable bookkeeping kept every Figure
# reachable across trials, so per-trial SNN model/network arrays never got
# collected — a linear ~0.6-1GB/trial RSS leak (confirmed 2026-07-03 via
# cluster/test_memory/test_snn_objects.jl, WeakRef + forced GC.gc(true) x2
# on the model showed it survived collection even with zero Optuna involved,
# until this empty!(fig) call was added). If you add a new *_artifact
# function or change how figures are built/returned here, keep every figure
# emptied (or otherwise dereferenced) right after it's saved.
function save_figure_artifact(fig, name, trial, study, Makie::Module)
    isnothing(study) && return fig
    if !isnothing(trial)
        file_path = joinpath(
            mkpath(joinpath(study.artifact_path, "tmp")),
            "$(name)_plot_$(trial.number)_$(randstring(10)).png",
        )
        @assert isa(fig, Makie.Figure)
        Makie.save(file_path, fig)
        # isa(fig, Plots.Plot) && SNNPlots.savefig(fig, file_path)

        artifact_id = upload_artifact[](
            artifact_store = get_artitfact_store(study),
            file_path = file_path,
            study_or_trial = trial,
        )
        trial.set_user_attr("artifact_$name", artifact_id)
        rm(file_path)
        empty!(fig)
    else
        file_path = joinpath(
            mkpath(joinpath(study.artifact_path, "plots")),
            "$(name)_plot.svg",
        )
        @assert isa(fig, Makie.Figure)
        Makie.save(file_path, fig)
        empty!(fig)
        fig
    end
end