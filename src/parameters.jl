
"""
    params_and_measured(trials::Vector; kwargs...)

Combine trial parameters and measured values into a DataFrame.

# Arguments
- `trials::Vector`: Vector of trial objects
- `kwargs...`: Additional keyword arguments

# Returns
- `DataFrame`: Combined DataFrame of trial parameters and measured values
"""
function params_and_measured(trials::Vector; kwargs...)
    if length(trials) == 0
        return DataFrame(), Dict{String,Any}()
    end
    trials_measures = []
    study_attrubutes = []
    trials_parameters = []
    trial_values = []
    field_keys = []
    extra = []
    for trial in trials
        push!(trial.user_attrs, "id"=>trial.id)
        push!(trial.user_attrs, "number"=>trial.number)
        push!(trial.user_attrs, "pruned"=>trial.pruned)
        push!(trials_measures, DataFrame(trial.user_attrs))
        push!(trials_parameters, DataFrame(trial.params))
        push!(trial_values, DataFrame(trial.values))
        push!(study_attrubutes, DataFrame(trial.study_attributes))
        field_key = Dict{String,Any}()
        field_key["user_attrs"] = keys(trial.user_attrs) |> collect |> x-> Symbol.(x)
        haskey(trial.params, "number") && delete!(trial.params, "number")
        field_key["params"] = keys(trial.params) |> collect |> x-> Symbol.(x)
        field_key["values"] = keys(trial.values) |> collect |> x-> Symbol.(x)
        field_key["study_attributes"] = keys(trial.study_attributes) |> collect |> x-> Symbol.(x)
        push!(field_keys, field_key)
        push!(extra, kwargs)
    end
    df = let
        # cols=:union: pruned trials carry fewer user_attrs than completed ones
        # (e.g. accuracy/squared_error are only set on successful evaluation),
        # so the naive vcat would error on mismatched columns.
        df1 = DataFrame(vcat(trials_measures...; cols = :union))
        df2 = DataFrame(vcat(trials_parameters...; cols = :union))
        df3 = DataFrame(vcat(trial_values...; cols = :union))
        df4 = DataFrame(vcat(extra...)) # extra holds raw kwargs Pairs, not DataFrames — no cols kwarg here
        df5 = DataFrame(vcat(study_attrubutes...; cols = :union))
        (df1, df2, df3, df4, df5)
    end
    hcat(df..., makeunique = true), Dict{String,Any}(field_keys[1]) |> dict2ntuple
end

"""
    params_and_measured(studies::Dict; kwargs...)

Combine trial parameters and measured values for multiple studies into a DataFrame.

# Arguments
- `studies::Dict`: Dictionary of study names to trial objects
- `kwargs...`: Additional keyword arguments

# Returns
- `DataFrame`: Combined DataFrame of trial parameters and measured values for all studies
- `Dict{String,Any}`: Dictionary of field keys for each study
"""
function params_and_measured(studies::Dict; kwargs...)
    dfs = []
    field_keys = Dict{String,Any}()
    for key in keys(studies)
            df, field_key = params_and_measured(
                studies[key];
                kwargs...,
            )
            push!(dfs, df)
            push!(field_keys, key=>field_key)

    end
    return vcat(dfs..., cols = :union), field_keys
end

"""
    params_keys(filtered_trials, args...)

Extract parameter keys from filtered trials.

# Arguments
- `filtered_trials`: Dictionary of filtered trials
- `args...`: Additional arguments

# Returns
- `DataFrame`: DataFrame containing parameter keys
"""
function params_keys(filtered_trials, args...)
    dfs = []
    for key in keys(filtered_trials)
        @show key
        if all([occursin(k, key) for k in args])
            trials_measures = []
            trials_parameters = []
            trial_values = []
            ids = []
            conditions = []
            for trial in filtered_trials[key]
                push!(ids, key)
                push!(trials_measures, (; measures = keys(trial.user_attrs)|>collect))
                push!(trials_parameters, (; params = keys(trial.params)|>collect))
                push!(trial_values, (; values = keys(trial.values)|>collect))
                network = guess_network(key)
                connectivity = guess_connectivity(key)
                plasticity = guess_plasticity(key)
                condition = (; network, connectivity, plasticity)
                push!(conditions, condition)
                break
            end

            df0 = DataFrame(vcat(conditions...))
            df1 = DataFrame(vcat(trials_measures...))
            df2 = DataFrame(vcat(trials_parameters...))
            df3 = DataFrame(vcat(trial_values...))
            df4 = DataFrame("study"=>Vector{String}(ids))

            push!(dfs, (df0, df1, df2, df3, df4))

        end
    end
    return vcat([hcat(df..., makeunique = true) for df in dfs]..., cols = :union)
end


"""
    parse_trial(trial, metric_names, study_attributes=Dict{String,String}())

Parse an Optuna trial into a named tuple.

# Arguments
- `trial`: Optuna trial object
- `metric_names`: Metric names from Optuna trial
- `study_attributes::Dict{String,String}`: Study attributes (default: empty dict)

# Returns
- `NamedTuple`: Parsed trial data
"""
function parse_trial(trial, metric_names, study_attributes=Dict{String,String}())
    state  = pyconvert(Int, trial.state)
    pruned = state == 2 # TrialState.PRUNED; distinct from "not COMPLETE" (FAIL/RUNNING/WAITING)

    values = if state == 1 # TrialState.COMPLETE — trial.values is a real Python list
        trial_values = pyconvert(Vector{Float32}, trial.values)
        names        = get_names(metric_names, trial_values)
        Dict(k=>v for (v, k) in zip(trial_values, names))
    else # PRUNED — trial.values is Python None, can't be pyconverted; report missing scores
        names = pyconvert(Vector{String}, metric_names)
        Dict(k=>missing for k in names)
    end

    return (
        params = pyconvert(Dict{String,Any}, trial.params),
        user_attrs = pyconvert(Dict{String,Any}, trial.user_attrs),
        values = values,
        study_attributes = study_attributes,
        number = pyconvert(Int, trial.number),
        state = state,
        pruned = pruned,
        id = pyconvert(Int, trial._trial_id),
    )
end

"""
    name_direction(d::Int)

Convert direction integer to string.

# Arguments
- `d::Int`: Direction integer (1 for minimize, 2 for maximize)

# Returns
- `String`: Direction string
"""
function name_direction(d::Int)
    if d == 2
        return "maximize"
    elseif d == 1
        return "minimize"
    else
        return "unknown"
    end
end

"""
    datetime_filter(trial; date)

Filter trials by datetime.

# Arguments
- `trial`: Optuna trial object
- `date`: Date string in "yyyy-mm-dd HH:MM:SS" format

# Returns
- `Bool`: True if trial datetime is after the specified date
"""
datetime_filter = (trial; date) -> begin
    fixed_datetime = Dates.DateTime(date, "yyyy-mm-dd HH:MM:SS")
    trial_datetime = Dates.DateTime(trial.datetime_start)
    trial_datetime > fixed_datetime && return true
    return false
end

"""
    trial_values(trials, func = mean, attr = "accuracy")

Calculate trial values using a specified function.

# Arguments
- `trials`: Vector of trial objects
- `func`: Function to apply (default: mean)
- `attr::String`: Attribute to calculate (default: "accuracy")

# Returns
- `Float64`: Calculated value
"""
function trial_values(trials, func = mean, attr = "accuracy")
    isempty(trials) && return 0.0
    accuracies = [trial.values[attr] for trial in trials]
    return func(accuracies)
end

"""
    trial_attr(trials, func = mean, attr = "accuracy", empty_val = 0)

Calculate trial attributes using a specified function.

# Arguments
- `trials`: Vector of trial objects
- `func`: Function to apply (default: mean)
- `attr::String`: Attribute to calculate (default: "accuracy")
- `empty_val`: Value to return if trials is empty (default: 0)

# Returns
- `Any`: Calculated value
"""
function trial_attr(trials, func = mean, attr = "accuracy", empty_val = 0)
    accuracies = isempty(trials) ? [0] : map(trials) do trial
        haskey(trial.user_attrs, attr) || return 0
        trial.user_attrs[attr]
    end
    return func(accuracies)
end

