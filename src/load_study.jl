
"""
    journal_storage(path; lock = JournalFileOpenLock)

Create a journal storage object for Optuna.

# Arguments
- `path::String`: Path to the journal file
- `lock`: Locking mechanism to use (default: JournalFileOpenLock)

# Returns
- `JournalStorage`: Optuna journal storage object
"""
function journal_storage(path; lock = JournalFileOpenLock)
    lock_obj = lock[](path)
    return JournalStorage[](JournalFileBackend[](path, lock_obj = lock_obj))
end

"""
    get_names(metric_names, metric_values)

Get metric names from Optuna trial.

# Arguments
- `metric_names`: Metric names from Optuna trial
- `metric_values`: Metric values from Optuna trial

# Returns
- `Vector{String}`: Vector of metric names
"""
function get_names(metric_names, metric_values)
    try
        pyconvert(Vector{String}, metric_names)
    catch e
        return string.(0:(length(metric_values)-1))
    end
end

"""
    set_study_attributes!(db_path::String, study_name::String, study_attributes::Dict{String, String})

Set study attributes in Optuna database.

# Arguments
- `db_path::String`: Path to the database file
- `study_name::String`: Name of the study
- `study_attributes::Dict{String, String}`: Study attributes to set
"""
function set_study_attributes!(
    db_path::String,
    study_name::String,
    study_attributes::Dict{String, String},
)
    storage = journal_storage(db_path; lock = JournalFileOpenLock)
    opt_study = optuna[].load_study(study_name = study_name, storage = storage)
    set_study_attributes!(opt_study, study_attributes)
end

function set_study_attributes!(
    opt_study::Py,
    study_attributes::Dict{String, String},
)
    for attribute in keys(study_attributes)
        opt_study.set_user_attr(pyconvert(String, attribute), pyconvert(String, study_attributes[attribute]))
    end
end

"""
    get_all_study_names(data_path, db_name)

Get all study names from an Optuna database.

# Arguments
- `data_path`: Path to the directory containing the database file
- `db_name`: Name of the database file

# Returns
- `Vector{String}`: Vector of study names
"""
function get_all_study_names(data_path, db_name)
    @assert isdir(data_path) "The path $(data_path) does not exist. Please check the paths.yml file"
    if !isfile(joinpath(data_path, db_name))
        @error "The $(db_name) file does not exist. Please check the path."
        return String[]
    end
    storage = journal_storage(joinpath(data_path, db_name); lock = JournalFileOpenLock)
    return pyconvert(Vector{String}, optuna[].study.get_all_study_names(storage))
end

"""
    get_all_study_names(data_path)

Get all study names from an Optuna database.

# Arguments
- `data_path`: Path to the database file

# Returns
- `Vector{String}`: Vector of study names
"""
function get_all_study_names(data_path)
    @assert isfile(data_path) "The path $(data_path) does not exist. Please check the paths.yml file"
    storage = journal_storage(data_path; lock = JournalFileOpenLock)
    return pyconvert(Vector{String}, optuna[].study.get_all_study_names(storage))
end

"""
    delete_study(data_path::String, db_name::String, study_name)

Delete a study from an Optuna database.

# Arguments
- `data_path::String`: Path to the directory containing the database file
- `db_name::String`: Name of the database file
- `study_name`: Name of the study to delete
"""
function delete_study(data_path::String, db_name::String, study_name)
    @assert isdir(data_path) "The path $(data_path) does not exist. Please check the paths.yml file"
    if !isfile(joinpath(data_path, db_name))
        @error "The $(db_name) file does not exist. Please check the path."
        return Dict{String,Any}()
    end
    storage = journal_storage(joinpath(data_path, db_name); lock = JournalFileOpenLock)
    optuna[].delete_study(study_name = study_name, storage = storage)
end

"""
    retrieve_trials(data_path::String, db_name::String, studies::Vector{String}, date=nothing; best=false)

Retrieve trials from multiple studies in an Optuna database.

# Arguments
- `data_path::String`: Path to the directory containing the database file
- `db_name::String`: Name of the Optuna database file
- `studies::Vector{String}`: List of study names to retrieve trials from
- `date`: Optional date filter (default: nothing)
- `best::Bool=false`: If true, retrieves only the best trials for each study

# Returns
- `Dict{String, Any}`: A dictionary mapping study names to their respective trials
"""
function retrieve_trials(
    data_path::String,
    db_name::String,
    studies::Vector{String};
    best = false,
)
    @assert isdir(data_path) "The path $(data_path) does not exist. Please check the paths.yml file"
    if !isfile(joinpath(data_path, db_name))
        @error "The $(db_name) file does not exist. Please check the path."
        return Dict{String,Any}()
    end
    trials = Dict{String,Any}()
    for study in studies
        @info "Retrieving trials for study: $(study)"
        trials[study] = retrieve_trials(data_path, db_name, study; best = best)
    end
    return trials
end

"""
    retrieve_trials(data_path::String, studies::Vector{String}; best = false)

Retrieve trials from multiple studies in an Optuna database.

# Arguments
- `data_path::String`: Path to the database file
- `studies::Vector{String}`: List of study names to retrieve trials from
- `best::Bool=false`: If true, retrieves only the best trials for each study

# Returns
- `Dict{String, Any}`: A dictionary mapping study names to their respective trials
"""
function retrieve_trials(
    data_path::String,
    studies::Vector{String};
    best = false,
)
    @assert isfile(data_path) "The file $(data_path) does not exist. Please check the paths.yml file"
    trials = Dict{String,Any}()
    for study in studies
        @info "Retrieving trials for study: $(study)"
        trials[study] = retrieve_trials(data_path, study; best = best)
    end
    return trials
end

"""
    retrieve_trials(db_path::String, study_name::String; best = false)

Retrieve trials from a single study in an Optuna database.

# Arguments
- `db_path::String`: Path to the database file
- `study_name::String`: Name of the study to retrieve trials from
- `best::Bool=false`: If true, retrieves only the best trials

# Returns
- `Vector`: Vector of trial objects
"""
function retrieve_trials(db_path::String, study_name::String; best = false)
    @assert isfile(db_path) "The path $(db_path) does not exist. Please check the paths.yml file"
    storage = journal_storage(db_path; lock = JournalFileOpenLock)
    opt_study = optuna[].load_study(study_name = study_name, storage = storage)
    study_attributes = opt_study.user_attrs |> x-> pyconvert(Dict{String,String}, x)
    _trials = best ? opt_study.best_trials : opt_study.trials
    trials = []
    for trial in _trials
        state = pyconvert(Int, trial.state)
        if state == 1 || state == 2 # TrialState.COMPLETE or TrialState.PRUNED
            if pyconvert(Int, length(trial.params)) == 0
                # Trial recorded with no params attached (e.g. an enqueued/seeded
                # trial completed without its distributions) — not a real sample,
                # and downstream params_and_measured assumes every trial has the
                # full param set, so it must be dropped here rather than passed on.
                @warn "Skipping trial with no recorded params" study=study_name number=pyconvert(Int, trial.number)
                continue
            end
            push!(trials, parse_trial(trial, opt_study.metric_names, study_attributes))
        end
    end
    return trials
end

"""
    retrieve_trials(data_path::String, db_name::String, study_name::String, date = nothing; best = false)

Retrieve trials from a single study in an Optuna database.

# Arguments
- `data_path::String`: Path to the directory containing the database file
- `db_name::String`: Name of the database file
- `study_name::String`: Name of the study to retrieve trials from
- `date`: Optional date filter (default: nothing)
- `best::Bool=false`: If true, retrieves only the best trials

# Returns
- `Vector`: Vector of trial objects
"""
retrieve_trials(
    data_path::String,
    db_name::String,
    study_name::String,
    date = nothing;
    best = false,
) = retrieve_trials(
    joinpath(data_path, db_name),
    study_name::String;
    best = best,
)

"""
    retrieve_best_trials(data_path, db_name, studies; best = true)

Retrieve best trials from multiple studies in an Optuna database.

# Arguments
- `data_path`: Path to the directory containing the database file
- `db_name`: Name of the database file
- `studies`: List of study names to retrieve trials from
- `best::Bool=true`: If true, retrieves only the best trials for each study

# Returns
- `Dict{String, Any}`: A dictionary mapping study names to their respective trials
"""
retrieve_best_trials(data_path, db_name, studies) =
    retrieve_trials(data_path, db_name, studies; best = true)

function retrieve_optuna_study(db_path, study_name)
    @assert isfile(db_path) "The database file does not exist. Please check the path."
    storage = journal_storage(db_path; lock = JournalFileOpenLock)
    opt_study = optuna[].load_study(study_name = study_name, storage = storage)
    return opt_study
end



"""
    get_study_data(db, studies; percentile = 0.95, attr = "accuracy")

Get study data statistics.

# Arguments
- `db`: Database of trials
- `studies`: List of study names
- `percentile::Float64`: Percentile to use for calculation (default: 0.95)
- `attr::String`: Attribute to calculate (default: "accuracy")

# Returns
- `Tuple`: Tuple containing means, standard deviations, and all points
"""
function get_study_data(db, studies; percentile = 0.95, attr = "accuracy")
    if !haskey(db, studies[1])
        "The database does not contain the studies: $(studies)"
        return zeros(100), zeros(100), zeros(100)
    end
    means = [
        trial_attr(db[condition], x->quantile(x, percentile), attr) for condition in studies
    ]
    stds = [
        trial_attr(db[condition], x->std(filter(y->y>quantile(x, 0.98), x)), attr) for
        condition in studies
    ]
    all_points = [trial_attr(db[condition], collect, attr, [0]) for condition in studies]
    return means, stds, all_points
end


