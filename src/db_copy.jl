"""
    copy_valid(data_path::String, db_name::String, study_name::String;
               pruned::Bool = true, clean_db_name::String = "optimization_clean.db")

Copy the valid trials of an Optuna study into a fresh database, dropping
trials stuck in RUNNING/WAITING/FAIL state (e.g. from crashed or timed-out
workers that never reported back — Optuna cannot reap these on its own).

Read-only with respect to the original database: writes only go to a new
file `clean_db_name` in the same directory. Artifact files are never
touched — each trial's `user_attrs` (including any `artifact_*` references)
are copied verbatim as part of the FrozenTrial, so existing artifacts remain
valid for the new study.

# Arguments
- `data_path::String`: Directory containing the original database file
- `db_name::String`: Name of the original database file
- `study_name::String`: Name of the study to copy

# Keyword Arguments
- `pruned::Bool = true`: If true, keep both COMPLETE and PRUNED trials; if
  false, keep only COMPLETE trials
- `clean_db_name::String = "optimization_clean.db"`: Name of the new database
  file to create (in the same directory as `db_name`)

# Returns
- `NamedTuple`: `(kept, dropped, clean_path)` — number of trials copied,
  number of trials dropped, and the path to the new database file
"""
function copy_valid(
    data_path::String,
    db_name::String,
    study_name::String;
    pruned::Bool = true,
    clean_db_name::String = "optimization_clean.db",
)
    @assert isdir(data_path) "The path $(data_path) does not exist. Please check the paths.yml file"
    original_path = joinpath(data_path, db_name)
    @assert isfile(original_path) "The $(db_name) file does not exist. Please check the path."
    clean_path = joinpath(data_path, clean_db_name)
    @assert !isfile(clean_path) "The target $(clean_db_name) already exists — refusing to overwrite. Remove it first or choose a different clean_db_name."

    # flock()-based locking (JournalFileOpenLock, the default) is unreliable
    # over NFS; use the symlink-based lock instead — see the depot-corruption
    # saga elsewhere in this project for the same underlying NFS-locking
    # fragility.
    original_storage = journal_storage(original_path; lock = JournalFileSymlinkLock)
    original_study = optuna[].load_study(study_name = study_name, storage = original_storage)

    clean_storage = journal_storage(clean_path; lock = JournalFileSymlinkLock)
    clean_study = optuna[].create_study(
        study_name = study_name,
        storage = clean_storage,
        directions = original_study.directions,
    )

    # Multi-objective label names (e.g. "response", "CV", "FF", ...) live
    # separately from directions/user_attrs — without copying them the new
    # study falls back to generic "0", "1", "2", ... labels.
    try
        clean_study.set_metric_names(original_study.metric_names)
    catch e
        @warn "Could not copy metric_names to the new study" exception = e
    end

    for (key, value) in pyconvert(Dict{String,String}, original_study.user_attrs)
        clean_study.set_user_attr(key, value)
    end

    valid_states = pruned ? (1, 2) : (1,) # TrialState.COMPLETE = 1, TrialState.PRUNED = 2
    trials = collect(original_study.trials)
    kept = 0
    dropped = 0
    for trial in ProgressBar(trials)
        state = pyconvert(Int, trial.state)
        if state in valid_states
            clean_study.add_trial(trial)
            kept += 1
        else
            dropped += 1
        end
    end

    @info "Clean study created at $(clean_path): kept $(kept) trials, dropped $(dropped) hanging/incomplete trials"
    return (kept = kept, dropped = dropped, clean_path = clean_path)
end
