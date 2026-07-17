# %%
using DrWatson
findproject(@__DIR__) |> quickactivate
using Logging
using ArgParse
using OptunaLoader

include(projectdir("loaders", "filesystem.jl"))

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--dir"
            help = "target directory"
            arg_type = String
            default = "models/SpatioTemporalTransformation/optuna_studies"
        "--db"
            help = "target database"
            arg_type = String
            default = nothing
        "--rest-time"
            help = "a positional argument"
            arg_type = Int
            default = 1200
        "--log-level"
            help = "Logging level (Debug, Info, Warn, Error)"
            arg_type = String
            default = "Info"
        "--to-scratch"
            help = "Move large artifacts to scratch"
            action = :store_true
    end

    return parse_args(s)
end


function run_delete_artifacts(path, db)
    # Threads.@threads 
    studies = get_all_study_names(path, db)
    no_trials = true
    for study in studies
        @info "===> Cleaning study: $study"
        some_trials = remove_non_best_trials_artifacts(path, db, study)
        no_trials = no_trials || some_trials
        # some_trial = remove_recordings_from_best_trials_artifacts(path, db, study)
        # no_trials = no_trials || some_trials
    end
    if no_trials
        @info "No more trials to clean."
    end
end

function run_move_artifacts(path, db)
    # Threads.@threads 
    studies = get_all_study_names(path, db)
    no_trials = true
    for study in studies
        @info "===> Cleaning study: $study"
        some_trials = move_large_artifact_to_scratch(path, db, study)
        no_trials = no_trials || some_trials
        # some_trial = remove_recordings_from_best_trials_artifacts(path, db, study)
        # no_trials = no_trials || some_trials
    end
    if no_trials
        @info "No more trials to clean."
    end
end

function main()
    parsed_args = parse_commandline()
    root = "/pasteur/helix/projects/Bathellierlab/User_folders/aquaresi/data"
    dir = joinpath(root, parsed_args["dir"])
    log_level = getfield(Logging, Symbol(parsed_args["log-level"]))
    rest_time = parsed_args["rest-time"]
    db = parsed_args["db"]
    scratch = parsed_args["to-scratch"]

    if scratch
        scratch_folder = "/pasteur/appa/scratch/aquaresi/optuna_artifacts/"
        isdir(scratch_folder) || mkpath(scratch_folder)
        @info "Using scratch folder: $scratch_folder"
    end

    with_logger(ConsoleLogger(stderr, log_level)) do
        while true
            dbs = isnothing(db) ? readdir(dir) : db
            @info "Databases found:" dbs
            for db in dbs
                db == "artifacts" && continue
                !endswith(db, ".db") && continue
                @info "=> Cleaning DB: $db"
                scratch ? run_move_artifacts(dir, db) : run_delete_artifacts(dir, db)
            end
            sleep(rest_time) # Sleep for 10 minutes before checking again
        end
    end
end

# root = "/pasteur/helix/projects/Bathellierlab/User_folders/aquaresi/data"
# dir = joinpath(root, "models/SpatioTemporalTransformation/optuna_studies")
# db = nothing
# with_logger(ConsoleLogger(stderr, Logging.)) do
#     while true
#         dbs = isnothing(db) ? readdir(dir) : db
#         @info "Databases found:" dbs
#         for db in dbs
#             db == "artifacts" && continue
#             @info "=> Cleaning DB: $db"
#             run_cleaning(dir, db)
#         end
#         sleep(rest_time) # Sleep for 10 minutes before checking again
#     end
# end

main()
# s = ArgParseSettings()
# @add_arg_table s begin
# end
# parsed_args = parse_args(ARGS, s)


# root = data_root("helix")

# rest_time = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 1200
# data_path = length()
# @info "Path:" ARGS[1]
# @info "DB:" ARGS[2]
# @info "Run intervals: $(rest_time) seconds to allow for logger setup..."
# data_path = joinpath(root, "data/models/SpatioTemporalTransformation", ARGS[1])

# path = ARGS[1]
# db = ARGS[2]
