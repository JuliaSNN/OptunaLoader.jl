module OptunaLoader
    using DrWatson
    using JLD2
    using DataFrames
    using Statistics
    using Random
    using PythonCall
    using ProgressBars


    MB = 1024 * 1024

    # Python imports
    const optuna = Ref{Py}()
    const JournalStorage = Ref{Py}()
    const JournalFileBackend = Ref{Py}()
    const JournalFileOpenLock = Ref{Py}()
    const JournalFileSymlinkLock = Ref{Py}()
    const ImportanceEvaluator = Ref{Py}()
    const Importance = Ref{Py}()
    const SuccessiveHalvingPruner = Ref{Py}()
    const TrialPruned = Ref{Py}()
    const TPESampler = Ref{Py}()
    const FileSystemArtifactStore = Ref{Py}()
    const upload_artifact = Ref{Py}()
    const download_artifact = Ref{Py}()

    function __init__()
        optuna[] = PythonCall.pyimport("optuna")
        # Optuna types
        JournalStorage[] = optuna[].storages.JournalStorage
        JournalFileBackend[] = optuna[].storages.journal.JournalFileBackend
        JournalFileOpenLock[] = optuna[].storages.journal.JournalFileOpenLock
        JournalFileSymlinkLock[] = optuna[].storages.journal.JournalFileSymlinkLock
        ImportanceEvaluator[] = optuna[].importance.PedAnovaImportanceEvaluator
        Importance[] = optuna[].importance
        SuccessiveHalvingPruner[] = optuna[].pruners.SuccessiveHalvingPruner()
        TrialPruned[] = optuna[].exceptions.TrialPruned
        TPESampler[] = optuna[].samplers.TPESampler
        FileSystemArtifactStore[] = optuna[].artifacts.FileSystemArtifactStore
        upload_artifact[] = optuna[].artifacts.upload_artifact
        download_artifact[] = optuna[].artifacts.download_artifact
    end

    include("load_study.jl")
    include("db_copy.jl")
    include("importance.jl")
    include("pareto.jl")
    include("parameters.jl")
    include("artifacts.jl")

    export pylist, pyconvert

    export journal_storage,
        get_names,
        parse_trial,
        name_direction,
        datetime_filter,
        trial_values,
        trial_attr,
        set_study_attributes!,
        get_all_study_names,
        delete_study,
        retrieve_trials,
        retrieve_best_trials,
        retrieve_PedAnova_importance,
        aggregate_PDE_importance,
        compute_study_importance,
        identify_pareto,
        hypervolume_history,
        plot_hypervolume_history,
        retrieve_optuna_study,
        params_and_measured,
        params_keys,
        get_study_data,
        copy_valid,
        remove_non_best_trials_artifacts,
        move_large_artifact_to_scratch,
        remove_recordings_from_best_trials_artifacts,
        pyconvert
    
    export JournalStorage,
        JournalFileBackend,
        JournalFileOpenLock,
        JournalFileSymlinkLock,
        ImportanceEvaluator,
        Importance,
        SuccessiveHalvingPruner,
        TrialPruned,
        TPESampler,
        FileSystemArtifactStore,
        upload_artifact,
        download_artifact

end # module OptunaLoader
