# Optuna setup for a Julia simulation project

Uses PythonCall.jl to drive Optuna from Julia. See `objective_skeleton.jl` for the code
skeleton this document explains.

## Storage: JournalStorage, not an RDB

Use Optuna's file-based `JournalStorage` with an explicit file lock, not a SQLite/Postgres
backend:

```julia
storage = optuna.storages.JournalStorage(
    optuna.storages.journal.JournalFileBackend(
        db_path;
        lock_obj = optuna.storages.journal.JournalFileOpenLock(db_path),
    ),
)
```

(Exact PythonCall call shape depends on your Optuna version — check `optuna.storages` for
the current API; the pattern — one journal file per study, explicit open-lock — is what
matters.)

- One journal file per study (`<study_root>/<study_name>/optimization.db` is a reasonable
  convention). All parallel workers of a run point at the SAME file; Optuna's own file lock
  serializes concurrent access, so you do not need to shard storage across workers.
- `create_study(..., load_if_exists=true)` — re-running with the same `study_name` resumes
  the study rather than erroring or overwriting.
- Keep a separate `study_attributes` dict (configuration name, db name, artifact path, any
  other provenance you want recoverable from the study object itself) and set it via
  `opt_study.set_user_attr`-equivalent calls after `create_study`.

## Samplers

Three are generally useful; keep all three configured even if you default to one, since
switching later costs nothing if the hyperparameters are already there:

- **TPE** (`optuna.samplers.TPESampler`) — good default for a single continuous-parameter
  search without a strong population-based reason to prefer NSGA. Key knobs:
  - `n_startup_trials` — random-sampling phase before the TPE kernel starts fitting; rule of
    thumb ~3x parameter count, rounded up to a multiple of your worker count so the startup
    phase doesn't leave idle workers.
  - `multivariate=true, group=true` — multivariate (MOTPE) sampling, with correlated
    parameters grouped. Worth testing without `group=true` first if you see broadcast-shape
    errors under high worker concurrency — this interaction is sampler-version-sensitive.
  - `constant_liar=true` — penalises in-flight (RUNNING) trials so parallel workers don't
    all sample the same region before any of them finishes. Essential for >1 worker.
  - `n_ei_candidates` — candidates evaluated per expected-improvement step; raise above the
    default (24) for high-dimensional parameter spaces (e.g. 48 for ~30+ parameters).
- **NSGA-II** (`optuna.samplers.NSGAIISampler`) — population-based, `population_size` should
  be a multiple of your worker count (so every worker always has a next individual to
  evaluate) and large enough relative to objective count for crowding-distance diversity.
  Crowding distance degrades for >=4 objectives — prefer NSGA-III there.
- **NSGA-III** (`optuna.samplers.NSGAIIISampler`) — reference-point-based, better for >=4
  objectives. Reference point count `H = C(n_obj + p - 1, p)` where `p = dividing_parameter`;
  pick `p` so `H` (hence a reasonable `population_size`, roughly `H` or a small multiple) is
  sized for your actual worker count.

All three: set an explicit `seed` for any run whose exact trial sequence you want to replay;
leave worker-level parameter *noise* (not the sampler's own exploration) explicitly random if
you want real split-to-split variance in evaluation (see the Pitfalls note in
`cluster_setup.md`'s sibling doc about accidentally reusing one fixed seed across repeats —
that is a real class of bug, not a hypothetical).

## Multi-objective directions

```julia
_metric_directions = pylist(["minimize", "minimize", "maximize", ...])  # one per objective
_metric_names      = pylist(["obj1", "obj2", "obj3", ...])
opt_study = optuna.create_study(;
    directions = _metric_directions,
    study_name,
    storage,
    load_if_exists = true,
    sampler,
)
opt_study.set_metric_names(_metric_names)
```

Keep the objective name -> direction mapping as a `Dict{Symbol,String}` in your project
config, sort by key before building the two parallel lists, and build your trial's return
vector (`metrics(scores; config)` in the skeleton) in that SAME sorted order every time —
order mismatches between the declared directions and the returned vector silently
mis-score every trial without erroring.

## Pruning: a real pruner usually does NOT work for multi-objective studies

Optuna's built-in pruners (`MedianPruner`, `PercentilePruner`, etc.) require
`trial.report()`/`trial.should_prune()`, which raise `NotImplementedError` for
multi-objective studies. If your objective function is one-shot (run the full
simulation, then evaluate every objective at the end — no natural "intermediate value" to
report mid-simulation), there is nothing for a pruner to act on anyway.

**Pattern that works instead: a manual pre-screen that raises `TrialPruned` directly**, before
running the expensive part of the trial:

```julia
if !(cheap_validation_passes(trial_config))
    throw(PyException(optuna.TrialPruned()))  # or your PythonCall exception wrapper
end
```

Put this after the cheapest possible validation step (e.g. a short stability check, a quick
feasibility test) and before the costly simulation/evaluation. This is exactly "network
stability as a hard constraint" or any other go/no-go gate you'd otherwise reach for a
pruner to express — do it by hand, early, cheaply.

## Memory guard

Long TPE/NSGA studies with many trials can grow a worker's RSS unboundedly even with
`gc_after_trial=true` passed to `optimize()` — samplers that keep trial history in memory for
their own model-fitting (TPE's KDE, in particular) are a known, acknowledged-upstream cause
(check your sampler's issue tracker before assuming it's your own code leaking). Add an
Optuna callback that stops the study cleanly BETWEEN trials once RSS crosses a budget
fraction:

```julia
function memory_guard_callback(task_mem_gb; fraction = 0.75)
    limit = fraction * task_mem_gb
    return (study, trial) -> begin
        rss = process_rss_gb()  # read /proc/self/status VmRSS on Linux
        rss > limit && study.stop()
    end
end
```

Optuna calls callbacks AFTER a trial has already completed and been recorded, so this always
stops the loop between trials — never leaves one hanging in RUNNING state.

## Walltime guard

Pass `timeout=` to `optimize()`, derived from your SLURM `--time=` budget minus a safety
margin (e.g. 1h+), via an env var your sbatch script sets explicitly to match `--time`:

```julia
function walltime_timeout_seconds(; margin_seconds = 4500)
    walltime_s = parse(Float64, get(ENV, "WALLTIME_SECONDS", "0"))
    return walltime_s > 0 ? max(walltime_s - margin_seconds, 0.0) : nothing
end
```

Lets `optimize()` stop cleanly before SLURM kills the job outright, rather than losing an
in-flight trial's work to a hard kill.

## Putting it together

```julia
opt_study.optimize(
    objective,
    n_trials = n_trials,
    gc_after_trial = true,
    timeout = walltime_timeout_seconds(),
    callbacks = pylist([memory_guard_callback(task_mem_gb)]),
)
```
