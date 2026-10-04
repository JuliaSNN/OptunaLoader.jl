# Objective skeleton for a Julia + Optuna (PythonCall) multi-objective study.
#
# Copy this file into a new project and fill in the marked sections. It has no
# project-specific science in it — the shape is: a per-trial config builder
# (`set_config`), a cheap stability pre-screen + expensive run + score
# (`run_config`), and the glue (`Objective`, `metrics`) Optuna calls. See
# `optuna_setup.md` for the reasoning behind each piece.

using PythonCall  # or OptunaLoader.jl / whatever your project's Optuna wrapper is called

# ── Parameter specs ──────────────────────────────────────────────────────────
# Declare every optimized parameter once, as a plain spec. `range` is (lo, hi);
# `log` = true means log-uniform sampling (use for scale parameters spanning
# orders of magnitude); `type` is :float or :int.
struct ParamSpec
    name::Symbol
    type::Symbol          # :float or :int
    range::Tuple{Float64,Float64}
    log::Bool
    group::Symbol         # for organizing / filtering specs by subsystem
end

function suggest_from_spec(trial, spec::ParamSpec)
    name = string(spec.name)
    if spec.type == :float
        return pyconvert(Float64, trial.suggest_float(name, spec.range[1], spec.range[2]; log = spec.log))
    elseif spec.type == :int
        return pyconvert(Int, trial.suggest_int(name, Int(spec.range[1]), Int(spec.range[2])))
    else
        error("Unknown parameter type: $(spec.type)")
    end
end

specs_for(all_specs, groups...) = filter(s -> s.group in groups, all_specs)

function suggest_group(trial, all_specs, groups...)
    return Dict(s.name => suggest_from_spec(trial, s) for s in specs_for(all_specs, groups...))
end

# ── Per-trial config ─────────────────────────────────────────────────────────
# Deep-copy the base config, draw fresh per-trial random seeds (IMPORTANT: use
# a genuinely random seed per trial/repeat, e.g. `rand(1:typemax(Int))`, NOT a
# fixed or trial-number-derived one, unless you specifically want
# reproducible replay — a fixed seed reused across "repeats" silently turns
# reported error bars into numerical noise instead of real sampling
# variance, which is an easy, hard-to-notice bug).
function set_config(trial, base_config)
    config = deepcopy(base_config)

    seed = rand(1:typemax(Int32))
    # TODO: wire `seed` into whatever your simulation's RNG entry points are

    sampled = suggest_group(trial, base_config.param_specs, :all)  # TODO: real group names
    # TODO: merge `sampled` into `config` at the right nested fields

    return config
end

# ── Metric bookkeeping ───────────────────────────────────────────────────────
# `config.targets` is a Dict{Symbol,String} of objective_name => "minimize"/
# "maximize". Keep everything sorted by key so the directions list built at
# study-creation time and the vector returned per-trial are in the SAME
# order — this is the #1 place a silent mis-score bug hides.
function metrics(scores = nothing; config)
    sorted_keys = sort(collect(keys(config.targets)))
    if isnothing(scores)
        directions = pylist([config.targets[k] for k in sorted_keys])
        names      = pylist(string.(sorted_keys))
        return directions, names
    else
        return [scores[k] for k in sorted_keys]
    end
end

# ── Per-trial run: cheap gate, then expensive simulation + evaluation ───────
function run_config(trial, config)
    # 1. Build whatever your simulation needs from `config`.
    # model = init_model(config)

    # 2. CHEAP stability/validity pre-screen. This replaces a real Optuna
    #    pruner (which does not work for multi-objective studies — see
    #    optuna_setup.md). Keep this step as cheap as possible; it exists to
    #    avoid paying for the expensive step 3 on a trial that's going to be
    #    thrown out anyway.
    stable = true  # TODO: real cheap check
    if !stable
        throw(PyException(optuna.TrialPruned()))
    end

    # 3. The expensive part: run the actual simulation/experiment.
    # result = run_simulation(config)

    # 4. Evaluate every objective into a Dict{Symbol,Any}.
    scores = Dict{Symbol,Any}()
    # scores[:obj1] = ...
    # scores[:obj2] = ...

    return metrics(scores; config)
end

# ── Objective glue ───────────────────────────────────────────────────────────
mutable struct Objective
    config::NamedTuple
end

function (obj::Objective)(trial)
    config = set_config(trial, obj.config)
    return run_config(trial, config)
end

# ── Guards (see optuna_setup.md) ─────────────────────────────────────────────
function process_rss_gb()
    for line in eachline("/proc/self/status")
        if startswith(line, "VmRSS:")
            return parse(Int, split(line)[2]) / 1024^2
        end
    end
    return NaN
end

function memory_guard_callback(task_mem_gb::Real; fraction::Real = 0.75)
    limit = fraction * task_mem_gb
    return (study, trial) -> begin
        if process_rss_gb() > limit
            study.stop()
        end
    end
end

function walltime_timeout_seconds(; margin_seconds = 4500)
    walltime_s = parse(Float64, get(ENV, "WALLTIME_SECONDS", "0"))
    return walltime_s > 0 ? max(walltime_s - margin_seconds, 0.0) : nothing
end

function guarded_optimize!(opt_study, objective; n_trials, task_mem_gb = 60.0, mem_fraction = 0.75)
    opt_study.optimize(
        objective,
        n_trials = n_trials,
        gc_after_trial = true,
        timeout = walltime_timeout_seconds(),
        callbacks = pylist([memory_guard_callback(task_mem_gb; fraction = mem_fraction)]),
    )
end

# ── Study setup ───────────────────────────────────────────────────────────────
# TODO: fill in db_path, study_name, sampler choice/hyperparameters per
# optuna_setup.md, then:
#
# storage = <JournalStorage as in optuna_setup.md>
# directions, names = metrics(; config)
# opt_study = optuna.create_study(; directions, study_name, storage, load_if_exists = true, sampler)
# opt_study.set_metric_names(names)
# opt_objective = Objective(config)
# guarded_optimize!(opt_study, opt_objective; n_trials = 150)
