# OptunaLoader.jl

Julia loader and analysis utilities for [Optuna](https://optuna.org/)
hyperparameter-optimization studies, bridged via
[PythonCall.jl](https://github.com/JuliaPy/PythonCall.jl). Extracted from the
`network_models` repo's `loaders/optuna_old.jl` monolith into its own
package/submodule.

## Requirements

- A Python environment with `optuna` installed, resolved automatically by
  PythonCall/CondaPkg on first use (see `.gitignore`'d `.CondaPkg/`)
- Optuna studies stored as journal-file (`JournalFileBackend`) databases

## Capabilities

- **Study/trial access** (`load_study.jl`): `journal_storage`, `retrieve_trials`,
  `retrieve_best_trials`, `retrieve_optuna_study`, `get_all_study_names`,
  `delete_study`, `set_study_attributes!`, `get_study_data`
- **Parameter importance** (`importance.jl`): `retrieve_PedAnova_importance`,
  `aggregate_PDE_importance`, `compute_study_importance` — PedAnova importance
  per study, aggregated by parameter group and objective group across the
  three target quantiles (0.1/0.5/0.8) in both global and local mode
- **Pareto analysis** (`pareto.jl`): `identify_pareto` (non-dominated set from
  a scores matrix), `hypervolume_history`/`plot_hypervolume_history`
- **DataFrame conversion** (`parameters.jl`): `parse_trial`, `params_and_measured`,
  `params_keys`, `trial_values`, `trial_attr`, `datetime_filter`
- **Artifact maintenance** (`artifacts.jl`): `remove_non_best_trials_artifacts`,
  `move_large_artifact_to_scratch`, `remove_recordings_from_best_trials_artifacts`,
  `save_figure_artifact` — disk-space cleanup for large per-trial artifact files

## Scripts

`scripts/optuna-dispatch` is the CLI entry point wrapping the two tasks below:

| Command | Wraps | Purpose |
|---|---|---|
| `optuna-dispatch clean [--dir DIR] [--db FILE] [--rest-time N]` | `remove_trials.jl` | Daemon: every `--rest-time` seconds, deletes artifact files >160MB belonging to non-Pareto trials |
| `optuna-dispatch scratch [options]` | `remove_trials.jl --to-scratch` | Same daemon loop, but moves files >1000MB to scratch instead of deleting |
| `optuna-dispatch check-json FILE` | `check_json.py` | Read-only diagnostic for a journal/JSON file that fails to parse — pinpoints the exact break point and suggests fixes |
| `optuna-dispatch fix-json FILE [--no-backup]` | `fix_json.py` | Repairs trailing commas / unbalanced braces in place (writes a `.bak` first) |

Run `optuna-dispatch help` for full option documentation.

## Known limitations

- **`params_keys`** calls `guess_network`/`guess_connectivity`/`guess_plasticity`
  (parse a network/connectivity/plasticity label out of a study-name string),
  which exist in the original `loaders/optuna_old.jl` but were never carried
  into this package during extraction. Not fixed here: every current caller
  of `params_keys` lives under `papers/SpatioTemporalTransformation/old_stuff/`
  (archived analysis scripts), so nothing live depends on it. If you need it,
  copy the three functions over from `loaders/optuna_old.jl`.
- **`datetime_filter`** uses `Dates.DateTime` but the module doesn't `using Dates`
  — same reasoning: unused by any current caller, so left as-is rather than
  silently patched.
- **`get_study_data`**'s missing-study branch is a bare string literal instead
  of `@warn`/`@error` (a no-op) — inherited unchanged from `loaders/optuna_old.jl`,
  where it has the same issue; not fixed for the same reason (no live callers
  found beyond archived/dead-code scripts).
- **`trial_attr`**'s `empty_val` keyword is accepted but never actually
  returned when `trials` is empty (returns `func([0])` instead) — real but
  harmless in practice, since its only call site always passes an `empty_val`
  that happens to already match what the buggy fallback produces.
