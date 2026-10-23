# Changelog

All notable changes to OptunaLoader.jl are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow semantic versioning.

## [0.2.0] - 2026-11-15

Integrates the samplers and lock handling developed inside SNN2AC_Optim
(running there since 2026-10-23) on top of the 0.1.0 line.

### Added
- `TPESampler_parallel` (`python/tpe_sampler_parallel.py`, exposed from
  `OptunaLoader`), 2026-10-23: deterministic windowed TPE. Trials are
  sampled in windows so that parallel workers reproduce the same suggestions
  regardless of completion order.
- `RetryJournalFileOpenLock` (`python/retry_journal_lock.py`, exposed from
  `OptunaLoader`), 2026-11-15: journal lock that retries ESTALE/EIO errors on
  the lock file, which occur on CIFS/NFS mounts under heavy concurrency.

### Changed
- `TPESampler_parallel`: PRUNED trials are never placed in the TPE "below"
  (good) group.
- `TPESampler_parallel`: while waiting for earlier trials in the window, the
  journal is polled every 2 s (was 0.2 s; briefly 10 s), balancing journal
  load against the per-step cost in a chain.

### Notes
- The `__pycache__/` ignore from the SNN2AC line was already present in 0.1.0.
- No Manifest.toml is tracked (unchanged convention).

## [0.1.0] - 2026-07-26 and earlier

Initial extraction of OptunaLoader.jl from `network_models`
(`loaders/optuna_old.jl`) into its own package.

- Study/trial access, PedAnova parameter importance, Pareto/hypervolume
  analysis, artifact handling, parameter utilities.
- `scripts/remove_trials.jl` fixes (inverted `no_trials` logic, single `--db`
  handling), `copy_valid` database-hygiene tool, hypervolume progress logging.
- Trials with empty params dropped before importance/retrieval.
- docs: Optuna setup guide and objective skeleton.
- `strip_recordings!` (`src/jld2_strip.jl`): the JLD2 artifact is kept when
  stripping recordings fails (temporary file moved over only on success);
  tests in `test/runtests.jl`.
- Manifest.toml no longer tracked.
