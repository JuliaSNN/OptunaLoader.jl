"""TPESampler_parallel: TPE whose proposals do not depend on how many trials run in parallel.

Scheme ("lagged history", window W).  For trial number n:
  * HISTORY = finished trials with number <= n-W-1 (COMPLETE/PRUNED enter the Parzen split; FAIL is
    ignored, as in stock Optuna).
  * LIARS   = trials n-W .. n-1, whatever their real state, treated exactly as stock
    constant_liar treats RUNNING trials (copied with state=RUNNING -> "above" group, no value).
  * Everything else (number >= n) is invisible.
Before sampling, the ask waits (polls storage) until all HISTORY trials are finished and every
LIAR has its full parameter set stored.  Dead trials must be closed (FAIL) by external tooling,
which also unblocks the wait.  The RNG of trial n is seeded from (base_seed, n) so the proposal is
a pure function of (base_seed, n, storage content of trials < n) -> independent of the process
asking, the time, and the number of workers.  Start-up (random) phase: a trial is random while
fewer than n_startup_trials HISTORY trials are COMPLETE/PRUNED (so TPE starts at about
n = n_startup_trials + W + 1; stock async with k in flight has the same effective delay ~ k).
No waiting happens in the random phase.

Optuna 4.9.0 internals relied upon (samplers/_tpe/sampler.py, pin to ==4.9.0 and re-check on bump):
  * Trial.__init__ calls sampler.before_trial(study, frozen_trial) first (trial/_trial.py:65):
    the hook used for wait + reseed + window build.
  * self._rng (LazyRandomState, l.415, used in _sample l.623) and self._random_sampler (l.416,
    start-up and fallback sampling) are replaced per trial.
  * _sample_relative (l.510), sample_independent (l.524) and _sample (l.593) read trials only via
    study._get_trials(...) -> we hand them a thin study view returning the window; _sample's
    constant_liar branch (l.597-608) and _split_trials (l.757: RUNNING -> above group) then do the
    rest unchanged.  Liar params come from system attrs "tpe:relative_params:*" via _get_params
    (l.558) for RUNNING trials -> constant_liar=True is forced.
  * infer_relative_search_space is NOT windowed: assumes a fixed search space (true for SNNOptim).
  * FrozenTrial.state is a plain attribute (copy.copy + assignment gives a RUNNING copy).
  * PRUNED trials always go to the "above" group (module functions _split_pruned_trials, l.866, and
    _split_infeasible_trials, l.887 -- a PRUNED trial with violated constraints is classified
    "infeasible" before its state is looked at -- replaced when the first TPESampler_parallel is
    built; process-wide).  Stock TPE fills "below" with PRUNED trials once the feasible COMPLETE
    trials run out, and _calculate_weights_below_for_multi_objective (l.895) then crashes on their
    `values = None` ("inhomogeneous shape") whenever their stored constraints are <= 0 -- the v8
    campaign hit this at trial 751 (97% of the history PRUNED by the hard rate limits, pruned trials
    with Exc < Exc_min have feasible constraint values).  Pruned = implausible rates = bad region,
    so "above" is also the right semantics.
"""
from __future__ import annotations

import copy
import hashlib
import logging
import random
import time

import optuna
from optuna.samplers import TPESampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.samplers._random import RandomSampler
from optuna.trial import TrialState

_log = logging.getLogger("tpe_sampler_parallel")


def _pruned_always_above(trials, study, n_below):
    """Replacement for optuna.samplers._tpe.sampler._split_pruned_trials: no PRUNED trial in "below"."""
    return [], list(trials)


def _infeasible_complete_only_below(trials, n_below):
    """Replacement for _split_infeasible_trials: PRUNED trials with violated constraints (classified
    infeasible before their state is looked at) never enter "below" either; COMPLETE infeasible trials
    are split as in stock (least violation first)."""
    from optuna.samplers._tpe.sampler import _get_infeasible_trial_score
    complete = sorted((t for t in trials if t.state == TrialState.COMPLETE), key=_get_infeasible_trial_score)
    pruned = [t for t in trials if t.state != TrialState.COMPLETE]
    return complete[:n_below], complete[n_below:] + pruned


def _patch_pruned_split():
    import optuna.samplers._tpe.sampler as _m
    if _m._split_pruned_trials is not _pruned_always_above:
        _m._split_pruned_trials = _pruned_always_above
        _m._split_infeasible_trials = _infeasible_complete_only_below


class _WindowView:
    """Study proxy: _get_trials returns the precomputed window; everything else is delegated."""

    def __init__(self, study, trials):
        self._study, self._trials = study, trials

    def __getattr__(self, name):
        return getattr(self._study, name)

    def _get_trials(self, deepcopy=True, states=None, use_cache=False):
        t = [x for x in self._trials if states is None or x.state in states]
        return copy.deepcopy(t) if deepcopy else t


class TPESampler_parallel(TPESampler):
    def __init__(self, *, seed: int, window: int = 100, wait_timeout_s: float = 6 * 3600,
                 poll_s: float = 2.0, on_trial_created=None, **kwargs):
        # poll_s: every poll re-reads the journal tail; ~100 waiting workers at the mock's 0.2 s is
        # pointless load on a network file system, but the poll interval adds directly to each step
        # of the serial ask chain (trial n needs liar n-1's params), so keep it a few seconds:
        # 2 s (jittered 1-3 s).
        if seed is None:
            raise ValueError("TPESampler_parallel needs an explicit base seed")
        kwargs["constant_liar"] = True  # needed: stores/reads relative params of pending trials
        _patch_pruned_split()  # see module docstring: PRUNED never in "below"
        super().__init__(seed=seed, **kwargs)
        self._base_seed, self._W = int(seed), int(window)
        self._timeout, self._poll = wait_timeout_s, poll_s
        self._on_trial_created = on_trial_created  # called with the trial before any wait
        self._win_key = None
        self._win_trials: list = []
        self.last_wait_s = 0.0  # diagnostics: seconds spent waiting in the last ask

    # -- determinism ---------------------------------------------------------------------------
    def trial_seed(self, n: int) -> int:
        h = hashlib.blake2b(f"{self._base_seed}:{n}".encode(), digest_size=8).digest()
        return int.from_bytes(h, "little") % (2**32)

    def reseed_rng(self) -> None:  # study.optimize(n_jobs>1) calls this; we never want it
        pass

    # -- window ----------------------------------------------------------------------------------
    def before_trial(self, study, trial):
        if self._on_trial_created is not None:
            self._on_trial_created(trial)
        n = trial.number
        s = self.trial_seed(n)
        self._rng = LazyRandomState(s)
        self._random_sampler = RandomSampler(seed=s)
        self._win_key = (study._study_id, n)
        self._win_trials = self._build_window(study, trial)

    def _build_window(self, study, trial):
        n, W = trial.number, self._W
        self.last_wait_s = 0.0
        if n - W < self._n_startup_trials:  # even a fully finished history is too small
            return []
        t0 = time.time()
        storage, sid = study._storage, study._study_id
        while True:
            allt = storage.get_all_trials(sid, deepcopy=False)
            hist = [t for t in allt if t.number <= n - W - 1]
            liars = [t for t in allt if n - W <= t.number < n]
            unfinished = [t.number for t in hist if not t.state.is_finished()]
            ref = max((len(t.params) for t in hist if t.state in (TrialState.COMPLETE, TrialState.PRUNED)),
                      default=0)
            lcopies = []
            for t in liars:
                c = copy.copy(t)
                c.state = TrialState.RUNNING
                lcopies.append(c)
            incomplete = [c.number for c, t in zip(lcopies, liars)
                          if not t.state.is_finished() and len(self._get_params(c)) < ref]
            if not unfinished and not incomplete:
                break
            if time.time() - t0 > self._timeout:
                msg = (f"TPESampler_parallel: trial {n} waited {self._timeout:.0f}s; unfinished history "
                       f"trials {unfinished[:10]} (+{max(0, len(unfinished) - 10)}), liars without params "
                       f"{incomplete[:10]}. Close dead trials as FAIL to unblock.")
                _log.error(msg)
                storage.set_trial_state_values(trial._trial_id, TrialState.FAIL)
                raise RuntimeError(msg)
            time.sleep(self._poll * (0.5 + random.random()))  # jitter: desynchronise pollers
        self.last_wait_s = time.time() - t0
        hist = [t for t in hist if t.state in (TrialState.COMPLETE, TrialState.PRUNED)]
        return hist + lcopies  # already sorted by number

    def _view(self, study, trial):
        assert self._win_key == (study._study_id, trial.number), "before_trial was not called"
        return _WindowView(study, self._win_trials)

    def sample_relative(self, study, trial, search_space):
        return super().sample_relative(self._view(study, trial), trial, search_space)

    def sample_independent(self, study, trial, param_name, param_distribution):
        return super().sample_independent(self._view(study, trial), trial, param_name, param_distribution)
