"""JournalFileOpenLock that survives transient NFS/CIFS "Stale file handle" errors on the lock file.

Stock `JournalFileOpenLock` creates `<journal>.lock` with open(O_CREAT|O_EXCL) and removes it with
rename+unlink. On the helix share, a lock file created, removed and re-created by many workers in
quick succession sometimes raises OSError(ESTALE, "Stale file handle") from either call; the error
escapes `Study.optimize` and kills the worker (SNN2AC v14 mix-opt job 41910142, worker 6, after
4 h). The lock file itself is fine on the next attempt, so both operations are retried a bounded
number of times with a short exponential back-off before the error is allowed to propagate.

Only ESTALE (and EIO, seen on CIFS) is retried; EEXIST (lock held) keeps the stock handling.
"""
import errno
import time

from optuna.storages.journal import JournalFileOpenLock

TRANSIENT = (errno.ESTALE, errno.EIO)


class RetryJournalFileOpenLock(JournalFileOpenLock):
    def __init__(self, filepath, grace_period=30, retries=12, base_sleep=0.25, max_sleep=8.0):
        super().__init__(filepath, grace_period)
        self._retries = retries
        self._base_sleep = base_sleep
        self._max_sleep = max_sleep

    def _pause(self, attempt):
        time.sleep(min(self._base_sleep * 2 ** attempt, self._max_sleep))

    def acquire(self):
        for attempt in range(self._retries + 1):
            try:
                return super().acquire()
            except OSError as err:
                if err.errno not in TRANSIENT or attempt == self._retries:
                    raise
                self._pause(attempt)

    def release(self):
        retried = False
        for attempt in range(self._retries + 1):
            try:
                return super().release()
            except RuntimeError as err:
                cause = err.__context__
                transient = isinstance(cause, OSError) and cause.errno in TRANSIENT
                gone = retried and isinstance(cause, OSError) and cause.errno == errno.ENOENT
                if gone:
                    return  # a previous attempt removed it before the stale error surfaced
                if not transient or attempt == self._retries:
                    raise
                retried = True
                self._pause(attempt)
