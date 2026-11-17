"""Infrastructure helpers for in-memory job runtime state."""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from config import JOBS_MAX_CACHE, MODEL_IDLE_TIMEOUT_SEC
from pipeline.contracts import TERMINAL_JOB_STATUSES

_MISSING = object()
_T = TypeVar("_T")


class _LRUJobsDict:
    """Keep active jobs pinned and bound only the terminal history cache."""

    def __init__(self, maxsize: int = 200):
        self._d: OrderedDict = OrderedDict()
        self._active: dict = {}
        self._lock = threading.Lock()
        self._maxsize = maxsize

    def __setitem__(self, key, value):
        with self._lock:
            # Legacy callers may still mutate a returned state in place.
            for active_key, state in list(self._active.items()):
                if state.get("status") in TERMINAL_JOB_STATUSES:
                    self._d[active_key] = self._active.pop(active_key)
            self._active.pop(key, None)
            self._d.pop(key, None)
            if value.get("status") in TERMINAL_JOB_STATUSES:
                self._d[key] = value
            else:
                self._active[key] = value
            while len(self._d) > max(self._maxsize, 0):
                self._d.popitem(last=False)

    def __getitem__(self, key):
        with self._lock:
            if key in self._active:
                return self._active[key]
            return self._d[key]

    def __contains__(self, key):
        with self._lock:
            return key in self._active or key in self._d

    def __delitem__(self, key):
        with self._lock:
            if key in self._active:
                del self._active[key]
            else:
                del self._d[key]

    def get(self, key, default=None):
        with self._lock:
            if key in self._active:
                return self._active[key]
            return self._d.get(key, default)

    def pop(self, key, default=_MISSING):
        with self._lock:
            if key in self._active:
                return self._active.pop(key)
            if default is _MISSING:
                return self._d.pop(key)
            return self._d.pop(key, default)


jobs: _LRUJobsDict = _LRUJobsDict(maxsize=JOBS_MAX_CACHE)

# Serialise GPU work: only one transcription runs at a time.
# Concurrent HTTP uploads are fine; they queue here before touching the GPU.
_gpu_sem = threading.Semaphore(1)
_runtime_state_lock = threading.Lock()
_last_gpu_job_finished_at: float | None = None

# In-flight dedup: prevents two concurrent requests with identical audio from
# both burning GPU. Cleared when the job reaches a terminal state.
_in_flight_hashes: dict[str, str] = {}
_in_flight_lock = threading.Lock()
_active_job_ids: set[str] = set()


class JobAdmissionError(RuntimeError):
    """A new unique job would exceed an atomic runtime budget."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class IdleModelUnloadDaemon:
    thread: threading.Thread
    stop_event: threading.Event

    def stop(self, timeout: float = 5.0) -> None:
        self.stop_event.set()
        self.thread.join(timeout=timeout)


def flush_torch_cuda_cache(
    logger: logging.Logger | None = None,
    *,
    phase: str,
) -> None:
    """Best-effort CUDA cache flush used around serialized GPU work."""

    try:
        import torch as _torch

        # Full Python GC can hold the GIL long enough to make FastAPI liveness
        # probes time out after large alignment results. Keep active job
        # boundaries lightweight; the idle-unload path remains the heavy cleanup
        # point because it runs after the GPU pipeline has been idle.
        if phase == "idle-unload":
            _collect_python_gc()
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
    except Exception as exc:  # pragma: no cover - guarded for runtime-only failures
        if logger is not None:
            logger.warning("%s CUDA cache flush failed: %s", phase, exc)


def _collect_python_gc() -> None:
    import gc as _gc

    _gc.collect()


def run_serialized_gpu_work(
    work: Callable[[], _T],
    *,
    logger: logging.Logger | None = None,
) -> _T:
    """Execute GPU work under the shared semaphore with current cache policy."""

    with _gpu_sem:
        flush_torch_cuda_cache(logger, phase="pre-whisper")
        try:
            result = work()
        finally:
            record_gpu_job_finished()
    flush_torch_cuda_cache(logger, phase="post-pipeline")
    return result


def record_gpu_job_finished(*, finished_at: float | None = None) -> None:
    """Record the latest completed GPU job time for idle-unload decisions."""

    global _last_gpu_job_finished_at
    with _runtime_state_lock:
        _last_gpu_job_finished_at = (
            time.monotonic() if finished_at is None else finished_at
        )


def _last_finished_at() -> float | None:
    with _runtime_state_lock:
        return _last_gpu_job_finished_at


def _model_is_loaded(pipeline) -> bool:
    has_loaded_models = getattr(pipeline, "has_loaded_models", None)
    if has_loaded_models is None:
        return False
    return bool(has_loaded_models())


def _is_idle_due(timeout_s: float, *, now: Callable[[], float]) -> bool:
    finished_at = _last_finished_at()
    if finished_at is None:
        return False
    return now() - finished_at >= timeout_s


def unload_idle_pipeline_if_due(
    pipeline,
    *,
    timeout_s: float = MODEL_IDLE_TIMEOUT_SEC,
    now: Callable[[], float] = time.monotonic,
    logger: logging.Logger | None = None,
) -> bool:
    """Unload loaded models when the serialized GPU runtime has been idle.

    The idle state is checked before waiting and again after acquiring the GPU
    semaphore. The second check is what prevents unloading based on a stale
    observation made before a newer queued job completed.
    """

    if timeout_s <= 0 or not _model_is_loaded(pipeline):
        return False
    if not _is_idle_due(timeout_s, now=now):
        return False

    _gpu_sem.acquire()
    try:
        if not _model_is_loaded(pipeline):
            return False
        if not _is_idle_due(timeout_s, now=now):
            return False
        pipeline.unload_models()
        flush_torch_cuda_cache(logger, phase="idle-unload")
        return True
    finally:
        _gpu_sem.release()


def _idle_model_unload_worker(
    pipeline,
    *,
    timeout_s: float,
    interval_s: float,
    stop_event: threading.Event,
    logger: logging.Logger | None,
) -> None:
    while not stop_event.wait(interval_s):
        try:
            unload_idle_pipeline_if_due(
                pipeline,
                timeout_s=timeout_s,
                logger=logger,
            )
        except Exception:
            if logger is not None:
                logger.exception("idle model unload worker tick failed")


def start_idle_model_unload_daemon(
    pipeline,
    *,
    timeout_s: float = MODEL_IDLE_TIMEOUT_SEC,
    interval_s: float | None = None,
    logger: logging.Logger | None = None,
) -> IdleModelUnloadDaemon | None:
    """Start the optional idle-unload daemon, or return None when disabled."""

    if timeout_s <= 0:
        return None

    stop_event = threading.Event()
    tick_interval = (
        interval_s if interval_s is not None else min(max(timeout_s / 2, 1.0), 60.0)
    )
    thread = threading.Thread(
        target=_idle_model_unload_worker,
        kwargs={
            "pipeline": pipeline,
            "timeout_s": timeout_s,
            "interval_s": tick_interval,
            "stop_event": stop_event,
            "logger": logger,
        },
        daemon=True,
        name="idle-model-unload",
    )
    thread.start()
    return IdleModelUnloadDaemon(thread=thread, stop_event=stop_event)


def register_in_flight(file_hash: str, job_id: str) -> str | None:
    """Register hash as in-flight.

    Returns existing job_id if a concurrent job is already processing the same
    content, None if registered successfully.
    """
    with _in_flight_lock:
        if file_hash in _in_flight_hashes:
            return _in_flight_hashes[file_hash]
        _in_flight_hashes[file_hash] = job_id
        return None


def unregister_in_flight(file_hash: str, job_id: str | None = None) -> bool:
    with _in_flight_lock:
        current_job = _in_flight_hashes.get(file_hash)
        if current_job is None:
            return False
        if job_id is not None and current_job != job_id:
            return False
        _in_flight_hashes.pop(file_hash, None)
        return True


def admit_job(
    file_hash: str | None,
    job_id: str,
    *,
    max_active_jobs: int,
    max_in_flight_jobs: int,
    prepare: Callable[[], None],
) -> str | None:
    """Deduplicate, check budgets and publish durable ownership atomically.

    ``prepare`` persists the queued record before another request can see its
    hash. It must not call runtime reservation helpers while this lock is held.
    """

    with _in_flight_lock:
        if file_hash and file_hash in _in_flight_hashes:
            return _in_flight_hashes[file_hash]
        if max_active_jobs > 0 and len(_active_job_ids) >= max_active_jobs:
            raise JobAdmissionError("active_job_budget_exceeded")
        if (
            file_hash
            and max_in_flight_jobs > 0
            and len(_in_flight_hashes) >= max_in_flight_jobs
        ):
            raise JobAdmissionError("in_flight_job_budget_exceeded")
        prepare()
        _active_job_ids.add(job_id)
        if file_hash:
            _in_flight_hashes[file_hash] = job_id
        return None


def release_active_job(job_id: str) -> None:
    with _in_flight_lock:
        _active_job_ids.discard(job_id)


def admission_counts() -> tuple[int, int]:
    with _in_flight_lock:
        return len(_active_job_ids), len(_in_flight_hashes)


__all__ = [
    "JobAdmissionError",
    "_LRUJobsDict",
    "admission_counts",
    "admit_job",
    "flush_torch_cuda_cache",
    "jobs",
    "record_gpu_job_finished",
    "register_in_flight",
    "release_active_job",
    "run_serialized_gpu_work",
    "start_idle_model_unload_daemon",
    "unload_idle_pipeline_if_due",
    "unregister_in_flight",
]
