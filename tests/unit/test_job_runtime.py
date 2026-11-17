"""Unit tests for GPU runtime serialization helpers."""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import pytest

import infra.job_runtime as job_runtime


@pytest.fixture(autouse=True)
def _reset_idle_runtime_state(monkeypatch):
    monkeypatch.setattr(job_runtime, "_last_gpu_job_finished_at", None, raising=False)


class _FakePipeline:
    def __init__(self, *, loaded: bool = True):
        self.loaded = loaded
        self.unload_calls = 0

    def has_loaded_models(self) -> bool:
        return self.loaded

    def unload_models(self) -> None:
        self.unload_calls += 1
        self.loaded = False


def test_run_serialized_gpu_work_flushes_before_and_after_success(monkeypatch):
    events = []

    monkeypatch.setattr(
        job_runtime,
        "flush_torch_cuda_cache",
        lambda logger=None, *, phase: events.append(phase),
    )

    result = job_runtime.run_serialized_gpu_work(lambda: events.append("work") or "ok")

    assert result == "ok"
    assert events == ["pre-whisper", "work", "post-pipeline"]


def test_run_serialized_gpu_work_skips_post_flush_on_error(monkeypatch):
    events = []

    monkeypatch.setattr(
        job_runtime,
        "flush_torch_cuda_cache",
        lambda logger=None, *, phase: events.append(phase),
    )

    with pytest.raises(RuntimeError, match="boom"):
        job_runtime.run_serialized_gpu_work(
            lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        )

    assert events == ["pre-whisper"]


def test_run_serialized_gpu_work_releases_semaphore_after_error(monkeypatch):
    events = []

    monkeypatch.setattr(
        job_runtime,
        "flush_torch_cuda_cache",
        lambda logger=None, *, phase: events.append(phase),
    )

    with pytest.raises(RuntimeError, match="boom"):
        job_runtime.run_serialized_gpu_work(
            lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        )

    result = job_runtime.run_serialized_gpu_work(lambda: events.append("retry") or "ok")

    assert result == "ok"
    assert events == ["pre-whisper", "pre-whisper", "retry", "post-pipeline"]


def test_flush_torch_cuda_cache_skips_python_gc_for_active_job_phases(monkeypatch):
    events = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: events.append("empty_cache"),
        )
    )

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        job_runtime,
        "_collect_python_gc",
        lambda: events.append("gc_collect"),
    )

    job_runtime.flush_torch_cuda_cache(phase="post-pipeline")

    assert events == ["empty_cache"]


def test_flush_torch_cuda_cache_keeps_full_gc_for_idle_unload(monkeypatch):
    events = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: events.append("empty_cache"),
        )
    )

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        job_runtime,
        "_collect_python_gc",
        lambda: events.append("gc_collect"),
    )

    job_runtime.flush_torch_cuda_cache(phase="idle-unload")

    assert events == ["gc_collect", "empty_cache"]


def test_idle_unload_daemon_disabled_when_timeout_zero():
    pipeline = _FakePipeline(loaded=True)

    daemon = job_runtime.start_idle_model_unload_daemon(
        pipeline,
        timeout_s=0,
        interval_s=0.01,
    )
    unloaded = job_runtime.unload_idle_pipeline_if_due(
        pipeline,
        timeout_s=0,
        now=lambda: 100.0,
    )

    assert daemon is None
    assert unloaded is False
    assert pipeline.unload_calls == 0


def test_idle_unload_runs_after_loaded_model_exceeds_timeout(monkeypatch):
    pipeline = _FakePipeline(loaded=True)
    events = []

    monkeypatch.setattr(
        job_runtime,
        "flush_torch_cuda_cache",
        lambda logger=None, *, phase: events.append(phase),
    )

    job_runtime.record_gpu_job_finished(finished_at=10.0)

    unloaded = job_runtime.unload_idle_pipeline_if_due(
        pipeline,
        timeout_s=5,
        now=lambda: 16.0,
    )

    assert unloaded is True
    assert pipeline.loaded is False
    assert pipeline.unload_calls == 1
    assert events == ["idle-unload"]


def test_idle_unload_skips_when_no_model_is_loaded():
    pipeline = _FakePipeline(loaded=False)
    job_runtime.record_gpu_job_finished(finished_at=10.0)

    unloaded = job_runtime.unload_idle_pipeline_if_due(
        pipeline,
        timeout_s=5,
        now=lambda: 16.0,
    )

    assert unloaded is False
    assert pipeline.unload_calls == 0


def test_idle_unload_rechecks_idle_after_waiting_for_gpu_semaphore(monkeypatch):
    pipeline = _FakePipeline(loaded=True)
    job_runtime.record_gpu_job_finished(finished_at=10.0)

    class UpdatingSemaphore:
        def __init__(self):
            self.released = False

        def acquire(self):
            # Simulate the daemon waiting while a newer job finishes. The
            # post-acquire idle check must see this fresher completion time.
            job_runtime.record_gpu_job_finished(finished_at=98.0)
            return True

        def release(self):
            self.released = True

    semaphore = UpdatingSemaphore()
    monkeypatch.setattr(job_runtime, "_gpu_sem", semaphore)

    unloaded = job_runtime.unload_idle_pipeline_if_due(
        pipeline,
        timeout_s=5,
        now=lambda: 100.0,
    )

    assert unloaded is False
    assert pipeline.unload_calls == 0
    assert semaphore.released is True


@pytest.mark.parametrize("cache_size, active_count", [(200, 201), (1, 2), (0, 2)])
def test_active_jobs_are_not_evicted_by_history_limit(cache_size, active_count):
    cache = job_runtime._LRUJobsDict(maxsize=cache_size)
    for index in range(active_count):
        cache[str(index)] = {"status": "queued"}
    cache["history"] = {"status": "completed", "result": {}}

    for index in range(active_count):
        assert cache[str(index)]["status"] == "queued"
        cache[str(index)] = {"status": "transcribing"}
    if cache_size == 0:
        assert cache.get("history") is None
    else:
        assert "history" in cache


def test_terminal_history_is_bounded_without_evicting_active_job():
    cache = job_runtime._LRUJobsDict(maxsize=1)
    cache["active"] = {"status": "queued"}
    cache["first"] = {"status": "completed", "result": {}}
    cache["second"] = {"status": "failed", "error": "synthetic failure"}

    assert "active" in cache
    assert "first" not in cache
    assert cache["second"]["status"] == "failed"
    cache["active"] = {"status": "completed", "result": {}}
    assert "second" not in cache
    assert cache.pop("active")["status"] == "completed"
    assert cache.pop("missing", None) is None
    with pytest.raises(KeyError):
        cache.pop("missing")


def test_legacy_in_place_terminal_update_enters_bounded_history():
    cache = job_runtime._LRUJobsDict(maxsize=1)
    cache["first"] = {"status": "queued"}
    cache["first"]["status"] = "failed"
    cache["second"] = {"status": "completed"}
    assert "first" not in cache


def test_admission_atomically_limits_concurrent_unique_hashes(monkeypatch):
    monkeypatch.setattr(job_runtime, "_in_flight_hashes", {})
    monkeypatch.setattr(job_runtime, "_active_job_ids", set())
    barrier = Barrier(8)
    prepared = []

    def submit(index):
        barrier.wait(timeout=10)
        try:
            job_runtime.admit_job(
                f"hash-{index}",
                f"job-{index}",
                max_active_jobs=200,
                max_in_flight_jobs=4,
                prepare=lambda: prepared.append(index),
            )
            return True
        except job_runtime.JobAdmissionError:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(submit, range(8)))
    assert sum(outcomes) == 4
    assert len(prepared) == 4
    assert job_runtime.admission_counts() == (4, 4)


def test_admission_same_hash_has_one_durable_owner_at_capacity(monkeypatch):
    monkeypatch.setattr(job_runtime, "_in_flight_hashes", {})
    monkeypatch.setattr(job_runtime, "_active_job_ids", set())
    barrier = Barrier(8)
    prepared = []

    def submit(index):
        barrier.wait(timeout=10)
        job_id = f"job-{index}"
        existing = job_runtime.admit_job(
            "same-hash",
            job_id,
            max_active_jobs=1,
            max_in_flight_jobs=1,
            prepare=lambda: prepared.append(job_id),
        )
        return existing or job_id

    with ThreadPoolExecutor(max_workers=8) as pool:
        owners = list(pool.map(submit, range(8)))
    assert len(prepared) == 1
    assert set(owners) == set(prepared)
    assert job_runtime.admission_counts() == (1, 1)


def test_failed_prepare_does_not_publish_hash_or_consume_capacity(monkeypatch):
    monkeypatch.setattr(job_runtime, "_in_flight_hashes", {})
    monkeypatch.setattr(job_runtime, "_active_job_ids", set())

    def fail():
        raise OSError("synthetic persistence error")

    with pytest.raises(OSError, match="synthetic persistence error"):
        job_runtime.admit_job(
            "hash", "job", max_active_jobs=1, max_in_flight_jobs=1, prepare=fail
        )
    assert job_runtime.admission_counts() == (0, 0)


def test_stale_worker_cannot_release_new_hash_owner(monkeypatch):
    monkeypatch.setattr(job_runtime, "_in_flight_hashes", {"hash": "new-job"})
    assert job_runtime.unregister_in_flight("hash", "old-job") is False
    assert job_runtime.register_in_flight("hash", "retry") == "new-job"
