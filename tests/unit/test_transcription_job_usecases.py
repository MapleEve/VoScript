"""Behavioral regressions for terminal state, cache pressure and ownership."""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import pytest


@pytest.fixture
def job_usecase(monkeypatch, tmp_path):
    worker = importlib.import_module("application.transcription_jobs")
    runtime = importlib.import_module("infra.job_runtime")
    persistence = importlib.import_module("infra.job_persistence")
    cache = runtime._LRUJobsDict(maxsize=1)
    monkeypatch.setattr(worker, "jobs", cache)
    monkeypatch.setattr(worker, "TRANSCRIPTIONS_DIR", tmp_path)
    monkeypatch.setattr(persistence, "TRANSCRIPTIONS_DIR", tmp_path)
    monkeypatch.setattr(worker, "_write_status", persistence._write_status)
    monkeypatch.setattr(worker, "register_hash", lambda *args: None)
    monkeypatch.setattr(worker, "run_serialized_gpu_work", lambda work, **kw: work())
    monkeypatch.setattr(runtime, "_in_flight_hashes", {})
    monkeypatch.setattr(runtime, "_active_job_ids", set())
    monkeypatch.setattr(worker, "unregister_in_flight", runtime.unregister_in_flight)
    monkeypatch.setattr(worker, "release_active_job", runtime.release_active_job)
    return SimpleNamespace(worker=worker, runtime=runtime, cache=cache, root=tmp_path)


def _admit(context, job_id):
    context.runtime.admit_job(
        f"hash-{job_id}",
        job_id,
        max_active_jobs=0,
        max_in_flight_jobs=0,
        prepare=lambda: context.cache.__setitem__(job_id, {"status": "queued"}),
    )


def _run(context, job_id, process):
    context.worker.run_transcription(
        job_id=job_id,
        audio_path=context.root / "synthetic.wav",
        language="en",
        min_speakers=0,
        max_speakers=0,
        pipeline=SimpleNamespace(process=process),
        voiceprint_db=None,
        file_hash=f"hash-{job_id}",
    )


def _successful_process(path, **kwargs):
    kwargs["status_callback"]("converting")
    kwargs["status_callback"]("identifying")
    return {"transcription": {"segments": [], "speaker_map": {}}}


def test_two_workers_complete_with_one_entry_history_cache(job_usecase):
    context = job_usecase
    for job_id in ("first", "second"):
        _admit(context, job_id)
    _run(context, "first", _successful_process)
    assert context.cache["second"]["status"] == "queued"
    _run(context, "second", _successful_process)

    for job_id in ("first", "second"):
        payload = json.loads((context.root / job_id / "status.json").read_text())
        assert payload["status"] == "completed"
    assert context.runtime.admission_counts() == (0, 0)
    assert "first" not in context.cache
    assert context.cache["second"]["result"]["speaker_map"] == {}


@pytest.mark.parametrize("failed", [False, True])
def test_cache_miss_cannot_prevent_terminal_persistence_or_release(job_usecase, failed):
    context = job_usecase
    _admit(context, "missing")
    context.cache.pop("missing")

    def process(path, **kwargs):
        if failed:
            raise RuntimeError("synthetic pipeline failure")
        return _successful_process(path, **kwargs)

    _run(context, "missing", process)

    payload = json.loads((context.root / "missing" / "status.json").read_text())
    assert payload["status"] == ("failed" if failed else "completed")
    assert context.runtime.admission_counts() == (0, 0)


@pytest.mark.parametrize("write_raises", [False, True])
def test_reporting_failure_preserves_original_error_and_releases(
    job_usecase, monkeypatch, write_raises
):
    context = job_usecase
    _admit(context, "failure")
    original = RuntimeError("synthetic original pipeline error")

    def process(path, **kwargs):
        raise original

    def write_status(*args, **kwargs):
        if write_raises:
            raise OSError("synthetic disk error")
        return False

    monkeypatch.setattr(context.worker, "_write_status", write_status)
    with pytest.raises(RuntimeError) as error:
        _run(context, "failure", process)
    assert error.value is original
    assert context.cache["failure"]["status"] == "failed"
    assert context.cache["failure"]["error"] == str(original)
    assert context.runtime.admission_counts() == (0, 0)


def test_completed_status_failure_does_not_register_success_hash(
    job_usecase, monkeypatch
):
    context = job_usecase
    _admit(context, "failure")
    writes = []
    registrations = []

    def write_status(job_id, status, **kwargs):
        writes.append(status)
        return status != "completed"

    monkeypatch.setattr(context.worker, "_write_status", write_status)
    monkeypatch.setattr(
        context.worker, "register_hash", lambda *args: registrations.append(args)
    )
    _run(context, "failure", _successful_process)

    assert writes == ["converting", "identifying", "completed", "failed"]
    assert registrations == []
    assert context.cache["failure"]["status"] == "failed"
    assert context.runtime.admission_counts() == (0, 0)


def test_worker_cache_write_error_still_persists_failure_and_releases(
    job_usecase, monkeypatch
):
    context = job_usecase
    _admit(context, "cache-error")
    original = OSError("synthetic cache publication failure")

    class BrokenCache(dict):
        def __setitem__(self, key, value):
            raise original

    monkeypatch.setattr(context.worker, "jobs", BrokenCache())
    with pytest.raises(OSError) as caught:
        _run(context, "cache-error", lambda path, **kw: {"transcription": {}})
    assert caught.value is original
    payload = json.loads((context.root / "cache-error" / "status.json").read_text())
    assert payload["status"] == "failed"
    assert payload["error"] == str(original)
    assert context.runtime.admission_counts() == (0, 0)


def test_worker_finally_does_not_release_replacement_hash_owner(job_usecase):
    context = job_usecase
    _admit(context, "old")

    def process(path, **kwargs):
        assert context.runtime.unregister_in_flight("hash-old", "old") is True
        context.runtime.admit_job(
            "hash-old",
            "new",
            max_active_jobs=0,
            max_in_flight_jobs=0,
            prepare=lambda: context.cache.__setitem__("new", {"status": "queued"}),
        )
        raise RuntimeError("synthetic superseded worker failure")

    _run(context, "old", process)
    assert context.runtime.admission_counts() == (1, 1)
    assert context.runtime.register_in_flight("hash-old", "retry") == "new"
    assert context.cache["new"]["status"] == "queued"
