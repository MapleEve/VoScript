"""Admission and route failure regressions using synthetic uploads only."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def submission(monkeypatch, app_client):
    router = importlib.import_module("api.routers.transcriptions")
    runtime = importlib.import_module("infra.job_runtime")
    admission = importlib.import_module("application.admission")
    monkeypatch.setattr(runtime, "_in_flight_hashes", {})
    monkeypatch.setattr(runtime, "_active_job_ids", set())
    monkeypatch.setattr(router, "TRANSCRIPTION_MAX_ACTIVE_JOBS", 200)
    monkeypatch.setattr(router, "TRANSCRIPTION_MAX_IN_FLIGHT_JOBS", 4)
    monkeypatch.setattr(router, "TRANSCRIPTION_MIN_FREE_DISK_BYTES", 1024)
    monkeypatch.setattr(
        admission.shutil, "disk_usage", lambda path: SimpleNamespace(free=2048)
    )
    started = []

    class DeferredThread:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start(self):
            started.append(self.kwargs)

    monkeypatch.setattr(router, "Thread", DeferredThread)
    return SimpleNamespace(
        client=app_client,
        router=router,
        runtime=runtime,
        admission=admission,
        started=started,
    )


def _submit(context, content=b"synthetic audio"):
    return context.client.post(
        "/api/transcribe", files={"file": ("synthetic.wav", content, "audio/wav")}
    )


def test_duplicate_reuses_durable_job_at_capacity_and_under_disk_pressure(
    submission, monkeypatch
):
    context = submission
    monkeypatch.setattr(context.router, "TRANSCRIPTION_MAX_ACTIVE_JOBS", 1)
    monkeypatch.setattr(context.router, "TRANSCRIPTION_MAX_IN_FLIGHT_JOBS", 1)
    first = _submit(context)
    assert first.status_code == 200
    first_id = first.json()["id"]
    assert (context.router.TRANSCRIPTIONS_DIR / first_id / "status.json").is_file()

    monkeypatch.setattr(
        context.admission.shutil, "disk_usage", lambda path: SimpleNamespace(free=0)
    )
    duplicate = _submit(context)
    assert duplicate.status_code == 200
    assert duplicate.json() == {
        "id": first_id,
        "status": "queued",
        "deduplicated": True,
    }
    assert len(context.started) == 1
    assert context.runtime.admission_counts() == (1, 1)
    assert len(list(context.router.UPLOADS_DIR.iterdir())) == 1


@pytest.mark.parametrize("budget", ["active", "inflight", "disk", "disk_error"])
def test_unique_upload_rejection_leaves_no_new_runtime_or_files(
    submission, monkeypatch, budget
):
    context = submission
    if budget == "disk":
        monkeypatch.setattr(
            context.admission.shutil, "disk_usage", lambda path: SimpleNamespace(free=0)
        )
    elif budget == "disk_error":

        def fail(path):
            raise OSError("synthetic disk error")

        monkeypatch.setattr(context.admission.shutil, "disk_usage", fail)
    else:
        name = (
            "TRANSCRIPTION_MAX_ACTIVE_JOBS"
            if budget == "active"
            else "TRANSCRIPTION_MAX_IN_FLIGHT_JOBS"
        )
        monkeypatch.setattr(context.router, name, 1)
        assert _submit(context, b"first synthetic upload").status_code == 200
    before = context.runtime.admission_counts()
    before_jobs = dict(context.router.jobs)
    before_hashes = dict(context.runtime._in_flight_hashes)
    before_active = set(context.runtime._active_job_ids)
    before_started = list(context.started)
    before_uploads = set(context.router.UPLOADS_DIR.iterdir())
    before_records = set(context.router.TRANSCRIPTIONS_DIR.iterdir())
    index_lock = context.router.TRANSCRIPTIONS_DIR / "hash_index.json.lock"
    lock_inode = index_lock.stat().st_ino if index_lock.exists() else None
    response = _submit(context, b"second synthetic upload")
    assert response.status_code == 503
    assert context.runtime.admission_counts() == before
    assert dict(context.router.jobs) == before_jobs
    assert context.runtime._in_flight_hashes == before_hashes
    assert context.runtime._active_job_ids == before_active
    assert context.started == before_started
    assert set(context.router.UPLOADS_DIR.iterdir()) == before_uploads
    # Hash lookup keeps this flock inode after release; it is not a job artifact.
    assert set(context.router.TRANSCRIPTIONS_DIR.iterdir()) == before_records | {
        index_lock
    }
    assert index_lock.is_file()
    if lock_inode is not None:
        assert index_lock.stat().st_ino == lock_inode


@pytest.mark.parametrize("mode", ["false", "raise", "cache", "upload", "lookup"])
def test_bootstrap_failures_release_all_transient_state(submission, monkeypatch, mode):
    context = submission
    if mode in {"false", "raise"}:

        def write_status(*args, **kwargs):
            if mode == "raise":
                raise OSError("synthetic status failure")
            return False

        monkeypatch.setattr(context.router, "_write_status", write_status)
    elif mode == "cache":

        class BrokenCache(dict):
            def __setitem__(self, key, value):
                raise OSError("synthetic cache failure")

        monkeypatch.setattr(context.router, "jobs", BrokenCache())
    elif mode == "lookup":

        def fail_lookup(file_hash):
            raise OSError("synthetic hash lookup failure")

        monkeypatch.setattr(context.router, "lookup_hash", fail_lookup)
    else:

        async def fail_upload(file, path, *args):
            path.write_bytes(b"partial synthetic upload")
            raise OSError("synthetic upload failure")

        monkeypatch.setattr(context.router, "save_upload_and_hash", fail_upload)

    if mode == "false":
        assert _submit(context).status_code == 503
    else:
        with pytest.raises(OSError):
            _submit(context)
    assert context.runtime.admission_counts() == (0, 0)
    assert list(context.router.UPLOADS_DIR.iterdir()) == []
    assert list(context.router.TRANSCRIPTIONS_DIR.glob("*/status.json")) == []
    assert context.started == []


@pytest.mark.parametrize("factory_failure", [False, True])
@pytest.mark.parametrize("report_failure", [False, True])
def test_thread_creation_or_start_failure_releases_even_if_reporting_fails(
    submission, monkeypatch, factory_failure, report_failure
):
    context = submission

    class BrokenThread:
        def __init__(self, **kwargs):
            if factory_failure:
                raise RuntimeError("synthetic thread creation failure")

        def start(self):
            raise RuntimeError("synthetic thread start failure")

    original_writer = context.router._write_status

    def write_status(job_id, status, **kwargs):
        if status == "failed" and report_failure:
            raise OSError("synthetic reporting failure")
        return original_writer(job_id, status, **kwargs)

    monkeypatch.setattr(context.router, "Thread", BrokenThread)
    monkeypatch.setattr(context.router, "_write_status", write_status)
    response = _submit(context)
    assert response.status_code == 503
    assert context.runtime.admission_counts() == (0, 0)
    assert list(context.router.UPLOADS_DIR.iterdir()) == []
    (status_path,) = context.router.TRANSCRIPTIONS_DIR.glob("*/status.json")
    job_id = status_path.parent.name
    assert context.client.get(f"/api/jobs/{job_id}").json()["status"] == "failed"
    if not report_failure:
        assert json.loads(status_path.read_text())["status"] == "failed"


@pytest.mark.parametrize(
    "filename",
    ["../outside.wav", "/outside.wav", "nested/source.wav", "..\\outside.wav", None],
)
def test_download_rejects_non_basename_artifact_metadata(
    submission, monkeypatch, filename
):
    monkeypatch.setattr(
        submission.router,
        "_load_transcription_result",
        lambda _: {"filename": filename},
    )
    response = submission.client.get("/api/transcriptions/tr_synthetic/audio")
    assert response.status_code == 400


def test_download_rejects_symlink_escape_and_allows_source_inside_uploads(
    submission, monkeypatch, tmp_path
):
    context = submission
    sibling = context.router.UPLOADS_DIR.with_name(
        context.router.UPLOADS_DIR.name + "-other"
    )
    sibling.mkdir()
    outside = sibling / "outside.wav"
    outside.write_bytes(b"synthetic outside source")
    link = context.router.UPLOADS_DIR / "source.wav"
    link.symlink_to(outside)
    monkeypatch.setattr(
        context.router, "_load_transcription_result", lambda _: {"filename": link.name}
    )
    assert (
        context.client.get("/api/transcriptions/tr_synthetic/audio").status_code == 400
    )
    link.unlink()
    link.write_bytes(b"synthetic inside source")
    response = context.client.get("/api/transcriptions/tr_synthetic/audio")
    assert response.status_code == 200
    assert response.content == b"synthetic inside source"


def test_transcription_directory_containment_rejects_prefix_collision(
    monkeypatch, tmp_path
):
    paths = importlib.import_module("infra.audio.paths")
    root = tmp_path / "transcriptions"
    sibling = tmp_path / "transcriptions-other"
    root.mkdir()
    sibling.mkdir()
    (root / "tr_synthetic").symlink_to(sibling, target_is_directory=True)
    monkeypatch.setattr(paths, "TRANSCRIPTIONS_DIR", root)
    with pytest.raises(Exception) as caught:
        paths.safe_tr_dir("tr_synthetic")
    assert caught.value.status_code == 400


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_get_job_disk_fallback_rejects_symlink_before_reading_artifacts(
    submission, monkeypatch, tmp_path, status
):
    context = submission
    outside = tmp_path / "external-synthetic-job"
    outside.mkdir()
    (outside / "status.json").write_text(json.dumps({"status": status}))
    (outside / "result.json").write_text(json.dumps({"synthetic": "outside artifact"}))
    job_id = "tr_symlink_probe"
    (context.router.TRANSCRIPTIONS_DIR / job_id).symlink_to(
        outside, target_is_directory=True
    )
    original_read_text = Path.read_text

    def reject_external_read(path, *args, **kwargs):
        if path.resolve().is_relative_to(outside):
            pytest.fail("job fallback must reject the symlink before reading artifacts")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", reject_external_read)
    response = context.client.get(f"/api/jobs/{job_id}")
    assert response.status_code == 400
    assert "result" not in response.json()
