"""Upload/download filename contracts using synthetic files, without GPU work."""

from __future__ import annotations

import asyncio
import importlib
import json
from io import BytesIO
from types import SimpleNamespace
from urllib.parse import quote

import pytest
from fastapi import Request, UploadFile


@pytest.fixture
def filename_routes(monkeypatch, app_client):
    router = importlib.import_module("api.routers.transcriptions")
    monkeypatch.setattr(router, "TRANSCRIPTION_MAX_ACTIVE_JOBS", 200)
    monkeypatch.setattr(router, "TRANSCRIPTION_MAX_IN_FLIGHT_JOBS", 4)
    monkeypatch.setattr(router, "TRANSCRIPTION_MIN_FREE_DISK_BYTES", 0)
    started = []

    class DeferredThread:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start(self):
            started.append(self.kwargs)

    monkeypatch.setattr(router, "Thread", DeferredThread)
    return SimpleNamespace(client=app_client, router=router, started=started)


def _seed_result(context, job_id, filename):
    tr_dir = context.router.TRANSCRIPTIONS_DIR / job_id
    tr_dir.mkdir(parents=True, exist_ok=True)
    result_path = tr_dir / "result.json"
    result_path.write_text(json.dumps({"filename": filename}), encoding="utf-8")
    return result_path


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("source.wav", "source.wav"),
        ("nested/source.wav", "source.wav"),
        ("/private/source.wav", "source.wav"),
        (r"nested\source.wav", "source.wav"),
        (r"C:\fakepath\source.wav", "source.wav"),
        (r"C:/fakepath\source.wav", "source.wav"),
        ("C:source.wav", "source.wav"),
        (r"\\server\share\source.wav", "source.wav"),
        ("//server/share/source.wav", "source.wav"),
        ("../source.wav", "source.wav"),
        (r"..\source.wav", "source.wav"),
        (r"nested/..\source.wav", "source.wav"),
        ("source.wav:stream", "source.wav_stream"),
        ("C:source.wav:stream", "source.wav_stream"),
        ("-source with spaces.wav", "-source with spaces.wav"),
        ("\u97f3\u9891.wav", "\u97f3\u9891.wav"),
        (None, "upload"),
        ("", "upload"),
        (".", "upload"),
    ]
    + [(f"source{chr(code)}.wav", "source?.wav") for code in (*range(32), 127)],
)
def test_upload_filename_is_safe_and_downloadable(filename_routes, filename, expected):
    context = filename_routes
    content = f"synthetic upload {filename!r}".encode("utf-8")
    # Bypass multipart's own Windows-path stripping and control-char escaping.
    upload = UploadFile(filename=filename, file=BytesIO(content))
    try:
        result = asyncio.run(
            context.router.transcribe(
                Request({"type": "http", "app": context.client.app}),
                file=upload,
                language=None,
                min_speakers=0,
                max_speakers=0,
                denoise_model=None,
                snr_threshold=None,
                no_repeat_ngram_size="0",
            )
        )
    finally:
        upload.file.close()

    job_id = result["id"]
    assert result == {"id": job_id, "status": "queued"}
    assert context.router.jobs[job_id]["filename"] == expected
    status_path = context.router.TRANSCRIPTIONS_DIR / job_id / "status.json"
    assert json.loads(status_path.read_text())["filename"] == expected
    (saved_path,) = context.router.UPLOADS_DIR.iterdir()
    assert saved_path.name == f"{job_id}_{expected}"
    assert saved_path.read_bytes() == content
    assert saved_path.resolve().parent == context.router.UPLOADS_DIR.resolve()
    assert len(context.started) == 1
    assert context.started[0]["args"][1] == saved_path

    # The artifacts provider persists Path(request.audio_path).name, not the
    # original multipart filename; exercise the download with that exact key.
    _seed_result(context, job_id, saved_path.name)
    response = context.client.get(f"/api/transcriptions/{job_id}/audio")
    assert response.status_code == 200
    assert response.content == content
    assert quote(saved_path.name) in response.headers["content-disposition"]


def test_windows_relative_multipart_filename_round_trips(filename_routes):
    context = filename_routes
    content = b"synthetic multipart Windows filename"
    response = context.client.post(
        "/api/transcribe", files={"file": (r"nested\source.wav", content, "audio/wav")}
    )
    assert response.status_code == 200
    job_id = response.json()["id"]
    (saved_path,) = context.router.UPLOADS_DIR.iterdir()
    assert saved_path.name == f"{job_id}_source.wav"
    _seed_result(context, job_id, saved_path.name)
    download = context.client.get(f"/api/transcriptions/{job_id}/audio")
    assert download.status_code == 200
    assert download.content == content


@pytest.mark.parametrize("source_exists", [False, True])
def test_legacy_safe_basename_uses_exact_filename(filename_routes, source_exists):
    context = filename_routes
    source = context.router.UPLOADS_DIR / "source.wav"
    if source_exists:
        source.write_bytes(b"synthetic legacy safe basename")
    result_path = _seed_result(context, "tr_synthetic", source.name)
    original_result = result_path.read_bytes()
    response = context.client.get("/api/transcriptions/tr_synthetic/audio")
    assert response.status_code == (200 if source_exists else 404)
    if source_exists:
        assert response.content == b"synthetic legacy safe basename"
        assert (
            response.headers["content-disposition"]
            == 'attachment; filename="source.wav"'
        )
    assert result_path.read_bytes() == original_result


@pytest.mark.parametrize(
    "filename",
    [
        None,
        123,
        [],
        "",
        ".",
        "..",
        "../source.wav",
        "/source.wav",
        "./source.wav",
        "nested/source.wav",
        "source.wav/",
        r"..\source.wav",
        r"nested\source.wav",
        "C:source.wav",
        r"C:\source.wav",
        "C:/source.wav",
        r"\source.wav",
        r"\\server\share\source.wav",
        "//server/share/source.wav",
        "source.wav:stream",
    ]
    + [f"source{chr(code)}.wav" for code in (*range(32), 127)],
)
def test_download_rejects_unsafe_metadata_without_basename_fallback(
    filename_routes, filename
):
    context = filename_routes
    (context.router.UPLOADS_DIR / "source.wav").write_bytes(b"wrong basename fallback")
    result_path = _seed_result(context, "tr_synthetic", filename)
    original_result = result_path.read_bytes()
    response = context.client.get("/api/transcriptions/tr_synthetic/audio")
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid audio filename"
    assert result_path.read_bytes() == original_result


@pytest.mark.parametrize(
    "filename",
    [
        r"tr_synthetic_nested\source.wav",
        r"tr_synthetic_C:\fakepath\source.wav",
        "tr_synthetic_source.wav:stream",
    ],
)
def test_legacy_literal_windows_names_stay_rejected_without_migration(
    filename_routes, filename
):
    context = filename_routes
    legacy_file = context.router.UPLOADS_DIR / filename
    legacy_file.write_bytes(b"synthetic legacy literal filename")
    result_path = _seed_result(context, "tr_synthetic", filename)
    original_result = result_path.read_bytes()
    response = context.client.get("/api/transcriptions/tr_synthetic/audio")
    assert response.status_code == 400
    assert legacy_file.read_bytes() == b"synthetic legacy literal filename"
    assert result_path.read_bytes() == original_result
