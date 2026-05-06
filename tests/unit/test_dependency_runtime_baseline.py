"""Regression tests for runtime dependency pins that affect Docker GPU loads."""

from __future__ import annotations

from pathlib import Path


def _requirements_lines() -> list[str]:
    root = Path(__file__).resolve().parents[2]
    return [
        line.strip()
        for line in (root / "app" / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _dockerfile_text() -> str:
    root = Path(__file__).resolve().parents[2]
    return (root / "app" / "Dockerfile").read_text()


def test_faster_whisper_runtime_stays_on_cudnn9_compatible_ctranslate2():
    lines = _requirements_lines()

    assert "faster-whisper>=1.2.1,<2.0.0" in lines
    assert "ctranslate2>=4.7.1,<5.0" in lines
    assert "faster-whisper==1.1.0" not in lines


def test_docker_installs_whisperx_without_replacing_asr_runtime_stack():
    lines = _requirements_lines()
    dockerfile = _dockerfile_text()

    assert "nltk>=3.9,<4.0" in lines
    assert "whisperx==3.3.1" not in lines
    assert (
        'pip install --no-cache-dir -i "$PIP_INDEX_URL" --no-deps whisperx==3.3.1'
        in dockerfile
    )
    assert "pip install --no-cache-dir --no-deps whisperx==3.3.1" in dockerfile
