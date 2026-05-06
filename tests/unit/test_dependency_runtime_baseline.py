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


def test_faster_whisper_runtime_stays_on_cudnn9_compatible_ctranslate2():
    lines = _requirements_lines()

    assert "faster-whisper>=1.2.1,<2.0.0" in lines
    assert "ctranslate2>=4.7.1,<5.0" in lines
    assert "faster-whisper==1.1.0" not in lines
