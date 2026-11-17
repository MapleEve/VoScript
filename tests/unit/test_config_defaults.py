"""Regression tests for public configuration defaults."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _fresh_config(
    monkeypatch,
    model_idle_timeout_sec: str | None = None,
    rust_kernel_mode: str | None = None,
):
    if model_idle_timeout_sec is None:
        monkeypatch.delenv("MODEL_IDLE_TIMEOUT_SEC", raising=False)
    else:
        monkeypatch.setenv("MODEL_IDLE_TIMEOUT_SEC", model_idle_timeout_sec)

    if rust_kernel_mode is None:
        monkeypatch.delenv("RUST_KERNEL_MODE", raising=False)
    else:
        monkeypatch.setenv("RUST_KERNEL_MODE", rust_kernel_mode)

    sys.modules.pop("config", None)
    return importlib.import_module("config")


def test_model_idle_timeout_defaults_to_three_minutes(monkeypatch):
    config = _fresh_config(monkeypatch)

    assert config.MODEL_IDLE_TIMEOUT_SEC == 180.0


def test_model_idle_timeout_explicit_zero_disables_idle_unload(monkeypatch):
    config = _fresh_config(monkeypatch, model_idle_timeout_sec="0")

    assert config.MODEL_IDLE_TIMEOUT_SEC == 0.0


def test_rust_kernel_mode_defaults_to_required(monkeypatch):
    config = _fresh_config(monkeypatch)

    assert config.RUST_KERNEL_MODE == "required"


def test_rust_kernel_mode_is_normalized(monkeypatch):
    config = _fresh_config(monkeypatch, rust_kernel_mode=" REQUIRED ")

    assert config.RUST_KERNEL_MODE == "required"


def test_admission_and_stage_budget_defaults(monkeypatch):
    defaults = {
        "TRANSCRIPTION_MAX_ACTIVE_JOBS": 200,
        "TRANSCRIPTION_MAX_IN_FLIGHT_JOBS": 4,
        "TRANSCRIPTION_MIN_FREE_DISK_BYTES": 1024 * 1024 * 1024,
        "DENOISE_MAX_AUDIO_DURATION_SEC": 7200.0,
        "WHISPERX_ALIGN_MAX_AUDIO_DURATION_SEC": 7200.0,
        "EMBEDDING_PRELOAD_MAX_AUDIO_DURATION_SEC": 1800.0,
    }
    for name in defaults:
        monkeypatch.delenv(name, raising=False)
    config = _fresh_config(monkeypatch)
    for name, value in defaults.items():
        assert getattr(config, name) == value


def test_alignment_device_is_forwarded_by_compose_and_accepts_override(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    compose_environment = {}
    for line in (root / "docker-compose.yml").read_text().splitlines():
        entry = line.strip()
        if entry.startswith("- ") and "=" in entry:
            key, value = entry[2:].split("=", 1)
            compose_environment[key] = value
    assert (
        compose_environment["WHISPERX_ALIGN_DEVICE"] == "${WHISPERX_ALIGN_DEVICE:-cpu}"
    )
    assert (
        "WHISPERX_ALIGN_DEVICE=cpu" in (root / ".env.example").read_text().splitlines()
    )
    monkeypatch.setenv("WHISPERX_ALIGN_DEVICE", " CUDA:1 ")
    assert _fresh_config(monkeypatch).WHISPERX_ALIGN_DEVICE == "cuda:1"
