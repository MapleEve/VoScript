"""Regression tests for Docker GPU exposure defaults."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_compose_defaults_to_all_visible_gpus_without_cuda_visible_devices_pin():
    compose = (ROOT / "docker-compose.yml").read_text()

    assert "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}" not in compose
    assert "count: all" in compose
    assert "count: 1" not in compose


def test_env_example_does_not_default_to_gpu_zero_visibility_pin():
    env_example = (ROOT / ".env.example").read_text()

    assert "CUDA_VISIBLE_DEVICES=0" not in env_example
