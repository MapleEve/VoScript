"""Unit tests for idle-unload daemon and GPU device selection.

Covers:
  - select_best_cuda_device() fallback and happy-path behaviour
  - last_gpu_work_elapsed() return values
  - run_serialized_gpu_work() updating _last_gpu_work_at on success/failure
  - IdleUnloadDaemon construction validation
  - IdleUnloadDaemon lifecycle (start / stop)
  - IdleUnloadDaemon unload triggering and suppression
"""

from __future__ import annotations

import sys
import time
import types
from unittest.mock import MagicMock

import pytest

import infra.job_runtime as job_runtime
from infra.idle_unload import IdleUnloadDaemon, select_best_cuda_device


# ---------------------------------------------------------------------------
# select_best_cuda_device
# ---------------------------------------------------------------------------


def _make_torch_stub(
    *,
    is_available: bool,
    device_count: int = 0,
    free_vram: dict[int, int] | None = None,
) -> types.ModuleType:
    """Build a minimal torch stub for testing GPU device selection."""
    stub = types.ModuleType("torch")
    stub.cuda = types.SimpleNamespace(
        is_available=lambda: is_available,
        device_count=lambda: device_count,
        mem_get_info=lambda idx: (
            (free_vram or {}).get(idx, 0),
            12 * 1024**3,
        ),
    )
    return stub


def test_select_best_cuda_device_cuda_not_available(monkeypatch):
    """Returns 'cuda' when CUDA is not available."""
    monkeypatch.setitem(sys.modules, "torch", _make_torch_stub(is_available=False))
    assert select_best_cuda_device() == "cuda"


def test_select_best_cuda_device_no_gpus(monkeypatch):
    """Returns 'cuda' when CUDA is available but device_count is 0."""
    stub = _make_torch_stub(is_available=True, device_count=0)
    monkeypatch.setitem(sys.modules, "torch", stub)
    assert select_best_cuda_device() == "cuda"


def test_select_best_cuda_device_single_gpu(monkeypatch):
    """Returns 'cuda:0' when there is exactly one GPU."""
    stub = _make_torch_stub(
        is_available=True,
        device_count=1,
        free_vram={0: 8 * 1024**3},
    )
    monkeypatch.setitem(sys.modules, "torch", stub)
    assert select_best_cuda_device() == "cuda:0"


def test_select_best_cuda_device_picks_most_free(monkeypatch):
    """Returns the device with the most free VRAM among multiple GPUs."""
    stub = _make_torch_stub(
        is_available=True,
        device_count=3,
        free_vram={0: 4 * 1024**3, 1: 10 * 1024**3, 2: 2 * 1024**3},
    )
    monkeypatch.setitem(sys.modules, "torch", stub)
    assert select_best_cuda_device() == "cuda:1"


def test_select_best_cuda_device_import_error(monkeypatch):
    """Falls back to 'cuda' when torch raises on import."""
    monkeypatch.setitem(sys.modules, "torch", None)
    # None in sys.modules triggers ImportError on 'import torch'
    assert select_best_cuda_device() == "cuda"


# ---------------------------------------------------------------------------
# last_gpu_work_elapsed
# ---------------------------------------------------------------------------


def test_last_gpu_work_elapsed_returns_inf_when_never_run(monkeypatch):
    """Returns float('inf') when no GPU work has completed yet."""
    monkeypatch.setattr(job_runtime, "_last_gpu_work_at", 0.0)
    assert job_runtime.last_gpu_work_elapsed() == float("inf")


def test_last_gpu_work_elapsed_returns_small_value_just_after_work(monkeypatch):
    """Returns a small non-negative elapsed time immediately after work."""
    monkeypatch.setattr(job_runtime, "_last_gpu_work_at", time.monotonic())
    elapsed = job_runtime.last_gpu_work_elapsed()
    assert 0.0 <= elapsed < 1.0


# ---------------------------------------------------------------------------
# run_serialized_gpu_work timestamp tracking
# ---------------------------------------------------------------------------


def test_run_serialized_updates_last_gpu_work_at_on_success(monkeypatch):
    """_last_gpu_work_at is updated after successful GPU work."""
    monkeypatch.setattr(job_runtime, "_last_gpu_work_at", 0.0)
    monkeypatch.setattr(
        job_runtime,
        "flush_torch_cuda_cache",
        lambda logger=None, *, phase: None,
    )

    job_runtime.run_serialized_gpu_work(lambda: "ok")

    assert job_runtime._last_gpu_work_at > 0.0


def test_run_serialized_does_not_update_timestamp_on_failure(monkeypatch):
    """_last_gpu_work_at is NOT updated when work raises."""
    monkeypatch.setattr(job_runtime, "_last_gpu_work_at", 0.0)
    monkeypatch.setattr(
        job_runtime,
        "flush_torch_cuda_cache",
        lambda logger=None, *, phase: None,
    )

    with pytest.raises(RuntimeError, match="boom"):
        job_runtime.run_serialized_gpu_work(
            lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        )

    assert job_runtime._last_gpu_work_at == 0.0


# ---------------------------------------------------------------------------
# IdleUnloadDaemon — construction
# ---------------------------------------------------------------------------


def test_idle_unload_daemon_rejects_zero_timeout():
    """Raises ValueError when idle_timeout_sec=0."""
    with pytest.raises(ValueError, match="idle_timeout_sec"):
        IdleUnloadDaemon(MagicMock(), idle_timeout_sec=0)


def test_idle_unload_daemon_rejects_negative_timeout():
    """Raises ValueError when idle_timeout_sec is negative."""
    with pytest.raises(ValueError, match="idle_timeout_sec"):
        IdleUnloadDaemon(MagicMock(), idle_timeout_sec=-10)


def test_idle_unload_daemon_default_poll_interval_within_bounds():
    """Default poll_interval is between 5 s and 30 s."""
    daemon = IdleUnloadDaemon(MagicMock(), idle_timeout_sec=120.0)
    assert 5.0 <= daemon._poll_interval <= 30.0


def test_idle_unload_daemon_explicit_poll_interval():
    """Explicit poll_interval_sec is accepted as-is."""
    daemon = IdleUnloadDaemon(MagicMock(), idle_timeout_sec=300, poll_interval_sec=15)
    assert daemon._poll_interval == 15.0


# ---------------------------------------------------------------------------
# IdleUnloadDaemon — lifecycle
# ---------------------------------------------------------------------------


def test_idle_unload_daemon_start_creates_named_daemon_thread():
    """start() spawns a daemon thread named 'idle-unload'."""
    pipeline = MagicMock()
    daemon = IdleUnloadDaemon(pipeline, idle_timeout_sec=300, poll_interval_sec=60)
    daemon.start()
    try:
        assert daemon._thread is not None
        assert daemon._thread.is_alive()
        assert daemon._thread.daemon
        assert daemon._thread.name == "idle-unload"
    finally:
        daemon.stop(timeout=2)


def test_idle_unload_daemon_stop_terminates_thread():
    """stop() sets the stop event and joins the thread."""
    pipeline = MagicMock()
    daemon = IdleUnloadDaemon(pipeline, idle_timeout_sec=300, poll_interval_sec=60)
    daemon.start()
    daemon.stop(timeout=2)
    assert daemon._thread is not None
    assert not daemon._thread.is_alive()


# ---------------------------------------------------------------------------
# IdleUnloadDaemon — unload triggering
# ---------------------------------------------------------------------------


def test_daemon_calls_unload_when_idle(monkeypatch):
    """Daemon calls pipeline.unload_models() when idle for >= timeout."""
    pipeline = MagicMock()
    pipeline.models_loaded.return_value = True

    # Pretend GPU work finished a long time ago.
    monkeypatch.setattr(job_runtime, "_last_gpu_work_at", time.monotonic() - 3600)

    unloaded = []

    def _fake_serialized(work, logger=None):
        work()
        unloaded.append(True)

    monkeypatch.setattr(job_runtime, "run_serialized_gpu_work", _fake_serialized)
    monkeypatch.setattr(
        job_runtime, "flush_torch_cuda_cache", lambda logger=None, *, phase: None
    )

    daemon = IdleUnloadDaemon(pipeline, idle_timeout_sec=60, poll_interval_sec=0.05)
    daemon.start()
    time.sleep(0.4)
    daemon.stop(timeout=2)

    pipeline.unload_models.assert_called()
    assert unloaded


def test_daemon_no_unload_when_models_not_loaded(monkeypatch):
    """Daemon skips unload when no models are in memory."""
    pipeline = MagicMock()
    pipeline.models_loaded.return_value = False

    monkeypatch.setattr(job_runtime, "_last_gpu_work_at", time.monotonic() - 3600)
    monkeypatch.setattr(
        job_runtime, "flush_torch_cuda_cache", lambda logger=None, *, phase: None
    )

    daemon = IdleUnloadDaemon(pipeline, idle_timeout_sec=60, poll_interval_sec=0.05)
    daemon.start()
    time.sleep(0.3)
    daemon.stop(timeout=2)

    pipeline.unload_models.assert_not_called()


def test_daemon_no_unload_when_recently_active(monkeypatch):
    """Daemon does NOT unload when pipeline completed work recently."""
    pipeline = MagicMock()
    pipeline.models_loaded.return_value = True

    # Mark GPU work as having just completed.
    monkeypatch.setattr(job_runtime, "_last_gpu_work_at", time.monotonic())
    monkeypatch.setattr(
        job_runtime, "flush_torch_cuda_cache", lambda logger=None, *, phase: None
    )

    daemon = IdleUnloadDaemon(pipeline, idle_timeout_sec=300, poll_interval_sec=0.05)
    daemon.start()
    time.sleep(0.3)
    daemon.stop(timeout=2)

    pipeline.unload_models.assert_not_called()


def test_daemon_survives_tick_exception(monkeypatch):
    """Daemon thread stays alive when a tick raises an exception."""
    pipeline = MagicMock()
    pipeline.models_loaded.side_effect = RuntimeError("simulated tick failure")

    monkeypatch.setattr(job_runtime, "_last_gpu_work_at", time.monotonic() - 3600)
    monkeypatch.setattr(
        job_runtime, "flush_torch_cuda_cache", lambda logger=None, *, phase: None
    )

    daemon = IdleUnloadDaemon(pipeline, idle_timeout_sec=60, poll_interval_sec=0.05)
    daemon.start()
    time.sleep(0.3)

    assert daemon._thread is not None
    assert daemon._thread.is_alive(), "Daemon thread must stay alive after tick failure"

    daemon.stop(timeout=2)
