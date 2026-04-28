"""Idle-unload daemon: releases GPU VRAM after a configurable idle period.

When ``MODEL_IDLE_TIMEOUT_SEC`` is set to a positive value the
``IdleUnloadDaemon`` background thread monitors GPU inactivity and, once the
pipeline has been idle for the configured duration, calls
``pipeline.unload_models()`` under the shared GPU semaphore so that no
in-flight transcription job is interrupted.

On the next transcription request the pipeline's lazy-loading properties
transparently reload every model.  If ``DEVICE="cuda"`` (no explicit index),
the reload targets the CUDA device with the most free VRAM at that moment —
useful on multi-GPU hosts where available memory shifts between jobs.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)


def select_best_cuda_device() -> str:
    """Return the CUDA device string with the most free VRAM.

    Iterates over all visible CUDA devices and returns ``"cuda:<idx>"`` for
    the one reporting the highest number of free bytes.  Falls back to
    ``"cuda"`` (PyTorch default) when:

    * ``torch`` is not importable, or
    * CUDA is not available, or
    * ``torch.cuda.mem_get_info`` raises for any reason.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return "cuda"
        device_count = torch.cuda.device_count()
        if device_count == 0:
            return "cuda"
        best_idx = 0
        best_free = -1
        for idx in range(device_count):
            free, _ = torch.cuda.mem_get_info(idx)
            if free > best_free:
                best_free = free
                best_idx = idx
        return f"cuda:{best_idx}"
    except Exception:
        return "cuda"


class IdleUnloadDaemon:
    """Background daemon that unloads pipeline GPU models after an idle timeout.

    The daemon polls :func:`infra.job_runtime.last_gpu_work_elapsed` and,
    when the pipeline has been idle for at least *idle_timeout_sec* seconds
    **and** models are currently loaded, calls ``pipeline.unload_models()``
    through :func:`infra.job_runtime.run_serialized_gpu_work` so that no
    concurrent transcription job is interrupted mid-inference.

    Parameters
    ----------
    pipeline:
        The ``TranscriptionPipeline`` instance whose models should be managed.
    idle_timeout_sec:
        Seconds of GPU inactivity before models are unloaded.  Must be > 0.
    poll_interval_sec:
        How often the daemon checks for idleness.  Defaults to
        ``max(5, min(30, idle_timeout_sec / 2))`` seconds.
    """

    def __init__(
        self,
        pipeline,
        idle_timeout_sec: float,
        poll_interval_sec: float | None = None,
    ) -> None:
        if idle_timeout_sec <= 0:
            raise ValueError(f"idle_timeout_sec must be > 0, got {idle_timeout_sec!r}")
        self._pipeline = pipeline
        self._idle_timeout = float(idle_timeout_sec)
        if poll_interval_sec is None:
            poll_interval_sec = max(5.0, min(30.0, self._idle_timeout / 2))
        self._poll_interval = float(poll_interval_sec)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn the background daemon thread."""
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="idle-unload",
        )
        self._thread.start()
        logger.info(
            "Idle-unload daemon started (timeout=%.0fs, poll=%.0fs)",
            self._idle_timeout,
            self._poll_interval,
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the daemon to stop and wait for it to exit."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        logger.info("Idle-unload daemon stopped")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        from infra import job_runtime

        while not self._stop_event.wait(timeout=self._poll_interval):
            try:
                elapsed = job_runtime.last_gpu_work_elapsed()
                if elapsed >= self._idle_timeout and self._pipeline.models_loaded():
                    logger.info(
                        "Pipeline idle for %.0fs — unloading GPU models to free VRAM",
                        elapsed,
                    )
                    job_runtime.run_serialized_gpu_work(
                        self._pipeline.unload_models,
                        logger=logger,
                    )
            except Exception:
                logger.exception("idle-unload worker tick failed")


__all__ = [
    "IdleUnloadDaemon",
    "select_best_cuda_device",
]
