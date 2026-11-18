"""Pytest bootstrap: import real audio libraries before legacy conftest stubs."""

import os

import numpy
import soundfile
import torch
import torchaudio

if torch.__version__.split("+")[0] != "2.13.0":
    raise RuntimeError("CI requires the actual Torch 2.13.0 runtime, not a test stub")
if torchaudio.__version__.split("+")[0] != "2.11.0":
    raise RuntimeError(
        "CI requires the actual TorchAudio 2.11.0 runtime, not a test stub"
    )
if os.environ.get("CI_REQUIRE_CPU_WHEEL") == "1" and torch.version.cuda is not None:
    raise RuntimeError("Fast CI must use the official CPU-only Torch wheel")
if not numpy.__version__ or not soundfile.__version__:
    raise RuntimeError("Real NumPy and soundfile are required for audio-I/O tests")


def pytest_sessionfinish(session, exitstatus):
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    skipped = reporter.stats.get("skipped", []) if reporter else []
    if skipped:
        session.exitstatus = 1
        if reporter:
            reporter.write_line(
                f"CI rejects skipped regression tests: {len(skipped)}", red=True
            )
