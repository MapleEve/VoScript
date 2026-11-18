"""Header-only duration checks before memory-sensitive audio processing."""

from __future__ import annotations

import math
from pathlib import Path

from infra.audio import io_compat as audio_io


def audio_duration_seconds(path: Path | str) -> float | None:
    try:
        info = audio_io.info(str(path))
        if info.sample_rate <= 0 or info.num_frames < 0:
            return None
        duration = info.num_frames / info.sample_rate
        return duration if math.isfinite(duration) else None
    except Exception:
        return None
