"""Small admission policy around the existing transcription route and runtime."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

from infra.job_runtime import JobAdmissionError, admit_job


def admit_transcription(
    file_hash: str | None,
    job_id: str,
    *,
    uploads_dir: Path,
    max_active_jobs: int,
    max_in_flight_jobs: int,
    min_free_disk_bytes: int,
    prepare: Callable[[], None],
) -> str | None:
    """Reuse existing work before checking limits for a new durable job."""

    def _prepare():
        if min_free_disk_bytes > 0:
            try:
                free_bytes = shutil.disk_usage(uploads_dir).free
            except OSError as exc:
                raise JobAdmissionError("data_disk_unavailable") from exc
            if free_bytes < min_free_disk_bytes:
                raise JobAdmissionError("data_disk_pressure")
        prepare()

    return admit_job(
        file_hash,
        job_id,
        max_active_jobs=max_active_jobs,
        max_in_flight_jobs=max_in_flight_jobs,
        prepare=_prepare,
    )
