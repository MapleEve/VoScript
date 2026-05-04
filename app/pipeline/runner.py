"""Stable pipeline runner that wires stage slots to current implementations."""

from __future__ import annotations

import logging
import time
from typing import Any

from infra.audio import cleanup_generated_files

from .contracts import PipelineContext, PipelineRequest
from .registry import available_stage_slots, resolve_stage

logger = logging.getLogger(__name__)

DEFAULT_STAGE_ORDER = available_stage_slots()


def _safe_stage_metrics(context: PipelineContext, stage_name: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    stage_metadata = context.metadata.get(stage_name)
    if isinstance(stage_metadata, dict):
        for key in (
            "status",
            "model",
            "language",
            "segment_count",
            "speaker_count",
            "turn_count",
            "applied",
            "reason",
            "persisted",
        ):
            if key in stage_metadata:
                metrics[key] = stage_metadata[key]
    if "segment_count" not in metrics and context.aligned_segments:
        metrics["segment_count"] = len(context.aligned_segments)
    if "speaker_count" not in metrics:
        if context.speaker_embeddings:
            metrics["speaker_count"] = len(context.speaker_embeddings)
        elif context.diarization_turns:
            metrics["speaker_count"] = len(
                {turn.get("speaker") for turn in context.diarization_turns}
            )
    return metrics


class PipelineRunner:
    """Execute the stable stage order against the current pipeline implementation."""

    def __init__(
        self,
        stage_order: tuple[str, ...] | None = None,
        stage_overrides: dict[str, Any] | None = None,
    ):
        self.stage_order = tuple(stage_order or DEFAULT_STAGE_ORDER)
        self.stage_overrides = dict(stage_overrides or {})

    def resolve_stage(self, stage_name: str):
        return self.stage_overrides.get(stage_name) or resolve_stage(stage_name)

    def build_context(self, pipeline: Any, request: PipelineRequest) -> PipelineContext:
        return PipelineContext(pipeline=pipeline, request=request)

    def run_context(self, pipeline: Any, request: PipelineRequest) -> PipelineContext:
        context = self.build_context(pipeline, request)
        try:
            for stage_name in self.stage_order:
                logger.info("Running pipeline stage: %s", stage_name)
                stage = self.resolve_stage(stage_name)
                context.mark_stage(stage_name)
                context.metadata.setdefault("selected_providers", {})[stage_name] = (
                    request.provider_for(stage_name)
                )
                stage_started = time.perf_counter()
                stage(context)
                elapsed_s = time.perf_counter() - stage_started
                metrics = _safe_stage_metrics(context, stage_name)
                context.metadata.setdefault("stage_timings", {})[stage_name] = round(
                    elapsed_s,
                    3,
                )
                logger.info(
                    "pipeline_stage_timing stage=%s elapsed_s=%.3f provider=%s metrics=%s",
                    stage_name,
                    elapsed_s,
                    request.provider_for(stage_name),
                    metrics,
                )
            return context
        finally:
            cleanup_generated_files(context.temporary_paths)

    def run(self, pipeline: Any, request: PipelineRequest) -> dict[str, Any]:
        return self.run_context(pipeline, request).to_result()


__all__ = [
    "DEFAULT_STAGE_ORDER",
    "PipelineContext",
    "PipelineRequest",
    "PipelineRunner",
]
