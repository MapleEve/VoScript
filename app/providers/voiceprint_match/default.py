"""Default provider for voiceprint matching and AS-norm lookup."""

from __future__ import annotations

import logging
import time
from typing import Any

from pipeline.contracts import (
    VoiceprintMatchProvider,
    VoiceprintMatchRequest,
    VoiceprintMatchResult,
)

logger = logging.getLogger(__name__)


class DefaultVoiceprintMatchProvider(VoiceprintMatchProvider):
    """Use the current VoiceprintDB identify() API when available."""

    def match(self, request: VoiceprintMatchRequest) -> VoiceprintMatchResult:
        if not request.speaker_embeddings:
            logger.info(
                "voiceprint_match_processing_timing elapsed_s=0.000 speaker_count=0 applied=False reason=no_embeddings"
            )
            return VoiceprintMatchResult(
                speaker_map={},
                applied=False,
                threshold=request.threshold,
                reason="no_embeddings",
            )

        if request.voiceprint_db is None:
            logger.info(
                "voiceprint_match_processing_timing elapsed_s=0.000 speaker_count=%d applied=False reason=voiceprint_db_unavailable",
                len(request.speaker_embeddings),
            )
            return VoiceprintMatchResult(
                speaker_map={},
                applied=False,
                threshold=request.threshold,
                reason="voiceprint_db_unavailable",
            )

        speaker_map: dict[str, dict[str, Any]] = {}
        processing_started = time.perf_counter()
        for speaker_label, embedding in request.speaker_embeddings.items():
            if request.threshold is None:
                matched_id, matched_name, similarity = request.voiceprint_db.identify(
                    embedding
                )
            else:
                matched_id, matched_name, similarity = request.voiceprint_db.identify(
                    embedding,
                    threshold=request.threshold,
                )
            speaker_map[speaker_label] = {
                "matched_id": matched_id,
                "matched_name": matched_name or speaker_label,
                "similarity": round(similarity, 4),
                "embedding_key": speaker_label,
            }
        elapsed_s = time.perf_counter() - processing_started
        logger.info(
            "voiceprint_match_processing_timing elapsed_s=%.3f speaker_count=%d applied=True reason=matched",
            elapsed_s,
            len(speaker_map),
        )

        return VoiceprintMatchResult(
            speaker_map=speaker_map,
            applied=True,
            threshold=request.threshold,
            reason="matched",
        )


default_voiceprint_match_provider = DefaultVoiceprintMatchProvider()


__all__ = [
    "DefaultVoiceprintMatchProvider",
    "default_voiceprint_match_provider",
]
