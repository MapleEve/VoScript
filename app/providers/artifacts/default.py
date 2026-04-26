"""Default provider for assembling and persisting pipeline artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from config import DENOISE_MODEL, DENOISE_SNR_THRESHOLD
from infra.transcription_artifacts import persist_transcription_artifacts
from pipeline.contracts import PipelineContext, PipelineResult


class InMemoryArtifactsProvider:
    """Assemble the final transcript payload from the current context state."""

    @staticmethod
    def _build_display_names(
        speaker_labels: list[str],
        speaker_map: dict[str, dict],
    ) -> dict[str, str]:
        labels_by_name: dict[str, list[str]] = {}

        for speaker_label in speaker_labels:
            match = speaker_map.get(speaker_label, {})
            speaker_name = str(match.get("matched_name") or speaker_label)
            labels_by_name.setdefault(speaker_name, []).append(speaker_label)

        display_names: dict[str, str] = {}
        for speaker_name, labels in labels_by_name.items():
            for index, speaker_label in enumerate(labels, start=1):
                display_names[speaker_label] = (
                    speaker_name if index == 1 else f"{speaker_name} ({index})"
                )
        return display_names

    @staticmethod
    def _build_segments(
        aligned_segments: list[dict],
        speaker_map: dict[str, dict],
    ) -> tuple[list[dict], list[str]]:
        speaker_labels = list(
            dict.fromkeys(segment["speaker"] for segment in aligned_segments)
        )
        display_names = InMemoryArtifactsProvider._build_display_names(
            speaker_labels,
            speaker_map,
        )
        segments: list[dict] = []
        seen_speakers: set[str] = set()
        unique_speakers: list[str] = []

        for index, segment in enumerate(aligned_segments):
            speaker_label = segment["speaker"]
            match = speaker_map.get(speaker_label, {})
            speaker_name = display_names.get(speaker_label, speaker_label)
            output = {
                "id": index,
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "speaker_label": speaker_label,
                "speaker_id": match.get("matched_id"),
                "speaker_name": speaker_name,
                "similarity": match.get("similarity", 0),
            }
            if segment.get("words"):
                output["words"] = segment["words"]
            segments.append(output)

            if speaker_name not in seen_speakers:
                seen_speakers.add(speaker_name)
                unique_speakers.append(speaker_name)

        return segments, unique_speakers

    def _build_transcription(self, context: PipelineContext) -> dict | None:
        if context.request.artifact_dir is None:
            return None

        effective_denoise = (
            (context.request.denoise_model or DENOISE_MODEL).strip().lower()
        )
        effective_snr = (
            context.request.snr_threshold
            if context.request.snr_threshold is not None
            else DENOISE_SNR_THRESHOLD
        )
        segments, unique_speakers = self._build_segments(
            context.aligned_segments,
            context.voiceprint_matches,
        )
        warning = None
        if not context.voiceprint_matches and not context.speaker_embeddings:
            warning = "no_speakers_detected"

        transcription = {
            "id": context.request.artifact_dir.name,
            "filename": Path(context.request.audio_path).name,
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "status": "completed",
            "language": context.request.language,
            "segments": segments,
            "speaker_map": context.voiceprint_matches,
            "unique_speakers": unique_speakers,
            "params": {
                "language": context.request.language or "auto",
                "denoise_model": effective_denoise,
                "snr_threshold": effective_snr,
                "voiceprint_threshold": context.request.voiceprint_threshold,
                "min_speakers": context.request.min_speakers,
                "max_speakers": context.request.max_speakers,
                "no_repeat_ngram_size": context.request.no_repeat_ngram_size or 0,
            },
        }
        if context.transcription_result is not None:
            guard_report = context.transcription_result.get("hallucination_guard")
            if guard_report is not None:
                transcription["asr_hallucination_guard"] = guard_report
        alignment_metadata = context.metadata.get("diarization", {}).get("alignment")
        if alignment_metadata:
            transcription["alignment"] = alignment_metadata
        if warning is not None:
            transcription["warning"] = warning
        return transcription

    def build(self, context: PipelineContext) -> PipelineResult:
        transcription = self._build_transcription(context)
        artifact_paths = None
        if transcription is not None and context.request.artifact_dir is not None:
            persisted = persist_transcription_artifacts(
                context.request.artifact_dir,
                transcription,
                context.speaker_embeddings,
            )
            artifact_paths = {
                "result_path": str(persisted.result_path),
                "embedding_paths": {
                    label: str(path)
                    for label, path in persisted.embedding_paths.items()
                },
            }
            segments = transcription["segments"]
            unique_speakers = transcription["unique_speakers"]
        else:
            segments = context.aligned_segments
            unique_speakers = list(context.speaker_embeddings.keys())

        return PipelineResult(
            segments=segments,
            speaker_embeddings=context.speaker_embeddings,
            unique_speakers=unique_speakers,
            transcription=transcription,
            artifact_paths=artifact_paths,
        )


default_artifacts_provider = InMemoryArtifactsProvider()


__all__ = [
    "InMemoryArtifactsProvider",
    "default_artifacts_provider",
]
