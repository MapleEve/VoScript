"""Default provider for speaker embedding extraction."""

from __future__ import annotations

import logging
import time

import numpy as np
import torchaudio

from config import (
    EMBEDDING_PRELOAD_MAX_AUDIO_DURATION_SEC,
    MAX_EMBED_DURATION,
    MIN_EMBED_DURATION,
)
from infra.audio import io_compat as audio_io
from pipeline.contracts import (
    SpeakerEmbeddingProvider,
    SpeakerEmbeddingRequest,
    SpeakerEmbeddingResult,
)

logger = logging.getLogger(__name__)


def _load_full_waveform(audio_path: str):
    """Preload only after metadata selection and the duration budget check."""

    load_started = time.perf_counter()
    waveform, sample_rate = audio_io.load(audio_path)
    logger.info(
        "embedding_audio_load_timing backend=soundfile elapsed_s=%.3f sample_rate=%d channels=%d frames=%d",
        time.perf_counter() - load_started,
        sample_rate,
        waveform.shape[0],
        waveform.shape[1],
    )
    return waveform, sample_rate


def extract_embeddings_for_turns(
    pipeline,
    audio_path: str,
    turns: list[dict[str, object]],
) -> dict[str, np.ndarray]:
    """Extract averaged embeddings for each speaker cluster."""

    info = audio_io.info(audio_path)
    native_sr = info.sample_rate
    audio_frames = getattr(info, "num_frames", None)
    if native_sr <= 0:
        raise ValueError("Invalid embedding audio sample rate")
    target_sr = 16000
    min_samples = int(MIN_EMBED_DURATION * native_sr)
    max_samples = int(MAX_EMBED_DURATION * native_sr)

    speaker_segments: dict[str, list[tuple[int, int, int]]] = {}
    for turn in turns:
        speaker = turn["speaker"]
        start_sample = int(turn["start"] * native_sr)
        end_sample = int(turn["end"] * native_sr)
        num_frames = end_sample - start_sample

        if num_frames < min_samples:
            continue
        if num_frames > max_samples:
            num_frames = max_samples

        available_frames = num_frames
        if audio_frames is not None:
            available_frames = min(num_frames, max(audio_frames - start_sample, 0))
        if available_frames <= 0:
            continue
        # Match the old post-resample length ranking, including stable ties
        # and turns clipped by EOF, without materializing every turn's audio.
        ranked_frames = (available_frames * target_sr + native_sr - 1) // native_sr
        speaker_segments.setdefault(speaker, []).append(
            (start_sample, num_frames, ranked_frames)
        )

    for segments in speaker_segments.values():
        segments.sort(key=lambda segment: segment[2], reverse=True)
    if not speaker_segments:
        return {}

    waveform = None
    duration_s = None if audio_frames is None else audio_frames / native_sr
    if duration_s is not None and (
        EMBEDDING_PRELOAD_MAX_AUDIO_DURATION_SEC <= 0
        or duration_s <= EMBEDDING_PRELOAD_MAX_AUDIO_DURATION_SEC
    ):
        try:
            waveform, _ = _load_full_waveform(audio_path)
        except Exception as exc:
            logger.warning(
                "Embedding preload failed; using selected segments error_type=%s",
                type(exc).__name__,
            )
    else:
        logger.info(
            "embedding_full_audio_preload_skipped reason=duration_budget_exceeded "
            "max_duration_s=%.3f",
            EMBEDDING_PRELOAD_MAX_AUDIO_DURATION_SEC,
        )

    embeddings: dict[str, np.ndarray] = {}
    model_processing_elapsed_s = 0.0
    processed_chunk_count = 0
    for speaker, segments in speaker_segments.items():
        emb_list = []
        for start_sample, num_frames, _ in segments:
            if len(emb_list) >= 10:
                break
            if waveform is not None:
                chunk = waveform[
                    :, start_sample : start_sample + num_frames
                ].contiguous()
                chunk_sr = native_sr
            else:
                try:
                    chunk, chunk_sr = audio_io.load(
                        audio_path,
                        frame_offset=start_sample,
                        num_frames=num_frames,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to load embedding audio segment [%d:%d]: %s",
                        start_sample,
                        start_sample + num_frames,
                        type(exc).__name__,
                    )
                    continue
            if chunk.shape[1] <= 0:
                continue

            if chunk_sr != target_sr:
                chunk = torchaudio.functional.resample(chunk, chunk_sr, target_sr)
            if chunk.shape[0] > 1:
                chunk = chunk.mean(dim=0, keepdim=True)

            embedding_model = pipeline.embedding_model
            embedding_device = getattr(pipeline, "embedding_device", pipeline.device)
            processing_started = time.perf_counter()
            emb = embedding_model(
                {"waveform": chunk.to(embedding_device), "sample_rate": target_sr}
            )
            model_processing_elapsed_s += time.perf_counter() - processing_started
            processed_chunk_count += 1
            emb_list.append(np.asarray(emb))
        if emb_list:
            embeddings[speaker] = np.mean(emb_list, axis=0)
    logger.info(
        "embedding_processing_timing model=wespeaker elapsed_s=%.3f device=%s speaker_count=%d chunk_count=%d",
        model_processing_elapsed_s,
        getattr(pipeline, "embedding_device", getattr(pipeline, "device", "")),
        len(embeddings),
        processed_chunk_count,
    )
    return embeddings


class PipelineMethodSpeakerEmbeddingProvider(SpeakerEmbeddingProvider):
    """Extract speaker embeddings through pipeline-owned model resources."""

    def extract_embeddings(
        self, request: SpeakerEmbeddingRequest
    ) -> SpeakerEmbeddingResult:
        return SpeakerEmbeddingResult(
            speaker_embeddings=extract_embeddings_for_turns(
                request.pipeline,
                request.audio_path,
                request.diarization_turns,
            )
        )


default_speaker_embedding_provider = PipelineMethodSpeakerEmbeddingProvider()


__all__ = [
    "PipelineMethodSpeakerEmbeddingProvider",
    "default_speaker_embedding_provider",
    "extract_embeddings_for_turns",
]
