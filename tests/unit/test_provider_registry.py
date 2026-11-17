"""Unit tests for provider registry and compatibility entrypoints."""

from __future__ import annotations

import sys
import os
import subprocess
import textwrap
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import pytest

from pipeline import TranscriptionPipeline
from pipeline.contracts import (
    ASRRequest,
    ASRResult,
    AudioEnhancementResult,
    AudioNormalizationResult,
    DiarizationRequest,
    DiarizationResult,
    PipelineRequest,
    SpeakerEmbeddingRequest,
    SpeakerEmbeddingResult,
    VoiceprintMatchResult,
)
from pipeline.registry import (
    ProviderNotFoundError,
    available_providers,
    register_provider,
    resolve_provider,
    unregister_provider,
)
from providers import maybe_denoise
import providers.asr.default as asr_default
from providers.asr.default import default_asr_provider
import providers.diarization.default as diarization_default
from providers.diarization.default import default_diarization_provider
from providers.embedding import default_speaker_embedding_provider
import providers.embedding.default as embedding_default
import pipeline.orchestrator as orchestrator
from providers.normalize import convert_to_wav


@pytest.fixture(autouse=True)
def _short_synthetic_alignment_metadata(monkeypatch):
    # Alignment-path tests use stub bytes/models rather than real audio headers.
    monkeypatch.setattr(diarization_default, "audio_duration_seconds", lambda path: 1.0)


class StubNormalizer:
    def normalize(self, request):
        return AudioNormalizationResult(
            source_path=request.input_path,
            normalized_path=request.input_path.with_suffix(".stub.wav"),
            reused_source=False,
        )


class StubEnhancer:
    def enhance(self, request):
        return AudioEnhancementResult(
            input_path=request.wav_path,
            output_path=request.wav_path.with_suffix(".boost.wav"),
            applied=True,
            model="stub",
        )


def test_default_providers_are_listed_and_resolvable():
    asr_provider = resolve_provider("asr", "default")
    diarization_provider = resolve_provider("diarization", "default")
    embedding_provider = resolve_provider("embedding", "default")
    voiceprint_provider = resolve_provider("voiceprint_match", "default")
    ingest_provider = resolve_provider("ingest", "default")
    normalizer = resolve_provider("normalize", "default")
    enhancer = resolve_provider("enhance", "default")
    vad_provider = resolve_provider("vad", "default")
    punc_provider = resolve_provider("punc", "default")
    postprocess_provider = resolve_provider("postprocess", "default")
    artifacts_provider = resolve_provider("artifacts", "default")

    assert asr_provider.__class__.__name__ == "PipelineMethodASRProvider"
    assert (
        diarization_provider.__class__.__name__ == "PipelineMethodDiarizationProvider"
    )
    assert (
        embedding_provider.__class__.__name__
        == "PipelineMethodSpeakerEmbeddingProvider"
    )
    assert voiceprint_provider.__class__.__name__ == "DefaultVoiceprintMatchProvider"
    assert ingest_provider.__class__.__name__ == "DefaultIngestProvider"
    assert normalizer.__class__.__name__ == "FFmpegInputNormalizer"
    assert enhancer.__class__.__name__ == "ConditionalDenoiseEnhancer"
    assert vad_provider.__class__.__name__ == "DefaultVADProvider"
    assert punc_provider.__class__.__name__ == "DefaultPunctuationProvider"
    assert postprocess_provider.__class__.__name__ == "DefaultPostprocessProvider"
    assert artifacts_provider.__class__.__name__ == "InMemoryArtifactsProvider"
    assert (
        resolve_provider("input_normalization", "default").__class__.__name__
        == "FFmpegInputNormalizer"
    )
    assert (
        resolve_provider("enhancement", "default").__class__.__name__
        == "ConditionalDenoiseEnhancer"
    )
    assert available_providers("ingest") == ("default",)
    assert available_providers("asr") == ("default",)
    assert available_providers("diarization") == ("default",)
    assert available_providers("embedding") == ("default",)
    assert available_providers("voiceprint_match") == ("default",)
    assert available_providers("normalize") == ("default",)
    assert available_providers("enhance") == ("default",)
    assert available_providers("vad") == ("default",)
    assert available_providers("punc") == ("default",)
    assert available_providers("postprocess") == ("default",)
    assert available_providers("artifacts") == ("default",)
    assert available_providers("input_normalization") == ("default",)
    assert available_providers("enhancement") == ("default",)


def test_default_asr_provider_times_materialized_segments(monkeypatch, caplog):
    events = []
    perf_values = iter([10.0, 12.5])

    class FakeSegment:
        start = 0.0
        end = 1.25
        text = " hello "

    class FakeWhisper:
        def transcribe(self, audio_path, **kwargs):
            events.append(("transcribe", audio_path, kwargs["language"]))

            def iter_segments():
                events.append("materialized")
                yield FakeSegment()

            return iter_segments(), SimpleNamespace(language="en")

    pipeline = SimpleNamespace(whisper=FakeWhisper())
    monkeypatch.setattr(
        asr_default.time,
        "perf_counter",
        lambda: events.append("perf") or next(perf_values),
    )

    with caplog.at_level("INFO", logger=asr_default.logger.name):
        result = default_asr_provider.transcribe(
            ASRRequest(
                pipeline=pipeline,
                audio_path="/private/audio.wav",
                language="en",
            )
        )

    assert result.transcription_result["segments"] == [
        {"start": 0.0, "end": 1.25, "text": "hello"}
    ]
    assert events == [
        "perf",
        ("transcribe", "/private/audio.wav", "en"),
        "materialized",
        "perf",
    ]
    assert "asr_processing_timing model=faster-whisper elapsed_s=2.500" in caplog.text
    assert "segment_count=1" in caplog.text
    assert "/private" not in caplog.text


def test_registry_named_overrides_drive_compatibility_helpers(tmp_path):
    input_path = tmp_path / "sample.mp3"
    input_path.write_bytes(b"stub")

    register_provider("normalize", "stub", StubNormalizer())
    register_provider("enhance", "stub", StubEnhancer())
    try:
        normalized = convert_to_wav(input_path, provider_name="stub")
        enhanced = maybe_denoise(normalized, provider_name="stub")
    finally:
        unregister_provider("normalize", "stub")
        unregister_provider("enhance", "stub")

    assert normalized.name == "sample.stub.wav"
    assert enhanced.name == "sample.stub.boost.wav"


def test_unknown_provider_raises_lookup_error():
    with pytest.raises(ProviderNotFoundError):
        resolve_provider("enhance", "missing")


def test_pipeline_request_normalizes_explicit_provider_selection_aliases():
    request = PipelineRequest(
        audio_path="demo.wav",
        provider_selection={
            "input-normalization": "FFmpeg-Basic",
            "enhancement": "DeepFilter-Net",
            "artifacts": "filesystem",
        },
    )

    assert request.provider_for("normalize") == "ffmpeg_basic"
    assert request.provider_for("input_normalization") == "ffmpeg_basic"
    assert request.provider_for("enhance") == "deepfilter_net"
    assert request.provider_for("enhancement") == "deepfilter_net"
    assert request.provider_for("artifacts") == "filesystem"
    assert request.provider_for("voiceprint_match") == "default"


class StubASRProvider:
    def transcribe(self, request):
        return ASRResult(
            transcription_result={
                "segments": [{"start": 0.0, "end": 1.0, "text": "stub"}],
                "language": request.language or "stub",
            }
        )


class StubDiarizationProvider:
    def diarize(self, request):
        return DiarizationResult(
            turns=[{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_STUB"}],
            aligned_segments=[
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "stub",
                    "speaker": "SPEAKER_STUB",
                }
            ],
        )


class StubEmbeddingProvider:
    def extract_embeddings(self, request):
        return SpeakerEmbeddingResult(speaker_embeddings={"SPEAKER_STUB": [0.1, 0.2]})


class StubVoiceprintMatchProvider:
    def match(self, request):
        return VoiceprintMatchResult(
            speaker_map={
                "SPEAKER_STUB": {
                    "matched_id": "spk_stub",
                    "matched_name": "Stub Speaker",
                    "similarity": 0.9876,
                    "embedding_key": "SPEAKER_STUB",
                }
            },
            applied=True,
            threshold=0.7,
            reason="matched",
        )


def test_default_asr_provider_uses_pipeline_whisper_resource():
    pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
    calls = []

    class FakeWhisper:
        def transcribe(self, audio_path, **kwargs):
            calls.append((audio_path, kwargs))
            segments = [SimpleNamespace(start=0.0, end=1.25, text=" hello ")]
            return iter(segments), SimpleNamespace(language="zh")

    pipeline._whisper = FakeWhisper()

    result = default_asr_provider.transcribe(
        ASRRequest(
            pipeline=pipeline,
            audio_path="demo.wav",
            language="zh",
            no_repeat_ngram_size=4,
        )
    )

    assert result.transcription_result == {
        "segments": [{"start": 0.0, "end": 1.25, "text": "hello"}],
        "language": "zh",
        "hallucination_guard": {
            "status": "pass",
            "input_segment_count": 1,
            "output_segment_count": 1,
            "removed_segment_count": 0,
            "removed_duration": 0,
        },
    }
    assert calls == [
        (
            "demo.wav",
            {
                "language": "zh",
                "beam_size": 5,
                "vad_filter": True,
                "vad_parameters": {"min_silence_duration_ms": 500},
                "initial_prompt": None,
                "condition_on_previous_text": False,
                "no_repeat_ngram_size": 4,
            },
        )
    ]


def test_default_diarization_provider_uses_pipeline_diarizer_and_alignment(
    monkeypatch, caplog
):
    pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
    pipeline.device = "cpu"
    calls = []
    perf_values = iter([5.0, 8.0, 10.0, 13.0, 20.0, 21.25])

    class FakeDiarizationResult:
        def itertracks(self, yield_label=False):
            assert yield_label is True
            yield SimpleNamespace(start=0.0, end=1.2), None, "SPEAKER_00"

    class FakeDiarizer:
        def __call__(self, audio_path, **kwargs):
            calls.append(("diarizer", audio_path, kwargs))
            return FakeDiarizationResult()

    pipeline._diarization = FakeDiarizer()
    whisperx = sys.modules["whisperx"]
    monkeypatch.setattr(
        whisperx,
        "load_audio",
        lambda audio_path: f"audio:{audio_path}",
        raising=False,
    )
    monkeypatch.setattr(
        whisperx,
        "load_align_model",
        lambda language_code, device: (
            "align-model",
            {"language": language_code, "device": device},
        ),
        raising=False,
    )
    monkeypatch.setattr(
        whisperx,
        "align",
        lambda segments, align_model, align_metadata, audio, device, return_char_alignments=False: {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.2,
                    "text": "hello",
                    "words": [{"start": 0.0, "end": 0.5, "word": "hello"}],
                }
            ]
        },
        raising=False,
    )
    monkeypatch.setattr(
        diarization_default.time,
        "perf_counter",
        lambda: next(perf_values),
    )

    with caplog.at_level("INFO", logger=diarization_default.logger.name):
        result = default_diarization_provider.diarize(
            DiarizationRequest(
                pipeline=pipeline,
                audio_path="demo.wav",
                transcription_result={"segments": [], "language": "en"},
                min_speakers=1,
                max_speakers=2,
            )
        )

    assert calls == [("diarizer", "demo.wav", {"min_speakers": 1, "max_speakers": 2})]
    assert "diarization_processing_timing model=pyannote elapsed_s=3.000" in caplog.text
    assert "Loaded WhisperX alignment model in 3.00s (cold_load=True" in caplog.text
    assert "alignment_processing_timing model=whisperx elapsed_s=1.250" in caplog.text
    assert result.turns == [{"start": 0.0, "end": 1.2, "speaker": "SPEAKER_00"}]
    assert result.aligned_segments == [
        {
            "start": 0.0,
            "end": 1.2,
            "text": "hello",
            "speaker": "SPEAKER_00",
            "words": [{"start": 0.0, "end": 0.5, "word": "hello", "score": 0.0}],
        }
    ]
    assert result.dedup_removed == 0


def test_default_diarization_provider_uses_zh_alignment_override(monkeypatch):
    pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
    pipeline.device = "cpu"
    calls = []

    class FakeDiarizationResult:
        def itertracks(self, yield_label=False):
            assert yield_label is True
            yield SimpleNamespace(start=0.0, end=1.2), None, "SPEAKER_00"

    class FakeDiarizer:
        def __call__(self, audio_path, **kwargs):
            return FakeDiarizationResult()

    pipeline._diarization = FakeDiarizer()
    monkeypatch.setattr(
        diarization_default,
        "WHISPERX_ALIGN_MODEL_MAP",
        {"zh": "safe/zh-align-model"},
    )
    whisperx = sys.modules["whisperx"]
    monkeypatch.setattr(
        whisperx,
        "load_audio",
        lambda audio_path: f"audio:{audio_path}",
        raising=False,
    )

    def fake_load_align_model(language_code, device, model_name):
        calls.append(("load_align_model", language_code, device, model_name))
        return "align-model", {"language": language_code, "device": device}

    monkeypatch.setattr(
        whisperx,
        "load_align_model",
        fake_load_align_model,
        raising=False,
    )
    monkeypatch.setattr(
        whisperx,
        "align",
        lambda segments, align_model, align_metadata, audio, device, return_char_alignments=False: {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.2,
                    "text": "你好",
                    "words": [{"start": 0.0, "end": 0.5, "word": "你"}],
                }
            ]
        },
        raising=False,
    )

    result = default_diarization_provider.diarize(
        DiarizationRequest(
            pipeline=pipeline,
            audio_path="demo.wav",
            transcription_result={
                "segments": [{"start": 0.0, "end": 1.2, "text": "你好"}],
                "language": "zh",
            },
        )
    )

    assert calls == [("load_align_model", "zh", "cpu", "safe/zh-align-model")]
    assert result.aligned_segments[0]["words"] == [
        {"start": 0.0, "end": 0.5, "word": "你", "score": 0.0}
    ]
    assert result.metadata["alignment"] == {
        "status": "succeeded",
        "language": "zh",
        "model": "safe/zh-align-model",
        "model_source": "override",
        "cache_only": False,
    }


def test_default_diarization_provider_applies_model_dir_and_cache_only(
    monkeypatch,
):
    pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
    pipeline.device = "cpu"
    calls = []

    class FakeDiarizationResult:
        def itertracks(self, yield_label=False):
            assert yield_label is True
            yield SimpleNamespace(start=0.0, end=1.2), None, "SPEAKER_00"

    class FakeDiarizer:
        def __call__(self, audio_path, **kwargs):
            return FakeDiarizationResult()

    pipeline._diarization = FakeDiarizer()
    monkeypatch.setattr(
        diarization_default,
        "WHISPERX_ALIGN_MODEL_MAP",
        {"zh": "safe/zh-align-model"},
    )
    monkeypatch.setattr(diarization_default, "WHISPERX_ALIGN_MODEL_DIR", "/cache")
    monkeypatch.setattr(diarization_default, "WHISPERX_ALIGN_CACHE_ONLY", True)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    hub_constants = ModuleType("huggingface_hub.constants")
    hub_constants.HF_HUB_OFFLINE = False
    transformers_hub = ModuleType("transformers.utils.hub")
    transformers_hub._is_offline_mode = False
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", hub_constants)
    monkeypatch.setitem(sys.modules, "transformers.utils.hub", transformers_hub)
    whisperx = sys.modules["whisperx"]
    monkeypatch.setattr(
        whisperx,
        "load_audio",
        lambda audio_path: f"audio:{audio_path}",
        raising=False,
    )

    def fake_load_align_model(language_code, device, model_name, model_dir):
        calls.append(
            (
                language_code,
                device,
                model_name,
                model_dir,
                os.environ.get("HF_HUB_OFFLINE"),
                os.environ.get("TRANSFORMERS_OFFLINE"),
                hub_constants.HF_HUB_OFFLINE,
                transformers_hub._is_offline_mode,
            )
        )
        return "align-model", {"language": language_code, "device": device}

    monkeypatch.setattr(
        whisperx,
        "load_align_model",
        fake_load_align_model,
        raising=False,
    )
    monkeypatch.setattr(
        whisperx,
        "align",
        lambda segments, align_model, align_metadata, audio, device, return_char_alignments=False: {
            "segments": segments
        },
        raising=False,
    )

    result = default_diarization_provider.diarize(
        DiarizationRequest(
            pipeline=pipeline,
            audio_path="demo.wav",
            transcription_result={
                "segments": [{"start": 0.0, "end": 1.2, "text": "你好"}],
                "language": "zh",
            },
        )
    )

    assert calls == [
        ("zh", "cpu", "safe/zh-align-model", "/cache", "1", "1", True, True),
    ]
    assert os.environ.get("HF_HUB_OFFLINE") is None
    assert os.environ.get("TRANSFORMERS_OFFLINE") is None
    assert hub_constants.HF_HUB_OFFLINE is False
    assert transformers_hub._is_offline_mode is False
    assert result.metadata["alignment"]["cache_only"] is True


def test_default_diarization_provider_attempts_zh_alignment_by_default(monkeypatch):
    pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
    pipeline.device = "cpu"
    calls = []

    class FakeDiarizationResult:
        def itertracks(self, yield_label=False):
            assert yield_label is True
            yield SimpleNamespace(start=0.0, end=1.2), None, "SPEAKER_00"

    class FakeDiarizer:
        def __call__(self, audio_path, **kwargs):
            return FakeDiarizationResult()

    pipeline._diarization = FakeDiarizer()
    whisperx = sys.modules["whisperx"]
    monkeypatch.setattr(
        whisperx,
        "load_audio",
        lambda audio_path: f"audio:{audio_path}",
        raising=False,
    )

    def fake_load_align_model(language_code, device):
        calls.append(("load_align_model", language_code, device))
        return "align-model", {"language": language_code, "device": device}

    monkeypatch.setattr(
        whisperx,
        "load_align_model",
        fake_load_align_model,
        raising=False,
    )
    monkeypatch.setattr(
        whisperx,
        "align",
        lambda segments, align_model, align_metadata, audio, device, return_char_alignments=False: {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.2,
                    "text": "你好",
                    "words": [{"start": 0.0, "end": 0.5, "word": "你"}],
                }
            ]
        },
        raising=False,
    )

    result = default_diarization_provider.diarize(
        DiarizationRequest(
            pipeline=pipeline,
            audio_path="demo.wav",
            transcription_result={
                "segments": [{"start": 0.0, "end": 1.2, "text": "你好"}],
                "language": "zh",
            },
        )
    )

    assert calls == [("load_align_model", "zh", "cpu")]
    assert result.aligned_segments[0]["words"] == [
        {"start": 0.0, "end": 0.5, "word": "你", "score": 0.0}
    ]
    assert result.metadata["alignment"] == {
        "status": "succeeded",
        "language": "zh",
        "model": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
        "model_source": "whisperx_default",
        "cache_only": False,
    }
    assert result.dedup_removed == 0


def test_default_diarization_provider_caches_alignment_model_on_configured_device(
    monkeypatch,
    caplog,
):
    pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
    pipeline.device = "cuda:1"
    calls = []

    class FakeDiarizationResult:
        def itertracks(self, yield_label=False):
            assert yield_label is True
            yield SimpleNamespace(start=0.0, end=1.2), None, "SPEAKER_00"

    class FakeDiarizer:
        def __call__(self, audio_path, **kwargs):
            return FakeDiarizationResult()

    pipeline._diarization = FakeDiarizer()
    monkeypatch.setattr(diarization_default, "WHISPERX_ALIGN_DEVICE", "cpu")
    whisperx = sys.modules["whisperx"]
    monkeypatch.setattr(
        whisperx,
        "load_audio",
        lambda audio_path: f"audio:{audio_path}",
        raising=False,
    )

    def fake_load_align_model(language_code, device):
        calls.append(("load_align_model", language_code, device))
        return object(), {"language": language_code, "device": device}

    def fake_align(
        segments,
        align_model,
        align_metadata,
        audio,
        device,
        return_char_alignments=False,
    ):
        calls.append(("align", align_metadata["device"], device))
        return {"segments": segments}

    monkeypatch.setattr(
        whisperx,
        "load_align_model",
        fake_load_align_model,
        raising=False,
    )
    monkeypatch.setattr(whisperx, "align", fake_align, raising=False)

    request = DiarizationRequest(
        pipeline=pipeline,
        audio_path="demo.wav",
        transcription_result={
            "segments": [{"start": 0.0, "end": 1.2, "text": "你好"}],
            "language": "zh",
        },
    )

    with caplog.at_level("INFO", logger=diarization_default.logger.name):
        default_diarization_provider.diarize(request)
        default_diarization_provider.diarize(request)

    assert calls == [
        ("load_align_model", "zh", "cpu"),
        ("align", "cpu", "cpu"),
        ("align", "cpu", "cpu"),
    ]
    assert "Loaded WhisperX alignment model" in caplog.text
    assert "cold_load=True" in caplog.text
    assert "Reusing WhisperX alignment model (hot reuse" in caplog.text
    assert pipeline._alignment_device == "cpu"


def test_default_diarization_provider_skips_zh_alignment_when_explicitly_disabled(
    monkeypatch,
):
    pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
    pipeline.device = "cpu"

    class FakeDiarizationResult:
        def itertracks(self, yield_label=False):
            assert yield_label is True
            yield SimpleNamespace(start=0.0, end=1.2), None, "SPEAKER_00"

    class FakeDiarizer:
        def __call__(self, audio_path, **kwargs):
            return FakeDiarizationResult()

    pipeline._diarization = FakeDiarizer()
    monkeypatch.setattr(
        diarization_default,
        "WHISPERX_ALIGN_DISABLED_LANGUAGES",
        frozenset({"zh"}),
    )
    whisperx = sys.modules["whisperx"]
    monkeypatch.setattr(
        whisperx,
        "load_audio",
        lambda audio_path: pytest.fail(
            "explicitly disabled zh alignment should not load audio"
        ),
        raising=False,
    )
    monkeypatch.setattr(
        whisperx,
        "load_align_model",
        lambda **kwargs: pytest.fail(
            "explicitly disabled zh alignment should not load a model"
        ),
        raising=False,
    )

    result = default_diarization_provider.diarize(
        DiarizationRequest(
            pipeline=pipeline,
            audio_path="demo.wav",
            transcription_result={
                "segments": [{"start": 0.0, "end": 1.2, "text": "你好"}],
                "language": "zh",
            },
        )
    )

    assert result.aligned_segments == [
        {
            "start": 0.0,
            "end": 1.2,
            "text": "你好",
            "speaker": "SPEAKER_00",
        }
    ]
    assert result.metadata["alignment"] == {
        "status": "skipped",
        "language": "zh",
        "model": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
        "reason": "language_disabled",
        "actionable_hint": (
            "Remove zh from WHISPERX_ALIGN_DISABLED_LANGUAGES to retry alignment, "
            "or set WHISPERX_ALIGN_MODEL_MAP=zh=<model> for a replacement model."
        ),
    }
    assert result.dedup_removed == 0


def test_default_diarization_provider_classifies_torch_safety_block(
    monkeypatch,
    caplog,
):
    pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
    pipeline.device = "cpu"

    class FakeDiarizationResult:
        def itertracks(self, yield_label=False):
            assert yield_label is True
            yield SimpleNamespace(start=0.0, end=1.2), None, "SPEAKER_00"

    class FakeDiarizer:
        def __call__(self, audio_path, **kwargs):
            return FakeDiarizationResult()

    pipeline._diarization = FakeDiarizer()
    whisperx = sys.modules["whisperx"]
    monkeypatch.setattr(
        whisperx,
        "load_audio",
        lambda audio_path: f"audio:{audio_path}",
        raising=False,
    )
    monkeypatch.setattr(
        whisperx,
        "load_align_model",
        lambda language_code, device: (_ for _ in ()).throw(
            RuntimeError(
                "Due to a serious vulnerability issue in torch.load, even with "
                "weights_only=True, we now require users to upgrade torch to at "
                "least v2.6 in order to use the function. This version restriction "
                "does not apply when loading files with safetensors."
            )
        ),
        raising=False,
    )

    with caplog.at_level("WARNING", logger="providers.diarization.default"):
        result = default_diarization_provider.diarize(
            DiarizationRequest(
                pipeline=pipeline,
                audio_path="demo.wav",
                transcription_result={
                    "segments": [{"start": 0.0, "end": 1.2, "text": "你好"}],
                    "language": "zh",
                },
            )
        )

    alignment = result.metadata["alignment"]
    assert alignment["status"] == "failed"
    assert alignment["language"] == "zh"
    assert alignment["model"] == "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
    assert alignment["reason"] == "torch_version_blocked"
    assert alignment["error_type"] == "RuntimeError"
    assert "torch>=2.6" in alignment["actionable_hint"]
    assert "safetensors" in alignment["actionable_hint"]
    assert alignment["reason"] != "not_found"
    assert "could not be found" not in caplog.text


def test_default_diarization_provider_sanitizes_alignment_failure_metadata(
    monkeypatch,
    caplog,
):
    pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
    pipeline.device = "cpu"

    class FakeDiarizationResult:
        def itertracks(self, yield_label=False):
            assert yield_label is True
            yield SimpleNamespace(start=0.0, end=1.2), None, "SPEAKER_00"

    class FakeDiarizer:
        def __call__(self, audio_path, **kwargs):
            return FakeDiarizationResult()

    pipeline._diarization = FakeDiarizer()
    whisperx = sys.modules["whisperx"]
    monkeypatch.setattr(
        whisperx,
        "load_audio",
        lambda audio_path: f"audio:{audio_path}",
        raising=False,
    )
    monkeypatch.setattr(
        whisperx,
        "load_align_model",
        lambda language_code, device: (_ for _ in ()).throw(
            RuntimeError("token=secret /private/cache/model.bin")
        ),
        raising=False,
    )

    with caplog.at_level("WARNING", logger="providers.diarization.default"):
        result = default_diarization_provider.diarize(
            DiarizationRequest(
                pipeline=pipeline,
                audio_path="demo.wav",
                transcription_result={
                    "segments": [{"start": 0.0, "end": 1.2, "text": "hello"}],
                    "language": "en",
                },
            )
        )

    assert result.metadata["alignment"] == {
        "status": "failed",
        "language": "en",
        "model": None,
        "reason": "load_or_align_failed",
        "error_type": "RuntimeError",
        "model_source": "whisperx_default",
        "cache_only": False,
        "actionable_hint": (
            "Check WHISPERX_ALIGN_MODEL_MAP, WHISPERX_ALIGN_MODEL_DIR, "
            "WHISPERX_ALIGN_CACHE_ONLY, network access, and model compatibility."
        ),
    }
    assert "token=secret" not in caplog.text
    assert "/private/cache" not in caplog.text


def test_default_embedding_provider_uses_pipeline_embedding_resource(monkeypatch):
    pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
    pipeline.device = "cpu"
    calls = []

    class FakeTensor:
        def __init__(self, channels, frames):
            self.shape = (channels, frames)

        def mean(self, dim=0, keepdim=True):
            assert dim == 0
            return FakeTensor(1, self.shape[1])

        def to(self, device):
            calls.append(("to", device, self.shape[1]))
            return self

    class FakeEmbeddingModel:
        def __call__(self, payload):
            calls.append(
                (
                    "embedding_model",
                    payload["sample_rate"],
                    payload["waveform"].shape[1],
                )
            )
            return [float(payload["waveform"].shape[1]), 1.0]

    class FakeInfo:
        sample_rate = 16000

    pipeline._embedding_model = FakeEmbeddingModel()
    monkeypatch.setattr(
        embedding_default.audio_io, "info", lambda audio_path: FakeInfo()
    )
    monkeypatch.setattr(
        embedding_default.audio_io,
        "load",
        lambda audio_path, frame_offset, num_frames: (FakeTensor(1, num_frames), 16000),
    )

    result = default_speaker_embedding_provider.extract_embeddings(
        SpeakerEmbeddingRequest(
            pipeline=pipeline,
            audio_path="demo.wav",
            diarization_turns=[
                {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0},
                {"speaker": "SPEAKER_00", "start": 2.0, "end": 4.0},
                {"speaker": "SPEAKER_SKIP", "start": 0.0, "end": 1.0},
            ],
        )
    )

    assert list(result.speaker_embeddings) == ["SPEAKER_00"]
    assert result.speaker_embeddings["SPEAKER_00"].tolist() == [32000.0, 1.0]
    assert calls == [
        ("to", "cpu", 32000),
        ("embedding_model", 16000, 32000),
        ("to", "cpu", 32000),
        ("embedding_model", 16000, 32000),
    ]


def test_default_embedding_provider_moves_chunks_to_embedding_device(
    monkeypatch, caplog
):
    pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
    pipeline.device = "cuda:0"
    pipeline._embedding_device = "cuda:1"
    calls = []
    perf_values = iter([30.0, 30.75])

    class FakeTensor:
        def __init__(self, channels, frames):
            self.shape = (channels, frames)

        def mean(self, dim=0, keepdim=True):
            assert dim == 0
            return FakeTensor(1, self.shape[1])

        def to(self, device):
            calls.append(("to", device, self.shape[1]))
            return self

    class FakeEmbeddingModel:
        def __call__(self, payload):
            calls.append(("embedding_model", payload["waveform"].shape[1]))
            return [1.0, 2.0]

    class FakeInfo:
        sample_rate = 16000

    pipeline._embedding_model = FakeEmbeddingModel()
    monkeypatch.setattr(
        embedding_default.audio_io, "info", lambda audio_path: FakeInfo()
    )
    monkeypatch.setattr(
        embedding_default.audio_io,
        "load",
        lambda audio_path, frame_offset, num_frames: (FakeTensor(1, num_frames), 16000),
    )
    monkeypatch.setattr(
        embedding_default.time,
        "perf_counter",
        lambda: next(perf_values),
    )

    with caplog.at_level("INFO", logger=embedding_default.logger.name):
        result = default_speaker_embedding_provider.extract_embeddings(
            SpeakerEmbeddingRequest(
                pipeline=pipeline,
                audio_path="demo.wav",
                diarization_turns=[
                    {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0},
                ],
            )
        )

    assert result.speaker_embeddings["SPEAKER_00"].tolist() == [1.0, 2.0]
    assert "embedding_processing_timing model=wespeaker elapsed_s=0.750" in caplog.text
    assert "speaker_count=1" in caplog.text
    assert calls == [
        ("to", "cuda:1", 32000),
        ("embedding_model", 32000),
    ]


def test_default_embedding_provider_prefers_bounded_short_audio_preload(monkeypatch):
    pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
    pipeline.device = "cpu"
    calls = []

    class FakeTensor:
        def __init__(self, channels, frames):
            self.shape = (channels, frames)

        def __getitem__(self, key):
            channel_key, frame_key = key
            assert isinstance(channel_key, slice)
            assert channel_key == slice(None)
            start = frame_key.start or 0
            stop = frame_key.stop or self.shape[1]
            return FakeTensor(self.shape[0], max(stop - start, 0))

        def contiguous(self):
            calls.append(("contiguous", self.shape[1]))
            return self

        def mean(self, dim=0, keepdim=True):
            assert dim == 0
            assert keepdim is True
            return FakeTensor(1, self.shape[1])

        def to(self, device):
            calls.append(("to", device, self.shape[1]))
            return self

    class FakeEmbeddingModel:
        def __call__(self, payload):
            calls.append(("embedding_model", payload["waveform"].shape[1]))
            return [float(payload["waveform"].shape[1]), 2.0]

    pipeline._embedding_model = FakeEmbeddingModel()
    monkeypatch.setattr(
        embedding_default.audio_io,
        "info",
        lambda audio_path: SimpleNamespace(sample_rate=16000, num_frames=64000),
    )
    monkeypatch.setattr(
        embedding_default.audio_io,
        "load",
        lambda audio_path: calls.append(("preload", audio_path)) or (FakeTensor(1, 64000), 16000),
    )

    result = default_speaker_embedding_provider.extract_embeddings(
        SpeakerEmbeddingRequest(
            pipeline=pipeline,
            audio_path="demo.wav",
            diarization_turns=[
                {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0},
                {"speaker": "SPEAKER_00", "start": 2.0, "end": 4.0},
            ],
        )
    )

    assert result.speaker_embeddings["SPEAKER_00"].tolist() == [32000.0, 2.0]
    assert calls == [
        ("preload", "demo.wav"),
        ("contiguous", 32000),
        ("to", "cpu", 32000),
        ("embedding_model", 32000),
        ("contiguous", 32000),
        ("to", "cpu", 32000),
        ("embedding_model", 32000),
    ]


def test_default_embedding_provider_uses_selected_device_after_first_lazy_load(
    monkeypatch,
):
    pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
    pipeline.device = "cuda"
    pipeline._configured_device = "cuda"
    pipeline._embedding_device = None
    pipeline._embedding_model = None
    pipeline.hf_token = None
    calls = []

    class FakeTensor:
        def __init__(self, channels, frames):
            self.shape = (channels, frames)

        def mean(self, dim=0, keepdim=True):
            assert dim == 0
            return FakeTensor(1, self.shape[1])

        def to(self, device):
            calls.append(("chunk_to", str(device), self.shape[1]))
            return self

    class FakeInfo:
        sample_rate = 16000

    class FakeEmbeddingModel:
        @classmethod
        def from_pretrained(cls, model_ref, use_auth_token=None):
            calls.append(("embedding_load", model_ref, use_auth_token))
            return cls()

        def to(self, device):
            calls.append(("model_to", str(device)))
            return self

    class FakeInference:
        def __init__(self, model, window):
            calls.append(("inference", window))

        def __call__(self, payload):
            calls.append(("embedding_model", payload["waveform"].shape[1]))
            return [1.0, 2.0]

    pyannote_audio = ModuleType("pyannote.audio")
    pyannote_audio.Model = FakeEmbeddingModel
    pyannote_audio.Inference = FakeInference

    monkeypatch.setitem(sys.modules, "pyannote.audio", pyannote_audio)
    monkeypatch.setattr(
        orchestrator,
        "resolve_hf_model_ref",
        lambda repo_id, *, token, purpose: repo_id,
    )
    monkeypatch.setattr(
        orchestrator,
        "select_best_cuda_device",
        lambda configured: calls.append(("select", configured)) or "cuda:1",
    )
    monkeypatch.setattr(
        embedding_default.audio_io, "info", lambda audio_path: FakeInfo()
    )
    monkeypatch.setattr(
        embedding_default.audio_io,
        "load",
        lambda audio_path, frame_offset, num_frames: (FakeTensor(1, num_frames), 16000),
    )

    result = default_speaker_embedding_provider.extract_embeddings(
        SpeakerEmbeddingRequest(
            pipeline=pipeline,
            audio_path="demo.wav",
            diarization_turns=[
                {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0},
            ],
        )
    )

    assert result.speaker_embeddings["SPEAKER_00"].tolist() == [1.0, 2.0]
    assert pipeline._embedding_device == "cuda:1"
    assert calls == [
        ("select", "cuda"),
        (
            "embedding_load",
            "pyannote/wespeaker-voxceleb-resnet34-LM",
            None,
        ),
        ("model_to", "cuda:1"),
        ("inference", "whole"),
        ("chunk_to", "cuda:1", 32000),
        ("embedding_model", 32000),
    ]


@pytest.mark.parametrize("preload", [False, True])
def test_embedding_selects_metadata_top_ten_per_speaker_before_decode(
    monkeypatch, preload
):
    import numpy as np

    native_sr = 8000
    total_frames = 40 * native_sr
    decoded = []
    processed = []
    events = []

    class ObservedTurns(list):
        def __iter__(self):
            yield from super().__iter__()
            events.append("metadata_selected")

    class Chunk:
        def __init__(self, offset, frames, channels=2):
            self.offset = offset
            self.shape = (channels, frames)

        def __getitem__(self, key):
            _, frames = key
            return Chunk(frames.start, min(frames.stop, self.shape[1]) - frames.start)

        def contiguous(self):
            return self

        def mean(self, dim=0, keepdim=True):
            assert dim == 0 and keepdim
            return Chunk(self.offset, self.shape[1], 1)

        def to(self, device):
            assert device == "cpu"
            return self

    def info(path):
        assert path == "synthetic-original.wav"
        return SimpleNamespace(sample_rate=native_sr, num_frames=total_frames)

    def load(path, frame_offset=0, num_frames=-1):
        assert "metadata_selected" in events
        assert path == "synthetic-original.wav"
        decoded.append((frame_offset, num_frames))
        frames = total_frames if num_frames == -1 else min(num_frames, total_frames - frame_offset)
        return Chunk(frame_offset, frames), native_sr

    def resample(chunk, source_sr, target_sr):
        assert source_sr == native_sr and target_sr == 16000
        return Chunk(chunk.offset, chunk.shape[1] * 2, chunk.shape[0])

    def embedding(payload):
        chunk = payload["waveform"]
        assert payload["sample_rate"] == 16000
        assert chunk.shape[0] == 1
        processed.append((chunk.offset, chunk.shape[1]))
        return [float(chunk.offset), float(chunk.shape[1])]

    turns = []
    for speaker in ("SPEAKER_A", "SPEAKER_B"):
        for index in range(13):
            start = index * 2.0
            turns.append({"speaker": speaker, "start": start, "end": start + 2 + index})
    turns += [
        {"speaker": "SPEAKER_A", "start": 39.0, "end": 49.0},
        {"speaker": "SPEAKER_A", "start": 50.0, "end": 60.0},
        {"speaker": "TOO_SHORT", "start": 0.0, "end": 1.0},
    ]
    monkeypatch.setattr(embedding_default, "MIN_EMBED_DURATION", 1.5)
    monkeypatch.setattr(embedding_default, "MAX_EMBED_DURATION", 10.0)
    monkeypatch.setattr(embedding_default, "EMBEDDING_PRELOAD_MAX_AUDIO_DURATION_SEC", 1800 if preload else 1)
    monkeypatch.setattr(embedding_default.audio_io, "info", info)
    monkeypatch.setattr(embedding_default.audio_io, "load", load)
    monkeypatch.setattr(embedding_default.torchaudio.functional, "resample", resample, raising=False)

    result = embedding_default.extract_embeddings_for_turns(
        SimpleNamespace(device="cpu", embedding_model=embedding),
        "synthetic-original.wav",
        ObservedTurns(turns),
    )

    expected_decodes = []
    expected_processed = []
    assert list(result) == ["SPEAKER_A", "SPEAKER_B"]
    for speaker in result:
        candidates = []
        for turn in turns:
            if turn["speaker"] != speaker:
                continue
            start = int(turn["start"] * native_sr)
            length = int(turn["end"] * native_sr) - start
            if length < int(1.5 * native_sr):
                continue
            length = min(length, 10 * native_sr)
            actual_length = min(length, max(total_frames - start, 0))
            if actual_length:
                candidates.append((start, length, actual_length * 2))
        selected = sorted(candidates, key=lambda item: item[2], reverse=True)[:10]
        expected_decodes += [(start, length) for start, length, _ in selected]
        expected = [(start, actual_length) for start, _, actual_length in selected]
        expected_processed += expected
        np.testing.assert_allclose(result[speaker], np.mean(expected, axis=0))
    assert processed == expected_processed
    assert decoded == ([(0, -1)] if preload else expected_decodes)


def test_embedding_preload_failure_falls_back_to_selected_segments(monkeypatch):
    calls = []

    class Chunk:
        shape = (1, 32000)

        def to(self, device):
            return self

    def load(path, frame_offset=0, num_frames=-1):
        calls.append((frame_offset, num_frames))
        if num_frames == -1:
            raise OSError("synthetic preload failure")
        return Chunk(), 16000

    monkeypatch.setattr(embedding_default.audio_io, "info", lambda _: SimpleNamespace(sample_rate=16000, num_frames=64000))
    monkeypatch.setattr(embedding_default.audio_io, "load", load)
    monkeypatch.setattr(embedding_default, "EMBEDDING_PRELOAD_MAX_AUDIO_DURATION_SEC", 1800)
    result = embedding_default.extract_embeddings_for_turns(
        SimpleNamespace(device="cpu", embedding_model=lambda payload: [1.0, 2.0]),
        "synthetic.wav",
        [{"speaker": "SPEAKER_A", "start": 0.0, "end": 2.0}],
    )
    assert result["SPEAKER_A"].tolist() == [1.0, 2.0]
    assert calls == [(0, -1), (0, 32000)]


@pytest.mark.parametrize("failure", [False, True])
def test_alignment_offline_flags_restore_existing_values_on_all_exits(monkeypatch, failure):
    monkeypatch.setattr(diarization_default, "WHISPERX_ALIGN_CACHE_ONLY", True)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "original")
    hub = SimpleNamespace(HF_HUB_OFFLINE=False)
    transformers_hub = SimpleNamespace(_is_offline_mode=False)
    modules = {"huggingface_hub.constants": hub, "transformers.utils.hub": transformers_hub}
    monkeypatch.setattr(diarization_default, "import_module", modules.__getitem__)

    def run():
        with diarization_default._cache_only_alignment_environment():
            assert os.environ["HF_HUB_OFFLINE"] == "1"
            assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
            assert hub.HF_HUB_OFFLINE is True
            assert transformers_hub._is_offline_mode is True
            if failure:
                raise OSError("synthetic model load failure")

    if failure:
        with pytest.raises(OSError):
            run()
    else:
        run()
    assert os.environ["HF_HUB_OFFLINE"] == "0"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "original"
    assert hub.HF_HUB_OFFLINE is False
    assert transformers_hub._is_offline_mode is False


def test_alignment_budget_skips_before_whisperx_audio_or_model_load(monkeypatch):
    monkeypatch.setattr(diarization_default, "WHISPERX_ALIGN_MAX_AUDIO_DURATION_SEC", 7200)
    monkeypatch.setattr(diarization_default, "WHISPERX_ALIGN_DISABLED_LANGUAGES", frozenset())
    monkeypatch.setattr(diarization_default, "audio_duration_seconds", lambda _: 7201.0)
    whisperx = sys.modules["whisperx"]
    for name in ("load_audio", "load_align_model", "align"):
        monkeypatch.setattr(whisperx, name, lambda *a, **kw: pytest.fail("over-budget alignment must not decode or load"), raising=False)
    segments, metadata = diarization_default.align_diarized_segments_with_metadata(
        SimpleNamespace(device="cpu"),
        {"language": "en", "segments": [{"start": 0.0, "end": 2.0, "text": "synthetic"}]},
        [{"speaker": "SPEAKER_A", "start": 0.0, "end": 2.0}],
        "synthetic.wav",
    )
    assert metadata["status"] == "skipped"
    assert metadata["reason"] == "duration_budget_exceeded"
    assert segments[0]["speaker"] == "SPEAKER_A"


@pytest.mark.parametrize("metadata_failure", ["read_error", "missing_frames", "invalid_rate", "nonfinite_duration"])
def test_alignment_unknown_duration_skips_without_loading_and_keeps_speakers(
    monkeypatch, caplog, metadata_failure
):
    from copy import deepcopy
    from infra.audio import metadata

    def info(path):
        if metadata_failure == "read_error":
            raise OSError("synthetic header read failure")
        if metadata_failure == "missing_frames":
            return SimpleNamespace(sample_rate=16000)
        if metadata_failure == "invalid_rate":
            return SimpleNamespace(sample_rate=0, num_frames=16000)
        return SimpleNamespace(sample_rate=16000, num_frames=float("nan"))

    monkeypatch.setattr(metadata.audio_io, "info", info)
    monkeypatch.setattr(diarization_default, "audio_duration_seconds", metadata.audio_duration_seconds)
    monkeypatch.setattr(diarization_default, "WHISPERX_ALIGN_DISABLED_LANGUAGES", frozenset())
    whisperx = sys.modules["whisperx"]
    for name in ("load_audio", "load_align_model", "align"):
        monkeypatch.setattr(whisperx, name, lambda *a, **kw: pytest.fail("unknown-duration alignment must not decode or load"), raising=False)
    transcription = {
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "synthetic first"},
            {"start": 2.0, "end": 4.0, "text": "synthetic second"},
        ],
    }
    turns = [
        {"speaker": "SPEAKER_A", "start": 0.0, "end": 2.0},
        {"speaker": "SPEAKER_B", "start": 2.0, "end": 4.0},
    ]
    original_transcription = deepcopy(transcription)
    original_turns = deepcopy(turns)
    with caplog.at_level("WARNING", logger=diarization_default.logger.name):
        segments, result_metadata = diarization_default.align_diarized_segments_with_metadata(
            SimpleNamespace(device="cpu"), transcription, turns, "synthetic-original.wav"
        )
    assert result_metadata["status"] == "skipped"
    assert result_metadata["reason"] == "audio_duration_unavailable"
    assert result_metadata["duration_s"] is None
    assert [segment["speaker"] for segment in segments] == ["SPEAKER_A", "SPEAKER_B"]
    assert transcription == original_transcription
    assert turns == original_turns
    assert "reason=audio_duration_unavailable" in caplog.text


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("initial_flags", [(None, None), ("0", "1"), ("1", "0")])
def test_alignment_first_import_offline_flags_restore_in_fresh_process(
    failure, initial_flags
):
    provider_path = Path(__file__).resolve().parents[2] / "app/providers/diarization/default.py"
    script = textwrap.dedent("""\
        import importlib.abc
        import importlib.util
        import os
        import sys
        from types import ModuleType

        def stub(name, **attrs):
            module = ModuleType(name)
            module.__path__ = []
            module.__dict__.update(attrs)
            sys.modules[name] = module
            return module

        stub("config", WHISPERX_ALIGN_DEVICE="cpu", WHISPERX_ALIGN_CACHE_ONLY=True,
             WHISPERX_ALIGN_DISABLED_LANGUAGES=frozenset(),
             WHISPERX_ALIGN_MAX_AUDIO_DURATION_SEC=7200,
             WHISPERX_ALIGN_MODEL_DIR=None, WHISPERX_ALIGN_MODEL_MAP={})
        for name in ("pipeline", "pipeline.stages", "pipeline.stages.diarization",
                     "infra", "infra.audio"):
            stub(name)
        stub("pipeline.contracts", DiarizationProvider=object,
             DiarizationRequest=object, DiarizationResult=object)
        stub("pipeline.stages.diarization.alignment",
             build_aligned_segments=lambda *args: [], dedup_short_segments=lambda x: x)
        stub("infra.audio.metadata", audio_duration_seconds=lambda path: 1.0)

        flags = {
            "huggingface_hub.constants": ("HF_HUB_OFFLINE", "HF_HUB_OFFLINE"),
            "transformers.utils.hub": ("_is_offline_mode", "TRANSFORMERS_OFFLINE"),
        }
        packages = {"huggingface_hub", "transformers", "transformers.utils"}
        initialized_under = {}

        class FirstImportLoader(importlib.abc.Loader):
            def create_module(self, spec):
                return None

            def exec_module(self, module):
                if module.__name__ in flags:
                    attribute, variable = flags[module.__name__]
                    initialized_under[variable] = os.environ.get(variable)
                    setattr(module, attribute, os.environ.get(variable) == "1")

        class FirstImportFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in flags or fullname in packages:
                    return importlib.util.spec_from_loader(
                        fullname, FirstImportLoader(), is_package=fullname in packages
                    )
                return None

        sys.meta_path.insert(0, FirstImportFinder())
        spec = importlib.util.spec_from_file_location("alignment_under_test", sys.argv[1])
        provider = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(provider)
        previous = {key: os.environ.get(key) for key in
                    ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
        assert not any(name in sys.modules for name in flags)

        class ExpectedFailure(Exception):
            pass

        try:
            with provider._cache_only_alignment_environment():
                for module_name, (attribute, variable) in flags.items():
                    assert os.environ[variable] == "1"
                    assert getattr(sys.modules[module_name], attribute) is True
                if sys.argv[2] == "1":
                    raise ExpectedFailure("synthetic context failure")
        except ExpectedFailure:
            assert sys.argv[2] == "1"

        assert initialized_under == previous
        assert {key: os.environ.get(key) for key in previous} == previous
        for module_name, (attribute, variable) in flags.items():
            assert getattr(sys.modules[module_name], attribute) is (previous[variable] == "1")
        print("offline_restore_ok")
    """)
    environment = os.environ.copy()
    for variable, value in zip(("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"), initial_flags):
        if value is None:
            environment.pop(variable, None)
        else:
            environment[variable] = value
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(provider_path), "1" if failure else "0"],
        env=environment, capture_output=True, text=True, check=False, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "offline_restore_ok"
