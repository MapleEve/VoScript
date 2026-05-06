"""Unit tests for the new audio provider / infra layering."""

from __future__ import annotations

import json
import subprocess
import sys
from contextlib import contextmanager
from inspect import signature
from pathlib import Path
from types import ModuleType

import pytest
import infra.audio.hash_index as hash_index_module
import infra.audio as audio_infra
import providers
import providers.enhance.default as enhance_default
import providers.normalize.default as normalize_default
import providers.voiceprint_match.default as voiceprint_match_default
from infra.audio import JsonAudioArtifactIndex
from pipeline.contracts import (
    AudioEnhancementRequest,
    AudioNormalizationRequest,
    UploadPersistenceRequest,
    VoiceprintMatchRequest,
)
from api.routers.transcriptions import transcribe


def test_audio_layer_entrypoints_point_to_new_modules():
    assert providers.convert_to_wav.__module__ == "providers.normalize"
    assert providers.maybe_denoise.__module__ == "providers.enhance"
    assert audio_infra.lookup_hash.__module__ == "infra.audio.hash_index"
    assert audio_infra.register_hash.__module__ == "infra.audio.hash_index"
    assert audio_infra.safe_tr_dir.__module__ == "infra.audio.paths"


def test_contract_entrypoints_are_instantiable():
    normalization = AudioNormalizationRequest(input_path=Path("sample.mp3"))
    enhancement = AudioEnhancementRequest(wav_path=Path("sample.wav"))
    persistence = UploadPersistenceRequest(
        file=None,
        save_path=Path("sample.wav"),
        max_bytes=1024,
        chunk_size=256,
    )

    assert normalization.target_format == "wav"
    assert enhancement.wav_path.name == "sample.wav"
    assert persistence.chunk_size == 256


def test_transcribe_omitted_denoise_model_preserves_service_default():
    denoise_default = signature(transcribe).parameters["denoise_model"].default
    snr_default = signature(transcribe).parameters["snr_threshold"].default

    assert getattr(denoise_default, "default", denoise_default) is None
    assert getattr(snr_default, "default", snr_default) is None


def test_denoise_env_default_applies_when_api_omits_model(monkeypatch, tmp_path):
    wav_path = tmp_path / "clean.wav"
    wav_path.write_bytes(b"stub")
    monkeypatch.setattr(enhance_default, "DENOISE_MODEL", "deepfilternet")
    monkeypatch.setattr(enhance_default, "DENOISE_SNR_THRESHOLD", 10.0)
    monkeypatch.setattr(enhance_default, "_estimate_snr", lambda path: 15.0)

    result = enhance_default.ConditionalDenoiseEnhancer().enhance(
        AudioEnhancementRequest(wav_path=wav_path)
    )

    assert result.applied is False
    assert result.model == "deepfilternet"
    assert result.output_path == wav_path


def test_denoise_api_none_explicitly_disables_env_default(monkeypatch, tmp_path):
    wav_path = tmp_path / "sample.wav"
    wav_path.write_bytes(b"stub")
    monkeypatch.setattr(enhance_default, "DENOISE_MODEL", "deepfilternet")

    result = enhance_default.ConditionalDenoiseEnhancer().enhance(
        AudioEnhancementRequest(wav_path=wav_path, model="none")
    )

    assert result.applied is False
    assert result.model == "none"
    assert result.output_path == wav_path


def test_denoise_api_snr_threshold_overrides_env_default(monkeypatch, tmp_path):
    wav_path = tmp_path / "speech.wav"
    wav_path.write_bytes(b"stub")
    monkeypatch.setattr(enhance_default, "DENOISE_MODEL", "deepfilternet")
    monkeypatch.setattr(enhance_default, "DENOISE_SNR_THRESHOLD", 100.0)
    monkeypatch.setattr(enhance_default, "_estimate_snr", lambda path: 15.0)

    result = enhance_default.ConditionalDenoiseEnhancer().enhance(
        AudioEnhancementRequest(
            wav_path=wav_path,
            model="deepfilternet",
            snr_threshold=10.0,
        )
    )

    assert result.applied is False
    assert result.model == "deepfilternet"
    assert result.output_path == wav_path


def test_unknown_denoise_model_is_a_noop(tmp_path, caplog):
    wav_path = tmp_path / "sample.wav"
    wav_path.write_bytes(b"stub")

    with caplog.at_level("WARNING", logger=enhance_default.logger.name):
        result = enhance_default.ConditionalDenoiseEnhancer().enhance(
            AudioEnhancementRequest(wav_path=wav_path, model="unsupported")
        )

    assert result.applied is False
    assert result.output_path == wav_path
    assert result.model == "unsupported"
    assert "Unknown DENOISE_MODEL='unsupported'" in caplog.text


def test_deepfilternet_lazy_load_logs_elapsed_time(monkeypatch, caplog):
    monkeypatch.setattr(enhance_default, "_df_model", None)
    monkeypatch.setattr(enhance_default, "_df_state", None)
    perf_values = iter([1.0, 4.5])
    df_module = ModuleType("df")
    df_module.init_df = lambda: ("model", "state", None)

    monkeypatch.setitem(sys.modules, "df", df_module)
    monkeypatch.setattr(
        enhance_default.time,
        "perf_counter",
        lambda: next(perf_values),
    )

    with caplog.at_level("INFO", logger=enhance_default.logger.name):
        model, state = enhance_default._load_deepfilternet()

    assert (model, state) == ("model", "state")
    assert "Loaded DeepFilterNet model in 3.50s (cold_load=True)" in caplog.text


def test_deepfilternet_hot_reuse_logs_without_reloading(monkeypatch, caplog):
    monkeypatch.setattr(enhance_default, "_df_model", "model")
    monkeypatch.setattr(enhance_default, "_df_state", "state")

    with caplog.at_level("INFO", logger=enhance_default.logger.name):
        model, state = enhance_default._load_deepfilternet()

    assert (model, state) == ("model", "state")
    assert "Reusing DeepFilterNet model (hot reuse)" in caplog.text


def test_deepfilternet_processing_timing_log_is_public_safe(
    monkeypatch, tmp_path, caplog
):
    wav_path = tmp_path / "private-meeting.wav"
    wav_path.write_bytes(b"stub")

    class FakeAudio:
        device = "cpu"

        def contiguous(self):
            return self

        def dim(self):
            return 2

    class FakeState:
        def sr(self):
            return 16_000

    @contextmanager
    def fake_cudnn_flags(*, enabled):
        assert enabled is False
        yield

    fake_cudnn = type(
        "Cudnn",
        (),
        {"flags": staticmethod(fake_cudnn_flags)},
    )()
    torchaudio_module = ModuleType("torchaudio")
    torchaudio_module.load = lambda path: (FakeAudio(), 48_000)
    torchaudio_module.save = lambda path, audio, sample_rate: Path(path).write_bytes(
        b"enhanced"
    )
    torchaudio_module.functional = ModuleType("torchaudio.functional")
    torchaudio_module.functional.resample = lambda audio, sr, target_sr: audio

    torch_module = ModuleType("torch")
    torch_module.backends = type(
        "Backends",
        (),
        {"cudnn": fake_cudnn},
    )()

    df_module = ModuleType("df")
    df_module.enhance = lambda model, state, audio: audio

    perf_values = iter([10.0, 12.5])
    monkeypatch.setitem(sys.modules, "torchaudio", torchaudio_module)
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "df", df_module)
    monkeypatch.setattr(enhance_default, "_estimate_snr", lambda path: 4.2)
    monkeypatch.setattr(
        enhance_default, "_load_deepfilternet", lambda: ("model", FakeState())
    )
    monkeypatch.setattr(
        enhance_default.time,
        "perf_counter",
        lambda: next(perf_values),
    )

    with caplog.at_level("INFO", logger=enhance_default.logger.name):
        result = enhance_default.ConditionalDenoiseEnhancer().enhance(
            AudioEnhancementRequest(
                wav_path=wav_path,
                model="deepfilternet",
                snr_threshold=10.0,
            )
        )

    assert result.applied is True
    assert (
        "enhance_processing_timing model=deepfilternet elapsed_s=2.500" in caplog.text
    )
    assert "applied=True reason=enhanced snr_db=4.2 threshold=10.0" in caplog.text
    assert "device=cpu input_sample_rate=48000 output_sample_rate=16000" in caplog.text
    assert "private-meeting.wav" not in caplog.text
    assert "private-meeting.denoised.wav" not in caplog.text


def test_noisereduce_processing_timing_log_is_public_safe(
    monkeypatch, tmp_path, caplog
):
    wav_path = tmp_path / "private-call.wav"
    wav_path.write_bytes(b"stub")

    soundfile_module = ModuleType("soundfile")
    soundfile_module.read = lambda path, dtype: ([0.0, 0.1], 44_100)
    soundfile_module.write = lambda path, data, sample_rate: Path(path).write_bytes(
        b"reduced"
    )

    noisereduce_module = ModuleType("noisereduce")
    noisereduce_module.reduce_noise = lambda y, sr, stationary: (
        y if stationary and sr == 44_100 else []
    )

    perf_values = iter([20.0, 21.25])
    monkeypatch.setitem(sys.modules, "soundfile", soundfile_module)
    monkeypatch.setitem(sys.modules, "noisereduce", noisereduce_module)
    monkeypatch.setattr(
        enhance_default.time,
        "perf_counter",
        lambda: next(perf_values),
    )

    with caplog.at_level("INFO", logger=enhance_default.logger.name):
        result = enhance_default.ConditionalDenoiseEnhancer().enhance(
            AudioEnhancementRequest(wav_path=wav_path, model="noisereduce")
        )

    assert result.applied is True
    assert "enhance_processing_timing model=noisereduce elapsed_s=1.250" in caplog.text
    assert "applied=True reason=enhanced sample_rate=44100" in caplog.text
    assert "private-call.wav" not in caplog.text
    assert "private-call.denoised.wav" not in caplog.text


def test_hash_index_infra_requires_completed_result(monkeypatch, tmp_path):
    monkeypatch.setattr(hash_index_module, "TRANSCRIPTIONS_DIR", tmp_path)

    store = JsonAudioArtifactIndex(index_path=tmp_path / "hash_index.json")
    store.register("hash-a", "tr_missing")
    assert store.lookup("hash-a") is None

    tr_dir = tmp_path / "tr_ready"
    tr_dir.mkdir(parents=True, exist_ok=True)
    (tr_dir / "result.json").write_text(json.dumps({"id": "tr_ready"}))

    store.register("hash-b", "tr_ready")
    assert store.lookup("hash-b") == "tr_ready"


def test_ffmpeg_normalizer_reuses_existing_target_format(tmp_path):
    wav_path = tmp_path / "already.wav"
    wav_path.write_bytes(b"wav")

    result = normalize_default.FFmpegInputNormalizer().normalize(
        AudioNormalizationRequest(input_path=wav_path)
    )

    assert result.reused_source is True
    assert result.source_path == wav_path
    assert result.normalized_path == wav_path


def test_ffmpeg_normalizer_invokes_ffmpeg_for_non_wav(monkeypatch, tmp_path):
    source = tmp_path / "meeting.ogg"
    source.write_bytes(b"ogg")
    calls = []

    def fake_run(args, *, check, timeout):
        calls.append((args, check, timeout))

    monkeypatch.setattr(normalize_default.subprocess, "run", fake_run)

    result = normalize_default.FFmpegInputNormalizer().normalize(
        AudioNormalizationRequest(input_path=source)
    )

    args, check, timeout = calls[0]
    assert check is True
    assert timeout == normalize_default.FFMPEG_TIMEOUT_SEC
    assert args[:6] == ["ffmpeg", "-y", "-v", "error", "-i", str(source)]
    assert args[-2:] == ["--", str(tmp_path / "meeting.wav")]
    assert result.reused_source is False
    assert result.normalized_path == tmp_path / "meeting.wav"


def test_ffmpeg_normalizer_timeout_cleans_partial(monkeypatch, tmp_path):
    source = tmp_path / "meeting.mp3"
    source.write_bytes(b"mp3")
    partial = tmp_path / "meeting.wav"
    partial.write_bytes(b"partial")

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=1)

    monkeypatch.setattr(normalize_default.subprocess, "run", fake_run)

    with pytest.raises(Exception) as excinfo:
        normalize_default.FFmpegInputNormalizer().normalize(
            AudioNormalizationRequest(input_path=source)
        )

    assert getattr(excinfo.value, "status_code", None) == 504
    assert not partial.exists()


def test_voiceprint_match_provider_reports_no_embeddings():
    result = voiceprint_match_default.DefaultVoiceprintMatchProvider().match(
        VoiceprintMatchRequest(
            speaker_embeddings={},
            voiceprint_db=object(),
            threshold=0.72,
        )
    )

    assert result.applied is False
    assert result.speaker_map == {}
    assert result.threshold == 0.72
    assert result.reason == "no_embeddings"


def test_voiceprint_match_provider_reports_missing_db():
    result = voiceprint_match_default.DefaultVoiceprintMatchProvider().match(
        VoiceprintMatchRequest(
            speaker_embeddings={"SPEAKER_00": [1.0, 0.0]},
            voiceprint_db=None,
            threshold=None,
        )
    )

    assert result.applied is False
    assert result.speaker_map == {}
    assert result.reason == "voiceprint_db_unavailable"


def test_voiceprint_match_provider_uses_identify_threshold_when_supplied():
    class FakeDB:
        def __init__(self):
            self.calls = []

        def identify(self, embedding, threshold=None):
            self.calls.append((embedding, threshold))
            return "spk_1", "Maple", 0.87654

    fake_db = FakeDB()

    result = voiceprint_match_default.DefaultVoiceprintMatchProvider().match(
        VoiceprintMatchRequest(
            speaker_embeddings={"SPEAKER_00": [0.1, 0.9]},
            voiceprint_db=fake_db,
            threshold=0.75,
        )
    )

    assert fake_db.calls == [([0.1, 0.9], 0.75)]
    assert result.applied is True
    assert result.reason == "matched"
    assert result.threshold == 0.75
    assert result.speaker_map == {
        "SPEAKER_00": {
            "matched_id": "spk_1",
            "matched_name": "Maple",
            "similarity": 0.8765,
            "embedding_key": "SPEAKER_00",
        }
    }
