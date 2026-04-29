"""Unit tests for pipeline model unload and reload-time device selection."""

from __future__ import annotations

import sys
from types import ModuleType

if "numpy" not in sys.modules:
    numpy_stub = ModuleType("numpy")
    numpy_stub.ndarray = object
    sys.modules["numpy"] = numpy_stub

from pipeline import TranscriptionPipeline
import pipeline.orchestrator as orchestrator


def _new_pipeline(*, device="cuda"):
    pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
    pipeline.device = device
    pipeline._configured_device = device
    pipeline._whisper_device = None
    pipeline._diarization_device = None
    pipeline._embedding_device = None
    pipeline.model_size = "tiny"
    pipeline.hf_token = None
    pipeline._whisper = None
    pipeline._diarization = None
    pipeline._embedding_model = None
    pipeline._runner = None
    return pipeline


def _install_fake_faster_whisper(monkeypatch, loaded_models):
    class FakeWhisperModel:
        def __init__(self, model_ref, **kwargs):
            loaded_models.append((model_ref, kwargs))

    faster_whisper = ModuleType("faster_whisper")
    faster_whisper.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", faster_whisper)
    return FakeWhisperModel


def test_unload_models_drops_loaded_references_without_selecting_device(monkeypatch):
    pipeline = _new_pipeline(device="cuda")
    pipeline._whisper = object()
    pipeline._diarization = object()
    pipeline._embedding_model = object()
    calls = []

    monkeypatch.setattr(
        orchestrator,
        "select_best_cuda_device",
        lambda configured: calls.append(configured) or "cuda:1",
    )

    assert pipeline.has_loaded_models() is True

    pipeline.unload_models()

    assert pipeline.has_loaded_models() is False
    assert pipeline._whisper is None
    assert pipeline._diarization is None
    assert pipeline._embedding_model is None
    assert calls == []


def test_whisper_lazy_reload_selects_best_cuda_device(monkeypatch):
    pipeline = _new_pipeline(device="cuda")
    calls = []
    loaded_models = []
    fake_model = _install_fake_faster_whisper(monkeypatch, loaded_models)

    monkeypatch.setattr(orchestrator.Path, "exists", lambda self: False)
    monkeypatch.setattr(
        orchestrator,
        "select_best_cuda_device",
        lambda configured: calls.append(configured) or "cuda:1",
    )

    assert pipeline.whisper.__class__ is fake_model

    assert calls == ["cuda"]
    assert loaded_models == [
        ("tiny", {"device": "cuda", "device_index": 1, "compute_type": "float16"})
    ]
    assert pipeline.device == "cuda:1"


def test_cpu_lazy_load_does_not_probe_cuda(monkeypatch):
    pipeline = _new_pipeline(device="cpu")
    loaded_models = []
    fake_model = _install_fake_faster_whisper(monkeypatch, loaded_models)

    def fail_if_called(configured):
        raise AssertionError("CPU-only loads must not probe CUDA")

    monkeypatch.setattr(orchestrator.Path, "exists", lambda self: False)
    monkeypatch.setattr(orchestrator, "select_best_cuda_device", fail_if_called)

    assert pipeline.whisper.__class__ is fake_model

    assert loaded_models == [("tiny", {"device": "cpu", "compute_type": "int8"})]
    assert pipeline.device == "cpu"


def test_whisper_lazy_load_keeps_unindexed_cuda_supported(monkeypatch):
    pipeline = _new_pipeline(device="cuda")
    loaded_models = []
    fake_model = _install_fake_faster_whisper(monkeypatch, loaded_models)

    monkeypatch.setattr(orchestrator.Path, "exists", lambda self: False)
    monkeypatch.setattr(
        orchestrator, "select_best_cuda_device", lambda configured: configured
    )

    assert pipeline.whisper.__class__ is fake_model

    assert loaded_models == [("tiny", {"device": "cuda", "compute_type": "float16"})]
    assert pipeline.device == "cuda"


def test_whisper_lazy_load_normalizes_cuda_zero_for_faster_whisper(monkeypatch):
    pipeline = _new_pipeline(device="cuda:0")
    loaded_models = []
    fake_model = _install_fake_faster_whisper(monkeypatch, loaded_models)

    monkeypatch.setattr(orchestrator.Path, "exists", lambda self: False)
    monkeypatch.setattr(
        orchestrator, "select_best_cuda_device", lambda configured: configured
    )

    assert pipeline.whisper.__class__ is fake_model

    assert loaded_models == [
        ("tiny", {"device": "cuda", "device_index": 0, "compute_type": "float16"})
    ]
    assert pipeline.device == "cuda:0"


def test_whisper_lazy_load_normalizes_fallback_cuda_index_for_faster_whisper(
    monkeypatch,
):
    pipeline = _new_pipeline(device="cuda:1")
    loaded_models = []
    fake_model = _install_fake_faster_whisper(monkeypatch, loaded_models)

    monkeypatch.setattr(orchestrator.Path, "exists", lambda self: False)
    monkeypatch.setattr(
        orchestrator, "select_best_cuda_device", lambda configured: configured
    )

    assert pipeline.whisper.__class__ is fake_model

    assert loaded_models == [
        ("tiny", {"device": "cuda", "device_index": 1, "compute_type": "float16"})
    ]
    assert pipeline.device == "cuda:1"


def test_fixed_cuda_device_does_not_probe_best_device(monkeypatch):
    pipeline = _new_pipeline(device="cuda:0")
    loaded_models = []
    fake_model = _install_fake_faster_whisper(monkeypatch, loaded_models)

    def fail_if_called(configured):
        raise AssertionError("fixed cuda device must not select a different GPU")

    monkeypatch.setattr(orchestrator.Path, "exists", lambda self: False)
    monkeypatch.setattr(orchestrator, "select_best_cuda_device", fail_if_called)

    assert pipeline.whisper.__class__ is fake_model

    assert loaded_models == [
        ("tiny", {"device": "cuda", "device_index": 0, "compute_type": "float16"})
    ]


def test_each_model_lazy_load_selects_its_own_cuda_device(monkeypatch):
    pipeline = _new_pipeline(device="cuda")
    loaded_models = []
    fake_model = _install_fake_faster_whisper(monkeypatch, loaded_models)
    selected_devices = iter(["cuda:0", "cuda:1", "cuda:1"])
    selector_calls = []
    pyannote_calls = []

    monkeypatch.setattr(orchestrator.Path, "exists", lambda self: False)
    monkeypatch.setattr(
        orchestrator,
        "select_best_cuda_device",
        lambda configured: selector_calls.append(configured) or next(selected_devices),
    )
    monkeypatch.setattr(
        orchestrator,
        "resolve_hf_model_ref",
        lambda repo_id, *, token, purpose: repo_id,
    )
    monkeypatch.setattr(
        orchestrator,
        "_localize_pyannote_diarization_config",
        lambda model_ref, *, token: model_ref,
    )

    class FakeDiarization:
        def to(self, device):
            pyannote_calls.append(("diarization_to", device))
            return self

    class FakePyannotePipeline:
        @classmethod
        def from_pretrained(cls, model_ref, use_auth_token=None):
            pyannote_calls.append(("diarization_load", model_ref, use_auth_token))
            return FakeDiarization()

    class FakeEmbeddingModel:
        @classmethod
        def from_pretrained(cls, model_ref, use_auth_token=None):
            pyannote_calls.append(("embedding_load", model_ref, use_auth_token))
            return cls()

        def to(self, device):
            pyannote_calls.append(("embedding_to", device))
            return self

    class FakeInference:
        def __init__(self, model, window):
            pyannote_calls.append(("inference", window))

    pyannote_audio = ModuleType("pyannote.audio")
    pyannote_audio.Pipeline = FakePyannotePipeline
    pyannote_audio.Model = FakeEmbeddingModel
    pyannote_audio.Inference = FakeInference
    monkeypatch.setitem(sys.modules, "pyannote.audio", pyannote_audio)

    assert pipeline.whisper.__class__ is fake_model
    assert pipeline.diarization.__class__ is FakeDiarization
    assert pipeline.embedding_model.__class__ is FakeInference

    assert selector_calls == ["cuda", "cuda", "cuda"]
    assert pipeline._whisper_device == "cuda:0"
    assert pipeline._diarization_device == "cuda:1"
    assert pipeline._embedding_device == "cuda:1"
    assert loaded_models == [
        ("tiny", {"device": "cuda", "device_index": 0, "compute_type": "float16"})
    ]
    assert pyannote_calls == [
        (
            "diarization_load",
            "pyannote/speaker-diarization-3.1",
            None,
        ),
        ("diarization_to", "cuda:1"),
        (
            "embedding_load",
            "pyannote/wespeaker-voxceleb-resnet34-LM",
            None,
        ),
        ("embedding_to", "cuda:1"),
        ("inference", "whole"),
    ]


def test_unload_models_clears_per_model_devices_and_reload_reselects(monkeypatch):
    pipeline = _new_pipeline(device="cuda")
    loaded_models = []
    _install_fake_faster_whisper(monkeypatch, loaded_models)
    selected_devices = iter(["cuda:0", "cuda:1"])

    monkeypatch.setattr(orchestrator.Path, "exists", lambda self: False)
    monkeypatch.setattr(
        orchestrator,
        "select_best_cuda_device",
        lambda configured: next(selected_devices),
    )

    _ = pipeline.whisper
    assert pipeline._whisper_device == "cuda:0"

    pipeline.unload_models()

    assert pipeline._whisper_device is None
    assert pipeline._diarization_device is None
    assert pipeline._embedding_device is None

    _ = pipeline.whisper
    assert pipeline._whisper_device == "cuda:1"
