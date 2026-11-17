"""Soundfile I/O for explicitly adapted legacy audio consumers.

This is not a replacement for the torchaudio package. DSP, models and transforms
remain in the real package; only removed file I/O/backend APIs live here.
"""

from __future__ import annotations

import importlib
import operator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


@dataclass
class AudioMetaData:
    sample_rate: int
    num_frames: int
    num_channels: int
    bits_per_sample: int
    encoding: str


_BITS = {
    "PCM_S8": 8,
    "PCM_U8": 8,
    "PCM_16": 16,
    "PCM_24": 24,
    "PCM_32": 32,
    "FLOAT": 32,
    "DOUBLE": 64,
    "ULAW": 8,
    "ALAW": 8,
}


def _check_backend(backend):
    if backend not in (None, "soundfile"):
        raise ValueError("Only the soundfile audio backend is supported")


@contextmanager
def _reader(uri):
    import soundfile as sf

    if isinstance(uri, sf.SoundFile):
        position = uri.tell()
        try:
            yield uri
        finally:
            uri.seek(position)
        return

    # libsndfile virtual I/O does not own caller-provided Python handles.
    position = None
    if hasattr(uri, "seek") and hasattr(uri, "tell"):
        position = uri.tell()
        uri.seek(0)
    try:
        with sf.SoundFile(uri, mode="r") as stream:
            yield stream
    finally:
        if position is not None:
            uri.seek(position)


def info(uri, format=None, buffer_size=4096, backend=None):
    """Read the header only, preserving caller-owned handles and positions."""
    _check_backend(backend)
    with _reader(uri) as stream:
        subtype = stream.subtype
        if stream.format == "FLAC":
            encoding = "FLAC"
        elif subtype.startswith("PCM_"):
            encoding = "PCM_U" if subtype == "PCM_U8" else "PCM_S"
        elif subtype in ("FLOAT", "DOUBLE"):
            encoding = "PCM_F"
        else:
            encoding = (
                subtype if subtype in ("ULAW", "ALAW", "VORBIS", "OPUS") else "UNKNOWN"
            )
        return AudioMetaData(
            stream.samplerate,
            stream.frames,
            stream.channels,
            _BITS.get(subtype, 0),
            encoding,
        )


def load(
    uri,
    frame_offset=0,
    num_frames=-1,
    normalize=True,
    channels_first=True,
    format=None,
    buffer_size=4096,
    backend=None,
):
    """Read only the requested frames; normalized audio is always float32."""
    import numpy as np
    import torch

    _check_backend(backend)
    frame_offset = operator.index(frame_offset)
    num_frames = operator.index(num_frames)
    if frame_offset < 0 or num_frames < -1:
        raise ValueError("frame_offset must be >= 0 and num_frames must be >= -1")
    with _reader(uri) as stream:
        dtype = "float32"
        unsigned = False
        if not normalize and stream.format in ("WAV", "WAVEX", "RF64"):
            dtype = {
                "PCM_16": "int16",
                "PCM_24": "int32",
                "PCM_32": "int32",
                "PCM_U8": "int16",
                "DOUBLE": "float64",
            }.get(stream.subtype, "float32")
            unsigned = stream.subtype == "PCM_U8"
        stream.seek(min(frame_offset, stream.frames))
        samples = stream.read(frames=num_frames, dtype=dtype, always_2d=True)
        sample_rate = stream.samplerate
    if unsigned:
        samples = ((samples.astype(np.int32) + 32768) >> 8).astype(np.uint8)
    if channels_first:
        samples = samples.T
    return torch.from_numpy(np.ascontiguousarray(samples)), sample_rate


def _wav_subtype(dtype, encoding, bits):
    if encoding is None:
        if bits is None:
            return {
                "uint8": "PCM_U8",
                "int16": "PCM_16",
                "int32": "PCM_32",
                "float32": "FLOAT",
                "float64": "DOUBLE",
            }[dtype]
        encoding = "PCM_U" if bits == 8 else "PCM_S"
    subtypes = {
        ("PCM_U", 8): "PCM_U8",
        ("PCM_S", 16): "PCM_16",
        ("PCM_S", 24): "PCM_24",
        ("PCM_S", 32): "PCM_32",
        ("PCM_F", 32): "FLOAT",
        ("PCM_F", 64): "DOUBLE",
        ("ULAW", 8): "ULAW",
        ("ALAW", 8): "ALAW",
    }
    bits = (
        bits
        if bits is not None
        else {"PCM_U": 8, "PCM_S": 32, "PCM_F": 32, "ULAW": 8, "ALAW": 8}.get(encoding)
    )
    try:
        return subtypes[encoding, bits]
    except KeyError as exc:
        raise ValueError(
            "Unsupported WAV encoding/bits_per_sample combination"
        ) from exc


def save(
    uri,
    src,
    sample_rate,
    channels_first=True,
    format=None,
    encoding=None,
    bits_per_sample=None,
    buffer_size=4096,
    backend=None,
):
    """Write WAV without changing integer PCM scale or floating-point amplitude."""
    import numpy as np
    import soundfile as sf
    import torch

    _check_backend(backend)
    sample_rate = operator.index(sample_rate)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if isinstance(src, torch.Tensor):
        samples = src.detach().cpu().numpy()
    else:
        samples = np.asarray(src)
    if samples.ndim != 2:
        raise ValueError("Expected a 2D audio tensor")
    if str(samples.dtype) not in ("uint8", "int16", "int32", "float32", "float64"):
        raise ValueError("Unsupported audio dtype")
    container = format.upper() if format else None
    if container is None and isinstance(uri, (str, Path)):
        container = Path(uri).suffix.lstrip(".").upper()
    if container == "WAVE":
        container = "WAV"
    if not container:
        raise ValueError("format is required for file-like audio output")
    if container in ("WAV", "WAVEX", "RF64"):
        subtype = _wav_subtype(str(samples.dtype), encoding, bits_per_sample)
    elif container == "FLAC":
        if encoding not in (None, "PCM_S") or bits_per_sample not in (None, 8, 16, 24):
            raise ValueError("Unsupported FLAC encoding/bits_per_sample combination")
        subtype = f"PCM_{bits_per_sample or 16}"
        if subtype == "PCM_8":
            subtype = "PCM_S8"
    elif encoding is not None or bits_per_sample is not None:
        raise ValueError("Explicit encoding is supported only for WAV and FLAC")
    else:
        subtype = None
    if samples.dtype == np.uint8:
        samples = ((samples.astype(np.int16) - 128) << 8).astype(np.int16)
    if channels_first:
        samples = samples.T
    sf.write(uri, samples, sample_rate, format=container, subtype=subtype)


def list_audio_backends():
    import soundfile  # noqa: F401

    return ["soundfile"]


def get_audio_backend():
    return "soundfile"


def set_audio_backend(backend):
    _check_backend(backend)


soundfile_backend = SimpleNamespace(info=info, load=load, save=save)
backend = SimpleNamespace(
    common=SimpleNamespace(AudioMetaData=AudioMetaData),
    soundfile_backend=soundfile_backend,
)


def __getattr__(name):
    if name in ("functional", "compliance", "pipelines", "transforms", "__version__"):
        return getattr(importlib.import_module("torchaudio"), name)
    raise AttributeError(name)
