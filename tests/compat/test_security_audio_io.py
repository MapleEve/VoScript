"""Real CPU Tensor/I/O regression. Run directly, not through stubbed conftest.

CI entry: PYTHONPATH=app python tests/compat/test_security_audio_io.py
Requires the selected CPU/cu126 torch, torchaudio, numpy, soundfile and aiofiles.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
import torchaudio

from infra.audio import io_compat


class RealAudioCompatibilityTests(unittest.TestCase):
    def test_header_only_metadata_and_handle_ownership(self):
        handle = io.BytesIO()
        sf.write(
            handle,
            np.zeros((16000, 2), dtype=np.float32),
            16000,
            format="WAV",
            subtype="PCM_16",
        )
        handle.seek(12)
        with patch.object(
            sf.SoundFile, "read", side_effect=AssertionError("metadata decoded audio")
        ):
            metadata = io_compat.info(handle)
        self.assertEqual(handle.tell(), 12)
        self.assertFalse(handle.closed)
        self.assertEqual(
            metadata, io_compat.AudioMetaData(16000, 16000, 2, 16, "PCM_S")
        )

    def test_metadata_does_not_import_torch_in_a_fresh_process(self):
        code = """
import importlib.abc
import io
import sys
import wave
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('Metadata must not import torch')
sys.meta_path.insert(0, NoTorch())
from infra.audio.io_compat import info
handle = io.BytesIO()
with wave.open(handle, 'wb') as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(16000)
    wav.writeframes(b'\\x00\\x00' * 80)
handle.seek(0)
assert info(handle).num_frames == 80
assert 'torch' not in sys.modules
assert not handle.closed
"""
        subprocess.run([sys.executable, "-c", code], check=True, timeout=30)

    def test_float_stereo_partial_read_preserves_channels_and_amplitude(self):
        waveform = torch.tensor(
            [[0.0, 0.25, -0.5, 1.25], [-1.25, 0.5, -0.25, 0.0]], dtype=torch.float32
        )
        handle = io.BytesIO()
        io_compat.save(handle, waveform, 16000, format="wav")
        handle.seek(7)
        actual, sample_rate = io_compat.load(handle, frame_offset=1, num_frames=2)
        torch.testing.assert_close(actual, waveform[:, 1:3], rtol=0, atol=0)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(sample_rate, 16000)
        self.assertEqual(handle.tell(), 7)
        self.assertFalse(handle.closed)
        full, _ = io_compat.load(handle, channels_first=False)
        torch.testing.assert_close(full, waveform.T, rtol=0, atol=0)
        self.assertEqual(io_compat.info(handle).bits_per_sample, 32)
        self.assertEqual(io_compat.info(handle).encoding, "PCM_F")

    def test_integer_pcm_round_trip_does_not_rescale_on_save(self):
        cases = [
            torch.tensor([[0, 1, 127, 128, 254, 255]], dtype=torch.uint8),
            torch.tensor([[-32768, -1, 0, 1, 32767]], dtype=torch.int16),
            torch.tensor([[-2147483648, -256, 0, 256, 2147483392]], dtype=torch.int32),
        ]
        for waveform in cases:
            with self.subTest(dtype=str(waveform.dtype)):
                handle = io.BytesIO()
                io_compat.save(handle, waveform, 8000, format="WAV")
                actual, _ = io_compat.load(handle, normalize=False)
                self.assertEqual(actual.dtype, waveform.dtype)
                torch.testing.assert_close(actual, waveform, rtol=0, atol=0)
                normalized, _ = io_compat.load(handle)
                if waveform.dtype == torch.uint8:
                    expected = (waveform.float() - 128) / 128
                elif waveform.dtype == torch.int16:
                    expected = waveform.float() / 32768
                else:
                    expected = waveform.float() / 2147483648
                torch.testing.assert_close(normalized, expected)

    def test_pcm24_and_explicit_float_wav(self):
        waveform = torch.tensor(
            [[-2147483648, -256, 0, 256, 2147483392]], dtype=torch.int32
        )
        handle = io.BytesIO()
        io_compat.save(
            handle, waveform, 48000, format="wav", encoding="PCM_S", bits_per_sample=24
        )
        self.assertEqual(io_compat.info(handle).bits_per_sample, 24)
        actual, _ = io_compat.load(handle, normalize=False)
        torch.testing.assert_close(actual, waveform, rtol=0, atol=0)
        floats = io.BytesIO()
        io_compat.save(
            floats,
            torch.tensor([[1.5, -1.5]], dtype=torch.float32),
            48000,
            format="wav",
            encoding="PCM_F",
            bits_per_sample=32,
        )
        self.assertEqual(io_compat.info(floats).encoding, "PCM_F")
        values, _ = io_compat.load(floats)
        torch.testing.assert_close(values, torch.tensor([[1.5, -1.5]]))

    def test_explicit_wav_encoding_defaults_match_legacy_soundfile(self):
        for encoding, expected_bits in (("PCM_S", 32), ("PCM_F", 32), ("PCM_U", 8)):
            with self.subTest(encoding=encoding):
                handle = io.BytesIO()
                io_compat.save(
                    handle, torch.zeros(1, 32), 16000, format="wav", encoding=encoding
                )
                metadata = io_compat.info(handle)
                self.assertEqual(metadata.encoding, encoding)
                self.assertEqual(metadata.bits_per_sample, expected_bits)

    def test_double_wav_keeps_precision_only_when_normalization_is_disabled(self):
        waveform = torch.tensor([[0.0, 0.1234567890123, -1.25]], dtype=torch.float64)
        handle = io.BytesIO()
        io_compat.save(handle, waveform, 16000, format="wav")
        self.assertEqual(io_compat.info(handle).bits_per_sample, 64)
        actual, _ = io_compat.load(handle, normalize=False)
        self.assertEqual(actual.dtype, torch.float64)
        torch.testing.assert_close(actual, waveform, rtol=0, atol=0)
        normalized, _ = io_compat.load(handle)
        self.assertEqual(normalized.dtype, torch.float32)
        torch.testing.assert_close(normalized, waveform.float(), rtol=0, atol=0)

    def test_open_soundfile_and_binary_handles_remain_owned_by_caller(self):
        waveform = torch.tensor([[0.0, 0.25, -0.5, 0.75]], dtype=torch.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.wav"
            io_compat.save(path, waveform, 16000)
            with path.open("rb") as handle:
                handle.seek(9)
                actual, _ = io_compat.load(handle, frame_offset=1, num_frames=2)
                torch.testing.assert_close(actual, waveform[:, 1:3], rtol=0, atol=0)
                self.assertEqual(handle.tell(), 9)
                self.assertFalse(handle.closed)
            with sf.SoundFile(path, "r") as handle:
                handle.seek(2)
                with patch.object(
                    sf.SoundFile,
                    "read",
                    side_effect=AssertionError("metadata decoded audio"),
                ):
                    self.assertEqual(io_compat.info(handle).num_frames, 4)
                actual, _ = io_compat.load(handle, frame_offset=1, num_frames=2)
                torch.testing.assert_close(actual, waveform[:, 1:3], rtol=0, atol=0)
                self.assertEqual(handle.tell(), 2)
                self.assertFalse(handle.closed)

    def test_empty_tail_invalid_ranges_and_backend_truthfulness(self):
        handle = io.BytesIO()
        io_compat.save(handle, torch.zeros(2, 40), 16000, format="wav")
        for offset, count in ((40, -1), (400, -1), (10, 0)):
            actual, _ = io_compat.load(handle, frame_offset=offset, num_frames=count)
            self.assertEqual(tuple(actual.shape), (2, 0))
        for kwargs in ({"frame_offset": -1}, {"num_frames": -2}, {"backend": "ffmpeg"}):
            with self.assertRaises(ValueError):
                io_compat.load(handle, **kwargs)
        self.assertEqual(io_compat.list_audio_backends(), ["soundfile"])
        self.assertEqual(io_compat.get_audio_backend(), "soundfile")
        with self.assertRaises(ValueError):
            io_compat.set_audio_backend("sox")
        self.assertIs(io_compat.backend.common.AudioMetaData, io_compat.AudioMetaData)

    def test_paths_and_nonwav_normalize_false(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.flac"
            io_compat.save(path, torch.zeros(1, 32), 16000)
            actual, _ = io_compat.load(path, normalize=False)
            self.assertEqual(actual.dtype, torch.float32)
            self.assertEqual(io_compat.info(path).encoding, "FLAC")

    def test_dsp_is_the_real_torchaudio_and_not_monkeypatched(self):
        self.assertIs(io_compat.functional, torchaudio.functional)
        self.assertIs(io_compat.compliance, torchaudio.compliance)
        self.assertIs(io_compat.pipelines, torchaudio.pipelines)
        self.assertTrue(torchaudio.load.__module__.startswith("torchaudio"))
        self.assertNotEqual(torchaudio.load, io_compat.load)
        waveform = torch.zeros(1, 160)
        resampled = io_compat.functional.resample(waveform, 16000, 8000)
        self.assertEqual(tuple(resampled.shape), (1, 80))


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    unittest.main(verbosity=2)
