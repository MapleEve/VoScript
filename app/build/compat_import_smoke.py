"""CI-only, fresh-process imports of the actual image's legacy audio stack.

Run from the built image: python build/compat_import_smoke.py
This does not download models, open a service or use a GPU.
"""

from __future__ import annotations

import os
import subprocess
import sys

SMOKES = {
    "pyannote": """
import torchaudio
assert not hasattr(torchaudio, 'AudioMetaData')
from pyannote.audio import Model, Pipeline
from pyannote.audio.core import io
from pyannote.audio.tasks.segmentation import mixins
from infra.audio import io_compat
from infra.huggingface_compat import hf_hub_download
import pyannote.audio.core.model as model
import pyannote.audio.core.pipeline as pipeline
assert io.torchaudio is io_compat
assert io.torchaudio.functional is torchaudio.functional
assert mixins.AudioMetaData is io_compat.AudioMetaData
assert model.hf_hub_download is hf_hub_download
assert pipeline.hf_hub_download is hf_hub_download
assert io.Audio().backend == 'soundfile'
from io import BytesIO
import torch
from pyannote.core import Segment
handle = BytesIO()
io_compat.save(handle, torch.zeros(2, 400), 16000, format='wav')
waveform, sample_rate = io.Audio(mono='downmix').crop(handle, Segment(0, 0.01))
assert sample_rate == 16000 and tuple(waveform.shape) == (1, 160)
from types import SimpleNamespace
from unittest.mock import patch
from pyannote.audio.pipelines import speaker_verification
from infra.audio.speechbrain_compat import from_hparams
prepared = SimpleNamespace(prepared_data={
    'audio-path': ['synthetic.wav'],
    'audio-info': [{'sample_rate': 16000, 'num_frames': 400,
                    'num_channels': 2, 'bits_per_sample': 16}],
    'audio-encoding': ['PCM_S'],
})
prepared_file = mixins.SegmentationTask.get_file(prepared, 0)
assert prepared_file['audio'] == 'synthetic.wav'
assert prepared_file['torchaudio.info'] == io_compat.AudioMetaData(16000, 400, 2, 16, 'PCM_S')
assert not hasattr(torchaudio, 'AudioMetaData')
assert speaker_verification.SPEECHBRAIN_IS_AVAILABLE
assert speaker_verification.speechbrain_from_hparams is from_hparams
calls = []
classifier = SimpleNamespace(from_hparams=lambda **kwargs: calls.append(kwargs) or object())
with patch.object(speaker_verification, 'SpeechBrain_EncoderClassifier', classifier):
    embedding = speaker_verification.SpeechBrainPretrainedSpeakerEmbedding(
        embedding='synthetic/repo@fixed', use_auth_token=False,
    )
    embedding.to(torch.device('cpu'))
assert len(calls) == 2
for call in calls:
    assert call['source'] == 'synthetic/repo'
    assert call['fetch_config'].revision == 'fixed'
    assert call['fetch_config'].token is False
    assert call['fetch_config'].allow_network is False
    assert call['run_opts']['device'] == 'cpu'
""",
    "speechbrain": """
import speechbrain
from speechbrain.inference import EncoderClassifier
from speechbrain.dataio import audio_io
from speechbrain.utils.torch_audio_backend import check_torchaudio_backend
assert speechbrain.__version__ == '1.1.2'
assert audio_io.list_audio_backends() == ['soundfile']
check_torchaudio_backend()
import tempfile
from pathlib import Path
import torch
import torchaudio
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'synthetic.wav'
    audio_io.save(path, torch.zeros(2, 160), 16000)
    assert audio_io.info(path).num_frames == 160
    waveform, sample_rate = audio_io.load(path, frame_offset=10, num_frames=80)
    assert sample_rate == 16000 and tuple(waveform.shape) == (2, 80)
assert torchaudio.load.__module__.startswith('torchaudio')
""",
    "deepfilternet": """
import df
from df import io
from infra.audio import io_compat
assert callable(df.init_df) and callable(df.enhance)
assert io.ta is io_compat
assert io.AudioMetaData is io_compat.AudioMetaData
from io import BytesIO
import torch
handle = BytesIO()
io_compat.save(handle, torch.zeros(1, 160), 16000, format='wav')
waveform, metadata = io.load_audio(handle, verbose=False)
assert metadata.num_frames == 160 and tuple(waveform.shape) == (1, 160)
""",
    "torch-audiomentations": """
from torch_audiomentations.utils import io
from infra.audio import io_compat
assert io.torchaudio is io_compat
import tempfile
from pathlib import Path
import torch
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'synthetic.wav'
    io_compat.save(path, torch.zeros(1, 160), 16000)
    assert io.Audio.get_audio_metadata(path) == (160, 16000)
    assert tuple(io.Audio(sample_rate=16000)(path, num_samples=80).shape) == (1, 80)
""",
    "alignment": """
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import whisperx
assert callable(whisperx.load_align_model) and callable(whisperx.align)
assert callable(whisperx.load_audio)
""",
}


def main():
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )
    for name, code in SMOKES.items():
        subprocess.run(
            [sys.executable, "-c", code], check=True, env=environment, timeout=120
        )
        print(f"PASS: fresh-process {name} import", flush=True)


if __name__ == "__main__":
    main()
