# Runtime Compatibility Tools

The supported deployment is the CI-built Linux CUDA image with Python 3.11,
the required Rust extension, and the exact runtime constraints in this directory's
parent. Installing `requirements.txt` alone is not a supported native deployment:
it neither applies the third-party compatibility adaptations nor installs the
Rust wheel. Native macOS, Windows and other source installations are not covered
by the published image's compatibility gate.

`requirements.constraints.txt` is platform-neutral. The Dockerfile additionally
uses `requirements.docker.constraints.txt` to select official PyTorch 2.13.0,
torchaudio 2.11.0 and torchvision 0.28.0 CUDA 12.6 wheels. TorchAudio 2.11.0
uses PyTorch's stable ABI and supports PyTorch 2.11 and later; see its
[compatibility matrix](https://docs.pytorch.org/audio/main/installation.html#compatibility-matrix).
The explicit `+cu126` builds preserve the CUDA12/cuDNN9 runtime used by
CTranslate2. Do not substitute the default PyPI CUDA build or use the Docker
constraint file for native macOS or CPU-only installations. Python 3.11,
NumPy 1.x, DeepFilterNet and the existing pyannote model line are retained.

During image build, `legacy_wheels.txt` authenticates exact upstream release
wheels. `patch_legacy_audio.py` verifies installed versions, wheel SHA256 values,
allowlisted file bytes and exact import/call structure before applying scoped
adaptations. Authenticated UTF-8 BOM and uniform CRLF payloads retain their
original encoding/newlines; installed bytes must still match the original or
patched authenticated wheel member exactly. Unknown artifacts fail closed. The image retains a per-file audit
manifest at `/usr/local/share/voscript/legacy-compat.json`.

The Torch update includes the `GHSA-rrmf-rvhw-rf47` fix in 2.13.0. It is not a
claim that arbitrary model archives are safe: `torch.export.load` still documents
pickle-based deserialization and must not receive untrusted input. The service
does not use this API; checkpoint loading remains scoped and weights-only.

To check the image's actual dependencies in CI without models, network or GPUs:

```sh
python build/compat_import_smoke.py
```

The separate real CPU Tensor/I/O regression entry point in the source checkout is:

```sh
PYTHONPATH=app python tests/compat/test_security_audio_io.py
```

Neither check substitutes for deployment validation of CUDA model loading and
processing against the published image digest.
