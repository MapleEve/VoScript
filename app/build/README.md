# Runtime Compatibility Tools

The supported deployment is the CI-built Linux CUDA image with Python 3.11,
the required Rust extension, and the exact runtime constraints in this directory's
parent. Installing `requirements.txt` alone is not a supported native deployment:
it neither applies the third-party compatibility adaptations nor installs the
Rust wheel. Native macOS, Windows and other source installations are not covered
by the published image's compatibility gate.

`requirements.constraints.txt` is platform-neutral. The Dockerfile additionally
uses `requirements.docker.constraints.txt` to select official PyTorch 2.10.0,
torchaudio 2.10.0 and torchvision 0.25.0 CUDA 12.6 wheels. Do not use the Docker
constraint file for native macOS or CPU-only installations.

During image build, `legacy_wheels.txt` authenticates exact upstream release
wheels. `patch_legacy_audio.py` verifies installed versions, wheel SHA256 values,
allowlisted file bytes and exact import/call structure before applying scoped
adaptations. Unknown artifacts fail closed. The image retains a per-file audit
manifest at `/usr/local/share/voscript/legacy-compat.json`.

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
