#!/usr/bin/env python3
"""Audit and smoke the exact local image that will be published, without GPU access."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from packaging.version import Version

from ci_release_contracts import ROOT, version

ENV = [
    "-e",
    "DEVICE=cpu",
    "-e",
    "ALLOW_NO_AUTH=1",
    "-e",
    "RUST_KERNEL_MODE=required",
    "-e",
    "HF_HUB_OFFLINE=1",
    "-e",
    "TRANSFORMERS_OFFLINE=1",
    "-e",
    "HF_ENDPOINT=https://hf-mirror.com",
    "-e",
    "CUDA_VISIBLE_DEVICES=",
    "-e",
    "HF_HUB_DISABLE_TELEMETRY=1",
]

CORE_SMOKE = """
from config import APP_VERSION
from providers.kernel_bridge import (
    artifact_manifest_contract, core_smoke, postprocess_segments,
    status_payload_contract, voiceprint_score,
)
import sys
assert APP_VERSION == sys.argv[1]
result = core_smoke({'source': 'ci'})
assert result['ok'] is True
assert result['version'] == APP_VERSION
decision = voiceprint_score({
    'query_embedding': [1.0, 0.0],
    'candidates': [{'speaker_id': 'spk_ci', 'name': 'CI', 'embedding': [1.0, 0.0],
                    'sample_count': 1, 'sample_spread': None}],
    'threshold': 0.75, 'asnorm_threshold': 0.5, 'cohort': None,
})
assert decision['matched_id'] == 'spk_ci'
processed = postprocess_segments({
    'aligned_segments': [
        {'start': 0.0, 'end': 1.0, 'text': 'synthetic one', 'speaker': 'SPEAKER_00'},
        {'start': 1.0, 'end': 2.0, 'text': 'synthetic two', 'speaker': 'SPEAKER_01'},
    ],
    'speaker_map': {},
})
assert processed['unique_speakers'] == ['SPEAKER_00', 'SPEAKER_01']
assert [segment['speaker_label'] for segment in processed['segments']] == ['SPEAKER_00', 'SPEAKER_01']
manifest = artifact_manifest_contract({
    'manifest_version': 'artifact_manifest.v1',
    'stable': [{'name': 'result', 'filename': 'result.json', 'role': 'primary_result',
                'media_type': 'application/json', 'required_for_result': True}],
    'optional': [], 'experimental': [],
})
assert manifest['stable'][0]['filename'] == 'result.json'
status = status_payload_contract({'status': 'queued', 'updated_at': '2026-01-01T00:00:00+00:00',
                                  'filename': 'synthetic/audio.wav'})
assert status['filename'] == 'audio.wav'
"""

NATIVE_GOLDENS = """
import math
import numpy as np
from postprocess.segments import build_result_segments
from providers.kernel_bridge import postprocess_segments, voiceprint_score
from voiceprints.scoring import VoiceprintScoreCandidate, score_voiceprint_candidates
aligned = [
    {'start': 0.0, 'end': 1.0, 'text': 'synthetic one', 'speaker': 'SPEAKER_00'},
    {'start': 1.0, 'end': 2.0, 'text': 'synthetic two', 'speaker': 'SPEAKER_01'},
]
speaker_map = {label: {'matched_id': 'spk_ci', 'matched_name': 'CI', 'similarity': 1.0}
               for label in ['SPEAKER_00', 'SPEAKER_01']}
native = postprocess_segments({'aligned_segments': aligned, 'speaker_map': speaker_map})
oracle_segments, oracle_speakers = build_result_segments(aligned, speaker_map)
assert native['segments'] == oracle_segments
assert native['unique_speakers'] == oracle_speakers == ['CI', 'CI (2)']
assert len(native['segments']) == 2
def vector(angle):
    return np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
candidates = [
    VoiceprintScoreCandidate('spk_first', 'First', vector(0.0), 3, 0.0),
    VoiceprintScoreCandidate('spk_second', 'Second', vector(0.005), 3, 0.0),
]
cohort = np.stack([vector(angle) for angle in [1.0,1.1,1.2,1.3,1.4,-1.0,-1.1,-1.2,-1.3,-1.4]])
oracle = score_voiceprint_candidates(query_embedding=vector(0.0), candidates=candidates,
    threshold=0.75, asnorm_threshold=0.5, cohort=cohort)
native = voiceprint_score({
    'query_embedding': vector(0.0).tolist(),
    'candidates': [{'speaker_id': c.speaker_id, 'name': c.name, 'embedding': c.embedding.tolist(),
                    'sample_count': c.sample_count, 'sample_spread': c.sample_spread} for c in candidates],
    'threshold': 0.75, 'asnorm_threshold': 0.5, 'cohort': cohort.tolist(),
})
assert native['reason'] == oracle.reason == 'ambiguous_margin'
assert native['matched_id'] == oracle.matched_id is None
assert native['asnorm_active'] == oracle.asnorm_active is True
assert abs(native['similarity'] - oracle.similarity) < 1e-5
"""

MISSING_CORE = """
import asyncio
import importlib.abc
import sys
class MissingCore(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'voscript_core' or fullname.startswith('voscript_core.'):
            raise ModuleNotFoundError('synthetic missing native extension')
sys.meta_path.insert(0, MissingCore())
from providers.kernel_bridge import RustKernelBridgeError
async def verify():
    try:
        from main import app
        async with app.router.lifespan_context(app):
            raise AssertionError('startup succeeded without required native extension')
    except RustKernelBridgeError:
        return
asyncio.run(verify())
"""

MODEL_IMPORT_SMOKE = """
import io
import tempfile
import torch
import torchaudio
import soundfile
from pyannote.audio import Pipeline, Model, Inference
from speechbrain.inference.classifiers import EncoderClassifier
from df import init_df, enhance
import df.io
import whisperx
from transformers import Wav2Vec2Config, Wav2Vec2ForCTC
from pipeline.orchestrator import _trusted_pyannote_checkpoint_context
from torch.torch_version import TorchVersion
assert torch.__version__.split('+')[0] == '2.13.0'
assert torchaudio.__version__.split('+')[0] == '2.11.0'
assert callable(whisperx.load_align_model)
assert callable(init_df) and callable(enhance)
assert callable(Pipeline.from_pretrained) and callable(Model.from_pretrained)
assert callable(Inference) and callable(EncoderClassifier.from_hparams)
# Exercise real weights-only loading with the application-scoped allowlist.
checkpoint = io.BytesIO()
torch.save({'version': TorchVersion(torch.__version__)}, checkpoint)
checkpoint.seek(0)
before = set(torch.serialization.get_safe_globals())
with _trusted_pyannote_checkpoint_context():
    restored = torch.load(checkpoint, weights_only=True, map_location='cpu')
assert isinstance(restored['version'], TorchVersion)
after = set(torch.serialization.get_safe_globals())
assert after == before, (
    f'Scoped safe globals changed: added={len(after - before)} missing={len(before - after)}'
)
# A tiny local alignment model exercises real Transformers/Torch loading without
# gated model downloads, credentials, production caches, or GPU allocation.
config = Wav2Vec2Config(vocab_size=8, hidden_size=16, num_hidden_layers=1,
    num_attention_heads=2, intermediate_size=32, conv_dim=(8,8,8),
    conv_stride=(5,2,2), conv_kernel=(10,3,3), num_conv_pos_embedding_groups=2)
with tempfile.TemporaryDirectory() as directory:
    Wav2Vec2ForCTC(config).save_pretrained(directory, safe_serialization=True)
    model = Wav2Vec2ForCTC.from_pretrained(directory, local_files_only=True).eval()
    with torch.inference_mode():
        assert model(torch.zeros(1, 1600)).logits.shape[-1] == 8
"""

INVENTORY = """
from importlib.metadata import distributions
import json
print(json.dumps([{'name': dist.metadata['Name'], 'version': dist.version}
    for dist in sorted(distributions(), key=lambda d: d.metadata['Name'].lower())]))
"""

# Exact official CUDA wheel build aliases, not vulnerability ignores or a generic
# local-version stripper. Advisory matching uses the same upstream release.
CUDA_UPSTREAM = {
    ("torch", "2.13.0+cu126"): "2.13.0",
    ("torchaudio", "2.11.0+cu126"): "2.11.0",
    ("torchvision", "0.28.0+cu126"): "0.28.0",
}


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def normalize_inventory(records: list[dict], source_version: str) -> tuple[dict, list]:
    expected = {}
    mappings = []
    seen = set()
    for record in records:
        name, installed = canonical_name(record["name"]), record["version"]
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name) or name in seen:
            raise ValueError("Invalid or duplicate installed distribution")
        seen.add(name)
        if name == "voscript-core":
            if installed != source_version:
                raise ValueError("First-party extension version does not match source")
            mappings.append(
                {
                    "name": name,
                    "installed_version": installed,
                    "audit": "Cargo.lock via cargo-audit, plus native image gates",
                    "cargo_lock_sha256": hashlib.sha256(
                        (ROOT / "Cargo.lock").read_bytes()
                    ).hexdigest(),
                }
            )
            continue
        upstream = CUDA_UPSTREAM.get((name, installed), installed)
        if Version(upstream).local is not None:
            raise ValueError(f"Unexplained local-version audit mapping: {name}")
        if upstream != installed:
            mappings.append(
                {
                    "name": name,
                    "installed_version": installed,
                    "audit_version": upstream,
                    "source": "https://download.pytorch.org/whl/cu126",
                    "reason": "Official CUDA build alias for upstream advisory matching; no advisory suppression",
                }
            )
        expected[name] = upstream
    if not {"torch", "torchaudio", "torchvision", "whisperx", "voscript-core"}.issubset(
        seen
    ):
        raise ValueError("Actual image inventory lacks required runtime distributions")
    return expected, mappings


def validate_audit_report(report: dict, expected: dict, returncode: int) -> None:
    dependencies = report.get("dependencies")
    if not isinstance(dependencies, list):
        raise ValueError("Audit did not produce a complete dependency report")
    seen = set()
    for dependency in dependencies:
        name = canonical_name(dependency["name"])
        if dependency.get("skip_reason") is not None:
            raise ValueError(f"Unexplained skipped distribution in image audit: {name}")
        if name in seen or name not in expected:
            raise ValueError(
                "Audit dependency identity differs from installed inventory"
            )
        seen.add(name)
        if Version(dependency["version"]) != Version(expected[name]):
            raise ValueError(
                f"Audit version differs from installed/upstream mapping: {name}"
            )
        if not isinstance(dependency.get("vulns"), list) or dependency["vulns"]:
            raise ValueError(f"Unresolved image advisory or incomplete audit: {name}")
    if seen != set(expected) or returncode != 0:
        raise ValueError(
            "Image audit is missing distributions or exited unsuccessfully"
        )


def command(*args: str, capture: bool = False, timeout: int = 300) -> str:
    result = subprocess.run(
        list(args),
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        timeout=timeout,
    )
    return (result.stdout or "").strip()


def audit(image: str) -> None:
    inventory = command(
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--entrypoint",
        "python",
        image,
        "-c",
        INVENTORY,
        capture=True,
    )
    evidence = (
        Path(os.environ.get("RUNNER_TEMP", tempfile.gettempdir()))
        / "voscript-image-audit"
    )
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "raw-inventory.json").write_text(inventory + "\n")
    expected, mappings = normalize_inventory(json.loads(inventory), version(ROOT))
    (evidence / "audit-mappings.json").write_text(json.dumps(mappings, indent=2) + "\n")
    requirements = evidence / "audit-inventory.txt"
    requirements.write_text(
        "".join(f"{name}=={release}\n" for name, release in sorted(expected.items()))
    )
    report = evidence / "pip-audit.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip_audit",
            "--strict",
            "--no-deps",
            "--disable-pip",
            "-r",
            str(requirements),
            "--format",
            "json",
            "--output",
            str(report),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    (evidence / "audit-stdout.txt").write_text(result.stdout)
    (evidence / "audit-stderr.txt").write_text(result.stderr)
    if not report.is_file():
        raise ValueError("Image audit failed without a machine-readable report")
    validate_audit_report(json.loads(report.read_text()), expected, result.returncode)


def smoke(image: str) -> None:
    expected = version(ROOT)
    command(
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        *ENV,
        "--workdir",
        "/app",
        "--entrypoint",
        "python",
        image,
        "build/compat_import_smoke.py",
    )
    for code in (MODEL_IMPORT_SMOKE, CORE_SMOKE, NATIVE_GOLDENS, MISSING_CORE):
        command(
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            *ENV,
            "--entrypoint",
            "python",
            image,
            "-c",
            code,
            expected,
        )
    cid = command("docker", "run", "-d", "--network", "none", *ENV, image, capture=True)
    try:
        probe = (
            "import json,sys,urllib.request; "
            "urllib.request.urlopen('http://127.0.0.1:8780/healthz',timeout=2).read(); "
            "payload=json.load(urllib.request.urlopen('http://127.0.0.1:8780/openapi.json',timeout=2)); "
            "assert payload['info']['version']==sys.argv[1]"
        )
        for _ in range(60):
            result = subprocess.run(
                ["docker", "exec", cid, "python", "-c", probe, expected],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
            )
            if result.returncode == 0:
                return
            if (
                command(
                    "docker",
                    "inspect",
                    "--format",
                    "{{.State.Running}}",
                    cid,
                    capture=True,
                )
                != "true"
            ):
                raise RuntimeError("Image exited before becoming healthy")
            time.sleep(2)
        raise RuntimeError("Image did not pass health/API-version smoke")
    finally:
        command("docker", "rm", "-f", "-v", cid)


def real_audio_compatibility(image: str) -> None:
    command(
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        *ENV,
        "-e",
        "PYTHONPATH=/app",
        "--mount",
        f"type=bind,source={ROOT / 'tests/compat'},target=/ci-compat,readonly",
        "--workdir",
        "/app",
        "--entrypoint",
        "python",
        image,
        "/ci-compat/test_security_audio_io.py",
    )


def unit_tests(image: str) -> None:
    # Install test-only tools into the disposable container writable layer, never
    # the published image. Bootstrap real audio libraries before conftest stubs.
    command(
        "docker",
        "run",
        "--rm",
        "--user",
        "0",
        *ENV,
        "-e",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "--mount",
        f"type=bind,source={ROOT},target=/ci,readonly",
        "--entrypoint",
        "sh",
        image,
        "-ec",
        "python -m pip install --target /opt/ci-tools pytest pyyaml httpx && "
        "cd /ci && PYTHONPATH=/opt/ci-tools:/ci/voscript-api/scripts "
        "python -m pytest tests/unit/ tests/test_security.py tests/test_job_service.py tests/test_voiceprint_db.py "
        "-p ci_real_audio -p no:cacheprovider -v --tb=short --no-header",
        timeout=1800,
    )


def main() -> None:
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--source-root", type=Path, default=ROOT)
    args = parser.parse_args()
    ROOT = args.source_root.resolve()
    audit(args.image)
    smoke(args.image)
    real_audio_compatibility(args.image)
    unit_tests(args.image)
    print(
        "Installed-image audit, model imports/checkpoint/alignment, native contracts, "
        "missing-extension, health smoke and real-audio unit regressions passed"
    )


if __name__ == "__main__":
    main()
