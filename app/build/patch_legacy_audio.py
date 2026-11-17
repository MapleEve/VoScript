"""Apply narrow import adaptations only to authenticated upstream wheel bytes.

Wheel hashes come from the linked PyPI release pages in legacy_wheels.txt.
No package is imported while patching. Every installed target must match the
corresponding member of a known wheel, byte for byte, before any file is written.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sysconfig
from importlib.metadata import distribution
from pathlib import Path
from zipfile import ZipFile

HUB_IMPORT = (
    "from huggingface_hub import hf_hub_download\n",
    "from infra.huggingface_compat import hf_hub_download\n",
    1,
)

PACKAGES = {
    "pyannote.audio": {
        "version": "3.3.2",
        "wheels": {
            "pyannote_audio-3.3.2-py2.py3-none-any.whl": "23e0dcedda920cb2e154e146bcd9663289ee7942d0e012663dad76f2e571ebeb",
            "pyannote.audio-3.3.2-py2.py3-none-any.whl": "599c694acd5d193215147ff82d0bf638bb191204ed502bd9fde8ff582e20aa1c",
        },
        "files": {
            "pyannote/audio/core/io.py": (
                (
                    "import torchaudio\n",
                    "from infra.audio import io_compat as torchaudio\n",
                    1,
                ),
            ),
            "pyannote/audio/core/model.py": (HUB_IMPORT,),
            "pyannote/audio/core/pipeline.py": (HUB_IMPORT,),
            "pyannote/audio/pipelines/speaker_verification.py": (
                (
                    "from pyannote.audio.core.io import AudioFile\n",
                    "from pyannote.audio.core.io import AudioFile\n"
                    "from infra.audio.speechbrain_compat import from_hparams as speechbrain_from_hparams\n",
                    1,
                ),
                (
                    "SpeechBrain_EncoderClassifier.from_hparams(\n",
                    "speechbrain_from_hparams(SpeechBrain_EncoderClassifier,\n",
                    2,
                ),
            ),
        },
    },
    "speechbrain": {
        "version": "1.1.2",
        "wheels": {
            "speechbrain-1.1.2-py3-none-any.whl": "9fea4144c917ea58bcf13f17a1a1759179ed7472a30ec8416e7c7c8cf21d9c9a",
        },
        "files": {
            "speechbrain/utils/torch_audio_backend.py": (
                (
                    "import torchaudio\n",
                    "from infra.audio import io_compat as torchaudio\n",
                    1,
                ),
            ),
        },
    },
    "deepfilternet": {
        "version": "0.5.6",
        "wheels": {
            "deepfilternet-0.5.6-py3-none-any.whl": "99f5688d954fcfa8f853bf8bb8c3b2a59e4f9dc5d95643c9e6a32053234ba7c6",
        },
        "files": {
            "df/io.py": (
                (
                    "import torchaudio as ta\n",
                    "from infra.audio import io_compat as ta\n",
                    1,
                ),
                (
                    "from torchaudio.backend.common import AudioMetaData\n",
                    "from infra.audio.io_compat import AudioMetaData\n",
                    1,
                ),
            ),
        },
    },
    "torch-audiomentations": {
        "version": "0.12.0",
        "wheels": {
            "torch_audiomentations-0.12.0-py3-none-any.whl": "1b80b91d2016ccf83979622cac8f702072a79b7dcc4c2bee40f00b26433a786b",
        },
        "files": {
            "torch_audiomentations/utils/io.py": (
                (
                    "import torchaudio\n",
                    "from infra.audio import io_compat as torchaudio\n",
                    1,
                ),
            ),
        },
    },
}


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def adapt_source(original: bytes, replacements, member: str) -> bytes:
    source = original.decode("utf-8")
    if source.startswith("\ufeff") or "\r" in source:
        raise ValueError(f"Unexpected source encoding/newlines: {member}")
    ast.parse(source, filename=member)
    for old, new, count in replacements:
        if source.count(old) != count:
            raise ValueError(f"Unknown upstream import/call structure: {member}")
        source = source.replace(old, new)
    ast.parse(source, filename=member)
    return source.encode("utf-8")


def checked_wheel(wheel_dir: Path, package: str, spec: dict) -> Path:
    matches = [
        wheel_dir / name for name in spec["wheels"] if (wheel_dir / name).is_file()
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one allowlisted wheel: {package}")
    wheel = matches[0]
    if sha256(wheel.read_bytes()) != spec["wheels"][wheel.name]:
        raise ValueError(f"Unrecognized wheel SHA256: {package}")
    return wheel


def checked_sources(archive: ZipFile, spec: dict):
    for member, replacements in spec["files"].items():
        if archive.namelist().count(member) != 1:
            raise ValueError(f"Missing or duplicate wheel source member: {member}")
        original = archive.read(member)
        yield member, original, adapt_source(original, replacements, member)


def plan_patches(wheel_dir: Path):
    plan = []
    report = []
    site_packages = Path(sysconfig.get_paths()["purelib"]).resolve()
    for package, spec in PACKAGES.items():
        wheel = checked_wheel(wheel_dir, package, spec)
        installed = distribution(package)
        if installed.version != spec["version"]:
            raise ValueError(f"Unsupported installed version: {package}")
        with ZipFile(wheel) as archive:
            for member, original, patched in checked_sources(archive, spec):
                target = Path(installed.locate_file(member)).resolve()
                if not target.is_relative_to(site_packages):
                    raise ValueError(f"Refusing a non-site-packages target: {member}")
                if target.read_bytes() not in (original, patched):
                    raise ValueError(
                        f"Installed source differs from authenticated wheel: {member}"
                    )
                plan.append((target, patched))
                report.append(
                    {
                        "package": package,
                        "version": installed.version,
                        "wheel": wheel.name,
                        "wheel_sha256": spec["wheels"][wheel.name],
                        "member": member,
                        "original_sha256": sha256(original),
                        "patched_sha256": sha256(patched),
                    }
                )
    return plan, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--verify-wheels-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.verify_wheels_only:
        for package, spec in PACKAGES.items():
            wheel = checked_wheel(args.wheel_dir, package, spec)
            with ZipFile(wheel) as archive:
                for _ in checked_sources(archive, spec):
                    pass
        print("Authenticated all legacy compatibility wheels and source structures")
        return
    # Validate the whole plan before touching any installed source file.
    plan, report = plan_patches(args.wheel_dir)
    for target, content in plan:
        target.write_bytes(content)
    serialized = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
