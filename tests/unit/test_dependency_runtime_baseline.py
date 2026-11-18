"""Regression tests for runtime dependency pins that affect Docker GPU loads."""

from __future__ import annotations

import shlex
from pathlib import Path


def _requirements_lines() -> list[str]:
    root = Path(__file__).resolve().parents[2]
    return [
        line.strip()
        for line in (root / "app" / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _dockerfile_text() -> str:
    root = Path(__file__).resolve().parents[2]
    return (root / "app" / "Dockerfile").read_text()


def test_idna_runtime_pin_excludes_cve_2026_45409():
    root = Path(__file__).resolve().parents[2]
    lines = _requirements_lines()
    constraints = (
        (root / "app" / "requirements.constraints.txt").read_text().splitlines()
    )
    docker_constraints = (
        (root / "app" / "requirements.docker.constraints.txt").read_text().splitlines()
    )

    assert "idna==3.15" in lines
    assert "idna==3.15" in constraints
    assert "-c requirements.constraints.txt" in lines
    assert "-c requirements.constraints.txt" in docker_constraints
    assert (
        "ENV PIP_CONSTRAINT=/app/requirements.docker.constraints.txt"
        in _dockerfile_text()
    )


def test_faster_whisper_runtime_stays_on_cudnn9_compatible_ctranslate2():
    lines = _requirements_lines()

    assert "faster-whisper>=1.2.1,<2.0.0" in lines
    assert "ctranslate2>=4.7.1,<5.0" in lines
    assert "faster-whisper==1.1.0" not in lines


def test_docker_installs_whisperx_without_replacing_asr_runtime_stack():
    lines = _requirements_lines()
    dockerfile = _dockerfile_text()
    root = Path(__file__).resolve().parents[2]

    assert not any(line.startswith("nltk") for line in lines)
    assert (root / "app" / "nltk" / "tokenize" / "punkt.py").exists()
    assert "whisperx==3.3.1" not in lines
    instructions = dockerfile.replace("\\\n", " ").splitlines()
    installs = [
        instruction
        for instruction in instructions
        if instruction.startswith("RUN ") and "whisperx==3.3.1" in instruction
    ]
    assert len(installs) == 1
    install = installs[0]
    commands = [shlex.split(command.strip()) for command in install.split(";")]
    whisperx = next(command for command in commands if "whisperx==3.3.1" in command)
    assert whisperx[:4] == ["python", "-m", "pip", "install"]
    assert "--no-cache-dir" in whisperx
    assert "--no-deps" in whisperx
    assert not any(
        token.startswith("whisperx") and token != "whisperx==3.3.1"
        for token in whisperx
    )
    # ARG is shell-local unless exported; require export in the very same RUN,
    # before the install, rather than an obsolete pip command's literal spelling.
    assert 'ARG PIP_INDEX_URL=""' in dockerfile
    assert 'if [ -n "$PIP_INDEX_URL" ]; then export PIP_INDEX_URL; fi;' in install
    assert install.index("export PIP_INDEX_URL") < install.index("whisperx==3.3.1")
