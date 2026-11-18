"""Tests for the repository-owned public release scanner."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCANNER = ROOT / "voscript-api" / "scripts" / "public_release_scan.py"


def _run_git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, stdout=subprocess.PIPE)


def _scan_fixture(
    content: str,
    filename: str = "fixture.md",
    *,
    stage: bool = True,
    include_untracked: bool = False,
) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _run_git(root, "init", "-q")
        fixture = root / filename
        fixture.parent.mkdir(parents=True, exist_ok=True)
        fixture.write_text(content, encoding="utf-8")
        if stage:
            _run_git(root, "add", "--", filename)
        command = [sys.executable, str(SCANNER), "--root", str(root)]
        if include_untracked:
            command.append("--include-untracked")
        return subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )


def test_public_release_scan_allows_placeholders():
    result = _scan_fixture(
        "\n".join(
            [
                "Authorization: Bearer <API_KEY>",
                "HF_TOKEN=${HF_TOKEN}",
                "API_KEY=your-api-key",
                "Use internal live validation for release notes.",
            ]
        )
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Public release scan passed" in result.stdout


def test_public_release_scan_blocks_secret_looking_assignments():
    synthetic_secret = "sk-live-" + "syntheticsecret123456789"

    result = _scan_fixture(f"API_KEY={synthetic_secret}")

    assert result.returncode == 1
    assert "secret-looking assignment" in result.stdout


def test_public_release_scan_blocks_private_paths_and_real_ids():
    local_path = "/" + "Users/example/private.log"
    transcription_id = "tr_" + "20260426_124218_abcdef"
    speaker_id = "spk_" + "1234abcd"
    fixture = "\n".join(
        [
            f"Read {local_path} before publishing.",
            f"Result id {transcription_id} should not be public.",
            f"Speaker id {speaker_id} should not be public.",
        ]
    )

    result = _scan_fixture(fixture)

    assert result.returncode == 1
    assert "machine-local path" in result.stdout
    assert "real transcription id" in result.stdout
    assert "real speaker id" in result.stdout


TEXT_SURFACE_SUFFIXES = (
    ".cfg",
    ".cjs",
    ".css",
    ".cts",
    ".dockerignore",
    ".env",
    ".gitignore",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".jsx",
    ".map",
    ".md",
    ".mjs",
    ".mts",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
)


@pytest.mark.parametrize("suffix", TEXT_SURFACE_SUFFIXES)
@pytest.mark.parametrize("stage", [True, False])
def test_all_text_surfaces_block_secret_and_internal_references(suffix, stage):
    secret = "sk-live-" + "syntheticcredentialvalue123456789"
    reference = "CONT" + "EXT.md"
    content = f"API_KEY={secret}\nRead {reference}."
    result = _scan_fixture(
        content, "public" + suffix, stage=stage, include_untracked=not stage
    )
    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert "secret-looking assignment" in output
    assert "internal document reference" in output
    assert "[source content redacted]" in output
    assert secret not in output
    assert reference not in output
    assert content not in output


@pytest.mark.parametrize("suffix", TEXT_SURFACE_SUFFIXES)
def test_all_text_surfaces_allow_public_placeholders(suffix):
    content = "API_KEY=<API_KEY>\nUsage: configure the service."
    result = _scan_fixture(content, "public" + suffix)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "filename",
    [
        str(Path(directory) / "synthetic-operator-note.md")
        for directory in (
            "adr",
            "adrs",
            "roadmap",
            "tmp",
            "internal",
            "local-docs",
            "local_docs",
            "validation-notes",
            "validation_notes",
        )
    ]
    + [
        "doc/" + "ADR" + "0012-synthetic-decision.md",
        "doc/" + "CONT" + "EXT.md",
        "doc/" + "CONT" + "EXT-MAP.md",
        "doc/" + "ROAD" + "MAP.md",
        "release" + ".local.md",
    ],
)
def test_internal_document_diagnostic_paths_are_redacted(filename):
    result = _scan_fixture("Synthetic operator-only note.", filename)
    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert "tracked local-only directory" in output
    assert "[sensitive path redacted]:1:" in output
    assert filename not in output
    assert Path(filename).name not in output
    assert "[source content redacted]" in output


def test_public_diagnostic_paths_remain_available():
    secret = "hf_" + "syntheticcredentialvalue123456789"
    result = _scan_fixture("API_KEY=" + secret, "doc/usage.md")
    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert "doc/usage.md:1: secret-looking assignment" in output
    assert secret not in output
