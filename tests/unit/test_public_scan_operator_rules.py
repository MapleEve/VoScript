"""Synthetic privacy regressions; no operator aliases, routes or ports are stored."""

import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCANNER = ROOT / "voscript-api/scripts/public_release_scan.py"
sys.path.insert(0, str(SCANNER.parent))
import public_release_scan as scanner  # noqa: E402


def scan(tmp_path, content):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "README.md").write_text(content)
    subprocess.run(["git", "-C", str(tmp_path), "add", "README.md"], check=True)
    return subprocess.run(
        [sys.executable, str(SCANNER), "--root", str(tmp_path)],
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "arguments",
    [
        " synthetic-deployment",
        " operator@synthetic-host",
        " -p 45678 synthetic-target",
        " -i <KEY_FILE> synthetic-host",
    ],
)
def test_generic_ssh_host_tokens_block_literals_without_knowing_aliases(
    tmp_path, arguments
):
    result = scan(tmp_path, "ssh" + arguments)
    assert result.returncode == 1
    assert "literal SSH host" in result.stdout
    assert arguments not in result.stdout + result.stderr


@pytest.mark.parametrize(
    "arguments",
    [" <SSH_ALIAS>", " -p <SSH_PORT> <SSH_ALIAS>", " <USER>@<HOST>", " ${SSH_ALIAS}"],
)
def test_explicit_ssh_placeholders_remain_public(tmp_path, arguments):
    result = scan(tmp_path, "ssh" + arguments)
    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize("separator", ["; ", " && ", " || ", "\n"])
def test_placeholder_ssh_cannot_hide_a_later_literal_host(tmp_path, separator):
    first = "ssh" + " <SSH_ALIAS>"
    result = scan(tmp_path, first + separator + "ssh" + " synthetic-host")
    assert result.returncode == 1
    assert "literal SSH host" in result.stdout
    assert "synthetic-host" not in result.stdout + result.stderr
    allowed = scan(tmp_path, first + separator + "ssh" + " <SSH_ALIAS>")
    assert allowed.returncode == 0, allowed.stdout


@pytest.mark.parametrize(
    "prefix", ["localhost:", "127.0.0.1:", "[::1]:", "SOCKS_PORT="]
)
def test_generic_nonpublic_local_debug_port_is_redacted(tmp_path, prefix):
    synthetic_port = str(45678)
    result = scan(tmp_path, prefix + synthetic_port)
    assert result.returncode == 1
    assert "non-public local debug port" in result.stdout
    assert synthetic_port not in result.stdout + result.stderr


def test_public_service_and_socks_placeholder_are_allowed(tmp_path):
    result = scan(tmp_path, "http://localhost:8780/healthz\nlocalhost:<SOCKS_PORT>\n")
    assert result.returncode == 0, result.stdout


def test_optional_markers_are_literal_external_input_and_never_echoed(
    tmp_path, monkeypatch
):
    marker = "synthetic-operator-marker[.]"
    monkeypatch.setenv(
        "VOSCRIPT_PUBLIC_SCAN_RULES_JSON", json.dumps({"markers": [marker]})
    )
    result = scan(tmp_path, "Deployment note: " + marker)
    assert result.returncode == 1
    assert "operator private marker" in result.stdout
    assert marker not in result.stdout + result.stderr
    assert marker not in SCANNER.read_text()


def test_invalid_optional_input_has_no_parse_details_or_marker_echo(
    tmp_path, monkeypatch
):
    invalid = "synthetic-private-value-invalid-json"
    monkeypatch.setenv("VOSCRIPT_PUBLIC_SCAN_RULES_JSON", invalid)
    result = scan(tmp_path, "Public usage instructions")
    assert result.returncode == 2
    assert invalid not in result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    assert "invalid optional operator rules" in result.stderr


def test_unknown_bare_alias_does_not_claim_complete_detection(tmp_path, monkeypatch):
    monkeypatch.delenv("VOSCRIPT_PUBLIC_SCAN_RULES_JSON", raising=False)
    result = scan(tmp_path, "An unclassified synthetic-bare-operator-name")
    assert result.returncode == 0, result.stdout
    assert not scanner.operator_rules_from_env()


def test_removed_machine_fact_constants_cannot_be_reintroduced_as_concatenations():
    tree = ast.parse(SCANNER.read_text())
    assigned = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    obsolete = {
        "PRIVATE_" + suffix
        for suffix in [
            "DIRECT_SSH_ALIAS",
            "WAN_SSH_ALIAS",
            "REMOTE_ALIASES",
            "PROXY_HOST_PORT",
        ]
    }
    assert not assigned.intersection(obsolete)
    assert scanner.PUBLIC_SERVICE_PORTS == {80, 443, 8780}
