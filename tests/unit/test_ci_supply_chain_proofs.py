"""Adversarial CI regressions for trusted source/proof and installed-image audit."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import urllib.error
import zipfile

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "voscript-api/scripts"))
import ci_release_contracts as contracts  # noqa: E402
import image_release_gate as images  # noqa: E402


def workflow(filename):
    return yaml.load(
        (ROOT / ".github/workflows" / filename).read_text(), Loader=yaml.BaseLoader
    )


def proof_fixture(monkeypatch):
    run = {
        "id": 123,
        "run_attempt": 2,
        "head_sha": "b" * 40,
        "head_branch": "main",
        "event": "workflow_dispatch",
        "path": ".github/workflows/rust-foundation-heavy.yml",
        "status": "completed",
        "conclusion": "success",
    }
    workflow_bytes = b"trusted workflow bytes\n"
    proof = {
        "schema": "heavy-target.v1",
        "repository": "example/project",
        "run_id": 123,
        "run_attempt": 2,
        "workflow_sha": run["head_sha"],
        "target_sha": "a" * 40,
        "workflow_ref": "example/project/.github/workflows/rust-foundation-heavy.yml@refs/heads/main",
        "workflow_file_sha256": hashlib.sha256(workflow_bytes).hexdigest(),
    }
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("proof.json", json.dumps(proof))
    archive_bytes = data.getvalue()
    digest = "sha256:" + hashlib.sha256(archive_bytes).hexdigest()
    artifact = {
        "id": 101,
        "name": "heavy-target-proof-123-2",
        "expired": False,
        "digest": digest,
        "workflow_run": {"id": 123, "head_sha": run["head_sha"]},
    }
    anchor = {"artifact_id": 101, "artifact_digest": digest}
    log = (
        "2026-11-17T00:00:00.000Z HEAVY_PROOF_ANCHOR=" + json.dumps(anchor) + "\n"
    ).encode()
    state = {
        "archive": archive_bytes,
        "artifact": artifact,
        "log": log,
        "workflow_bytes": workflow_bytes,
        "proof": proof,
    }
    jobs = [
        {"id": 456, "name": "resolve-source", "conclusion": "success"},
        {"id": 457, "name": "rust-wheel", "conclusion": "success"},
        {"id": 458, "name": "docker-packaging", "conclusion": "success"},
    ]

    def fake_api(path):
        if "/attempts/" in path and "/jobs?" in path:
            return {"jobs": jobs}
        if path == "actions/artifacts/101":
            return state["artifact"]
        raise AssertionError(path)

    def fake_bytes(path):
        if path == "actions/jobs/456/logs":
            return state["log"]
        if path == "actions/artifacts/101/zip":
            return state["archive"]
        raise AssertionError(path)

    monkeypatch.setenv("GITHUB_REPOSITORY", "example/project")
    monkeypatch.setattr(contracts, "api", fake_api)
    monkeypatch.setattr(contracts, "api_bytes", fake_bytes)
    monkeypatch.setattr(
        contracts, "git_file", lambda ref, path: state["workflow_bytes"]
    )
    monkeypatch.setattr(contracts.subprocess, "run", lambda *args, **kwargs: None)
    return run, state


def test_proof_binds_real_resolver_log_archive_digest_workflow_attempt_and_target(
    monkeypatch,
):
    run, _ = proof_fixture(monkeypatch)
    proof = contracts.manual_proof(run)
    assert proof["target_sha"] == "a" * 40
    assert proof["run_id"] == 123 and proof["run_attempt"] == 2
    check = {
        "name": "rust-wheel",
        "head_sha": "a" * 40,
        "status": "completed",
        "conclusion": "success",
        "app": {"id": 15368},
        "external_id": "arbitrary forged metadata",
    }
    assert contracts.check_evidence(
        check, run, "a" * 40, "rust-foundation-heavy.yml", proof
    )
    assert not contracts.check_evidence(
        check, run, "a" * 40, "rust-foundation-heavy.yml"
    )
    check["head_sha"] = "c" * 40
    check["external_id"] = "heavy:123:2:" + "c" * 40
    assert not contracts.check_evidence(
        check, run, "c" * 40, "rust-foundation-heavy.yml", proof
    )


@pytest.mark.parametrize(
    "attack",
    ["archive", "attempt", "run", "workflow", "anchor", "replacement", "branch"],
)
def test_manual_proof_rejects_replaced_or_replayed_evidence(monkeypatch, attack):
    run, state = proof_fixture(monkeypatch)
    if attack == "archive":
        state["archive"] += b"tampered"
    elif attack == "attempt":
        run["run_attempt"] = 3
    elif attack == "run":
        run["id"] = 124
    elif attack == "workflow":
        state["workflow_bytes"] = b"different workflow source\n"
    elif attack == "anchor":
        state["log"] = b"no trusted resolver anchor\n"
    elif attack == "replacement":
        state["artifact"]["id"] = 202
    else:
        run["head_branch"] = "candidate"
    with pytest.raises(ValueError):
        contracts.manual_proof(run)


@pytest.mark.parametrize(
    "scenario",
    [
        "replay-success",
        "valid-recovery",
        "other-target",
        "old-attempt",
        "failure-without-check",
    ],
)
def test_latest_actions_execution_not_check_creation_controls_acceptance(
    monkeypatch, scenario
):
    sha = "a" * 40
    workflow_name = "rust-foundation-heavy.yml"

    def run(run_id, conclusion, attempt=1, event="workflow_dispatch"):
        return {
            "id": run_id,
            "run_attempt": attempt,
            "head_sha": sha,
            "head_branch": "main",
            "event": event,
            "path": f".github/workflows/{workflow_name}",
            "run_started_at": f"2026-11-17T00:00:0{run_id + attempt - 1}Z",
            "status": "completed",
            "conclusion": conclusion,
        }

    current = {
        1: run(1, "failure" if scenario == "valid-recovery" else "success"),
        2: run(
            2,
            "success" if scenario in {"valid-recovery", "other-target"} else "failure",
        ),
    }
    if scenario == "other-target":
        current[1]["event"] = "push"
    if scenario == "old-attempt":
        current = {1: run(1, "failure", attempt=2)}
    listed = {run_id: dict(value) for run_id, value in current.items()}
    if scenario == "old-attempt":
        listed[1] = run(
            1, "success", attempt=1
        )  # stale list and a newly replayed old check

    def check(check_id, run_id, conclusion):
        return {
            "id": check_id,
            "name": "rust-wheel",
            "head_sha": sha,
            "app": {"id": 15368},
            "status": "completed",
            "conclusion": conclusion,
            "external_id": f"heavy:{run_id}:1:{sha}",
            "details_url": f"https://github.com/example/project/actions/runs/{run_id}/job/{run_id}",
        }

    checks = [check(999, 1, "success"), check(1, 1, "success")]
    if 2 in current and scenario != "failure-without-check":
        checks.append(check(2, 2, current[2]["conclusion"]))
    refreshed = []

    def fake_api(path):
        if path.startswith("commits/"):
            return {"check_runs": checks}
        if path.startswith("actions/workflows/"):
            event = "workflow_dispatch" if "event=workflow_dispatch" in path else "push"
            return {
                "workflow_runs": [
                    value for value in listed.values() if value["event"] == event
                ]
            }
        if "/attempts/" in path:
            parts = path.split("/")
            actual = current[int(parts[2])]
            assert int(parts[4]) == actual["run_attempt"]
            return {
                "jobs": [{"name": "rust-wheel", "conclusion": actual["conclusion"]}]
            }
        run_id = int(path.rsplit("/", 1)[1])
        refreshed.append(run_id)
        return current[run_id]

    def verified_proof(actual):
        assert actual["run_attempt"] == current[actual["id"]]["run_attempt"]
        return {
            "run_id": actual["id"],
            "run_attempt": actual["run_attempt"],
            "target_sha": "c" * 40
            if scenario == "other-target" and actual["id"] == 2
            else sha,
            "job_results": {
                "resolve-source": "success",
                "rust-wheel": actual["conclusion"],
                "docker-packaging": actual["conclusion"],
            },
        }

    monkeypatch.setenv("GITHUB_REPOSITORY", "example/project")
    monkeypatch.setattr(contracts, "REQUIRED_CHECKS", {"rust-wheel": workflow_name})
    monkeypatch.setattr(contracts, "api", fake_api)
    monkeypatch.setattr(contracts, "manual_proof", verified_proof)
    if scenario in {"valid-recovery", "other-target"}:
        contracts.complete_checks(sha)
    else:
        with pytest.raises(ValueError, match="rust-wheel"):
            contracts.complete_checks(sha)
    assert set(refreshed) == set(current)


def test_release_validator_is_read_only_main_code_before_candidate_checkout():
    release = workflow("release.yml")
    assert set(release["on"]) == {"workflow_dispatch"}
    validator = release["jobs"]["validate-source"]
    assert all(value == "read" for value in validator["permissions"].values())
    assert "secrets." not in str(validator)
    steps = validator["steps"]
    assert steps[0]["with"]["ref"] == "${{ github.workflow_sha }}"
    assert steps[0]["with"]["path"] == "control"
    verify = next(i for i, step in enumerate(steps) if step.get("id") == "source")
    candidate = next(
        i
        for i, step in enumerate(steps)
        if step.get("with", {}).get("path") == "candidate"
    )
    assert verify < candidate
    assert steps[verify]["working-directory"] == "control"
    assert (
        steps[-1]["run"]
        == "python control/voscript-api/scripts/public_release_scan.py --root candidate"
    )
    publish = release["jobs"]["publish"]
    assert publish["needs"] == "validate-source"
    assert publish["permissions"]["packages"] == "write"
    assert "../control/voscript-api/scripts/image_release_gate.py" in str(publish)
    publish_steps = publish["steps"]
    revalidate = next(
        i
        for i, step in enumerate(publish_steps)
        if "EXPECTED_SOURCE_SHA" in step.get("env", {})
    )
    smoke = next(
        i
        for i, step in enumerate(publish_steps)
        if "image_release_gate.py" in step.get("run", "")
    )
    authenticate = next(
        i
        for i, step in enumerate(publish_steps)
        if "docker login" in step.get("run", "")
    )
    push = next(
        i
        for i, step in enumerate(publish_steps)
        if "docker push" in step.get("run", "")
    )
    assert smoke < revalidate < authenticate < push
    assert publish_steps[revalidate]["working-directory"] == "control"
    assert (
        publish_steps[revalidate]["run"]
        == "python voscript-api/scripts/ci_release_contracts.py release"
    )
    assert (
        publish_steps[revalidate]["env"]["EXPECTED_SOURCE_SHA"]
        == "${{ needs.validate-source.outputs.sha }}"
    )
    heavy = workflow("rust-foundation-heavy.yml")
    assert heavy["on"]["pull_request"]["types"] == [
        "opened",
        "reopened",
        "ready_for_review",
    ]
    resolver = heavy["jobs"]["resolve-source"]
    proof_upload = next(step for step in resolver["steps"] if step.get("id") == "proof")
    assert proof_upload["with"]["overwrite"] == "false"
    assert "github.run_attempt" in proof_upload["with"]["name"]


@pytest.mark.parametrize("annotated", [False, True])
@pytest.mark.parametrize(
    "change", ["unchanged", "moved", "deleted", "network", "timeout"]
)
def test_release_revalidation_refreshes_remote_tag_despite_stale_local_green_source(
    tmp_path, monkeypatch, annotated, change
):
    sha, control = "a" * 40, "b" * 40
    event = tmp_path / "event.json"
    output = tmp_path / "output"
    event.write_text(json.dumps({"inputs": {"tag": "v1.2.3"}}))
    for name, value in {
        "GITHUB_EVENT_PATH": str(event),
        "GITHUB_OUTPUT": str(output),
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_REPOSITORY": "example/project",
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_WORKFLOW_REF": "example/project/.github/workflows/release.yml@refs/heads/main",
        "GITHUB_WORKFLOW_SHA": control,
        "GITHUB_SHA": control,
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("EXPECTED_SOURCE_SHA", raising=False)
    git_calls, remote_calls, checked = [], [], []
    state = {"changed": False}

    def fake_git(*args):
        git_calls.append(args)
        if args in {("rev-parse", "HEAD"), ("rev-parse", "origin/main")}:
            return control
        if args == ("rev-parse", "--verify", f"{sha}^{{commit}}"):
            return sha
        if args == ("rev-parse", "--verify", "refs/tags/v1.2.3^{commit}"):
            return sha  # The checkout still has the previously green tag.
        raise AssertionError(f"Unexpected git call: {args}")

    def fake_api(path):
        if path == "git/ref/tags/v1.2.3":
            remote_calls.append(path)
            if state["changed"]:
                if change == "deleted":
                    raise urllib.error.HTTPError(
                        "https://api.github.com/", 404, "Not Found", None, None
                    )
                if change == "network":
                    raise urllib.error.URLError("synthetic network failure")
                if change == "timeout":
                    raise TimeoutError("synthetic timeout")
            moved = state["changed"] and change == "moved"
            target = "c" * 40 if moved else sha
            return {
                "ref": "refs/tags/v1.2.3",
                "object": {
                    "type": "tag" if annotated else "commit",
                    "sha": ("e" if moved else "d") * 40 if annotated else target,
                },
            }
        if path in {f"git/tags/{'d' * 40}", f"git/tags/{'e' * 40}"}:
            remote_calls.append(path)
            return {
                "sha": path.rsplit("/", 1)[1],
                "object": {
                    "type": "commit",
                    "sha": "c" * 40 if path.endswith("e" * 40) else sha,
                },
            }
        if path == f"commits/{sha}/pulls?per_page=100":
            return [
                {
                    "merged_at": "synthetic",
                    "base": {"ref": "main"},
                    "merge_commit_sha": sha,
                }
            ]
        if path == "releases/tags/v1.2.3":
            return {"draft": False, "prerelease": False, "published_at": "synthetic"}
        if path == "branches/main":
            return {
                "protected": True,
                "commit": {"sha": control},
                "protection": {
                    "required_status_checks": {
                        "contexts": list(contracts.REQUIRED_CHECKS),
                        "checks": [
                            {"context": name, "app_id": 15368}
                            for name in contracts.REQUIRED_CHECKS
                        ],
                    }
                },
            }
        raise AssertionError(f"Unexpected API call: {path}")

    monkeypatch.setattr(contracts, "git", fake_git)
    monkeypatch.setattr(contracts, "api", fake_api)
    monkeypatch.setattr(contracts.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        contracts, "git_file", lambda ref, path: b'APP_VERSION = "1.2.3"\n'
    )
    monkeypatch.setattr(contracts, "complete_checks", checked.append)
    contracts.release_source()
    initial_output = output.read_text()
    assert f"sha={sha}\n" in initial_output
    assert checked == [sha]

    # Model the publish job's unchanged checkout after a successful first validation.
    state["changed"] = True
    monkeypatch.setenv("EXPECTED_SOURCE_SHA", sha)
    if change == "unchanged":
        contracts.release_source()
        assert checked == [sha, sha]
        assert output.read_text() == initial_output * 2
    else:
        error = {
            "moved": ValueError,
            "deleted": urllib.error.HTTPError,
            "network": urllib.error.URLError,
            "timeout": TimeoutError,
        }[change]
        with pytest.raises(error) as raised:
            contracts.release_source()
        if change == "moved":
            assert "Release tag moved" in str(raised.value)
        assert checked == [sha]
        assert output.read_text() == initial_output
    assert remote_calls.count("git/ref/tags/v1.2.3") == 2
    assert all("refs/tags/" not in arg for args in git_calls for arg in args)


def test_candidate_metadata_is_parsed_without_executing_candidate_code():
    source = 'APP_VERSION = "1.2.3"\nraise AssertionError("candidate code executed")\n'
    assert contracts.version_source(source) == "1.2.3"


def cuda_inventory():
    return [
        {"name": name, "version": installed}
        for name, installed in [
            ("torch", "2.13.0+cu126"),
            ("torchaudio", "2.11.0+cu126"),
            ("torchvision", "0.28.0+cu126"),
            ("whisperx", "3.3.1"),
            ("voscript_core", "0.8.5"),
        ]
    ]


def test_cuda_advisory_aliases_are_exact_and_raw_inventory_is_not_mutated(
    tmp_path, monkeypatch
):
    (tmp_path / "Cargo.lock").write_text("synthetic lock evidence\n")
    monkeypatch.setattr(images, "ROOT", tmp_path)
    raw = cuda_inventory()
    expected, mappings = images.normalize_inventory(raw, "0.8.5")
    assert raw[0]["version"] == "2.13.0+cu126"
    assert expected["torch"] == "2.13.0"
    assert expected["torchaudio"] == "2.11.0"
    assert expected["torchvision"] == "0.28.0"
    assert len(mappings) == 4
    assert "voscript-core" not in expected
    raw[0]["version"] = "2.13.0+unexplained"
    with pytest.raises(ValueError, match="Unexplained local-version"):
        images.normalize_inventory(raw, "0.8.5")


@pytest.mark.parametrize(
    "attack", ["skip", "missing", "wrong-version", "advisory", "nonzero"]
)
def test_image_audit_rejects_unexplained_skips_and_incomplete_results(attack):
    expected = {"torch": "2.13.0", "whisperx": "3.3.1"}
    report = {
        "dependencies": [
            {"name": name, "version": release, "vulns": []}
            for name, release in expected.items()
        ]
    }
    returncode = 0
    if attack == "skip":
        report["dependencies"][0] = {
            "name": "torch",
            "skip_reason": "unknown package version",
        }
    elif attack == "missing":
        report["dependencies"].pop()
    elif attack == "wrong-version":
        report["dependencies"][0]["version"] = "2.6.0"
    elif attack == "advisory":
        report["dependencies"][0]["vulns"] = [{"id": "synthetic-advisory"}]
    else:
        returncode = 1
    with pytest.raises(ValueError):
        images.validate_audit_report(report, expected, returncode)


def test_strict_mapped_image_audit_accepts_complete_clean_report():
    images.validate_audit_report(
        {"dependencies": [{"name": "torch", "version": "2.13.0", "vulns": []}]},
        {"torch": "2.13.0"},
        0,
    )


def test_audit_preserves_raw_cuda_inventory_and_the_actual_service_report(
    tmp_path, monkeypatch
):
    (tmp_path / "Cargo.lock").write_text("synthetic lock\n")
    monkeypatch.setattr(images, "ROOT", tmp_path)
    monkeypatch.setattr(images, "version", lambda root: "0.8.5")
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    raw = cuda_inventory()
    monkeypatch.setattr(images, "command", lambda *args, **kwargs: json.dumps(raw))
    expected, _ = images.normalize_inventory(raw, "0.8.5")

    def fake_audit(args, **kwargs):
        assert "--strict" in args and "--disable-pip" in args
        report = {
            "dependencies": [
                {"name": name, "version": release, "vulns": []}
                for name, release in expected.items()
            ]
        }
        Path(args[args.index("--output") + 1]).write_text(json.dumps(report))
        return SimpleNamespace(returncode=0, stdout="synthetic summary", stderr="")

    monkeypatch.setattr(images.subprocess, "run", fake_audit)
    images.audit("sha256:synthetic-image")
    evidence = tmp_path / "voscript-image-audit"
    assert json.loads((evidence / "raw-inventory.json").read_text()) == raw
    assert "torch==2.13.0\n" in (evidence / "audit-inventory.txt").read_text()
    assert json.loads((evidence / "pip-audit.json").read_text())["dependencies"]
    assert len(json.loads((evidence / "audit-mappings.json").read_text())) == 4
