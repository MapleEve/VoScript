"""Regression coverage for exact-source CI and public publication boundaries."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import urllib.error

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "voscript-api" / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contracts = load_script("ci_release_contracts")
scanner = load_script("public_release_scan")


def workflow(name):
    # BaseLoader keeps YAML's 'on' key literal, rather than YAML 1.1's boolean.
    return yaml.load(
        (ROOT / ".github/workflows" / name).read_text(), Loader=yaml.BaseLoader
    )


def test_required_nine_checks_exist_and_execute_real_work():
    assert len(contracts.REQUIRED_CHECKS) == 9
    for name, file in contracts.REQUIRED_CHECKS.items():
        job = workflow(file)["jobs"][name]
        assert job.get("name", name) == name
        assert any("run" in step for step in job["steps"])
    ci = workflow("ci.yml")
    assert "ci_release_contracts.py architecture" in str(
        ci["jobs"]["architecture-gate"]
    )
    assert "ci_release_contracts.py docs" in str(ci["jobs"]["docs-code-drift-gate"])


def test_all_official_actions_are_verified_node24_sha_pins():
    pins = {
        "checkout": "de0fac2e4500dabe0009e67214ff5f5447ce83dd",
        "setup-python": "83679a892e2d95755f2dac6acb0bfd1e9ac5d548",
        "upload-artifact": "b7c566a772e6b6bfb58ed0dc250532a479d7789f",
        "download-artifact": "37930b1c2abaa49bbe596cd826c3c89aef350131",
        "github-script": "ed597411d8f924073f98dfc5c65a23a2325f34cd",
    }
    for path in (ROOT / ".github/workflows").glob("*.yml"):
        for name, sha in re.findall(
            r"uses:\s*actions/([\w-]+)@([^\s]+)", path.read_text()
        ):
            assert pins[name] == sha, path


def test_heavy_first_round_policy_and_manual_target_binding():
    heavy = workflow("rust-foundation-heavy.yml")
    assert heavy["on"]["pull_request"]["types"] == [
        "opened",
        "reopened",
        "ready_for_review",
    ]
    assert heavy["on"]["workflow_dispatch"]["inputs"]["ref"]["required"] == "true"
    assert heavy["permissions"] == {"contents": "read"}
    jobs = heavy["jobs"]
    resolver = jobs["resolve-source"]
    assert "if" not in resolver
    assert "github.workflow_sha" in resolver["steps"][0]["with"]["ref"]
    source = next(step for step in resolver["steps"] if step.get("id") == "source")
    assert source["env"]["REQUESTED_SHA"] == (
        "${{ github.event_name != 'workflow_dispatch' && (github.event.pull_request.head.sha || github.sha) || inputs.ref }}"
    )
    for name in ("resolve-source", "rust-wheel", "docker-packaging"):
        assert "checks" not in jobs[name].get("permissions", {})
    proof_steps = [
        step
        for step in resolver["steps"]
        if step.get("id") == "proof" or "ARTIFACT_ID" in step.get("env", {})
    ]
    assert len(proof_steps) == 2
    for step in proof_steps:
        assert step["if"] == (
            "${{ github.event_name == 'workflow_dispatch' && github.ref == 'refs/heads/main' }}"
        )
    for name in ("rust-wheel", "docker-packaging"):
        assert jobs[name]["steps"][0]["with"]["ref"] == (
            "${{ needs.resolve-source.outputs.source-sha }}"
        )
    reporter = jobs["report-target-checks"]
    assert reporter["permissions"]["checks"] == "write"
    assert reporter["if"] == (
        "${{ always() && github.event_name == 'workflow_dispatch' && github.ref == 'refs/heads/main' && needs.resolve-source.outputs.source-sha != '' }}"
    )
    assert "head_sha: sha" in str(reporter)
    assert "external_id:" in str(reporter)


@pytest.mark.parametrize(
    "overrides,accepted",
    [
        pytest.param({}, True, id="candidate-self"),
        pytest.param({"REQUESTED_SHA": "b" * 40}, False, id="candidate-other-sha"),
        pytest.param({"REQUESTED_SHA": ""}, False, id="missing-target"),
        pytest.param({"REQUESTED_SHA": "a" * 39}, False, id="short-sha"),
        pytest.param({"REQUESTED_SHA": "main"}, False, id="mutable-target"),
        pytest.param({"REQUESTED_SHA": "g" * 40}, False, id="invalid-sha"),
        pytest.param({"GITHUB_WORKFLOW_SHA": "b" * 40}, False, id="workflow-mismatch"),
        pytest.param({"CHECKOUT_SHA": "b" * 40}, False, id="checkout-mismatch"),
        pytest.param(
            {
                "GITHUB_WORKFLOW_REF": "other/project/.github/workflows/rust-foundation-heavy.yml@refs/heads/candidate"
            },
            False,
            id="other-repository",
        ),
        pytest.param(
            {
                "GITHUB_WORKFLOW_REF": "example/project/.github/workflows/other.yml@refs/heads/candidate"
            },
            False,
            id="other-workflow",
        ),
        pytest.param({"GITHUB_REF": "refs/tags/candidate"}, False, id="tag-dispatch"),
        pytest.param({"GITHUB_REF": "refs/heads/main"}, True, id="main-self"),
        pytest.param(
            {"GITHUB_REF": "refs/heads/main", "REQUESTED_SHA": "b" * 40},
            True,
            id="main-other-sha",
        ),
        pytest.param(
            {
                "GITHUB_EVENT_NAME": "pull_request",
                "GITHUB_REF": "refs/pull/1/merge",
                "GITHUB_SHA": "b" * 40,
                "GITHUB_WORKFLOW_SHA": "b" * 40,
            },
            True,
            id="first-pr-head-not-merge-ref",
        ),
        pytest.param(
            {"GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/heads/main"},
            True,
            id="main-push",
        ),
    ],
)
def test_heavy_resolver_executes_exact_source_binding(tmp_path, overrides, accepted):
    step = next(
        step
        for step in workflow("rust-foundation-heavy.yml")["jobs"]["resolve-source"][
            "steps"
        ]
        if step.get("id") == "source"
    )
    env = {
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_REPOSITORY": "example/project",
        "GITHUB_REF": "refs/heads/candidate",
        "GITHUB_SHA": "a" * 40,
        "GITHUB_WORKFLOW_SHA": "a" * 40,
        "CHECKOUT_SHA": "a" * 40,
        "REQUESTED_SHA": "a" * 40,
        "GITHUB_OUTPUT": str(tmp_path / "output"),
        "TRACE": str(tmp_path / "trace"),
        **overrides,
    }
    env.setdefault(
        "GITHUB_WORKFLOW_REF",
        f"{env['GITHUB_REPOSITORY']}/.github/workflows/rust-foundation-heavy.yml@{env['GITHUB_REF']}",
    )
    # Execute the actual workflow shell, stubbing only git transport and proof writing.
    stubs = """
git() {
  case "$*" in
    "rev-parse HEAD") printf '%s\\n' "$CHECKOUT_SHA" ;;
    "cat-file -e $REQUESTED_SHA^{commit}") return 1 ;;
    "fetch --no-tags origin $REQUESTED_SHA") printf 'fetch\\n' >> "$TRACE" ;;
    *) return 2 ;;
  esac
}
python() {
  test "$*" = "voscript-api/scripts/ci_release_contracts.py write-proof" || return 2
  printf 'proof\\n' >> "$TRACE"
}
"""
    result = subprocess.run(
        ["bash", "-c", stubs + step["run"]],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert (result.returncode == 0) is accepted, result.stderr
    output = tmp_path / "output"
    trace = tmp_path / "trace"
    if not accepted:
        assert not output.exists()
        assert not trace.exists()
        return
    assert output.read_text() == f"sha={env['REQUESTED_SHA']}\n"
    if (
        env["GITHUB_EVENT_NAME"] == "workflow_dispatch"
        and env["GITHUB_REF"] == "refs/heads/main"
    ):
        assert trace.read_text() == "fetch\nproof\n"
    else:
        assert not trace.exists()


def test_release_builds_once_then_pushes_the_smoked_image():
    release = workflow("release.yml")
    assert "push" not in release["on"]
    assert "release" not in release["on"]
    assert release["jobs"]["publish"]["needs"] == "validate-source"
    steps = release["jobs"]["publish"]["steps"]
    commands = [step.get("run", "") for step in steps]
    assert sum(command.count("docker build ./app") for command in commands) == 1
    smoke = next(
        i for i, command in enumerate(commands) if "image_release_gate.py" in command
    )
    push = next(i for i, command in enumerate(commands) if "docker push" in command)
    assert smoke < push
    assert 'docker tag "$IMAGE_ID"' in commands[push]
    assert "releases/latest" in commands[push]
    assert "build-push-action" not in str(steps)


def test_no_blanket_python_or_rust_advisory_ignores():
    for file in ("ci.yml", "rust-foundation-heavy.yml", "release.yml"):
        text = (ROOT / ".github/workflows" / file).read_text()
        assert "--ignore-vuln" not in text
        assert "cargo audit --file Cargo.lock --deny warnings" in text
    image = (SCRIPTS / "image_release_gate.py").read_text()
    assert "distributions()" in image
    assert '"whisperx"' in image
    assert "MISSING_CORE" in image
    assert "lifespan_context(app)" in image
    assert '--network", "none' in image


def evidence():
    sha = "a" * 40
    run = {
        "id": 123,
        "path": ".github/workflows/rust-foundation-heavy.yml",
        "event": "workflow_dispatch",
        "head_branch": "main",
        "head_sha": "b" * 40,
        "status": "completed",
        "conclusion": "success",
    }
    check = {
        "name": "rust-wheel",
        "head_sha": sha,
        "app": {"id": 15368},
        "status": "completed",
        "conclusion": "success",
        "external_id": f"heavy:123:{sha}",
    }
    return sha, check, run


def test_manual_evidence_is_bound_to_target_not_workflow_sha():
    sha, check, run = evidence()
    proof = {"target_sha": sha, "job_results": {"rust-wheel": "success"}}
    assert contracts.check_evidence(check, run, sha, "rust-foundation-heavy.yml", proof)
    # User-controlled check metadata alone is never target evidence.
    assert not contracts.check_evidence(check, run, sha, "rust-foundation-heavy.yml")
    proof["target_sha"] = "c" * 40
    assert not contracts.check_evidence(
        check, run, sha, "rust-foundation-heavy.yml", proof
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("head_sha", "c" * 40),
        ("conclusion", "skipped"),
        ("conclusion", "neutral"),
        ("status", "in_progress"),
        ("app", {"id": 7}),
    ],
)
def test_check_evidence_fails_closed(field, value):
    sha, check, run = evidence()
    check[field] = value
    proof = {"target_sha": sha, "job_results": {"rust-wheel": "success"}}
    assert not contracts.check_evidence(
        check, run, sha, "rust-foundation-heavy.yml", proof
    )


def test_dispatch_from_unreviewed_workflow_or_wrong_path_is_rejected():
    sha, check, run = evidence()
    proof = {"target_sha": sha, "job_results": {"rust-wheel": "success"}}
    run["head_branch"] = "candidate"
    run["head_sha"] = sha
    assert not contracts.check_evidence(
        check, run, sha, "rust-foundation-heavy.yml", proof
    )
    run["head_branch"] = "main"
    run["path"] = ".github/workflows/untrusted.yml"
    assert not contracts.check_evidence(
        check, run, sha, "rust-foundation-heavy.yml", proof
    )


def test_candidate_dispatch_cannot_supply_release_proof_or_execution(monkeypatch):
    sha, check, run = evidence()
    run.update(head_branch="candidate", head_sha=sha)
    monkeypatch.setenv("GITHUB_REPOSITORY", "example/project")
    assert not contracts.check_evidence(check, run, sha, "rust-foundation-heavy.yml")
    with pytest.raises(ValueError, match="not from main"):
        contracts.manual_proof(run)
    monkeypatch.setattr(contracts, "paginated", lambda path, key: [run])

    def reject_candidate_lookup(path):
        raise AssertionError(
            f"Candidate must not enter the trusted execution index: {path}"
        )

    monkeypatch.setattr(contracts, "api", reject_candidate_lookup)
    with pytest.raises(ValueError, match="No trusted execution"):
        contracts.latest_execution("rust-foundation-heavy.yml", sha)


def test_latest_failed_run_cannot_borrow_old_green_check(monkeypatch):
    sha = "a" * 40
    checks = [
        {
            "id": 2,
            "name": "lint",
            "head_sha": sha,
            "app": {"id": 15368},
            "status": "completed",
            "conclusion": "failure",
            "details_url": "https://github.com/example/project/actions/runs/2/job/2",
        },
        {
            "id": 1,
            "name": "lint",
            "head_sha": sha,
            "app": {"id": 15368},
            "status": "completed",
            "conclusion": "success",
            "details_url": "https://github.com/example/project/actions/runs/1/job/1",
        },
    ]
    monkeypatch.setenv("GITHUB_REPOSITORY", "example/project")
    monkeypatch.setattr(contracts, "REQUIRED_CHECKS", {"lint": "ci.yml"})

    def fake_api(path):
        if path.startswith("commits/"):
            return {"check_runs": checks}
        if path.startswith("actions/workflows/"):
            return {
                "workflow_runs": [
                    {"id": 1, "event": "push", "head_sha": sha},
                    {"id": 2, "event": "push", "head_sha": sha},
                ]
            }
        if "/attempts/" in path:
            return {
                "jobs": [
                    {
                        "name": "lint",
                        "conclusion": "failure" if "/runs/2/" in path else "success",
                    }
                ]
            }
        run_id = int(path.rsplit("/", 1)[1])
        return {
            "id": run_id,
            "run_attempt": 1,
            "run_started_at": f"2026-11-17T00:00:0{run_id}Z",
            "path": ".github/workflows/ci.yml",
            "head_sha": sha,
            "event": "push",
            "status": "completed",
            "conclusion": "failure" if run_id == 2 else "success",
        }

    monkeypatch.setattr(contracts, "api", fake_api)
    with pytest.raises(ValueError, match="lint"):
        contracts.complete_checks(sha)


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_remote_release_tag_resolves_exact_ref_and_immutable_objects(
    monkeypatch, depth
):
    sha = "a" * 40
    tag_objects = ["b" * 40, "c" * 40][:depth]
    objects = [
        *({"type": "tag", "sha": value} for value in tag_objects),
        {"type": "commit", "sha": sha},
    ]
    responses = {
        "git/ref/tags/v1.2.3": {"ref": "refs/tags/v1.2.3", "object": objects[0]},
        **{
            f"git/tags/{value}": {"sha": value, "object": objects[index + 1]}
            for index, value in enumerate(tag_objects)
        },
    }
    calls = []

    def fake_api(path):
        calls.append(path)
        return responses[path]

    def reject_git(*args):
        raise AssertionError(f"Remote tag resolution must not use local git: {args}")

    monkeypatch.setattr(contracts, "api", fake_api)
    monkeypatch.setattr(contracts, "git", reject_git)
    assert contracts.remote_tag_commit("v1.2.3") == sha
    assert calls == ["git/ref/tags/v1.2.3", *(f"git/tags/{v}" for v in tag_objects)]


@pytest.mark.parametrize(
    "response",
    [
        None,
        [],
        {},
        {"ref": "refs/tags/v1.2.30", "object": {"type": "commit", "sha": "a" * 40}},
        {"ref": "refs/heads/v1.2.3", "object": {"type": "commit", "sha": "a" * 40}},
        {"ref": "refs/tags/v1.2.3"},
        {"ref": "refs/tags/v1.2.3", "object": []},
        {"ref": "refs/tags/v1.2.3", "object": {"type": "commit", "sha": None}},
        {"ref": "refs/tags/v1.2.3", "object": {"type": "commit", "sha": "a" * 39}},
        {"ref": "refs/tags/v1.2.3", "object": {"type": "commit", "sha": "g" * 40}},
        {"ref": "refs/tags/v1.2.3", "object": {"type": "tree", "sha": "a" * 40}},
        {"ref": "refs/tags/v1.2.3", "object": {"type": "blob", "sha": "a" * 40}},
        {"ref": "refs/tags/v1.2.3", "object": {"sha": "a" * 40}},
    ],
)
def test_remote_release_tag_rejects_inexact_or_malformed_refs(monkeypatch, response):
    monkeypatch.setattr(contracts, "api", lambda path: response)
    with pytest.raises(ValueError):
        contracts.remote_tag_commit("v1.2.3")


@pytest.mark.parametrize("mutation", ["missing", "wrong-sha", "cycle", "deep", "blob"])
def test_remote_release_tag_rejects_invalid_annotated_objects(monkeypatch, mutation):
    tag_sha = "b" * 40
    calls = []

    def fake_api(path):
        calls.append(path)
        if path == "git/ref/tags/v1.2.3":
            return {
                "ref": "refs/tags/v1.2.3",
                "object": {"type": "tag", "sha": tag_sha},
            }
        requested = path.removeprefix("git/tags/")
        if mutation == "cycle":
            return {"sha": requested, "object": {"type": "tag", "sha": requested}}
        if mutation == "deep":
            return {
                "sha": requested,
                "object": {"type": "tag", "sha": f"{len(calls):040x}"},
            }
        if mutation == "missing":
            return {"sha": requested}
        return {
            "sha": "c" * 40 if mutation == "wrong-sha" else requested,
            "object": {
                "type": "blob" if mutation == "blob" else "commit",
                "sha": "a" * 40,
            },
        }

    monkeypatch.setattr(contracts, "api", fake_api)
    with pytest.raises(ValueError):
        contracts.remote_tag_commit("v1.2.3")
    assert len(calls) <= 17


@pytest.mark.parametrize("stage", ["ref", "annotated"])
@pytest.mark.parametrize(
    "failure", ["deleted", "forbidden", "rate-limit", "network", "timeout", "json"]
)
def test_remote_release_tag_lookup_errors_never_fall_back(monkeypatch, stage, failure):
    url = "https://api.github.com/repos/example/project/git/ref/tags/v1.2.3"
    error = {
        "deleted": urllib.error.HTTPError(url, 404, "Not Found", None, None),
        "forbidden": urllib.error.HTTPError(url, 403, "Forbidden", None, None),
        "rate-limit": urllib.error.HTTPError(url, 429, "Too Many Requests", None, None),
        "network": urllib.error.URLError("synthetic network failure"),
        "timeout": TimeoutError("synthetic timeout"),
        "json": json.JSONDecodeError("synthetic invalid response", "", 0),
    }[failure]
    calls = []

    def fake_api(path):
        calls.append(path)
        if stage == "annotated" and path == "git/ref/tags/v1.2.3":
            return {
                "ref": "refs/tags/v1.2.3",
                "object": {"type": "tag", "sha": "b" * 40},
            }
        raise error

    def reject_git(*args):
        raise AssertionError(f"Failed remote lookup must not fall back to git: {args}")

    monkeypatch.setattr(contracts, "api", fake_api)
    monkeypatch.setattr(contracts, "git", reject_git)
    with pytest.raises(type(error)) as raised:
        contracts.remote_tag_commit("v1.2.3")
    assert raised.value is error
    assert calls == ["git/ref/tags/v1.2.3"] + (
        [f"git/tags/{'b' * 40}"] if stage == "annotated" else []
    )


@pytest.mark.parametrize(
    "mutation",
    ["accepted", "candidate", "unmerged", "wrong-tag", "moved-main", "weakened-checks"],
)
def test_release_source_rejects_unreviewed_or_mismatched_source(
    tmp_path, monkeypatch, mutation
):
    import json

    sha = "a" * 40
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"inputs": {"tag": "v1.2.3"}}))
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_REPOSITORY", "example/project")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setenv(
        "GITHUB_WORKFLOW_REF",
        "example/project/.github/workflows/release.yml@refs/heads/main",
    )
    monkeypatch.setenv("GITHUB_WORKFLOW_SHA", sha)
    monkeypatch.setenv("GITHUB_SHA", sha)
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    monkeypatch.setattr(
        contracts,
        "git",
        lambda *args: (
            "b" * 40
            if mutation == "wrong-tag" and args[-1].endswith("^{commit}")
            else sha
        ),
    )
    monkeypatch.setattr(contracts.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(contracts, "version", lambda root: "1.2.3")
    monkeypatch.setattr(
        contracts, "git_file", lambda ref, path: b'APP_VERSION = "1.2.3"\n'
    )
    monkeypatch.setattr(contracts, "complete_checks", lambda target: None)

    def fake_api(path):
        if path == "git/ref/tags/v1.2.3":
            return {
                "ref": "refs/tags/v1.2.3",
                "object": {"type": "commit", "sha": sha},
            }
        if path.endswith("/pulls?per_page=100"):
            return (
                []
                if mutation == "unmerged"
                else [
                    {
                        "merged_at": "synthetic",
                        "base": {"ref": "main"},
                        "merge_commit_sha": sha,
                    }
                ]
            )
        if path.startswith("releases/tags/"):
            return {
                "draft": False,
                "prerelease": mutation == "candidate",
                "published_at": "synthetic",
            }
        if path == "branches/main":
            return {
                "protected": True,
                "commit": {"sha": "b" * 40 if mutation == "moved-main" else sha},
                "protection": {
                    "required_status_checks": {
                        "contexts": []
                        if mutation == "weakened-checks"
                        else list(contracts.REQUIRED_CHECKS),
                        "checks": [
                            {"context": name, "app_id": 15368}
                            for name in contracts.REQUIRED_CHECKS
                        ],
                    }
                },
            }
        raise AssertionError(f"Unexpected API call: {path}")

    monkeypatch.setattr(contracts, "api", fake_api)
    if mutation == "accepted":
        contracts.release_source()
        assert f"sha={sha}" in (tmp_path / "output").read_text()
        return
    with pytest.raises(ValueError):
        contracts.release_source()


def test_static_gates_accept_source_and_reject_boundary_violation(tmp_path):
    contracts.architecture(ROOT)
    contracts.docs(ROOT)
    for layer in (
        "api",
        "application",
        "pipeline",
        "providers",
        "voiceprints",
        "infra",
    ):
        (tmp_path / "app" / layer).mkdir(parents=True)
    (tmp_path / "app/infra/unsafe.py").write_text("from application import jobs\n")
    with pytest.raises(ValueError, match="Upward layer import"):
        contracts.architecture(tmp_path)


def scan_fixture(tmp_path, path, content, *, stage=True, include_untracked=False):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    destination = tmp_path / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content)
    if stage:
        subprocess.run(["git", "-C", str(tmp_path), "add", "--", str(path)], check=True)
    command = [
        sys.executable,
        str(SCRIPTS / "public_release_scan.py"),
        "--root",
        str(tmp_path),
    ]
    if include_untracked:
        command.append("--include-untracked")
    return subprocess.run(
        command,
        text=True,
        capture_output=True,
    )


@pytest.mark.parametrize(
    "path",
    [
        Path("doc") / "adr" / "0012-design.md",
        Path("doc") / ("ADR" + "0012-design.md"),
        Path("nested") / ("CONT" + "EXT.md"),
        Path("nested") / ("CONT" + "EXT-MAP.md"),
        Path("nested") / ("AGENTS" + ".local.md"),
        Path("doc") / ("ROAD" + "MAP.md"),
        Path("doc") / ("A" + "DR.md"),
        Path("nested") / "roadmap" / "plan.md",
    ],
)
def test_scanner_rejects_tracked_internal_artifacts(tmp_path, path):
    result = scan_fixture(tmp_path, path, "Synthetic internal notes.")
    assert result.returncode == 1
    assert "tracked local-only directory" in result.stdout


@pytest.mark.parametrize(
    "reference",
    [
        "[details](doc/" + "adr" + "/0012-design.md)",
        "[details](https://github.com/example/project/blob/main/doc/"
        + "ADR"
        + "0012-design.md)",
        "See `" + "CONT" + "EXT.md` for details.",
        "[details](doc/" + "%61" + "%64" + "%72" + "/0012-design.md)",
        "[details](doc/" + "roadmap" + "/plan.md)",
        "Read " + "AGENTS" + ".local.md before deploying.",
    ],
)
def test_scanner_rejects_public_internal_references(tmp_path, reference):
    result = scan_fixture(tmp_path, Path("README.md"), reference)
    assert result.returncode == 1
    assert "internal document reference" in result.stdout


def test_scanner_allows_normal_usage_prose_and_ignore_rules(tmp_path):
    result = scan_fixture(
        tmp_path,
        Path("README.md"),
        "Usage: configure the service and consult API documentation. "
        "Architecture decisions, context and a product roadmap are generic terms.",
    )
    assert result.returncode == 0, result.stdout
    ignored = "CONT" + "EXT.md"
    (tmp_path / ".gitignore").write_text(ignored + "\n")
    (tmp_path / ignored).write_text("Private untracked synthetic notes.")
    subprocess.run(["git", "-C", str(tmp_path), "add", ".gitignore"], check=True)
    paths = scanner.tracked_files(tmp_path)
    assert not scanner.path_findings(paths)
    assert not scanner.line_findings(tmp_path, paths)


def test_public_sdk_identifier_is_not_a_raw_token_but_assignments_still_block(tmp_path):
    result = scan_fixture(
        tmp_path, Path("adapter.py"), "from huggingface_hub import hf_hub_download\n"
    )
    assert result.returncode == 0, result.stdout
    assert (
        scanner.secret_assignment_finding(
            "synthetic.py", 1, "API_KEY=" + "hf_" + "hub_download"
        )
        is not None
    )
    assert (
        scanner.raw_secret_token_finding(
            "synthetic.py", 1, "hf_" + "syntheticnotasecret1234"
        )
        is not None
    )


def test_python_variable_forwarding_is_not_a_credential_literal(tmp_path):
    source = "download(" + "token=" + "use_auth_token)\n"
    result = scan_fixture(tmp_path, Path("adapter.py"), source)
    assert result.returncode == 0, result.stdout
    literal = "download(" + "token=" + repr("use_auth_token") + ")\n"
    result = scan_fixture(tmp_path, Path("adapter.py"), literal)
    assert result.returncode == 1


def test_fast_audio_tests_use_real_cpu_runtime_and_reject_skips():
    job = workflow("ci.yml")["jobs"]["test"]
    assert "https://download.pytorch.org/whl/cpu" in str(job)
    assert "torch==2.13.0 torchaudio==2.11.0" in str(job)
    assert "-p ci_real_audio" in str(job)
    assert (
        "PYTHONPATH=app python -m coverage run --append --source=app tests/compat/test_security_audio_io.py"
        in str(job)
    )
    plugin = (SCRIPTS / "ci_real_audio.py").read_text()
    assert "session.exitstatus = 1" in plugin
    assert "import soundfile" in plugin
    steps = job["steps"]
    runner = next(step for step in steps if step.get("name") == "Run tests")
    assert runner["env"]["RUST_KERNEL_MODE"] == "off"
    assert "RUST_KERNEL_MODE" not in workflow("ci.yml").get("env", {})
    image = (SCRIPTS / "image_release_gate.py").read_text()
    assert "RUST_KERNEL_MODE=required" in image
    assert "tests/test_job_service.py tests/test_voiceprint_db.py" in image
    assert "SPEAKER_01" in image
    assert "build/compat_import_smoke.py" in image
    assert "/ci-compat/test_security_audio_io.py" in image
    assert "target=/ci-compat,readonly" in image


def test_combined_coverage_contains_real_io_and_preserves_thresholds():
    steps = workflow("ci.yml")["jobs"]["test"]["steps"]
    commands = [step.get("run", "") for step in steps]
    erase = next(i for i, command in enumerate(commands) if "coverage erase" in command)
    unit = next(
        i for i, command in enumerate(commands) if "pytest tests/unit/" in command
    )
    audio = next(
        i
        for i, command in enumerate(commands)
        if "coverage run --append --source=app tests/compat/test_security_audio_io.py"
        in command
    )
    xml = next(
        i
        for i, command in enumerate(commands)
        if "coverage xml -o coverage.xml" in command
    )
    upload = next(
        i
        for i, step in enumerate(steps)
        if step.get("name") == "Upload coverage to Codecov"
    )
    assert erase < unit < audio < xml < upload
    assert "--cov-append" in commands[unit]
    assert "--cov-report= " in commands[unit]
    assert "--cov-report=xml" not in commands[unit]
    assert sum("coverage erase" in command for command in commands) == 1
    assert steps[upload]["with"]["files"] == "./coverage.xml"
    policy = yaml.load((ROOT / "codecov.yml").read_text(), Loader=yaml.BaseLoader)
    assert policy["coverage"]["status"]["project"]["default"] == {
        "target": "90%",
        "threshold": "0%",
    }
    assert policy["coverage"]["status"]["patch"]["default"] == {
        "target": "80%",
        "threshold": "0%",
    }


def test_system_temporary_build_path_is_not_a_private_document_reference(tmp_path):
    path = "/" + "tmp" + "/synthetic-wheels"
    result = scan_fixture(tmp_path, Path("Dockerfile"), f"RUN mkdir -p {path}\n")
    assert result.returncode == 0, result.stdout
    reference = path + "/operator-notes.md"
    result = scan_fixture(tmp_path, Path("README.md"), f"[notes]({reference})\n")
    assert result.returncode == 1


@pytest.mark.parametrize("suffix", ["", "-private", "/private.txt", ".private"])
def test_public_container_path_exemption_is_exact_and_per_match(tmp_path, suffix):
    public = "/" + "data/" + "voiceprints"
    private = "/" + "Users/example/private.txt"
    content = f"`{public}{suffix}`"
    if not suffix:
        content += f" `{private}`"
    result = scan_fixture(tmp_path, Path("README.md"), content)
    assert result.returncode == 1
    assert "machine-local path" in result.stdout
    assert private not in result.stdout + result.stderr


def test_exact_public_container_paths_remain_allowed(tmp_path):
    content = "`/" + "data/voiceprints`, `/" + "data/transcriptions/asnorm_cohort.npy`"
    result = scan_fixture(tmp_path, Path("README.md"), content)
    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize(
    "marker, marker_categories",
    [
        ("/" + "Users/example/private.txt", {"machine-local path"}),
        (
            "road" + "map/private.md",
            {"internal document reference", "local-only directory reference"},
        ),
        ("private" + " corpus", {"private validation corpus name"}),
        ("~/." + "ssh/config", {"local ssh" + " config path"}),
    ],
)
def test_every_diagnostic_redacts_the_entire_source_line(
    tmp_path, marker, marker_categories
):
    secret = "sk-live-" + "syntheticcredentialvalue123456789"
    content = "API_KEY=" + secret + " " + marker
    result = scan_fixture(tmp_path, Path("README.md"), content)
    output = result.stdout + result.stderr
    assert result.returncode == 1
    assert secret not in output
    assert marker not in output
    assert content not in output
    assert scanner.REDACTED_EXCERPT in output
    findings = scanner.line_findings(tmp_path, [Path("README.md")])
    required_categories = {"secret-looking assignment"} | marker_categories
    assert required_categories <= {finding.category for finding in findings}
    assert all(category in output for category in required_categories)
    assert all(finding.excerpt == scanner.REDACTED_EXCERPT for finding in findings)
    assert all(
        finding.path == "README.md" and finding.line == 1 for finding in findings
    )
    assert all(
        secret not in repr(finding) and marker not in repr(finding)
        for finding in findings
    )


@pytest.mark.parametrize(
    "filename",
    [
        "AGENTS" + ".local.md",
        "CLAUDE" + ".local.md",
        "docs/release" + ".local.md",
        "CONT" + "EXT.md",
        "ROAD" + "MAP.md",
        "ADR" + "0012-design.md",
    ],
)
@pytest.mark.parametrize("form", ["prose", "inline", "link", "html", "url", "entity"])
def test_internal_document_references_allow_sentence_punctuation(
    tmp_path, filename, form
):
    references = {
        "prose": f"Read {filename}.",
        "inline": f"Read `{filename}`.",
        "link": f"[details]({filename}).",
        "html": f'<a href="{filename}">details</a>.',
        "url": "Read " + filename.replace(".", "%2E") + ".",
        "entity": "Read " + filename.replace(".", "&#46;") + ".",
    }
    content = references[form]
    result = scan_fixture(tmp_path, Path("README.md"), content)
    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert "internal document reference" in output
    assert scanner.REDACTED_EXCERPT in output
    assert filename not in output
    assert content not in output
    findings = scanner.line_findings(tmp_path, [Path("README.md")])
    assert any(
        finding.category == "internal document reference" for finding in findings
    )
    assert all(finding.excerpt == scanner.REDACTED_EXCERPT for finding in findings)


@pytest.mark.parametrize("suffix", [".example", "-template", "x"])
def test_local_override_reference_does_not_match_a_longer_filename(tmp_path, suffix):
    filename = "AGENTS" + ".local.md" + suffix
    result = scan_fixture(tmp_path, Path("README.md"), f"Read {filename}.")
    assert result.returncode == 0, result.stdout + result.stderr


def test_diagnostic_paths_cannot_echo_a_credential_in_a_tracked_filename(tmp_path):
    secret = "hf_" + "syntheticcredentialvalue123456789"
    result = scan_fixture(tmp_path, Path(secret + ".wav"), "synthetic fixture")
    assert result.returncode == 1
    assert "tracked media corpus file" in result.stdout
    assert secret not in result.stdout + result.stderr
    assert "[sensitive path redacted]" in result.stdout


@pytest.mark.parametrize(
    "filename",
    [
        "ADR" + "0012_design.md",
        "ADR" + "0012-design.md",
        "ADR" + "0012.md",
    ],
)
def test_numbered_adr_filenames_and_references_share_the_same_boundary(
    tmp_path, filename
):
    path = Path("doc") / filename
    assert scanner.is_internal_document(path)
    tracked = scan_fixture(tmp_path, path, "Synthetic decision text")
    assert tracked.returncode == 1
    assert "tracked local-only directory" in tracked.stdout
    (tmp_path / "README.md").write_text(f"[decision](doc/{filename})")
    findings = scanner.line_findings(tmp_path, [Path("README.md")])
    assert any(
        finding.category == "internal document reference" for finding in findings
    )


@pytest.mark.parametrize(
    "filename",
    [
        "validation" + "-results.json",
        "validation" + "_records.jsonl",
        "validation" + "-logs.txt",
    ],
)
def test_public_links_to_ignored_raw_validation_records_are_blocked(tmp_path, filename):
    (tmp_path / ".gitignore").write_text("logs/\n")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / filename).write_text("Synthetic ignored record")
    result = scan_fixture(tmp_path, Path("README.md"), f"[records](logs/{filename})")
    assert result.returncode == 1
    assert "raw validation artifact reference" in result.stdout
    assert Path("logs") / filename not in scanner.tracked_files(tmp_path)


def test_validation_usage_schema_is_not_a_raw_validation_record(tmp_path):
    result = scan_fixture(
        tmp_path,
        Path("README.md"),
        "Use validation-schema.json to validate request payloads.",
    )
    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize("artifact", ["adr", "readme"])
def test_include_untracked_scans_genuinely_unstaged_files_without_staging_them(
    tmp_path, artifact
):
    filename = "ADR" + "0012_design.md"
    path = Path("doc") / filename if artifact == "adr" else Path("README.md")
    content = (
        "Synthetic decision"
        if artifact == "adr"
        else "[records](logs/" + "validation" + "-results.json)"
    )
    tracked_only = scan_fixture(tmp_path, path, content, stage=False)
    assert tracked_only.returncode == 0, tracked_only.stdout
    assert scanner.tracked_files(tmp_path) == []
    assert path in scanner.tracked_files(tmp_path, include_untracked=True)
    full = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "public_release_scan.py"),
            "--root",
            str(tmp_path),
            "--include-untracked",
        ],
        text=True,
        capture_output=True,
    )
    assert full.returncode == 1
    assert scanner.tracked_files(tmp_path) == []
    assert (
        subprocess.check_output(
            ["git", "-C", str(tmp_path), "diff", "--cached", "--name-only"]
        )
        == b""
    )
