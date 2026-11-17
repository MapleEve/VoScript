#!/usr/bin/env python3
"""Fail-closed source, workflow, and publication contracts (standard library only)."""

from __future__ import annotations

import argparse
import ast
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tomllib
import urllib.request
from urllib.parse import urlsplit
import zipfile

ROOT = Path(__file__).resolve().parents[2]
REQUIRED_CHECKS = {
    "public-release-scan": "ci.yml",
    "architecture-gate": "ci.yml",
    "docs-code-drift-gate": "ci.yml",
    "lint": "ci.yml",
    "test": "ci.yml",
    "security-scan": "ci.yml",
    "resolve-source": "rust-foundation-heavy.yml",
    "rust-wheel": "rust-foundation-heavy.yml",
    "docker-packaging": "rust-foundation-heavy.yml",
}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def git_file(ref: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=ROOT)


def version(root: Path) -> str:
    return version_source((root / "app/config.py").read_text())


def version_source(source: str) -> str:
    tree = ast.parse(source)
    return next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "APP_VERSION" for t in node.targets)
    )


def architecture(root: Path) -> None:
    # Check the existing layer direction, without importing models or runtime code.
    for directory in (
        "api",
        "application",
        "pipeline",
        "providers",
        "voiceprints",
        "infra",
    ):
        if not (root / "app" / directory).is_dir():
            raise ValueError(f"Missing application layer: {directory}")
    for path in (root / "app").rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        layer = path.relative_to(root / "app").parts[0]
        if layer not in {"infra", "providers", "voiceprints", "pipeline"}:
            continue
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    package = list(path.relative_to(root / "app").parent.parts)
                    package = package[: len(package) - node.level + 1]
                    modules = [".".join([*package, node.module or ""])]
                else:
                    modules = [node.module or ""]
            for module in modules:
                module = module.removeprefix("app.")
                if module.split(".")[0] in {"api", "application", "main"}:
                    raise ValueError(
                        f"Upward layer import: {path.relative_to(root)}:{node.lineno}"
                    )


def docs(root: Path) -> None:
    current = version(root)
    for language in ("en", "zh"):
        text = (root / f"doc/changelog.{language}.md").read_text()
        headings = re.findall(r"^## (\d+\.\d+\.\d+)\b", text, re.M)
        if not headings or headings[0] != current:
            raise ValueError(f"Changelog {language} does not match application version")
        api = (root / f"doc/api.{language}.md").read_text()
        if "/healthz" not in api or "/openapi.json" not in api:
            raise ValueError(
                f"Missing documented health/API discovery endpoints: {language}"
            )
    manifest = tomllib.loads((root / "crates/voscript_core/Cargo.toml").read_text())
    if manifest["package"]["version"] != current:
        raise ValueError("Rust package and application versions disagree")


def api(path: str):
    request = urllib.request.Request(
        f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}/{path}",
        headers={
            "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        if urlsplit(newurl).scheme != "https":
            raise ValueError("Non-HTTPS Actions download redirect")
        redirected = super().redirect_request(request, fp, code, msg, headers, newurl)
        if urlsplit(newurl).netloc != "api.github.com":
            redirected.remove_header("Authorization")
        return redirected


def api_bytes(path: str) -> bytes:
    request = urllib.request.Request(
        f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}/{path}",
        headers={"Authorization": f"Bearer {os.environ['GH_TOKEN']}"},
    )
    with urllib.request.build_opener(SafeRedirect()).open(
        request, timeout=30
    ) as response:
        data = response.read(16 * 1024 * 1024 + 1)
    if len(data) > 16 * 1024 * 1024:
        raise ValueError("Actions evidence exceeds the size limit")
    return data


def paginated(path: str, key: str) -> list[dict]:
    values = []
    page = 1
    while True:
        separator = "&" if "?" in path else "?"
        batch = api(f"{path}{separator}per_page=100&page={page}")[key]
        values.extend(batch)
        if len(batch) < 100:
            return values
        page += 1


def proof_filename(run_id: int, attempt: int) -> str:
    return f"heavy-target-proof-{run_id}-{attempt}"


def write_target_proof() -> None:
    repository = os.environ["GITHUB_REPOSITORY"]
    workflow = (
        f"{repository}/.github/workflows/rust-foundation-heavy.yml@refs/heads/main"
    )
    control = git("rev-parse", "HEAD")
    if (
        os.environ["GITHUB_WORKFLOW_REF"].lower() != workflow.lower()
        or os.environ["GITHUB_WORKFLOW_SHA"] != control
        or os.environ["GITHUB_SHA"] != control
    ):
        raise ValueError("Manual proof must be produced by the immutable main workflow")
    target = os.environ["REQUESTED_SHA"]
    if not re.fullmatch(r"[0-9a-f]{40}", target):
        raise ValueError("Manual target must be an exact SHA")
    if git("rev-parse", "--verify", f"{target}^{{commit}}") != target:
        raise ValueError("Manual target is not a commit")
    proof = {
        "schema": "heavy-target.v1",
        "repository": repository.lower(),
        "run_id": int(os.environ["GITHUB_RUN_ID"]),
        "run_attempt": int(os.environ["GITHUB_RUN_ATTEMPT"]),
        "workflow_ref": workflow.lower(),
        "workflow_sha": control,
        "workflow_file_sha256": hashlib.sha256(
            (ROOT / ".github/workflows/rust-foundation-heavy.yml").read_bytes()
        ).hexdigest(),
        "target_sha": target,
    }
    destination = Path(os.environ["RUNNER_TEMP"]) / "heavy-target-proof"
    destination.mkdir()
    (destination / "proof.json").write_text(json.dumps(proof, sort_keys=True) + "\n")


def manual_proof(run: dict) -> dict:
    repository = os.environ["GITHUB_REPOSITORY"].lower()
    if run.get("event") != "workflow_dispatch" or run.get("head_branch") != "main":
        raise ValueError("Manual proof workflow is not from main")
    run_id, attempt = run["id"], run["run_attempt"]
    workflow_path = ".github/workflows/rust-foundation-heavy.yml"
    if run.get("path") != workflow_path:
        raise ValueError("Unexpected manual workflow path")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", run["head_sha"], "origin/main"],
        cwd=ROOT,
        check=True,
    )
    jobs = paginated(f"actions/runs/{run_id}/attempts/{attempt}/jobs", "jobs")
    resolvers = [job for job in jobs if job["name"] == "resolve-source"]
    if len(resolvers) != 1 or resolvers[0].get("conclusion") != "success":
        raise ValueError("Trusted resolver did not complete successfully")
    # A checks:write token cannot forge the Actions job identity or its runner log.
    log = api_bytes(f"actions/jobs/{resolvers[0]['id']}/logs").decode("utf-8")
    anchors = re.findall(
        r"(?m)^(?:\d{4}-\S+\s+)?HEAVY_PROOF_ANCHOR=(\{[^\r\n]+\})\r?$", log
    )
    if len(anchors) != 1:
        raise ValueError("Trusted resolver proof anchor is missing or ambiguous")
    anchor = json.loads(anchors[0])
    artifact = api(f"actions/artifacts/{anchor['artifact_id']}")
    expected_name = proof_filename(run_id, attempt)
    if (
        artifact.get("id") != anchor["artifact_id"]
        or artifact.get("name") != expected_name
        or artifact.get("expired")
        or artifact.get("workflow_run", {}).get("id") != run_id
        or artifact.get("workflow_run", {}).get("head_sha") != run["head_sha"]
        or artifact.get("digest") != anchor.get("artifact_digest")
    ):
        raise ValueError("Artifact identity differs from the trusted resolver anchor")
    archive = api_bytes(f"actions/artifacts/{artifact['id']}/zip")
    if "sha256:" + hashlib.sha256(archive).hexdigest() != artifact["digest"]:
        raise ValueError("Proof archive digest mismatch")
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        if (
            zipped.namelist() != ["proof.json"]
            or zipped.getinfo("proof.json").file_size > 16384
        ):
            raise ValueError("Unexpected proof archive content")
        proof = json.loads(zipped.read("proof.json"))
    expected = {
        "schema": "heavy-target.v1",
        "repository": repository,
        "run_id": run_id,
        "run_attempt": attempt,
        "workflow_ref": f"{repository}/{workflow_path}@refs/heads/main",
        "workflow_sha": run["head_sha"],
        "workflow_file_sha256": hashlib.sha256(
            git_file(run["head_sha"], workflow_path)
        ).hexdigest(),
    }
    if any(
        proof.get(key) != value for key, value in expected.items()
    ) or not re.fullmatch(r"[0-9a-f]{40}", proof.get("target_sha", "")):
        raise ValueError(
            "Proof does not bind this run, attempt, workflow source and target"
        )
    proof["job_results"] = {job["name"]: job.get("conclusion") for job in jobs}
    return proof


def check_evidence(
    check: dict, run: dict, sha: str, workflow: str, proof: dict | None = None
) -> bool:
    if (
        check.get("head_sha") != sha
        or check.get("app", {}).get("id") != 15368
        or check.get("status") != "completed"
        or check.get("conclusion") != "success"
        or run.get("path") != f".github/workflows/{workflow}"
        or run.get("status") != "completed"
        or run.get("conclusion") != "success"
    ):
        return False
    if run.get("head_sha") == sha and run.get("event") in {"push", "pull_request"}:
        return True
    return (
        workflow == "rust-foundation-heavy.yml"
        and run.get("event") == "workflow_dispatch"
        and run.get("head_branch") == "main"
        and proof is not None
        and proof.get("target_sha") == sha
        and proof.get("job_results", {}).get(check.get("name")) == "success"
    )


def execution_order(run: dict) -> tuple:
    started = run.get("run_started_at") or run["created_at"]
    return (
        datetime.fromisoformat(started.replace("Z", "+00:00")),
        run["id"],
        run["run_attempt"],
    )


def latest_execution(workflow: str, sha: str) -> tuple[dict, dict | None, dict]:
    path = f".github/workflows/{workflow}"
    listed = paginated(
        f"actions/workflows/{workflow}/runs?head_sha={sha}", "workflow_runs"
    )
    run_ids = {
        run["id"]
        for run in listed
        if run.get("event") in {"push", "pull_request"} and run.get("head_sha") == sha
    }
    if workflow == "rust-foundation-heavy.yml":
        manual = paginated(
            f"actions/workflows/{workflow}/runs?event=workflow_dispatch",
            "workflow_runs",
        )
        run_ids.update(run["id"] for run in manual if run.get("head_branch") == "main")
    # Check metadata is not an execution index. Refresh the actual run so an old
    # attempt/check cannot select a superseded attempt or replay an earlier run.
    runs = [api(f"actions/runs/{run_id}") for run_id in run_ids]
    for run in sorted(runs, key=execution_order, reverse=True):
        if run.get("path") != path:
            continue
        proof = None
        if run.get("event") == "workflow_dispatch":
            proof = manual_proof(run)
            if proof["target_sha"] != sha:
                continue
            results = proof["job_results"]
        elif (
            run.get("event") in {"push", "pull_request"} and run.get("head_sha") == sha
        ):
            jobs = paginated(
                f"actions/runs/{run['id']}/attempts/{run['run_attempt']}/jobs", "jobs"
            )
            results = {job["name"]: job.get("conclusion") for job in jobs}
        else:
            continue
        return run, proof, results
    raise ValueError(f"No trusted execution for exact-SHA workflow: {workflow}")


def complete_checks(sha: str) -> None:
    checks = paginated(f"commits/{sha}/check-runs?filter=all", "check_runs")
    repository = os.environ["GITHUB_REPOSITORY"].lower()
    selected = {}
    for name, workflow in REQUIRED_CHECKS.items():
        if workflow not in selected:
            selected[workflow] = latest_execution(workflow, sha)
        run, proof, results = selected[workflow]
        if (
            run.get("status") != "completed"
            or run.get("conclusion") != "success"
            or results.get(name) != "success"
        ):
            raise ValueError(f"Latest trusted exact-SHA execution did not pass: {name}")
        associated = False
        for check in checks:
            if check.get("name") != name:
                continue
            match = re.fullmatch(
                rf"https://github\.com/{re.escape(repository)}/actions/runs/(\d+)(?:/job/\d+)?",
                check.get("details_url") or "",
                re.I,
            )
            if match and int(match.group(1)) == run["id"]:
                if check_evidence(check, run, sha, workflow, proof):
                    associated = True
                    break
        if not associated:
            raise ValueError(
                f"Missing successful check associated with latest exact-SHA execution: {name}"
            )
    print(f"Verified {len(REQUIRED_CHECKS)} exact-SHA checks for {sha}")


def release_source() -> None:
    control = git("rev-parse", "HEAD")
    repository = os.environ["GITHUB_REPOSITORY"].lower()
    expected_workflow = f"{repository}/.github/workflows/release.yml@refs/heads/main"
    if (
        os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
        or os.environ.get("GITHUB_REF") != "refs/heads/main"
        or os.environ.get("GITHUB_WORKFLOW_REF", "").lower() != expected_workflow
        or os.environ.get("GITHUB_WORKFLOW_SHA") != control
        or os.environ.get("GITHUB_SHA") != control
    ):
        raise ValueError("Release validation must execute the immutable main workflow")
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    tag = event.get("inputs", {}).get("tag")
    if not tag or not re.fullmatch(r"v\d+\.\d+\.\d+", tag):
        raise ValueError("A stable published version tag is required")
    sha = git("rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Tag did not resolve to an immutable commit")
    if os.environ.get("EXPECTED_SOURCE_SHA", sha) != sha:
        raise ValueError("Release tag moved after read-only source validation")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", sha, "origin/main"], cwd=ROOT, check=True
    )
    merged = api(f"commits/{sha}/pulls?per_page=100")
    if not any(
        pr.get("merged_at")
        and pr["base"]["ref"] == "main"
        and pr.get("merge_commit_sha") == sha
        for pr in merged
    ):
        raise ValueError("Release source must be an integrated main pull request")
    release = api(f"releases/tags/{tag}")
    if release["draft"] or release["prerelease"] or not release.get("published_at"):
        raise ValueError(
            "Candidate/draft releases cannot be published as stable images"
        )
    # Parse source metadata as data; never import/execute any candidate helper.
    if tag != "v" + version_source(git_file(sha, "app/config.py").decode()):
        raise ValueError("Release tag and application version disagree")
    main = api("branches/main")
    protected = main.get("protection", {}).get("required_status_checks", {})
    bindings = {
        check.get("context"): check.get("app_id")
        for check in protected.get("checks", [])
    }
    if (
        not main.get("protected")
        or set(protected.get("contexts", [])) != set(REQUIRED_CHECKS)
        or bindings != dict.fromkeys(REQUIRED_CHECKS, 15368)
    ):
        raise ValueError(
            "Required-check protection changed; review the release contract"
        )
    if main["commit"]["sha"] != git("rev-parse", "origin/main"):
        raise ValueError(
            "Main changed after checkout; rerun against fresh main ancestry"
        )
    complete_checks(sha)
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write(f"sha={sha}\ntag={tag}\ncontrol-sha={control}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("architecture", "docs", "release", "write-proof")
    )
    args = parser.parse_args()
    if args.mode == "release":
        release_source()
    elif args.mode == "write-proof":
        write_target_proof()
    elif args.mode == "architecture":
        architecture(ROOT)
    else:
        docs(ROOT)
    print(f"{args.mode} contract passed")


if __name__ == "__main__":
    main()
