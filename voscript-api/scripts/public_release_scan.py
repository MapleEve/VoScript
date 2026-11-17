#!/usr/bin/env python3
"""Scan tracked files for public-release privacy leaks."""

from __future__ import annotations

import argparse
import ast
import html
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import unquote

MEDIA_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}
TEXT_EXTENSIONS = {
    ".dockerignore",
    ".env",
    ".gitignore",
    ".ini",
    ".html",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
ALLOW_CONTENT = {"voscript-api/scripts/public_release_scan.py"}
PUBLIC_CONTAINER_PATHS = {
    "/data/voiceprints",
    "/data/transcriptions/asnorm_cohort.npy",
}
# An exact public SDK identifier is not an opaque raw token. Secret assignments
# remain checked independently, including assignments containing this spelling.
PUBLIC_SDK_IDENTIFIERS = {"hf_hub_download"}
REDACTED_EXCERPT = "[source content redacted]"
NUMBERED_ADR_PATTERN = r"ADR[-_]?\d{3,}(?:[-_][\w.-]+)?(?:\.(?:md|rst|html|txt))?"
RAW_VALIDATION_BASENAME_PATTERN = (
    r"(?:validation|validation[-_](?:results?|records?|logs?|reports?|outputs?)"
    r"(?:[-_][\w.-]+)?)\.(?:jsonl?|log|txt|csv)"
)
RAW_VALIDATION_REFERENCE_RE = re.compile(
    rf"(?<![\w.-]){RAW_VALIDATION_BASENAME_PATTERN}(?![\w.-])", re.I
)

# Match concrete artifacts/paths, not generic prose about architecture or context.
INTERNAL_REFERENCE_RE = re.compile(
    r"(?<![\w.-])(?:"
    r"(?:adr[s]?|roadmap|internal|local[-_]docs|validation[-_]notes)[/\\]"
    r"[^\s<>\"'`)]*|"
    r"(?<![/\\])tmp[/\\][^\s<>\"'`)]*|"
    r"/tmp/[^\s<>\"'`)]*\.(?:md|rst|html)(?:#[\w-]+)?|"
    rf"{NUMBERED_ADR_PATTERN}|"
    r"(?:CONTEXT(?:[-_]MAP)?|ROADMAP|ADRS?)\.(?:md|rst|html|txt)|"
    r"[\w.-]+\.local\.(?:md|rst|html)"
    r")(?![\w.-])",
    re.I,
)


def is_internal_document(path: Path) -> bool:
    parts = {part.lower() for part in path.parts[:-1]}
    return bool(
        parts
        & {
            "adr",
            "adrs",
            "roadmap",
            "tmp",
            "internal",
            "local-docs",
            "local_docs",
            "validation-notes",
            "validation_notes",
        }
        or re.fullmatch(
            r"(?:context(?:[-_]map)?|roadmap|adrs?)\.(?:md|rst|html|txt)",
            path.name,
            re.I,
        )
        or re.fullmatch(NUMBERED_ADR_PATTERN, path.name, re.I)
        or re.search(r"\.local\.(?:md|rst|html)$", path.name, re.I)
    )


@dataclass(frozen=True)
class Finding:
    category: str
    path: str
    line: int
    excerpt: str
    advice: str


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]
    advice: str


def _rx(value: str, flags: int = 0) -> re.Pattern[str]:
    return re.compile(value, flags)


SECRET_ASSIGNMENT_RE = _rx(
    r"(?<![A-Za-z0-9_-])(?:export\s+)?['\"]?(?P<name>[A-Za-z_][A-Za-z0-9_-]*)['\"]?"
    r"\s*(?:=|:)\s*(?P<value>\"[^\"\n]*\"|'[^'\n]*'|`[^`\n]*`|[^\s,;]+)?",
    re.I,
)
SECRET_VALUE_PREFIX_RE = _rx(
    r"^(?:sk[-_](?:(?:live|test)[-_])?|hf_|ghp_|github_pat_|xox[baprs]-|AKIA|AIza|eyJ)"
    r"[A-Za-z0-9_+/\-.=]{8,}$"
)
RAW_BEARER_TOKEN_RE = _rx(
    r"(?<![A-Za-z0-9_-])Bearer\s+(?P<value>[A-Za-z0-9_+/\-.=]{12,})(?![A-Za-z0-9_-])",
    re.I,
)
RAW_SECRET_TOKEN_RE = _rx(
    r"(?<![A-Za-z0-9_+/\-.=])"
    r"(?P<value>(?:sk[-_](?:(?:live|test)[-_])?|hf_|ghp_|github_pat_|xox[baprs]-|AKIA|AIza|eyJ)"
    r"[A-Za-z0-9_+/\-.=]{8,})"
    r"(?![A-Za-z0-9_+/\-.=])"
)
PLACEHOLDER_VALUE_RE = _rx(
    r"^(?:<[^>\s]+>|\$\{[A-Za-z_][A-Za-z0-9_]*(?::-[^}]*)?\}|\$[A-Za-z_][A-Za-z0-9_]*|"
    r"(?:your|example|sample|dummy|fake|test)[-_]?[A-Za-z0-9_-]*"
    r"(?:api[-_]?key|key|token|password|secret)[A-Za-z0-9_-]*)$",
    re.I,
)
EXPLICIT_SECRET_NAMES = {
    "api_key",
    "hf_token",
    "password",
    "secret",
    "token",
    "voscript_api_key",
    "voscript_token",
}

PUBLIC_SERVICE_PORTS = {80, 443, 8780}
LOCALHOST_PORT_RE = _rx(r"(?:localhost|127\.0\.0\.1|\[::1\]):(?P<port>\d{2,5})\b", re.I)
DEBUG_PORT_ASSIGNMENT_RE = _rx(
    r"\b(?:SOCKS|PROXY|DEBUG)_PORT\s*(?:=|:)\s*['\"]?(?P<port>\d{2,5})\b", re.I
)
SSH_TOKENS_RE = _rx(r"(?:[^\s`\"'<>]|<[^>\s]+>)+")
SSH_VALUE_OPTIONS = {
    "-p",
    "-i",
    "-o",
    "-F",
    "-J",
    "-l",
    "-b",
    "-c",
    "-D",
    "-L",
    "-R",
    "-S",
    "-W",
    "-E",
    "-m",
    "-w",
}

LINE_RULES = [
    Rule(
        "private validation corpus name",
        _rx(r"\b(?:private E2E (?:corpus|sample)|private (?:corpus|sample))\b", re.I),
        "Use anonymized wording such as internal live validation.",
    ),
    Rule(
        "machine-local path",
        _rx(r"(?:/Users/|/data/)[^\s<>()\[\]\"'`,;]+"),
        "Move machine-specific paths to local-only notes.",
    ),
    Rule(
        "local-only directory reference",
        _rx(r"\broadmap[/\\]|(?<![/\\])\btmp[/\\]"),
        "Replace local-only directory paths with generic ignored archive wording.",
    ),
    Rule(
        "local ssh config path",
        _rx(r"(?:~[/\\])?\.ssh[/\\]config"),
        "Refer to local SSH config generically in public docs.",
    ),
    Rule(
        "real transcription id",
        _rx(r"\btr_[0-9]{8}_[0-9]{6}_[A-Za-z0-9_-]+\b"),
        "Replace real transcription IDs with <tr_id>.",
    ),
    Rule(
        "real speaker id",
        _rx(r"\bspk_[0-9a-f]{6,}\b", re.I),
        "Replace real speaker IDs with <speaker_id>.",
    ),
]


def literal_ssh_host(line: str) -> bool:
    for command in re.finditer(r"\bssh\s+([^;\r\n&|]+)", line):
        tokens = iter(SSH_TOKENS_RE.findall(command.group(1)))
        for token in tokens:
            if token in SSH_VALUE_OPTIONS:
                next(tokens, None)
                continue
            if token.startswith("-"):
                continue
            host = token.rsplit("@", 1)[-1].rstrip(".,;)")
            if re.fullmatch(
                r"(?:<[^>\s]+>|\$\{[A-Za-z_][A-Za-z0-9_]*\}|\$[A-Za-z_][A-Za-z0-9_]*)",
                host,
            ):
                break
            if re.fullmatch(r"[A-Za-z0-9_.:\[\]-]+", host):
                return True
            break
    return False


def operator_rules_from_env() -> list[Rule]:
    raw = os.environ.get("VOSCRIPT_PUBLIC_SCAN_RULES_JSON")
    if raw is None:
        return []
    try:
        config = json.loads(raw)
        if not isinstance(config, dict) or set(config) != {"markers"}:
            raise ValueError
        markers = config["markers"]
        if (
            not isinstance(markers, list)
            or len(markers) > 128
            or any(
                not isinstance(marker, str)
                or not 1 <= len(marker) <= 256
                or not marker.strip()
                or not marker.isprintable()
                for marker in markers
            )
        ):
            raise ValueError
    except (ValueError, TypeError):
        raise RuntimeError(
            "Invalid optional operator rules (details redacted)"
        ) from None
    return [
        Rule(
            "operator private marker",
            _rx(re.escape(marker)),
            "Remove operator-only details from public content.",
        )
        for marker in markers
    ]


def run_git(root: Path, args: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def tracked_files(root: Path, include_untracked: bool = False) -> list[Path]:
    args = ["ls-files", "-z", "--cached"]
    if include_untracked:
        args.extend(["--others", "--exclude-standard"])
    proc = run_git(root, args)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace").strip())
    return [Path(p.decode("utf-8")) for p in proc.stdout.split(b"\0") if p]


def is_text_file(path: Path) -> bool:
    if path.name in {".gitignore", ".dockerignore"}:
        return True
    return path.suffix.lower() in TEXT_EXTENSIONS or not path.suffix


def path_findings(
    paths: Iterable[Path], extra_rules: Sequence[Rule] = ()
) -> list[Finding]:
    findings: list[Finding] = []
    for rel in paths:
        rel_str = rel.as_posix()
        for rule in extra_rules:
            if rule.pattern.search(rel_str):
                findings.append(
                    Finding(rule.name, rel_str, 1, REDACTED_EXCERPT, rule.advice)
                )
        suffix = rel.suffix.lower()
        if is_internal_document(rel):
            findings.append(
                Finding(
                    "tracked local-only directory",
                    rel_str,
                    1,
                    REDACTED_EXCERPT,
                    "Remove tracked planning/tmp artifacts or move them to ignored local storage.",
                )
            )
        if rel.name in {"CLAUDE.local.md", "AGENTS.local.md"} or (
            rel.name.startswith(".env") and rel.name != ".env.example"
        ):
            findings.append(
                Finding(
                    "tracked local config file",
                    rel_str,
                    1,
                    REDACTED_EXCERPT,
                    "Untrack local config files and keep them ignored.",
                )
            )
        if suffix in MEDIA_EXTENSIONS:
            findings.append(
                Finding(
                    "tracked media corpus file",
                    rel_str,
                    1,
                    REDACTED_EXCERPT,
                    "Do not publish raw audio/video validation material.",
                )
            )
        if (
            suffix in {".log", ".json", ".txt"} and "validation" in rel_str.lower()
        ) or re.fullmatch(RAW_VALIDATION_BASENAME_PATTERN, rel.name, re.I):
            findings.append(
                Finding(
                    "tracked validation artifact",
                    rel_str,
                    1,
                    REDACTED_EXCERPT,
                    "Keep raw validation logs/results local-only.",
                )
            )
    return findings


def is_secret_name(name: str) -> bool:
    normalized = name.replace("-", "_").lower()
    return (
        normalized in EXPLICIT_SECRET_NAMES
        or "api_key" in normalized
        or normalized.endswith("_token")
        or normalized == "token"
        or "password" in normalized
        or "secret" in normalized
    )


def assigned_value(raw: str | None) -> str:
    value = (raw or "").strip()
    if not value:
        return ""
    if value[0] in {"'", '"', "`"}:
        quote = value[0]
        end = value.find(quote, 1)
        if end == -1:
            return value[1:].strip()
        return value[1:end].strip()
    return (
        value.split("#", 1)[0].strip().split(maxsplit=1)[0].rstrip("`.,;)]\"'").strip()
    )


def is_placeholder_secret_value(value: str) -> bool:
    normalized = value.strip()
    if normalized in {"", "..."}:
        return True
    if normalized.lower() in {
        "changeme",
        "change-me",
        "change_me",
        "placeholder",
        "redacted",
        "replace-me",
        "replace_me",
        "todo",
    }:
        return True
    return bool(PLACEHOLDER_VALUE_RE.fullmatch(normalized))


def looks_like_secret_value(value: str) -> bool:
    if is_placeholder_secret_value(value):
        return False
    compact = re.sub(r"\s+", "", value.strip())
    if SECRET_VALUE_PREFIX_RE.match(compact):
        return True
    if len(compact) < 12:
        return False
    if len([char for char in compact if char.isalnum()]) < 12:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9_+/\-.=]+", compact))


def is_backtick_wrapped_value(raw: str | None) -> bool:
    return bool(raw and raw.strip().startswith("`"))


def secret_assignment_finding(
    path: str, line_no: int, line: str, symbolic_values: set[str] | None = None
) -> Finding | None:
    for match in SECRET_ASSIGNMENT_RE.finditer(line):
        if not is_secret_name(match.group("name")):
            continue
        raw_value = match.group("value")
        value = assigned_value(raw_value)
        if (
            symbolic_values
            and value in symbolic_values
            and raw_value
            and not raw_value.startswith(("'", '"', "`"))
        ):
            continue
        if not value or is_placeholder_secret_value(value):
            continue
        if not looks_like_secret_value(value) and not is_backtick_wrapped_value(
            raw_value
        ):
            continue
        return Finding(
            "secret-looking assignment",
            path,
            line_no,
            REDACTED_EXCERPT,
            "Use placeholders such as <API_KEY>; rotate if this is a real secret.",
        )
    return None


def python_symbolic_values(source: str) -> dict[int, set[str]]:
    """Distinguish Python variable references from credential string literals."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    values: dict[int, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.keyword, ast.Assign, ast.AnnAssign)) and isinstance(
            node.value, ast.Name
        ):
            values.setdefault(node.lineno, set()).add(node.value.id)
    return values


def raw_secret_token_finding(path: str, line_no: int, line: str) -> Finding | None:
    for pattern in (RAW_BEARER_TOKEN_RE, RAW_SECRET_TOKEN_RE):
        for match in pattern.finditer(line):
            if (
                pattern is RAW_SECRET_TOKEN_RE
                and match.group("value") in PUBLIC_SDK_IDENTIFIERS
            ):
                continue
            if looks_like_secret_value(assigned_value(match.group("value"))):
                return Finding(
                    "secret-looking raw token",
                    path,
                    line_no,
                    REDACTED_EXCERPT,
                    "Use placeholders such as <API_KEY>; rotate if this is a real secret.",
                )
    return None


def diagnostic_path(path: str, extra_rules: Sequence[Rule] = ()) -> str:
    if (
        not path.isprintable()
        or secret_assignment_finding(path, 1, path)
        or raw_secret_token_finding(path, 1, path)
        or any(rule.pattern.search(path) for rule in (*LINE_RULES, *extra_rules))
    ):
        return "[sensitive path redacted]"
    return path


def line_findings(
    root: Path, paths: Iterable[Path], extra_rules: Sequence[Rule] = ()
) -> list[Finding]:
    findings: list[Finding] = []
    for rel in paths:
        rel_str = rel.as_posix()
        if rel.name in {".gitignore", ".npmignore", ".dockerignore"}:
            continue
        if rel_str in ALLOW_CONTENT or not is_text_file(rel):
            continue
        try:
            lines = (
                (root / rel).read_text(encoding="utf-8", errors="replace").splitlines()
            )
        except OSError:
            continue
        symbolic = (
            python_symbolic_values("\n".join(lines)) if rel.suffix == ".py" else {}
        )
        for line_no, line in enumerate(lines, start=1):
            if literal_ssh_host(line):
                findings.append(
                    Finding(
                        "literal SSH host",
                        rel_str,
                        line_no,
                        REDACTED_EXCERPT,
                        "Use an explicit placeholder such as <SSH_ALIAS>.",
                    )
                )
            if any(
                int(match.group("port")) not in PUBLIC_SERVICE_PORTS
                for pattern in (LOCALHOST_PORT_RE, DEBUG_PORT_ASSIGNMENT_RE)
                for match in pattern.finditer(line)
            ):
                findings.append(
                    Finding(
                        "non-public local debug port",
                        rel_str,
                        line_no,
                        REDACTED_EXCERPT,
                        "Use an explicit placeholder such as <SOCKS_PORT>.",
                    )
                )
            decoded = unquote(html.unescape(line))
            if INTERNAL_REFERENCE_RE.search(decoded):
                findings.append(
                    Finding(
                        "internal document reference",
                        rel_str,
                        line_no,
                        REDACTED_EXCERPT,
                        "Remove internal artifact references from public content.",
                    )
                )
            if RAW_VALIDATION_REFERENCE_RE.search(decoded):
                findings.append(
                    Finding(
                        "raw validation artifact reference",
                        rel_str,
                        line_no,
                        REDACTED_EXCERPT,
                        "Remove raw validation record links from public content.",
                    )
                )
            for maybe_finding in (
                secret_assignment_finding(
                    rel_str, line_no, line, symbolic.get(line_no)
                ),
                raw_secret_token_finding(rel_str, line_no, line),
            ):
                if maybe_finding is not None:
                    findings.append(maybe_finding)
            for rule in (*LINE_RULES, *extra_rules):
                matches = rule.pattern.finditer(line)
                if rule.name == "machine-local path":
                    matches = (
                        match
                        for match in matches
                        if match.group() not in PUBLIC_CONTAINER_PATHS
                    )
                if not any(matches):
                    continue
                findings.append(
                    Finding(rule.name, rel_str, line_no, REDACTED_EXCERPT, rule.advice)
                )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan Git-tracked files for VoScript public-release privacy leaks."
    )
    parser.add_argument("--root", default=".", help="Repository root to scan.")
    parser.add_argument(
        "--include-untracked",
        action="store_true",
        help="Also scan non-ignored new files before staging; ignored files stay private.",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print("root does not exist (path redacted)", file=sys.stderr)
        return 2

    try:
        paths = tracked_files(root, include_untracked=args.include_untracked)
    except (RuntimeError, OSError):
        print("not a git worktree or git failed (details redacted)", file=sys.stderr)
        return 2
    try:
        extra_rules = operator_rules_from_env()
    except RuntimeError:
        print("invalid optional operator rules (details redacted)", file=sys.stderr)
        return 2

    findings = path_findings(paths, extra_rules) + line_findings(
        root, paths, extra_rules
    )
    if findings:
        print("Public release scan failed:")
        for item in findings:
            print(
                f"- {diagnostic_path(item.path, extra_rules)}:{item.line}: {item.category}"
            )
            print(f"  {REDACTED_EXCERPT}")
            print(f"  {item.advice}")
        return 1

    surface = "public candidate files" if args.include_untracked else "tracked files"
    print(f"Public release scan passed ({len(paths)} {surface}).")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    sys.exit(main())
