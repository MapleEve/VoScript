# VoScript

## Naming
- Project: **VoScript** — Docker Hub: `mapleeve/voscript` — GitHub: `MapleEve/voscript`
- Integration client: **BetterAINote**
- License: **Custom — free for individuals, written authorization required for commercial use**

## Versioning
- Format: `MAJOR.MINOR.PATCH` — bump patch for fixes/small additions, minor for new features, major for breaking changes
- Version in `app/main.py` and `doc/changelog.*.md` must stay in sync

## Planning Scope
- For `v0.7.2`, keep documentation focused on roadmap structure, compatibility boundaries, and product positioning.
- Current pre-1.0 product narrative remains: self-hosted GPU transcription service + persistent voiceprints + HTTP API.
- Post-1.0 vision belongs in [`roadmap/vision-post-1.0.md`](./roadmap/vision-post-1.0.md), not in current-version scope statements.
- Future architecture reshaping, pipeline canonicalization, provider/plugin layering, and other large structural plans belong to later roadmap stages and must not be written here as `v0.7.2` committed scope.

## Documentation Boundaries
- `README.md` and `README.en.md` may add minimal route guidance so readers can find the roadmap and long-term vision quickly.
- `roadmap/` is the canonical place for version planning, compatibility boundaries, and post-1.0 direction.
- Do not rewrite the current product positioning inside collaboration docs. If future scope needs to be described, link to the relevant roadmap document instead of restating it here as an active commitment.
- Keep pre-1.0 commitments, post-1.0 vision, and compatibility policy clearly separated.

## Docs
- Update zh and en together
- Changelog: `doc/changelog.zh.md` + `doc/changelog.en.md`
- API / behavior docs must match the current implementation in `app/`; do not document fixed
  thresholds or legacy validation semantics after changing runtime behavior

## CI
- Lint: `ruff check app/ --ignore E501`
- Format check: `ruff format --check app/`
- CI test slice: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/unit/ tests/test_security.py -v --tb=short --no-header`
- Full live-server validation is outside CI: use `tests/e2e/` only when a running voscript service is available
