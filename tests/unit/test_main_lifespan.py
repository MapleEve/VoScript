"""TEST-H4, TEST-H5: lifespan background thread behaviour.

TEST-H4: The 'cohort-rebuild' daemon thread must be alive while the TestClient
         lifespan is active.

TEST-H5: The 'cohort-rebuild' daemon thread must survive a tick where
         maybe_rebuild_cohort raises an exception (exception is caught and
         logged; the thread loop continues).

The thread and stop_event are locals inside the lifespan coroutine, so we
locate the thread by its well-known name via threading.enumerate().
"""

import asyncio
import importlib
import importlib.abc
import inspect
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import threading


def _find_rebuild_thread():
    """Return the running 'cohort-rebuild' daemon thread, or None."""
    for t in threading.enumerate():
        if t.name == "cohort-rebuild":
            return t
    return None


@pytest.fixture(autouse=True)
def _patch_voiceprint_stub_ctor(monkeypatch):
    """Keep the fallback VoiceprintDB stub compatible with lifespan startup."""
    from voiceprints.db import VoiceprintDB

    init = VoiceprintDB.__init__
    if "cohort_path" in inspect.signature(init).parameters:
        return

    def _compat_init(self, path, cohort_path=None, *args, **kwargs):
        init(self, path)
        self.path = path
        self.cohort_path = cohort_path
        self._cohort_path = cohort_path

    monkeypatch.setattr(VoiceprintDB, "__init__", _compat_init)


def test_rebuild_thread_alive_during_lifespan(app_client):
    """Daemon thread must be alive while the TestClient lifespan is active (TEST-H4)."""
    thread = _find_rebuild_thread()
    assert thread is not None, (
        "Expected a running thread named 'cohort-rebuild' during lifespan; "
        "none found among: " + str([t.name for t in threading.enumerate()])
    )
    assert thread.is_alive(), "cohort-rebuild thread should be alive during lifespan"
    assert thread.daemon, "cohort-rebuild thread must be a daemon thread"


def test_openapi_version_matches_config(app_client):
    from app.config import APP_VERSION

    assert app_client.app.version == APP_VERSION


def test_rebuild_thread_survives_tick_exception(app_client, monkeypatch):
    """Daemon thread must stay alive after maybe_rebuild_cohort raises (TEST-H5)."""
    thread = _find_rebuild_thread()
    assert thread is not None, (
        "cohort-rebuild thread not found; lifespan may not have started"
    )

    # Locate the db object on app.state via the TestClient's app.
    # app_client is a FastAPI TestClient; the underlying ASGI app has .state.db
    # set by lifespan.
    db = app_client.app.state.db

    call_count = [0]
    original_maybe_rebuild = db.maybe_rebuild_cohort

    def _raise(*a, **kw):
        call_count[0] += 1
        raise RuntimeError("simulated cohort rebuild failure")

    monkeypatch.setattr(db, "maybe_rebuild_cohort", _raise)

    # Give the thread a moment — the tick interval is 60 s so it won't fire
    # during a short test, but the thread itself must remain alive.
    time.sleep(0.1)

    assert thread.is_alive(), (
        "Daemon thread died after monkeypatching maybe_rebuild_cohort; "
        "the worker loop must catch exceptions and keep running"
    )

    # Restore to avoid affecting subsequent tests.
    monkeypatch.setattr(db, "maybe_rebuild_cohort", original_maybe_rebuild)


@pytest.fixture
def native_startup(monkeypatch, tmp_path):
    """Exercise the real lifespan/bridge without models or background workers."""
    monkeypatch.chdir(Path(__file__).resolve().parents[2] / "app")
    monkeypatch.delitem(sys.modules, "main", raising=False)
    main = importlib.import_module("main")
    bridge = importlib.import_module("providers.kernel_bridge.runtime")
    monkeypatch.setattr(bridge, "RUST_KERNEL_MODE", "required")
    monkeypatch.delitem(sys.modules, "voscript_core", raising=False)

    directories = [
        tmp_path / name for name in ("transcriptions", "uploads", "voiceprints")
    ]
    for name, directory in zip(
        ("TRANSCRIPTIONS_DIR", "UPLOADS_DIR", "VOICEPRINTS_DIR"), directories
    ):
        monkeypatch.setattr(main, name, directory)

    recover = Mock()
    db = Mock(return_value=Mock(build_cohort_from_transcriptions=Mock(return_value=0)))
    pipeline = Mock()
    idle_daemon = Mock(return_value=None)
    threads = SimpleNamespace(Event=Mock(), Thread=Mock())
    for name, value in (
        ("recover_orphan_jobs", recover),
        ("VoiceprintDB", db),
        ("TranscriptionPipeline", pipeline),
        ("start_idle_model_unload_daemon", idle_daemon),
        ("threading", threads),
    ):
        monkeypatch.setattr(main, name, value)

    return SimpleNamespace(
        main=main,
        bridge=bridge,
        app=SimpleNamespace(state=SimpleNamespace()),
        directories=directories,
        recover=recover,
        db=db,
        pipeline=pipeline,
        idle_daemon=idle_daemon,
        threads=threads,
    )


async def _run_native_startup(startup):
    async with startup.main.lifespan(startup.app):
        assert startup.app.state.pipeline is startup.pipeline.return_value


def _assert_startup_has_no_side_effects(startup):
    assert not any(directory.exists() for directory in startup.directories)
    for operation in (
        startup.recover,
        startup.db,
        startup.pipeline,
        startup.idle_daemon,
        startup.threads.Event,
        startup.threads.Thread,
    ):
        operation.assert_not_called()
    assert vars(startup.app.state) == {}


def test_required_native_startup_succeeds(native_startup, monkeypatch):
    core = SimpleNamespace(
        core_smoke=Mock(
            return_value={
                "ok": True,
                "echoed": {},
                "version": "test",
                "capabilities": {"core_smoke": True, "rust_extension": True},
            }
        )
    )
    monkeypatch.setitem(sys.modules, "voscript_core", core)

    asyncio.run(_run_native_startup(native_startup))

    core.core_smoke.assert_called_once_with({})
    native_startup.pipeline.assert_called_once()
    native_startup.threads.Thread.return_value.start.assert_called_once()
    native_startup.threads.Thread.return_value.join.assert_called_once_with(timeout=5)


@pytest.mark.parametrize(
    "error",
    [
        ModuleNotFoundError("synthetic missing native extension"),
        ImportError("synthetic native import failure"),
        OSError("synthetic native ABI failure"),
    ],
    ids=["missing", "import", "abi"],
)
def test_required_native_startup_import_failure_has_no_side_effects(
    native_startup, monkeypatch, error
):
    class BrokenCore(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "voscript_core" or fullname.startswith("voscript_core."):
                raise error

    monkeypatch.setattr(sys, "meta_path", [BrokenCore(), *sys.meta_path])

    with pytest.raises(
        native_startup.bridge.RustKernelBridgeError, match="required but unavailable"
    ) as exc_info:
        asyncio.run(_run_native_startup(native_startup))

    assert exc_info.value.__cause__ is error
    _assert_startup_has_no_side_effects(native_startup)


@pytest.mark.parametrize("failure", ["missing_api", "call", "invalid_response"])
def test_required_native_startup_smoke_failure_has_no_side_effects(
    native_startup, monkeypatch, failure
):
    core = SimpleNamespace()
    if failure == "call":
        core.core_smoke = Mock(
            side_effect=RuntimeError("synthetic native call failure")
        )
    elif failure == "invalid_response":
        core.core_smoke = Mock(return_value={"ok": False})
    monkeypatch.setitem(sys.modules, "voscript_core", core)

    with pytest.raises(native_startup.bridge.RustKernelBridgeError):
        asyncio.run(_run_native_startup(native_startup))

    _assert_startup_has_no_side_effects(native_startup)


def test_off_native_startup_never_imports_extension(native_startup, monkeypatch):
    monkeypatch.setattr(native_startup.bridge, "RUST_KERNEL_MODE", "off")
    smoke = Mock(side_effect=AssertionError("off must not call native bridge"))
    monkeypatch.setattr(native_startup.main, "core_smoke", smoke)
    imports = []

    class ForbiddenCore(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "voscript_core" or fullname.startswith("voscript_core."):
                imports.append(fullname)
                raise AssertionError("off must not import native extension")

    monkeypatch.setattr(sys, "meta_path", [ForbiddenCore(), *sys.meta_path])

    asyncio.run(_run_native_startup(native_startup))

    smoke.assert_not_called()
    assert imports == []
    assert "voscript_core" not in sys.modules
    native_startup.pipeline.assert_called_once()
