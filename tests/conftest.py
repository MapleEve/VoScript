"""Shared pytest fixtures for voscript test suite."""

import importlib.util
import sys
import os
import types
import wave
import struct
import pytest

# Ensure 'from pipeline import ...' and 'from main import ...' work
_APP_DIR = os.path.join(os.path.dirname(__file__), "..", "app")
_APP_DIR = os.path.abspath(_APP_DIR)
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)


# ---------------------------------------------------------------------------
# Stub out heavy native dependencies that are absent in the test environment.
#
# pipeline package imports torchaudio, torch, and numpy at module level.
# main.py imports fastapi, pipeline, and voiceprints.db at module level.
# We register minimal stubs in sys.modules BEFORE any test module imports
# them so that ModuleNotFoundError never propagates.
# ---------------------------------------------------------------------------


def _make_stub(name: str, **attrs) -> types.ModuleType:
    """Return a stub module with the given attributes."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


class _ArrayStub(list):
    @property
    def shape(self):
        return _shape(self)

    def tolist(self):
        return list(self)

    def item(self):
        flat = _flatten(self)
        if len(flat) != 1:
            raise ValueError("can only convert an array of size 1 to a scalar")
        return flat[0]

    def mean(self):
        flat = [float(item) for item in _flatten(self)]
        if not flat:
            return _ArrayStub()
        return sum(flat) / len(flat)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        return _reshape(_flatten(self), tuple(int(item) for item in shape))

    def __getitem__(self, item):
        if isinstance(item, tuple):
            return _tuple_getitem(self, item)
        return _wrap_getitem(super().__getitem__(item), item)


def _shape(values):
    if isinstance(values, (list, tuple, _ArrayStub)):
        if not values:
            return (0,)
        return (len(values),) + _shape(values[0])
    return ()


def _wrap_getitem(value, item):
    if isinstance(item, slice) and not isinstance(value, _ArrayStub):
        return _ArrayStub(value)
    if isinstance(value, list) and not isinstance(value, _ArrayStub):
        return _ArrayStub(value)
    return value


def _tuple_getitem(values, items):
    if not items:
        return values

    index, *rest = items
    if isinstance(index, slice):
        raw = values[index]
        if rest:
            return _ArrayStub(_tuple_getitem(row, tuple(rest)) for row in raw)
        return _wrap_getitem(raw, index)

    raw = values[index]
    if rest:
        return _tuple_getitem(raw, tuple(rest))
    return _wrap_getitem(raw, index)


def _normalise(value):
    if isinstance(value, _ArrayStub):
        return value
    if isinstance(value, (list, tuple)):
        return _ArrayStub(_normalise(item) for item in value)
    return value


def _flatten(values):
    if isinstance(values, (list, tuple, _ArrayStub)):
        result = []
        for item in values:
            result.extend(_flatten(item))
        return result
    return [values]


def _reshape(flat, shape):
    if -1 in shape:
        known = 1
        unknown_count = 0
        for size in shape:
            if size == -1:
                unknown_count += 1
            else:
                known *= size
        if unknown_count != 1:
            raise ValueError("can only specify one unknown dimension")
        inferred = len(flat) // known
        shape = tuple(inferred if size == -1 else size for size in shape)

    total = 1
    for size in shape:
        total *= size
    if total != len(flat):
        raise ValueError("cannot reshape array")

    def build(offset, dims):
        if not dims:
            return flat[offset], offset + 1
        values = []
        for _ in range(dims[0]):
            value, offset = build(offset, dims[1:])
            values.append(value)
        return _ArrayStub(values), offset

    result, _ = build(0, shape)
    return result


def _make_numpy_stub() -> types.ModuleType:
    def _asarray(value, *args, **kwargs):
        if isinstance(value, _ArrayStub):
            return value
        if isinstance(value, (list, tuple)):
            return _ArrayStub(_normalise(item) for item in value)
        return _ArrayStub([value])

    def _full(shape, fill_value, *args, **kwargs):
        if isinstance(shape, int):
            shape = (shape,)

        def build(dims):
            if not dims:
                return fill_value
            return _ArrayStub(build(dims[1:]) for _ in range(dims[0]))

        return build(tuple(shape))

    def _zeros(shape, *args, **kwargs):
        return _full(shape, 0.0)

    def _ones(shape, *args, **kwargs):
        return _full(shape, 1.0)

    def _concatenate(values, axis=0):
        if axis != 0:
            raise NotImplementedError("numpy stub only supports axis=0")
        result = []
        for value in values:
            result.extend(list(_asarray(value)))
        return _ArrayStub(result)

    def _stack(values, axis=0):
        if axis != 0:
            raise NotImplementedError("numpy stub only supports axis=0")
        return _ArrayStub(_asarray(value) for value in values)

    def _mean(values, axis=None, keepdims=False, **kwargs):
        values = _asarray(values)
        flat = [float(item) for item in _flatten(values)]
        if not flat:
            return _ArrayStub()
        if axis is None:
            return sum(flat) / len(flat)
        if axis == 0:
            rows = list(values)
            if not rows or not isinstance(rows[0], (list, tuple, _ArrayStub)):
                result = sum(float(item) for item in rows) / len(rows)
            else:
                result = _ArrayStub(
                    sum(float(row[index]) for row in rows) / len(rows)
                    for index in range(len(rows[0]))
                )
            return _ArrayStub([result]) if keepdims else result
        if axis == 1:
            result = _ArrayStub(
                sum(float(item) for item in row) / len(row) for row in values
            )
            if keepdims:
                return _ArrayStub(_ArrayStub([item]) for item in result)
            return result
        raise NotImplementedError("numpy stub only supports axis=None, 0, or 1")

    def _squeeze(values, axis=None):
        values = _asarray(values)
        if axis is None:
            while isinstance(values, _ArrayStub) and len(values) == 1:
                values = values[0]
            return values
        if axis == 0 and isinstance(values, _ArrayStub) and len(values) == 1:
            return values[0]
        return values

    def _map(values, fn):
        if isinstance(values, _ArrayStub):
            return _ArrayStub(_map(item, fn) for item in values)
        if isinstance(values, (list, tuple)):
            return _ArrayStub(_map(item, fn) for item in values)
        return fn(values)

    def _power(values, power):
        return _map(values, lambda item: float(item) ** power)

    def _sqrt(values):
        import math

        return _map(values, lambda item: math.sqrt(float(item)))

    def _sort(values):
        return _ArrayStub(sorted(float(item) for item in _flatten(_asarray(values))))

    def _isscalar(value):
        return not isinstance(value, (list, tuple, _ArrayStub))

    def _save(path, value):
        with open(path, "wb") as fh:
            fh.write(repr(value).encode("utf-8"))

    return _make_stub(
        "numpy",
        ndarray=object,
        array=_asarray,
        asarray=_asarray,
        concatenate=_concatenate,
        float32=float,
        full=_full,
        isscalar=_isscalar,
        mean=_mean,
        ones=_ones,
        power=_power,
        save=_save,
        sort=_sort,
        sqrt=_sqrt,
        squeeze=_squeeze,
        stack=_stack,
        zeros=_zeros,
    )


def _ensure_stubs():
    # --- torch ---
    if "torch" not in sys.modules:
        _torch = _make_stub("torch")
        _torch.device = lambda x: x
        _torch.cuda = _make_stub("torch.cuda", is_available=lambda: False)
        sys.modules["torch"] = _torch
        sys.modules["torch.cuda"] = _torch.cuda

    # --- torchaudio ---
    if "torchaudio" not in sys.modules:

        class _FakeInfo:
            """Mimics torchaudio.info() return value."""

            num_frames: int = 80000  # 5 s @ 16 kHz
            sample_rate: int = 16000

        def _fake_info(path):
            # Read real frame count from the WAV header when possible.
            try:
                import wave as _wave

                with _wave.open(path, "r") as wf:
                    info = _FakeInfo()
                    info.num_frames = wf.getnframes()
                    info.sample_rate = wf.getframerate()
                    return info
            except Exception:
                return _FakeInfo()

        _torchaudio = _make_stub("torchaudio", info=_fake_info, load=None)
        _torchaudio.functional = _make_stub("torchaudio.functional")
        sys.modules["torchaudio"] = _torchaudio
        sys.modules["torchaudio.functional"] = _torchaudio.functional

    # --- numpy ---
    if "numpy" not in sys.modules:
        try:
            if importlib.util.find_spec("numpy") is not None:
                import numpy as _np  # noqa: F401
            else:
                _np = _make_numpy_stub()
                sys.modules["numpy"] = _np
        except ImportError:
            _np = _make_numpy_stub()
            sys.modules["numpy"] = _np

    # --- fastapi stubs (only if fastapi itself absent) ---
    # Prefer real fastapi when installed — our new TestClient-based tests
    # (security / voiceprint_db / job_service) depend on the genuine package.
    if "fastapi" not in sys.modules:
        try:
            import fastapi as _real_fa  # noqa: F401

            # Also pull in submodules the app uses so the stub branch below
            # doesn't accidentally shadow them later in the same process.
            import fastapi.responses  # noqa: F401
            import fastapi.middleware.cors  # noqa: F401
            import fastapi.staticfiles  # noqa: F401
            import fastapi.testclient  # noqa: F401
        except Exception:
            _real_fa = None

    if "fastapi" not in sys.modules:
        _fastapi = _make_stub("fastapi")

        class _FastAPI:
            def __init__(self, **kw):
                self.routes = []

            def add_middleware(self, *a, **kw):
                pass

            def mount(self, *a, **kw):
                pass

            def middleware(self, kind):
                def decorator(fn):
                    return fn

                return decorator

            def get(self, path, **kw):
                def decorator(fn):
                    return fn

                return decorator

            def post(self, path, **kw):
                def decorator(fn):
                    return fn

                return decorator

            def put(self, path, **kw):
                def decorator(fn):
                    return fn

                return decorator

            def delete(self, path, **kw):
                def decorator(fn):
                    return fn

                return decorator

        _fastapi.FastAPI = _FastAPI

        class _APIRouter:
            def __init__(self, *a, **kw):
                pass

            def get(self, path, **kw):
                def decorator(fn):
                    return fn

                return decorator

            def post(self, path, **kw):
                def decorator(fn):
                    return fn

                return decorator

            def put(self, path, **kw):
                def decorator(fn):
                    return fn

                return decorator

            def delete(self, path, **kw):
                def decorator(fn):
                    return fn

                return decorator

        class _HTTPException(Exception):
            def __init__(self, status_code=None, detail=None):
                super().__init__(detail)
                self.status_code = status_code
                self.detail = detail

        def _Path(default=None, **kw):
            return default

        _fastapi.APIRouter = _APIRouter
        _fastapi.HTTPException = _HTTPException
        _fastapi.Path = _Path
        _fastapi.Request = object

        class _Form:
            def __class_getitem__(cls, item):
                return item

            def __call__(self, default=None, **kw):
                return default

        class _File(_Form):
            pass

        class _UploadFile:
            pass

        _fastapi.Form = _Form()
        _fastapi.File = _File()
        _fastapi.Header = _Form()
        _fastapi.UploadFile = _UploadFile

        sys.modules["fastapi"] = _fastapi

        # fastapi sub-modules referenced by main.py
        _responses = _make_stub(
            "fastapi.responses",
            FileResponse=object,
            HTMLResponse=object,
            JSONResponse=object,
            PlainTextResponse=object,
        )
        sys.modules["fastapi.responses"] = _responses

        _middleware = _make_stub("fastapi.middleware")
        _cors = _make_stub("fastapi.middleware.cors", CORSMiddleware=object)
        sys.modules["fastapi.middleware"] = _middleware
        sys.modules["fastapi.middleware.cors"] = _cors

        # StaticFiles must be callable with keyword args (directory="static")
        class _StaticFiles:
            def __init__(self, *a, **kw):
                pass

        _static = _make_stub("fastapi.staticfiles", StaticFiles=_StaticFiles)
        sys.modules["fastapi.staticfiles"] = _static

        _testclient = _make_stub("fastapi.testclient")

        class _TestClient:
            """Thin stub — replaced by real TestClient when fastapi is present."""

            def __init__(self, app):
                self.app = app

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def post(self, *a, **kw):
                raise RuntimeError("fastapi not installed")

            def get(self, *a, **kw):
                raise RuntimeError("fastapi not installed")

        _testclient.TestClient = _TestClient
        sys.modules["fastapi.testclient"] = _testclient

    # --- pyannote stubs ---
    for mod_name in [
        "pyannote",
        "pyannote.audio",
        "pyannote.audio.pipelines",
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = _make_stub(mod_name)

    # --- other heavy deps used only inside methods (imported lazily in prod) ---
    for mod_name in ["whisperx", "faster_whisper", "df", "soundfile", "noisereduce"]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = _make_stub(mod_name)

    # --- clearvoice stub ---
    # ClearVoice is a heavy optional dependency (MossFormer2).  Register a
    # stub that creates the expected output files so separation tests can
    # verify the path-collection logic without a real GPU model.
    if "clearvoice" not in sys.modules:
        import pathlib as _pathlib

        class _FakeClearVoice:
            def __init__(self, task=None, model_names=None):
                self._model_names = model_names or []

            def __call__(self, input_path=None, online_write=False, output_path=None):
                # Create the output files that the real ClearVoice would produce.
                stem = _pathlib.Path(input_path).stem
                out_dir = _pathlib.Path(output_path)
                model_tag = (
                    self._model_names[0] if self._model_names else "MossFormer2_SS_16K"
                )
                for i in (1, 2):
                    out_file = out_dir / f"{stem}_{model_tag}_spk{i}.wav"
                    out_file.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

        _cv_mod = _make_stub("clearvoice", ClearVoice=_FakeClearVoice)
        sys.modules["clearvoice"] = _cv_mod

    # voiceprints.db — prefer the real module when sqlite-vec is available,
    # otherwise fall back to the stub so main.py can be imported.
    if "voiceprints.db" not in sys.modules:
        try:
            import sqlite_vec  # noqa: F401

            import voiceprints.db as _real_vdb  # noqa: F401
        except Exception:
            _real_vdb = None

    if "voiceprints.db" not in sys.modules:

        class _VoiceprintDB:
            def __init__(self, path, cohort_path=None):
                pass

            def list_speakers(self):
                return []

            def identify(self, emb, threshold=0.75):
                return (None, None, 0.0)

            def add_speaker(self, name, emb):
                return "stub_id"

            def update_speaker(self, sid, emb, name=None):
                pass

            def delete_speaker(self, sid):
                pass

            def rename_speaker(self, sid, name):
                pass

            def get_speaker(self, sid):
                return None

            def build_cohort_from_transcriptions(self, path, save_path=None):
                return 0

            def maybe_rebuild_cohort(self, path, debounce_s=30):
                pass

        _vdb = _make_stub("voiceprints.db", VoiceprintDB=_VoiceprintDB)
        _voiceprints = sys.modules.get("voiceprints", _make_stub("voiceprints"))
        _voiceprints.__path__ = getattr(_voiceprints, "__path__", [])
        _voiceprints.db = _vdb
        sys.modules["voiceprints"] = _voiceprints
        sys.modules["voiceprints.db"] = _vdb


_ensure_stubs()


# ---------------------------------------------------------------------------
# mock_osd_result
# ---------------------------------------------------------------------------


class _FakeSegment:
    """Minimal stand-in for a pyannote Segment."""

    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end


class _FakeAnnotation:
    """Minimal stand-in for a pyannote Annotation returned by OSD pipeline."""

    def __init__(self, intervals):
        # intervals: list of (start, end) tuples
        self._intervals = intervals

    def itertracks(self, yield_label=False):
        for start, end in self._intervals:
            seg = _FakeSegment(start, end)
            if yield_label:
                yield seg, None, "OVERLAP"
            else:
                yield seg, None


@pytest.fixture
def mock_osd_result():
    """Return a fake pyannote OSD Annotation with two overlapping intervals."""
    return _FakeAnnotation([(1.0, 2.5), (4.0, 5.0)])


# ---------------------------------------------------------------------------
# minimal_wav
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_wav(tmp_path):
    """Create a 5-second 16kHz mono silent WAV file using stdlib wave module.

    Returns the pathlib.Path to the file.
    """
    wav_path = tmp_path / "test_silence.wav"
    sample_rate = 16000
    duration_s = 5
    num_samples = sample_rate * duration_s
    num_channels = 1
    sampwidth = 2  # 16-bit PCM

    with wave.open(str(wav_path), "w") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        # Write all-zero frames (silence)
        silence = struct.pack("<" + "h" * num_samples, *([0] * num_samples))
        wf.writeframes(silence)

    return wav_path


# ---------------------------------------------------------------------------
# app_client
# ---------------------------------------------------------------------------


@pytest.fixture
def app_client(monkeypatch):
    """FastAPI TestClient wrapping app.main.app.

    DATA_DIR is pointed at a temporary directory so the app does not attempt
    to read /data.  The pipeline and voiceprints.db are NOT initialised with
    real models — any test that exercises GPU paths should monkeypatch them.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Set env vars before importing main so directory setup uses tmpdir.
        monkeypatch.setenv("DATA_DIR", tmpdir)

        # app/main.py mounts ./static — chdir to app/ so that path resolves
        # whether pytest is invoked from the repo root or from a sub-dir.
        monkeypatch.chdir(_APP_DIR)

        # Force a fresh import of main + config each time the fixture is used
        # so DATA_DIR is re-evaluated.
        for _m in ("main", "config"):
            if _m in sys.modules:
                del sys.modules[_m]

        # Drop any cached reference to transcriptions/voiceprints routers —
        # they import config at module top-level, so without a fresh import
        # they'd keep pointing at the previous tmpdir.
        for _m in list(sys.modules):
            if _m == "voiceprints" or _m.startswith("voiceprints."):
                del sys.modules[_m]
            elif (
                _m.startswith("api.")
                or _m.startswith("application.")
                or _m.startswith("infra.")
                or _m.startswith("providers.")
                or _m in ("api",)
            ):
                del sys.modules[_m]

        from fastapi.testclient import TestClient
        from main import app

        # Stub pipeline / voiceprint_db on app.state so routes that pull
        # from request.app.state don't blow up when the real model classes
        # are absent. The lifespan initialises these with stubs already, but
        # tests often want to replace them with mocks post-startup.
        with TestClient(app) as client:
            yield client
