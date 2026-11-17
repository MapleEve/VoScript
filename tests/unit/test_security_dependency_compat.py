"""Lightweight adapter/patch safety tests, separate from real-Tensor CI I/O."""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from zipfile import ZipFile

import pytest

from infra.huggingface_compat import hf_hub_download

PATCHER_PATH = (
    Path(__file__).resolve().parents[2] / "app" / "build" / "patch_legacy_audio.py"
)
SPEC = importlib.util.spec_from_file_location("security_legacy_patcher", PATCHER_PATH)
patcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(patcher)


@pytest.mark.parametrize("legacy", [None, False, True, "synthetic-placeholder"])
def test_hub_adapter_translates_only_legacy_token(monkeypatch, legacy):
    calls = []
    hub = ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda *args, **kwargs: (
        calls.append((args, kwargs)) or "cached-file"
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    result = hf_hub_download(
        "synthetic/repo",
        "config.yaml",
        use_auth_token=legacy,
        revision="fixed",
        local_files_only=True,
    )
    assert result == "cached-file"
    assert calls == [
        (
            ("synthetic/repo", "config.yaml"),
            {"token": legacy, "revision": "fixed", "local_files_only": True},
        )
    ]
    assert hub.hf_hub_download is not hf_hub_download


def test_hub_adapter_does_not_silently_drop_conflicting_tokens(monkeypatch):
    hub = ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda *args, **kwargs: pytest.fail(
        "must reject before download"
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    with pytest.raises(ValueError, match="Conflicting token"):
        hf_hub_download(
            "synthetic/repo", "config.yaml", token="one", use_auth_token="two"
        )


def test_speechbrain_factory_maps_revision_token_device_and_offline(monkeypatch):
    from infra.audio.speechbrain_compat import from_hparams

    constants = ModuleType("huggingface_hub.constants")
    constants.HF_HUB_OFFLINE = True
    fetching = ModuleType("speechbrain.utils.fetching")
    fetching.FetchConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants)
    monkeypatch.setitem(sys.modules, "speechbrain.utils.fetching", fetching)
    calls = []
    classifier = SimpleNamespace(
        from_hparams=lambda **kwargs: calls.append(kwargs) or "model"
    )
    options = {"device": SimpleNamespace(__str__=lambda: "cpu")}
    result = from_hparams(
        classifier,
        source="synthetic/repo",
        use_auth_token=False,
        revision="fixed",
        run_opts=options,
    )
    assert result == "model"
    assert calls[0]["fetch_config"].token is False
    assert calls[0]["fetch_config"].revision == "fixed"
    assert calls[0]["fetch_config"].allow_network is False
    assert isinstance(calls[0]["run_opts"]["device"], str)
    assert calls[0]["source"] == "synthetic/repo"
    assert not isinstance(options["device"], str)


def test_patch_rejects_unknown_source_structure_and_invalid_python():
    replacements = (
        ("import torchaudio\n", "from infra.audio import io_compat as torchaudio\n", 1),
    )
    with pytest.raises(ValueError, match="Unknown upstream"):
        patcher.adapt_source(b"import different_audio\n", replacements, "synthetic.py")
    with pytest.raises(SyntaxError):
        patcher.adapt_source(b"not valid python !!!\n", replacements, "synthetic.py")
    for malformed in (b"import torchaudio\r", b"import torchaudio\r\npass\n"):
        with pytest.raises(ValueError, match="encoding/newlines"):
            patcher.adapt_source(malformed, replacements, "synthetic.py")


def _synthetic_install(tmp_path, monkeypatch, original=b"import torchaudio\n"):
    member = "synthetic/io.py"
    wheel_name = "synthetic-0.0.1-py3-none-any.whl"
    wheel_bytes = io.BytesIO()
    with ZipFile(wheel_bytes, "w") as archive:
        archive.writestr(member, original)
    (tmp_path / wheel_name).write_bytes(wheel_bytes.getvalue())
    target = tmp_path / member
    target.parent.mkdir()
    target.write_bytes(original)
    package_spec = {
        "version": "0.0.1",
        "wheels": {wheel_name: patcher.sha256(wheel_bytes.getvalue())},
        "files": {
            member: (
                (
                    "import torchaudio\n",
                    "from infra.audio import io_compat as torchaudio\n",
                    1,
                ),
            )
        },
    }
    monkeypatch.setattr(patcher, "PACKAGES", {"synthetic": package_spec})
    installed = SimpleNamespace(
        version="0.0.1", locate_file=lambda name: tmp_path / name
    )
    monkeypatch.setattr(patcher, "distribution", lambda name: installed)
    monkeypatch.setattr(
        patcher.sysconfig, "get_paths", lambda: {"purelib": str(tmp_path)}
    )
    return target, installed, package_spec


@pytest.mark.parametrize("bom", [b"", b"\xef\xbb\xbf"])
@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_authenticated_patch_preserves_bom_and_uniform_newlines(
    tmp_path, monkeypatch, bom, newline
):
    original = bom + b"import torchaudio" + newline + b"pass" + newline
    target, _, _ = _synthetic_install(tmp_path, monkeypatch, original)
    plan, report = patcher.plan_patches(tmp_path)
    patched = (
        bom
        + b"from infra.audio import io_compat as torchaudio"
        + newline
        + b"pass"
        + newline
    )
    assert plan == [(target, patched)]
    assert target.read_bytes() == original
    assert report[0]["original_sha256"] == patcher.sha256(original)
    assert report[0]["patched_sha256"] == patcher.sha256(patched)
    target.write_bytes(patched)
    assert patcher.plan_patches(tmp_path)[0] == plan

    # Even harmless newline/BOM changes to installed bytes must not bypass the
    # authenticated wheel comparison, before or after patching.
    target.write_bytes(original + newline)
    with pytest.raises(ValueError, match="Installed source differs"):
        patcher.plan_patches(tmp_path)


def test_patch_plan_authenticates_wheel_and_installed_source_before_writes(
    tmp_path, monkeypatch
):
    target, installed, package_spec = _synthetic_install(tmp_path, monkeypatch)
    plan, report = patcher.plan_patches(tmp_path)
    assert target.read_bytes() == b"import torchaudio\n"
    assert plan[0][0] == target
    assert report[0]["original_sha256"] == patcher.sha256(target.read_bytes())
    assert report[0]["patched_sha256"] == patcher.sha256(plan[0][1])

    target.write_bytes(b"import malicious_audio\n")
    with pytest.raises(ValueError, match="Installed source differs"):
        patcher.plan_patches(tmp_path)
    installed.version = "0.0.2"
    with pytest.raises(ValueError, match="Unsupported installed version"):
        patcher.plan_patches(tmp_path)
    wheel = tmp_path / next(iter(package_spec["wheels"]))
    wheel.write_bytes(b"modified artifact")
    with pytest.raises(ValueError, match="Unrecognized wheel SHA256"):
        patcher.plan_patches(tmp_path)


def test_patch_path_allowlist_cannot_escape_site_packages(tmp_path, monkeypatch):
    _, installed, _ = _synthetic_install(tmp_path, monkeypatch)
    installed.locate_file = lambda member: tmp_path.parent / member
    with pytest.raises(ValueError, match="non-site-packages"):
        patcher.plan_patches(tmp_path)


def test_wheel_preflight_rejects_unknown_import_structure(tmp_path, monkeypatch):
    _, _, spec = _synthetic_install(tmp_path, monkeypatch)
    spec["files"]["synthetic/io.py"] = (("import unknown\n", "import adapted\n", 1),)
    wheel = patcher.checked_wheel(tmp_path, "synthetic", spec)
    with ZipFile(wheel) as archive:
        with pytest.raises(ValueError, match="Unknown upstream"):
            list(patcher.checked_sources(archive, spec))


def test_wheel_preflight_requires_exactly_one_allowlisted_source_member():
    wheel_bytes = io.BytesIO()
    with ZipFile(wheel_bytes, "w") as archive:
        archive.writestr("unrelated.py", "pass\n")
    with ZipFile(wheel_bytes) as archive:
        with pytest.raises(ValueError, match="Missing or duplicate"):
            list(patcher.checked_sources(archive, {"files": {"missing.py": ()}}))


def test_patch_manifest_keeps_dsp_imports_and_only_known_versions():
    assert patcher.PACKAGES["pyannote.audio"]["version"] == "3.3.2"
    assert patcher.PACKAGES["speechbrain"]["version"] == "1.1.2"
    assert patcher.PACKAGES["deepfilternet"]["version"] == "0.5.6"
    assert set(patcher.PACKAGES["deepfilternet"]["files"]) == {"df/io.py"}
    for spec in patcher.PACKAGES.values():
        for replacements in spec["files"].values():
            for old, new, count in replacements:
                assert "torchaudio.functional" not in old
                assert "torchaudio.compliance" not in old
                assert "sys.modules" not in new
                assert count > 0
