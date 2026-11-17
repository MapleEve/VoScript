"""Legacy pyannote download argument adapter, not a global Hub monkeypatch."""

from __future__ import annotations


def hf_hub_download(*args, **kwargs):
    from huggingface_hub import hf_hub_download as download

    if "use_auth_token" in kwargs:
        legacy_token = kwargs.pop("use_auth_token")
        if "token" in kwargs and kwargs["token"] != legacy_token:
            raise ValueError("Conflicting token and use_auth_token arguments")
        kwargs.setdefault("token", legacy_token)
    return download(*args, **kwargs)
