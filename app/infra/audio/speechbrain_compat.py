"""Adapt pyannote 3.3.2's two SpeechBrain factory calls to SpeechBrain 1.1.2."""

from __future__ import annotations


def from_hparams(
    classifier, *, use_auth_token=None, revision=None, run_opts=None, **kwargs
):
    from huggingface_hub.constants import HF_HUB_OFFLINE
    from speechbrain.utils.fetching import FetchConfig

    options = dict(run_opts or {})
    if "device" in options:
        options["device"] = str(options["device"])
    return classifier.from_hparams(
        fetch_config=FetchConfig(
            token=use_auth_token,
            revision=revision,
            allow_network=not HF_HUB_OFFLINE,
        ),
        run_opts=options,
        **kwargs,
    )
