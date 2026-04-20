"""E2E canary: verify server health and basic transcription API."""

import pytest
import requests

pytestmark = [pytest.mark.e2e]


def test_server_healthy(server_url):
    r = requests.get(f"{server_url}/healthz")
    assert r.status_code == 200
    assert r.json().get("ok") is True


def test_voiceprints_endpoint(server_url, api_headers):
    r = requests.get(f"{server_url}/api/voiceprints", headers=api_headers)
    assert r.status_code == 200
