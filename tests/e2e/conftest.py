# tests/e2e/conftest.py
import os
import pytest

# Config from env vars
VOSCRIPT_URL = os.environ.get("VOSCRIPT_URL", "http://localhost:8780")
VOSCRIPT_API_KEY = os.environ.get("VOSCRIPT_API_KEY", "")


@pytest.fixture(scope="session")
def server_url():
    return VOSCRIPT_URL


@pytest.fixture(scope="session")
def api_headers():
    if VOSCRIPT_API_KEY:
        return {"Authorization": f"Bearer {VOSCRIPT_API_KEY}"}
    return {}
