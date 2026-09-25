"""Shared fixtures: a fake BRAIN + credd server and clients pointed at it.

The fake starts at import time so WQMCP_BASE_URL / CREDD_URL are set before
brain_client / server read them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_brain import FakeBrain  # noqa: E402

FAKE = FakeBrain()
os.environ["WQMCP_BASE_URL"] = FAKE.url
os.environ["CREDD_URL"] = FAKE.url
os.environ.pop("CREDD_TOKEN", None)


@pytest.fixture
def fake():
    FAKE.reset()
    yield FAKE


@pytest.fixture
def client(fake):
    import brain_client
    c = brain_client.BrainClient(fake.url)
    yield c
    c._executor.shutdown(wait=False)


def pytest_sessionfinish(session, exitstatus):
    FAKE.close()
