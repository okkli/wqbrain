"""Shared fixtures: a fake BRAIN + credd server and the MCP module pointed at it.

The fake starts at import time so WQMCP_BASE_URL / CREDD_URL are set before
platform_functions reads them.
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


def pytest_sessionfinish(session, exitstatus):
    FAKE.close()
