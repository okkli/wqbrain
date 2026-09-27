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
# Correlation jobs end with the call that started them, so request counts are
# exact; the background queue has its own tests, which switch it on.
os.environ["WQMCP_CORR_BACKGROUND"] = "0"


@pytest.fixture
def fake():
    FAKE.reset()
    gate = getattr(getattr(sys.modules.get("platform_functions"), "brain_client", None),
                   "correlation_gate", None)
    if gate is not None:
        gate.reset()   # queue, slots and cached correlations are per test
    yield FAKE


def pytest_sessionfinish(session, exitstatus):
    FAKE.close()
