"""Shared fixtures for the HAMLET test suite.

The package is not installable yet (no pyproject.toml), so make the repository root importable
the same way `run.py` does.
"""
import datetime
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def timesteps():
    """Four hourly timesteps — the standard L2 horizon."""
    return pd.date_range("2021-03-24 00:00", periods=4, freq="1h", tz="UTC")


@pytest.fixture
def delta():
    """Timestep length matching the `timesteps` fixture."""
    return datetime.timedelta(hours=1)
