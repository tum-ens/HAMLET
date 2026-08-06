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


@pytest.fixture(scope='session')
def repo_root():
    """The repository root, so tests do not encode their own depth in the tree."""
    return REPO_ROOT


@pytest.fixture
def timesteps():
    """Four hourly timesteps — the standard L2 horizon.

    The index must be named `timesteps`: the executor builds its models with
    `linopy.Model(force_dim_names=True)`, which rejects unnamed coordinates, and the SoC
    recursion rolls along that dimension by name (`var_soc.roll(timesteps=1)`).
    """
    return pd.date_range("2021-03-24 00:00", periods=4, freq="1h", tz="UTC",
                         name="timesteps")


@pytest.fixture
def delta():
    """Timestep length matching the `timesteps` fixture."""
    return datetime.timedelta(hours=1)
