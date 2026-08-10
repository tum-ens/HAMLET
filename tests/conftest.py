"""Shared fixtures for the HAMLET test suite.

`hamlet` is imported as an installed package (`uv sync` installs the project in editable mode),
so nothing here touches `sys.path`. If an import fails, the environment is not set up -- see the
Installation section of `README.md` -- rather than the test being run from the wrong directory.

**The `hamlet` import below `pandas` is load-bearing on Windows and must stay first.** This is the
rootdir conftest, so it is the first thing pytest imports, and importing `pandas` before HAMLET
hands the process `pyarrow`'s private, too-old `MSVCP140.dll` -- which `framework: poi` then
refuses to solve against (#202). The suite is simply obeying the same ordering rule HAMLET asks
users to obey; see `hamlet/msvc_runtime.py`. Without it the POI tests fail with a `RuntimeError`
naming the DLL, which is a real failure and not one to work around by skipping them.
"""
import hamlet  # noqa: F401  -- must precede pandas on Windows; see the module docstring

import datetime
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


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
