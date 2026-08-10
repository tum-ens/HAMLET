"""Shared fixtures for the HAMLET test suite.

`hamlet` is imported as an installed package (`uv sync` installs the project in editable mode),
so nothing here touches `sys.path`. If an import fails, the environment is not set up -- see the
Installation section of `README.md` -- rather than the test being run from the wrong directory.

**The `hamlet` import below `pandas` stays first, and is no longer the only thing holding the
line.** This is the rootdir conftest, so it is the first thing pytest imports, and importing
`pandas` before HAMLET used to hand the process `pyarrow`'s private, too-old `MSVCP140.dll` --
which `framework: poi` then refuses to solve against (#202). A `.pth` startup hook now claims that
name before pytest itself starts, so this line is belt-and-braces rather than load-bearing; it is
kept because it costs nothing and still covers an environment where the hook cannot run. See
`hamlet/msvc_runtime.py` and `packaging/hamlet_msvcp140_hook.py`. Do not "simplify" it away on the
grounds that the hook makes it redundant -- and note that any test wanting to provoke the bad
ordering must set `HAMLET_NO_MSVCP140_HOOK=1`, or it will silently assert nothing.
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
