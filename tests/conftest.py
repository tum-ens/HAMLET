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


def pytest_configure(config):
    """Give the suite the same enumerated warning policy the runtime uses.

    `pytest.ini` used to carry `filterwarnings = ignore::DeprecationWarning`, which is the same
    blanket suppression #199 is about, one layer out: it hid every deprecation raised anywhere in
    the suite, HAMLET's own and its dependencies', and it would have hidden a new one just as
    effectively as the old one.

    The entries come from `hamlet.warning_policy.SUPPRESSED` rather than being written out in
    `pytest.ini`, so there is exactly one list. Two copies is how `{'OutputFlag': 0}` ended up
    being sent to HiGHS, and a second copy here would drift the same way -- silently, because a
    filter that matches nothing looks identical to one that matches.

    Registered here rather than as ini text because pytest applies `filterwarnings` per test item
    from `config.getini`, so appending at configure time reaches every test.
    """
    from hamlet.warning_policy import SUPPRESSED

    for category, message, _ in SUPPRESSED:
        # pytest's spelling is `action:message:category:module:lineno`; the message is matched as
        # a prefix regex, exactly as `warnings.filterwarnings` does.
        config.addinivalue_line(
            'filterwarnings',
            f'ignore:{message}:{category.__module__}.{category.__qualname__}')


def pytest_report_header(config):
    """State which solver x framework combinations will run, at the top of every run.

    Skip counts are load-bearing in this project -- !212 used the +5/+3 split across platforms as
    evidence that a fix had landed -- and a Gurobi cell that skips is the normal case rather than
    an anomaly. Printing the matrix in the header means a reader sees which of the four cells ran
    without having to remember `-rs`, and an environment that lost a cell is noticed on the run
    that lost it rather than three merge requests later.

    The probes are cached, so the cost is paid once per session and the matrix tests reuse the
    answers instead of re-probing.
    """
    from tests.backend_matrix import describe

    return describe()


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
