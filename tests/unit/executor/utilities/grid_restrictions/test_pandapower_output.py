"""Regression — pandapower must be quietened through its own flags, not a stdout hack.

Documents the two findings that decided how the upstream output-suppression change was ported:
the power-flow calls print nothing at all, and the progress bar has a documented flag.
"""
import contextlib
import inspect
import io

import pytest


@pytest.mark.solver
def test_power_flow_writes_nothing_to_stdout_or_stderr():
    """The power-flow calls do not print, so they need no output suppression.

    pandapower reports through `logging`, not `print`. The upstream fix wrapped these calls in
    a file-descriptor swap plus a ctypes hack; there is nothing here for it to catch.
    """
    import pandapower as pp
    import pandapower.networks as ppn

    net = ppn.example_simple()
    out, err = io.StringIO(), io.StringIO()

    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        pp.runpp(net)
        pp.rundcpp(net)

    assert out.getvalue() == ''
    assert err.getvalue() == ''


def test_run_timeseries_exposes_a_verbose_flag():
    """The progress bar -- the output that actually needed suppressing -- has its own flag.

    Pins the parameter so a pandapower upgrade that renames or drops it fails here rather
    than silently reintroducing a progress bar on every timestep.
    """
    from pandapower.timeseries.run_time_series import run_timeseries

    parameters = inspect.signature(run_timeseries).parameters

    assert 'verbose' in parameters
    assert parameters['verbose'].default is True  # hence we must pass it explicitly


def test_enwg_14a_passes_verbose_false():
    """The §14a hot path must not print a progress bar once per horizon per timestep."""
    from hamlet.executor.utilities.grid_restrictions import enwg_14a

    source = inspect.getsource(enwg_14a)

    assert 'run_timeseries(grid, range(int(horizon)), verbose=False)' in source
