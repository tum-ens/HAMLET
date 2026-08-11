"""Regression — pandapower must be quietened through its own flags, not a stdout hack.

Documents the two findings that decided how the upstream output-suppression change was ported:
the power-flow calls print nothing at all, and the progress bar has a documented flag.
"""
import contextlib
import inspect
import io

import pytest


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


def test_the_14a_horizon_flows_write_nothing_to_stdout_or_stderr():
    """The §14a hot path must not print once per horizon per timestep.

    This used to assert that `run_timeseries` was called with `verbose=False`, since its progress
    bar was the output needing suppression. §14a no longer calls it -- the horizon is a plain
    `pp.runpp` loop now, for cost rather than for output -- so this asserts the *property* instead
    of the mechanism that used to provide it, and survives the next change to how flows are run.
    """
    import numpy as np
    import pandapower as pp
    import pandas as pd
    from pandapower.control import ConstControl
    from pandapower.timeseries import DFData

    from hamlet.executor.utilities.grid_restrictions.enwg_14a import EnWG14a

    net = pp.create_empty_network()
    hv = pp.create_bus(net, vn_kv=20.0, zone='region')
    lv = pp.create_bus(net, vn_kv=0.4, zone='region')
    pp.create_ext_grid(net, bus=hv)
    pp.create_transformer(net, hv_bus=hv, lv_bus=lv, std_type='0.25 MVA 20/0.4 kV')
    bus = pp.create_bus(net, vn_kv=0.4, zone='region')
    pp.create_line(net, from_bus=lv, to_bus=bus, length_km=0.05, std_type='NAYY 4x50 SE')
    index = pp.create_load(net, bus=bus, p_mw=0.0)
    ConstControl(net, element='load', variable='p_mw', element_index=index,
                 data_source=DFData(pd.DataFrame({'p': np.full(4, 0.01)})), profile_name=['p'])

    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        EnWG14a._EnWG14a__run_horizon_power_flows(EnWG14a.__new__(EnWG14a), net, 4)

    assert out.getvalue() == ''
    assert err.getvalue() == ''
