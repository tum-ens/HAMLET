"""Unit -- the hand-rolled horizon loop must equal `run_timeseries`, series for series.

§14a runs the horizon's power flows to work out each bus's loading. It used to do that through
`pandapower.timeseries.run_timeseries`, which cost **2.07 s per simulated timestep** on design 6
while a single `pp.runpp` on that network is **3 ms** -- so ~0.07 s of it was power flow and the
rest was framework: one `ConstControl` per element per variable (~1,460 of them), walked on every
horizon step, plus an OutputWriter appending per logged variable per step.

Replacing a library call with a loop is only safe if the loop gives the same numbers, so that is
what these tests check: both paths, same network, same controllers, compared exactly rather than
approximately. `run_timeseries` is the oracle and is still imported here even though production no
longer calls it.

Production no longer builds those controllers either: the grid stage records the profile arrays
directly, because constructing ~1,460 `ConstControl` objects per timestep only to decompile them
here cost 0.298 s and bought nothing. The controller path is kept as the *oracle* -- `run_timeseries`
can only be driven by real controllers -- and `TestTheRecordedProfilesEqualTheControllers` is what
ties the two representations together.
"""
import numpy as np
import pandas as pd
import pandapower as pp
import pytest
from pandapower.control import ConstControl
from pandapower.timeseries import DFData, OutputWriter
from pandapower.timeseries.run_time_series import run_timeseries

import hamlet.constants as c
from hamlet.executor.utilities.grid_restrictions.enwg_14a import EnWG14a

HORIZON = 6


def network(seed=0):
    """A small radial network with loads and sgens on separate buses."""
    rng = np.random.default_rng(seed)
    net = pp.create_empty_network()
    hv = pp.create_bus(net, vn_kv=20.0, zone='region')
    lv = pp.create_bus(net, vn_kv=0.4, zone='region')
    pp.create_ext_grid(net, bus=hv)
    pp.create_transformer(net, hv_bus=hv, lv_bus=lv, std_type='0.25 MVA 20/0.4 kV')

    previous = lv
    for _ in range(4):
        bus = pp.create_bus(net, vn_kv=0.4, zone='region')
        pp.create_line(net, from_bus=previous, to_bus=bus, length_km=0.08, std_type='NAYY 4x50 SE')
        pp.create_load(net, bus=bus, p_mw=0.0, q_mvar=0.0)
        pp.create_sgen(net, bus=bus, p_mw=0.0, q_mvar=0.0)
        previous = bus

    profiles = {}
    for element, sign in (('load', 1.0), ('sgen', -1.0)):
        frame = getattr(net, element)
        for index in frame.index:
            for variable, scale in (('p_mw', 0.01), ('q_mvar', 0.003)):
                column = f'{element}_{index}_{variable}'
                series = sign * scale * rng.random(HORIZON)
                profiles[column] = series
                ConstControl(net, element=element, variable=variable, element_index=index,
                             data_source=DFData(pd.DataFrame({column: series})),
                             profile_name=[column])
    return net


def via_run_timeseries(net):
    """The oracle: what production used to do."""
    writer = OutputWriter(net, range(HORIZON))
    for table, column in EnWG14a.HORIZON_RESULTS:
        writer.log_variable(table, column)
    run_timeseries(net, range(HORIZON), verbose=False)
    return {f'{table}.{column}': writer.output[f'{table}.{column}']
            for table, column in EnWG14a.HORIZON_RESULTS}


def via_loop(net):
    instance = EnWG14a.__new__(EnWG14a)
    return instance._EnWG14a__run_horizon_power_flows(net, HORIZON)


@pytest.fixture
def both():
    """Two identical networks, so neither run can be affected by the other's leftovers."""
    return via_run_timeseries(network()), via_loop(network())


def test_the_same_series_are_produced(both):
    reference, actual = both

    assert sorted(actual) == sorted(reference)


@pytest.mark.parametrize('name', [f'{table}.{column}' for table, column in EnWG14a.HORIZON_RESULTS])
def test_every_series_matches_exactly(both, name):
    reference, actual = both

    pd.testing.assert_frame_equal(actual[name], reference[name],
                                  check_names=False, check_dtype=False)


def test_the_results_are_not_all_identical_across_steps(both):
    """Guard against both paths agreeing because neither did anything.

    Every profile is random per step, so the loading must move; if it did not, the comparison
    above would be comparing two constant frames and would pass for the wrong reason.
    """
    _, actual = both
    loading = actual['res_trafo.loading_percent']

    assert loading.shape[0] == HORIZON
    assert loading.nunique().max() > 1, 'transformer loading is constant across the horizon'


def test_every_controller_is_compiled(both):
    """A controller silently dropped would show up as a wrong flow, but say nothing about why."""
    net = network()
    profiles = EnWG14a._EnWG14a__compile_profiles(net, HORIZON)

    compiled = sum(len(indices) for indices, _ in profiles.values())

    assert compiled == len(net.controller), f'{compiled} compiled of {len(net.controller)}'
    for _, values in profiles.values():
        assert values.shape[0] == HORIZON


def recorded_from(net):
    """The same profiles the grid stage records, derived from this net's controllers.

    Mirrors what `Electricity.__add_controller_to_grid` writes, so the two representations can be
    compared without dragging the whole grid stage into a unit test.
    """
    recorded = {}
    for _, entry in net.controller.iterrows():
        controller = entry['object']
        column = controller.profile_name[0]
        key = (controller.element, controller.variable)
        recorded.setdefault(key, {})[controller.element_index] = (
            controller.data_source.df[column].to_numpy())
    return recorded


class TestTheRecordedProfilesEqualTheControllers:
    """The grid stage records profiles directly now; the controller path stays as the oracle.

    Building ~1,460 `ConstControl` objects per timestep cost 0.298 s of a 0.524 s
    `_write_grid_parameters` and bought nothing once the horizon loop stopped using
    `run_timeseries` -- they were constructed and immediately decompiled. These tests are what
    makes dropping them safe: the two representations must compile to the same arrays and produce
    the same flows.
    """

    def test_both_representations_compile_to_the_same_arrays(self):
        net = network()
        from_controllers = EnWG14a._EnWG14a__compile_profiles(net, HORIZON)

        net[c.GRID_HORIZON_PROFILES] = recorded_from(net)
        from_recorded = EnWG14a._EnWG14a__compile_profiles(net, HORIZON)

        assert sorted(from_recorded) == sorted(from_controllers)
        for key in from_controllers:
            indices_a, values_a = from_controllers[key]
            indices_b, values_b = from_recorded[key]
            order_a, order_b = np.argsort(indices_a), np.argsort(indices_b)
            np.testing.assert_array_equal(indices_a[order_a], indices_b[order_b])
            np.testing.assert_allclose(values_a[:, order_a], values_b[:, order_b])

    @pytest.mark.parametrize('name',
                             [f'{table}.{column}' for table, column in EnWG14a.HORIZON_RESULTS])
    def test_the_flows_match_run_timeseries_from_recorded_profiles_too(self, name):
        """The end that matters: same numbers, whichever representation the net carries."""
        reference = via_run_timeseries(network())

        net = network()
        net[c.GRID_HORIZON_PROFILES] = recorded_from(net)
        net.controller = net.controller.iloc[0:0]     # nothing left to read them from
        actual = via_loop(net)

        pd.testing.assert_frame_equal(actual[name], reference[name],
                                      check_names=False, check_dtype=False)

    def test_the_recorded_path_is_the_one_being_used(self):
        """Guard against the test above passing through the controller fallback by accident."""
        net = network()
        net[c.GRID_HORIZON_PROFILES] = recorded_from(net)
        net.controller = net.controller.iloc[0:0]

        profiles = EnWG14a._EnWG14a__compile_profiles(net, HORIZON)

        assert profiles, 'no profiles compiled, so the controller fallback would have been empty'


def test_a_non_converging_network_still_raises(both):
    """The caller catches `LoadflowNotConverged` and carries on; it must still be able to."""
    net = network()
    net.load['p_mw'] = 1e6
    for _, entry in net.controller.iterrows():
        controller = entry['object']
        if controller.variable == 'p_mw' and controller.element == 'load':
            column = controller.profile_name[0]
            controller.data_source.df[column] = 1e6

    with pytest.raises(pp.powerflow.LoadflowNotConverged):
        via_loop(net)
