"""L4 — accounting invariants for grid fees and levies.

Pins the rule that fees and levies are owed on **net** energy: what the agent actually drew
from or fed into the grid over a timestep, not the sum of its individual trades.
"""
import polars as pl
import pytest

import hamlet.constants as c
from hamlet.executor.markets.electricity import ElectricityMarket


def frame(rows):
    """Build a minimal transactions frame with signed energy columns."""
    return pl.DataFrame(
        {
            c.TC_ID_AGENT: [r[0] for r in rows],
            c.TC_ENERGY_IN: [r[1] for r in rows],
            c.TC_ENERGY_OUT: [r[2] for r in rows],
        },
        schema_overrides={c.TC_ENERGY_IN: pl.Int64, c.TC_ENERGY_OUT: pl.Int64},
    )


def net(rows):
    out = ElectricityMarket._to_net_energy(frame(rows))
    return list(zip(out[c.TC_ENERGY_IN].to_list(), out[c.TC_ENERGY_OUT].to_list()))


def test_net_consumer_is_charged_only_on_the_difference():
    """Regression: fees and levies were charged on gross energy, overcharging both sides.

    An agent that bought 5 kWh and sold 3 kWh in one timestep is a net consumer of 2 kWh.
    Charging it on 5 kWh of import *and* 3 kWh of export bills it for 8 kWh it never moved.
    """
    assert net([('agent1', 5000, 3000)]) == [(2000, 0)]


def test_net_producer_is_charged_only_on_the_difference():
    """The mirror case: a net exporter is credited on the net, not the gross."""
    assert net([('agent1', 3000, 5000)]) == [(0, 2000)]


def test_a_balanced_agent_pays_nothing():
    """Buying and selling the same amount is no net exchange with the grid at all."""
    assert net([('agent1', 4000, 4000)]) == [(0, 0)]


def test_only_one_direction_is_ever_non_zero():
    """Invariant: an agent cannot be a net importer and a net exporter simultaneously."""
    rows = [('a', 5000, 3000), ('b', 3000, 5000), ('c', 4000, 4000), ('d', 0, 7000)]

    for energy_in, energy_out in net(rows):
        assert energy_in == 0 or energy_out == 0


def test_pure_flows_are_left_untouched():
    """An agent that only imported, or only exported, is unaffected by the netting."""
    assert net([('a', 6000, 0), ('b', 0, 6000)]) == [(6000, 0), (0, 6000)]


@pytest.mark.parametrize('energy_in, energy_out', [(5000, 3000), (3000, 5000), (0, 0)])
def test_net_flow_is_preserved(energy_in, energy_out):
    """Netting must not change the signed flow, only how it is expressed."""
    (net_in, net_out), = net([('agent1', energy_in, energy_out)])

    assert net_in - net_out == energy_in - energy_out
