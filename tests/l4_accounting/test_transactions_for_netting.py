"""L4 — which transactions count towards a timestep's net energy.

Pins that the market trades cleared *in the current timestep* are part of the netting.
"""
import polars as pl

import hamlet.constants as c
from hamlet.executor.markets.electricity import ElectricityMarket


def frame(rows):
    return pl.DataFrame(
        {
            c.TC_ID_AGENT: [r[0] for r in rows],
            c.TC_ENERGY_IN: [r[1] for r in rows],
            c.TC_ENERGY_OUT: [r[2] for r in rows],
        },
        schema_overrides={c.TC_ENERGY_IN: pl.Int64, c.TC_ENERGY_OUT: pl.Int64},
    )


def market(previous, current):
    """An `ElectricityMarket` carrying only the two transaction tables."""
    m = object.__new__(ElectricityMarket)
    m.transactions_prev = previous
    m.transactions = current
    return m


def test_current_timestep_market_trades_are_included():
    """Regression: only the previous and the balancing transactions were netted.

    The trades cleared on the market in this timestep were dropped, so an agent's net energy
    -- and therefore its grid fees and levies -- was computed as though it had not traded.
    """
    previous = frame([('agent1', 1000, 0)])
    current = frame([('agent1', 5000, 0)])
    balancing = frame([('agent1', 0, 2000)])

    result = market(previous, current)._transactions_for_netting(balancing)

    assert result[c.TC_ENERGY_IN].sum() == 6000
    assert result[c.TC_ENERGY_OUT].sum() == 2000


def test_all_three_sources_are_concatenated():
    """One row per contributing transaction, from all three tables."""
    previous = frame([('agent1', 1000, 0), ('agent2', 500, 0)])
    current = frame([('agent1', 5000, 0)])
    balancing = frame([('agent2', 0, 2000)])

    result = market(previous, current)._transactions_for_netting(balancing)

    assert len(result) == 4


def test_an_empty_previous_table_is_tolerated():
    """The first timestep has nothing settled yet."""
    previous = frame([('agent1', 0, 0)]).clear()  # correct schema, no rows
    current = frame([('agent1', 5000, 0)])
    balancing = frame([('agent1', 0, 1000)])

    result = market(previous, current)._transactions_for_netting(balancing)

    assert result[c.TC_ENERGY_IN].sum() == 5000
