"""L2 — the balance slack variables and their penalties.

The slacks keep the balance equations solvable so a single infeasible agent cannot abort a run.
They are off by default: with an unbounded market variable the balance is always satisfiable,
and adding variables to an already-feasible problem changes which equally-optimal solution the
solver returns. When enabled they must never be cheaper than serving the load, or the optimiser
would shed demand instead of buying energy.
"""
import pandas as pd
import pytest
from linopy import Model

import hamlet.constants as c

# HAMLET carries prices as integers in units of 0.01 ct/kWh: the shipped retailer data uses
# 1992 for a 19.92 ct/kWh retail price. One unit is therefore 0.1 EUR/MWh.
UNIT_IN_EUR_PER_MWH = 0.1

# 10_000 units is 1 EUR/kWh, i.e. 1000 EUR/MWh -- far above any normal wholesale spike.
EXTREME_PRICE = 10_000


def test_slacks_are_off_by_default():
    """Default-off is what keeps this change from moving anyone's results.

    Measured on the shipped example: enabling the slacks shifts cleared bids and set-points even
    though every problem was already feasible, because the extra variables change which of
    several equally-optimal solutions HiGHS returns.
    """
    assert c.DEFAULT_SLACK_ENABLED is False


def test_fbc_penalty_is_a_value_of_lost_load():
    """The MPC objective is monetary, so its penalty must be a price.

    100_000 units at 0.1 EUR/MWh each is 10,000 EUR/MWh, inside the usual VOLL range.
    """
    assert c.FBC_DEFAULT_SLACK_PENALTY == 100_000

    eur_per_mwh = c.FBC_DEFAULT_SLACK_PENALTY * UNIT_IN_EUR_PER_MWH
    assert eur_per_mwh == pytest.approx(10_000)
    assert 3_000 <= eur_per_mwh <= 30_000


def test_shedding_is_dearer_than_buying_at_any_realistic_price():
    """The decisive property: the optimiser must always prefer to serve the load.

    If the penalty ever fell below the market price the slack would become an arbitrage and
    agents would silently shed demand.
    """
    assert c.FBC_DEFAULT_SLACK_PENALTY > EXTREME_PRICE


def test_rtc_penalty_outranks_every_deviation_weight():
    """The RTC objective is a weighted sum of set-point deviations, not a cost.

    Its penalty is a priority weight, and must exceed the largest component weight (market, 4)
    so that slack is the least preferred way to close the balance.
    """
    largest_deviation_weight = 4

    assert c.RTC_DEFAULT_SLACK_PENALTY > largest_deviation_weight


@pytest.mark.solver
def test_slack_stays_at_zero_when_the_balance_can_be_met():
    """Adding a penalised slack must not change the solution of a feasible problem.

    This is what makes the change behaviour-preserving on scenarios that already converged.
    """
    timesteps = pd.date_range('2021-03-24', periods=3, freq='1h', tz='UTC', name='timesteps')
    model = Model(force_dim_names=True)

    load = pd.Series([1000.0, 2000.0, 1500.0], index=timesteps)
    market = model.add_variables(name='market', lower=-1e9, upper=1e9, coords=[timesteps])
    slack = model.add_variables(name='slack', lower=0, coords=[timesteps])

    model.add_constraints(market + slack == load, name='balance')
    model.add_objective(1 * market.sum() + c.FBC_DEFAULT_SLACK_PENALTY * slack.sum(),
                        overwrite=True)
    model.solve(solver_name='highs', output_flag=False, log_to_console=False)

    assert model.status == 'ok'
    assert slack.solution.values.max() == pytest.approx(0)


@pytest.mark.solver
def test_slack_absorbs_an_otherwise_infeasible_balance():
    """When the market cannot cover the load the run continues instead of aborting."""
    timesteps = pd.date_range('2021-03-24', periods=3, freq='1h', tz='UTC', name='timesteps')
    model = Model(force_dim_names=True)

    load = pd.Series([1000.0, 5000.0, 1500.0], index=timesteps)
    # Market bounded below the load in the middle timestep
    market = model.add_variables(name='market', lower=-2000, upper=2000, coords=[timesteps])
    slack = model.add_variables(name='slack', lower=0, coords=[timesteps])

    model.add_constraints(market + slack == load, name='balance')
    model.add_objective(1 * market.sum() + c.FBC_DEFAULT_SLACK_PENALTY * slack.sum(),
                        overwrite=True)
    model.solve(solver_name='highs', output_flag=False, log_to_console=False)

    assert model.status == 'ok'
    assert slack.solution.values.max() == pytest.approx(3000)
