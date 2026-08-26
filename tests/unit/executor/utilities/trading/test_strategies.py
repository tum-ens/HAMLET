"""Unit -- the three trading strategies, priced by hand.

**Two of the three had never executed.** Measured for #223 over the whole suite, `Linear` and
`Zi` reached only their `class` and `def` lines: 15 and 7 statements respectively were dead,
because no scenario any test ran cleared a market, and `agent_base.create_bids_offers`
substitutes `Retailer` when there is no clearing method. This file drives all three directly.

**What each strategy is.** All three take the same two forecast prices -- what the agent expects
to be paid for feed-in (`energy_price_in`) and to pay for supply (`energy_price_out`) -- and turn
a net position into a bid or an offer price:

* `Retailer` quotes those two prices crossed, so the agent is indifferent between trading locally
  and trading with the retailer. It is the default when there is no clearing.
* `Linear` walks a ladder between them, indexed by a `multiplication_factor` running from 0 at
  the timestep **nearest delivery** to `len_table` at the far end of the horizon. So the nearest
  step quotes the *most aggressive* pair -- willing to pay the full supply price, willing to sell
  at the feed-in tariff -- and the far end quotes that pair reversed, bidding what it would sell
  at. It does not converge on the midpoint; it passes through it and keeps going.
* `Zi` -- zero intelligence -- draws each price uniformly from between the two.

**Prices are in EURe-7/Wh**, energies in Wh, and setpoints in W. The fixture's timestep is
15 minutes, so the last two differ by a factor of four; see `tests/trading_fixtures.py` for why
that matters and what else about the fixture is load-bearing.

**`Zi` is not reproducible in a real run, and this file is where that is written down.** It calls
`random.random()` on the module-global generator, which HAMLET never seeds --
`test_zi_is_reproducible_only_when_the_global_generator_is_seeded` demonstrates both halves. It
is the same shape as the market half of #216, which `ElectricityMarket._clearing_seed` fixed for
the clearing shuffle; the strategies were not touched then and no scenario runs `zi`, so it is
latent. The tests below therefore pin `Zi`'s *formula* -- exactly, via a degenerate forecast
where the draw cannot matter -- and its *range*, rather than a recorded sequence of draws.
"""
import random

import polars as pl
import pytest

import hamlet.constants as c
from hamlet.executor.utilities.trading.trading import Trading
from tests.trading_fixtures import (BOOK_COLUMNS, FORECAST_IN, FORECAST_OUT, NAME, STEPS,
                                    STEP_MINUTES, Agent, market_data, timetable)

#: The agent's positions as the strategies quote them, in Wh: `SETPOINTS` scaled by the 15-minute
#: timestep. Literals, so every expected table below states the energy rather than deriving it
#: from the conversion under test.
BID_0, OFFER_1, BID_2 = 250, 125, 500


def quotes(strategy, agent=None, market=None, **kwargs):
    """Run one strategy and return `{step: (agent_id, energy_in, energy_out, pu_in, pu_out)}`.

    The agent id is part of the tuple deliberately: `_preprocess_bids_offers` stamps it onto every
    row, and a fixture with one agent whose id nothing asserts lets that stamp be replaced by a
    constant -- which in a real run would attribute every agent's energy and money to one account.
    """
    agent = agent if agent is not None else Agent('agent_1', strategy, **kwargs)
    instance = Trading(strategy=strategy, timetable=timetable(), market=NAME,
                       market_data=market if market is not None else market_data(),
                       agent=agent).create_instance()

    result = instance.create_bids_offers()

    assert result is agent, 'the strategy must hand the agent back'
    index = {step: position for position, step in enumerate(STEPS)}
    return {index[row[c.TC_TIMESTEP]]: (row[c.TC_ID_AGENT], row[c.TC_ENERGY_IN],
                                        row[c.TC_ENERGY_OUT], row[c.TC_PRICE_PU_IN],
                                        row[c.TC_PRICE_PU_OUT])
            for row in agent.bids_offers.to_dicts()}


def prices(result):
    """Just the two price columns, per step, for the tests that are only about prices."""
    return {step: (row[3], row[4]) for step, row in result.items()}


def quantities(result):
    """`[(energy_in, energy_out), ...]` in step order."""
    return [(row[1], row[2]) for _, row in sorted(result.items())]


def test_the_registry_maps_every_shipped_strategy_name_to_a_distinct_class():
    """`agents.yaml` documents three names. All three resolve, and to three different classes."""
    classes = {name: Trading(strategy=name, timetable=timetable(), market=NAME,
                             market_data=market_data(),
                             agent=Agent('agent_1', name)).strategy
               for name in (c.MTS_ZI, c.MTS_LINEAR, c.MTS_RETAILER)}

    assert len(set(classes.values())) == 3
    assert [cls.__name__ for cls in classes.values()] == ['Zi', 'Linear', 'Retailer']


def test_an_unknown_strategy_is_refused_by_name():
    with pytest.raises(ValueError, match='vibes'):
        Trading(strategy='vibes', timetable=timetable(), market=NAME, market_data=market_data(),
                agent=Agent('agent_1', 'vibes'))


@pytest.mark.parametrize('strategy', [c.MTS_ZI, c.MTS_LINEAR, c.MTS_RETAILER])
def test_the_quoted_columns_are_the_ones_the_market_reads(strategy):
    """A real strategy's output, checked against the column list the market fixture is built on.

    This is the joint between the two halves of L3.
    `tests/unit/executor/markets/test_clearing_by_hand.py` hand-writes books from `BOOK_COLUMNS`;
    nothing checked that a *strategy* produces exactly those columns, so an internal column
    leaking out of `_postprocess_bids_offers` -- `time_to_trade`, a `Duration`, is one line away
    -- would reach `__create_bids_offers`, break `retailer.select(self.bids_offers.columns)`, and
    be invisible here. Order is asserted as well as membership, since the market selects
    positionally in places.
    """
    random.seed(3)
    agent = Agent('agent_1', strategy)
    Trading(strategy=strategy, timetable=timetable(), market=NAME, market_data=market_data(),
            agent=agent).create_instance().create_bids_offers()

    assert agent.bids_offers.columns == BOOK_COLUMNS
    assert dict(agent.bids_offers.schema) == {col: c.SCHEMA[col] for col in BOOK_COLUMNS}


# ----------------------------------------------------------------------------------------------
# Retailer.
#
# The agent quotes the two forecast prices crossed: it bids what it expects to pay (1500) and
# offers at what it expects to be paid (500). Every timestep gets the same pair, and the timestep
# with no net position is dropped.
# ----------------------------------------------------------------------------------------------
def test_retailer_quotes_the_retailers_own_prices_on_both_sides():
    result = quotes(c.MTS_RETAILER)

    assert result == {
        0: ('agent_1', BID_0, 0, FORECAST_OUT, FORECAST_IN),
        1: ('agent_1', 0, OFFER_1, FORECAST_OUT, FORECAST_IN),
        2: ('agent_1', BID_2, 0, FORECAST_OUT, FORECAST_IN),
    }


def test_retailer_does_not_cross_the_two_prices():
    """A regression shape, not a tautology: bidding 500 and offering 1500 would be free money.

    Asserted separately from the exact table above because the two prices being swapped is the
    one mutation of `Retailer` that keeps every row, every quantity and every timestep intact.
    """
    result = quotes(c.MTS_RETAILER)

    assert len(result) == 3, 'nothing was quoted, so the loop below asserts nothing'
    for step, (price_in, price_out) in prices(result).items():
        assert price_in > price_out, step


# ----------------------------------------------------------------------------------------------
# Linear.
#
# Derivation, with four timesteps in the horizon, `steps_to_final = 1` and `steps_from_init = 0`:
#
#     mul_factor = [0] * 1  ++  linspace(0, 4, 4 - 1 - 0)  ++  [4] * 0
#                = [0]      ++  [0, 2, 4]                  ++  []
#                = [0, 0, 2, 4]
#
#     price_pu_in  = 1500 + (500 - 1500) / 4 * factor  =  1500 - 250 * factor
#     price_pu_out = (500 - 1500) / 4 * (4 - factor) + 1500  =  500 + 250 * factor
#
#     step    factor   bid    offer
#       0        0     1500     500      <- nearest delivery, most aggressive
#       1        0     1500     500      <- still held there by steps_to_final = 1
#       2        2     1000    1000      <- passes through the midpoint
#       3        4      500    1500      <- crossed: bids below its own offer. No net position
#                                           here, so `_postprocess` drops it and no test sees it.
# ----------------------------------------------------------------------------------------------
def test_linear_walks_the_bid_down_and_the_offer_up_across_the_horizon():
    result = quotes(c.MTS_LINEAR)

    assert result == {
        0: ('agent_1', BID_0, 0, 1500, 500),
        1: ('agent_1', 0, OFFER_1, 1500, 500),
        2: ('agent_1', BID_2, 0, 1000, 1000),
    }


def test_linear_and_retailer_agree_while_the_ladder_is_held_at_its_initial_value():
    """The two strategies coincide wherever `mul_factor` is 0 -- here at **steps 0 and 1**.

    `steps_to_final = 1` prepends one extra zero to the factor sequence, so the agreement covers
    two steps and not just the nearest one. Both are asserted: a test claiming "only the nearest
    step" would be wrong about this fixture, and step 1 is exactly the step that falsifies it.

    This is why an end-to-end test cannot tell the two strategies apart from prices alone --
    `tests/e2e/test_market_clearing.py` reads a run-time receipt instead, and this test is why.
    """
    linear, retailer = prices(quotes(c.MTS_LINEAR)), prices(quotes(c.MTS_RETAILER))

    assert linear[0] == retailer[0]
    assert linear[1] == retailer[1]
    assert linear[2] != retailer[2]


def test_linears_prices_stay_within_the_retail_band():
    """Neither side ever quotes outside [feed-in tariff, supply price]."""
    result = quotes(c.MTS_LINEAR)

    assert len(result) == 3, 'nothing was quoted, so the loop below asserts nothing'
    for step, (price_in, price_out) in prices(result).items():
        assert FORECAST_IN <= price_in <= FORECAST_OUT, step
        assert FORECAST_IN <= price_out <= FORECAST_OUT, step


def test_steps_to_final_holds_the_aggressive_price_for_longer():
    """With `steps_to_final = 2` the factors become `[0, 0] ++ linspace(0, 4, 2) ++ []`
    = `[0, 0, 0, 4]`, so step 2 -- which quoted the midpoint above -- quotes the aggressive end.
    """
    result = quotes(c.MTS_LINEAR, agent=Agent('agent_1', c.MTS_LINEAR, steps_to_final=2))

    assert prices(result) == {0: (1500, 500), 1: (1500, 500), 2: (1500, 500)}


def test_steps_from_init_reaches_the_far_end_of_the_ladder_sooner():
    """The other shape parameter, which every other test here leaves at its shipped 0.

    With `steps_from_init = 1` the factors become `[0] ++ linspace(0, 4, 2) ++ [4]`
    = `[0, 0, 4, 4]`, so step 2 jumps past the midpoint to the crossed end: bid 500, offer 1500.
    Without this case `steps_initial` can be replaced by the literal 0 and nothing notices.
    """
    result = quotes(c.MTS_LINEAR, agent=Agent('agent_1', c.MTS_LINEAR, steps_from_init=1))

    assert prices(result) == {0: (1500, 500), 1: (1500, 500), 2: (500, 1500)}


def test_a_shorter_horizon_shortens_the_ladder_rather_than_truncating_it():
    """A three-step horizon makes `len_table` 3, which changes every price on the ladder.

    `Linear` filters on `within_horizon` itself before measuring `len_table`, and *then*
    `_postprocess_bids_offers` filters again. Deleting the first filter leaves the second one
    producing the same three rows -- but priced off a four-rung ladder, so step 2 would quote
    1000 instead of 500. The row count is identical either way, which is why this is asserted by
    price.
    """
    result = quotes(c.MTS_LINEAR,
                    agent=Agent('agent_1', c.MTS_LINEAR, horizon=3 * STEP_MINUTES * 60))

    assert set(result) == {0, 1, 2}
    assert prices(result) == {0: (1500, 500), 1: (1500, 500), 2: (500, 1500)}


# ----------------------------------------------------------------------------------------------
# Zi.
#
#     price = (forecast_in - forecast_out) * r + forecast_out,  r ~ U[0, 1)
#
# drawn independently for each side and each timestep. With `forecast_in == forecast_out` the
# draw cancels, which is what makes the formula checkable by value without pinning a sequence of
# random numbers.
# ----------------------------------------------------------------------------------------------
FLAT_PRICE = 900


def test_zi_reduces_to_the_forecast_when_there_is_no_spread():
    """The one Zi case with an exact answer: both prices are 900, whatever the draw.

    This pins the *formula* -- both endpoints, the multiplication and the addition -- without
    depending on the generator. Run under two seeds so a pass cannot be an accident of one.
    """
    for seed in (0, 12345):
        random.seed(seed)
        result = quotes(c.MTS_ZI, agent=Agent('agent_1', c.MTS_ZI, forecast_in=FLAT_PRICE,
                                              forecast_out=FLAT_PRICE))

        assert result == {
            0: ('agent_1', BID_0, 0, FLAT_PRICE, FLAT_PRICE),
            1: ('agent_1', 0, OFFER_1, FLAT_PRICE, FLAT_PRICE),
            2: ('agent_1', BID_2, 0, FLAT_PRICE, FLAT_PRICE),
        }


def test_zi_draws_both_prices_from_inside_the_retail_band():
    """Every quote lies in [500, 1500], and **each side separately** lands strictly inside it.

    Per side, not in total: the two prices come from two separate `pl.Series` comprehensions, and
    a mutation to one leaves the other producing interior values. Counting both together lets the
    surviving side vouch for the broken one -- measured, replacing the bid side's draw with a
    constant kept a combined count green.

    `len(drawn[column]) >= 5` is the half that catches a draw taken *once* and reused for every
    row; the interior count is the half that catches a draw ignored altogether.
    """
    inside = {c.TC_PRICE_PU_IN: 0, c.TC_PRICE_PU_OUT: 0}
    drawn = {c.TC_PRICE_PU_IN: set(), c.TC_PRICE_PU_OUT: set()}
    for seed in (1, 2, 3, 4):
        random.seed(seed)
        result = quotes(c.MTS_ZI)

        assert len(result) == 3, seed
        for step, (price_in, price_out) in prices(result).items():
            for column, price in ((c.TC_PRICE_PU_IN, price_in), (c.TC_PRICE_PU_OUT, price_out)):
                assert FORECAST_IN <= price <= FORECAST_OUT, (seed, step, column)
                inside[column] += FORECAST_IN < price < FORECAST_OUT
                drawn[column].add(price)

    for column in inside:
        assert inside[column] >= 8, f'{column} never landed strictly inside the band'
        assert len(drawn[column]) >= 5, f'{column} is not varying with the draw'


def test_zi_is_reproducible_only_when_the_global_generator_is_seeded():
    """Both halves of the finding, in one test.

    `Zi` calls `random.random()` on the module-global generator. Seed it and two runs agree;
    advance it between two runs and they do not. **HAMLET seeds it nowhere** -- verified across
    `hamlet/`, where the only seeding is the `seed=` argument `ElectricityMarket._clearing_seed`
    passes to `DataFrame.sample` -- so in a real simulation the second case is the one that
    applies and a `zi` scenario is not reproducible across processes. Same shape as #216. No
    shipped scenario selects `zi`, so this is latent.
    """
    random.seed(7)
    first = quotes(c.MTS_ZI)
    random.seed(7)
    again = quotes(c.MTS_ZI)

    assert first == again

    random.seed(7)
    random.random()  # advance the generator the way an unrelated caller would
    shifted = quotes(c.MTS_ZI)

    assert shifted != first


# ----------------------------------------------------------------------------------------------
# Shared behaviour of `TradingBase`, which all three inherit.
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize('strategy', [c.MTS_ZI, c.MTS_LINEAR, c.MTS_RETAILER])
def test_a_timestep_with_no_net_position_produces_no_quote(strategy):
    """Step 3's setpoint is zero, so no bid and no offer are submitted for it."""
    random.seed(11)

    assert set(quotes(strategy)) == {0, 1, 2}


@pytest.mark.parametrize('strategy', [c.MTS_ZI, c.MTS_LINEAR, c.MTS_RETAILER])
def test_a_purchase_becomes_a_bid_and_a_sale_becomes_an_offer(strategy):
    """The sign convention and the power-to-energy conversion, per strategy.

    `+1000 W` over a 15-minute step is a **250 Wh** bid and `-500 W` a **125 Wh** offer. The
    factor of four is the point: at an hourly step `dt_hours` is 1.0 and deleting the conversion
    changes nothing, while the shipped `timestep: 900` makes the same deletion a fourfold
    over-bid for every agent at every timestep.

    Asserted for all three strategies because it lives in `_preprocess_bids_offers`, which each of
    them calls, and a reversal there would flip every agent in a simulation from buying to
    selling.
    """
    random.seed(11)

    assert quantities(quotes(strategy)) == [(BID_0, 0), (0, OFFER_1), (BID_2, 0)]


@pytest.mark.parametrize('strategy', [c.MTS_ZI, c.MTS_LINEAR, c.MTS_RETAILER])
def test_energy_already_traded_is_netted_out_of_the_position(strategy):
    """An agent quotes for what it still needs, not for its whole setpoint.

    Step 0's setpoint is 250 Wh and the agent has already bought 100, so it bids **150**. Step 1's
    is a 125 Wh sale against 25 Wh already sold, so it offers **100**. Step 2 has no history and
    is unchanged at 500.

    **The empty-history case every other test here uses cannot see any of this.** With no
    transactions the subtraction is a subtraction of zero, so reversing it, deleting the
    `fill_null(0)` that guards it, or swapping the first/last bounds of the query window are all
    invisible -- measured, all three survive the whole suite without this test. Reversing it is
    the serious one: an agent that had already bought would be treated as having sold, and would
    re-bid for more rather than stopping.
    """
    random.seed(11)
    result = quotes(strategy, market=market_data(history=[(0, 100, 0), (1, 0, 25)]),
                    agent=Agent('agent_1', strategy))

    assert quantities(result) == [(150, 0), (0, 100), (BID_2, 0)]


@pytest.mark.parametrize('strategy', [c.MTS_ZI, c.MTS_LINEAR, c.MTS_RETAILER])
def test_positions_beyond_the_trading_horizon_are_dropped(strategy):
    """A two-step horizon admits steps 0 and 1 and excludes steps 2 and 3.

    Brackets the boundary rather than testing a value far from it: step 2 sits exactly *at* two
    steps' distance and is excluded, because the comparison is strict.
    """
    random.seed(11)

    assert set(quotes(strategy, agent=Agent('agent_1', strategy,
                                            horizon=2 * STEP_MINUTES * 60))) == {0, 1}


def test_a_timestep_the_forecaster_did_not_cover_is_quoted_at_a_null_price():
    """**A stated gap, not an endorsement.** A missing forecast produces a price-less bid.

    `_preprocess_bids_offers` joins the forecast on with `how='left'`, so a timestep the
    forecaster did not produce keeps its position and carries `null` prices into the market. It is
    pinned because the alternative -- an inner join -- would silently *drop* the position instead,
    and neither behaviour is obviously the intended one: a null price reaching
    `__split_bids_offers`' sort is its own hazard. Recorded so the choice is visible; no shipped
    configuration produces a short forecast.
    """
    agent = Agent('agent_1', c.MTS_RETAILER, setpoints=[1000, -500, 2000, 1000])
    agent.forecasts = agent.forecasts.head(3)

    result = quotes(c.MTS_RETAILER, agent=agent)

    assert set(result) == {0, 1, 2, 3}
    assert result[3] == ('agent_1', BID_0, 0, None, None)
    assert agent.bids_offers[c.TC_PRICE_PU_IN].null_count() == 1


def test_the_agent_id_is_stamped_onto_every_quote():
    """Two agents, two ids. A hard-coded id would attribute both to one account.

    `quotes()` carries the id in every tuple, so the tables above already assert it -- but only
    for one agent, where a constant is indistinguishable from the real thing.
    """
    first = quotes(c.MTS_RETAILER, agent=Agent('agent_1', c.MTS_RETAILER))
    second = quotes(c.MTS_RETAILER, agent=Agent('agent_2', c.MTS_RETAILER))

    assert {row[0] for row in first.values()} == {'agent_1'}
    assert {row[0] for row in second.values()} == {'agent_2'}


def test_the_quoted_id_column_is_categorical_as_the_market_schema_requires():
    """`__split_bids_offers` renames and joins on this column; the market schema types it."""
    agent = Agent('agent_1', c.MTS_RETAILER)
    quotes(c.MTS_RETAILER, agent=agent)

    assert agent.bids_offers.schema[c.TC_ID_AGENT] == pl.Categorical
