"""Unit -- the three trading strategies, priced by hand.

**Two of the three had never executed.** Measured for #223 over the whole suite, `Linear` and
`Zi` reached only their `class` and `def` lines: 15 and 7 statements respectively were dead,
because no scenario any test ran cleared a market, and `agent_base.create_bids_offers`
substitutes `Retailer` when there is no clearing method. This file drives all three directly.

**What each strategy is.** All three take the same two forecast prices -- what the agent expects
to pay the retailer (`energy_price_out`) and to be paid by it (`energy_price_in`) -- and turn a
net position into a bid or an offer price:

* `Retailer` quotes the retailer's own prices, so the agent is indifferent between trading
  locally and trading with the retailer. It is the default when there is no clearing.
* `Linear` walks the bid from the retailer's selling price down towards its buying price as
  delivery approaches, and the offer the other way. Closest to delivery, the agent is most
  willing to pay and least fussy about what it receives.
* `Zi` -- zero intelligence -- draws each price uniformly from between the two.

**Prices are in EURe-7/Wh** and energies in Wh, as everywhere in the executor.

**`Zi` is not reproducible in a real run, and this file is where that is written down.** It calls
`random.random()` on the module-global generator, which HAMLET never seeds --
`test_zi_is_reproducible_only_when_the_global_generator_is_seeded` demonstrates both halves. It
is the same shape as the market half of #216, which `ElectricityMarket._clearing_seed` fixed for
the clearing shuffle; the strategies were not touched then and no scenario runs `zi`, so it is
latent. The tests below therefore pin `Zi`'s *formula* -- exactly, via a degenerate forecast
where the draw cannot matter -- and its *range*, rather than a recorded sequence of draws.
"""
import random

import pytest

import hamlet.constants as c
from hamlet.executor.utilities.trading.trading import Trading
from tests.trading_fixtures import (FORECAST_IN, FORECAST_OUT, NAME, STEPS, Agent, market_data,
                                    timetable)


def quotes(strategy, agent=None, **kwargs):
    """Run one strategy and return `{timestep_index: (energy_in, energy_out, pu_in, pu_out)}`."""
    agent = agent if agent is not None else Agent('agent_1', strategy, **kwargs)
    instance = Trading(strategy=strategy, timetable=timetable(), market=NAME,
                       market_data=market_data(), agent=agent).create_instance()

    result = instance.create_bids_offers()

    assert result is agent, 'the strategy must hand the agent back'
    index = {step: position for position, step in enumerate(STEPS)}
    return {index[row[c.TC_TIMESTEP]]: (row[c.TC_ENERGY_IN], row[c.TC_ENERGY_OUT],
                                        row[c.TC_PRICE_PU_IN], row[c.TC_PRICE_PU_OUT])
            for row in agent.bids_offers.to_dicts()}, agent


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


# ----------------------------------------------------------------------------------------------
# Retailer.
#
# The agent quotes the retailer's own two prices, crossed: it bids what the retailer charges
# (1500) and offers at what the retailer pays (500). Every timestep gets the same pair, and the
# timestep with no net position is dropped.
# ----------------------------------------------------------------------------------------------
def test_retailer_quotes_the_retailers_own_prices_on_both_sides():
    result, _ = quotes(c.MTS_RETAILER)

    assert result == {
        0: (1000, 0, FORECAST_OUT, FORECAST_IN),
        1: (0, 500, FORECAST_OUT, FORECAST_IN),
        2: (2000, 0, FORECAST_OUT, FORECAST_IN),
    }


def test_retailer_does_not_cross_the_two_prices():
    """A regression shape, not a tautology: bidding 500 and offering 1500 would be free money.

    Asserted separately from the exact table above because the two prices being swapped is the
    one mutation of `Retailer` that keeps every row, every quantity and every timestep intact.
    """
    result, _ = quotes(c.MTS_RETAILER)

    for _, (_, _, price_in, price_out) in result.items():
        assert price_in > price_out


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
#       0        0     1500     500
#       1        0     1500     500
#       2        2     1000    1000
#       3        4      500    1500     <- no net position, dropped by _postprocess
#
# So the nearest steps quote the aggressive end (willing to pay the full retail price, willing to
# sell at the feed-in tariff) and the ladder converges on the midpoint as the horizon recedes.
# ----------------------------------------------------------------------------------------------
def test_linear_walks_the_bid_down_and_the_offer_up_across_the_horizon():
    result, _ = quotes(c.MTS_LINEAR)

    assert result == {
        0: (1000, 0, 1500, 500),
        1: (0, 500, 1500, 500),
        2: (2000, 0, 1000, 1000),
    }


def test_linear_and_retailer_agree_only_at_the_nearest_step():
    """The two strategies coincide at factor 0, and this is why an e2e test cannot tell them apart
    from the first timestep's book alone.

    Recorded because it is a real trap: a `scenario_with_market` clearing at the first timestep
    produces bids of exactly 1500 under either strategy.
    `tests/e2e/test_market_clearing.py` uses prices strictly *inside* the band as its evidence,
    and this test is why.
    """
    linear, _ = quotes(c.MTS_LINEAR)
    retailer, _ = quotes(c.MTS_RETAILER)

    assert linear[0] == retailer[0]
    assert linear[2] != retailer[2]


def test_linears_prices_stay_within_the_retail_band():
    """Neither side ever quotes outside [feed-in tariff, retail price]."""
    result, _ = quotes(c.MTS_LINEAR)

    for step, (_, _, price_in, price_out) in result.items():
        assert FORECAST_IN <= price_in <= FORECAST_OUT, step
        assert FORECAST_IN <= price_out <= FORECAST_OUT, step


def test_linear_shape_parameters_move_the_ladder():
    """`steps_to_final` holds the aggressive price for longer, and the table says so.

    With `steps_to_final = 2` the factor sequence becomes `[0, 0] ++ linspace(0, 4, 2) ++ []`
    = `[0, 0, 0, 4]`, so step 2 -- which quoted the midpoint above -- now quotes the aggressive
    end instead. Without this, `steps_to_final` could be ignored entirely and the file stay green.
    """
    agent = Agent('agent_1', c.MTS_LINEAR)
    agent.account[c.K_EMS][c.K_MARKET][c.MTS_LINEAR]['steps_to_final'] = 2

    result, _ = quotes(c.MTS_LINEAR, agent=agent)

    assert result == {
        0: (1000, 0, 1500, 500),
        1: (0, 500, 1500, 500),
        2: (2000, 0, 1500, 500),
    }


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
    depending on the generator. Run under two different seeds so a passing result cannot be an
    accident of one of them.
    """
    for seed in (0, 12345):
        random.seed(seed)
        result, _ = quotes(c.MTS_ZI, forecast_in=FLAT_PRICE, forecast_out=FLAT_PRICE)

        assert result == {
            0: (1000, 0, FLAT_PRICE, FLAT_PRICE),
            1: (0, 500, FLAT_PRICE, FLAT_PRICE),
            2: (2000, 0, FLAT_PRICE, FLAT_PRICE),
        }


def test_zi_draws_both_prices_from_inside_the_retail_band():
    """Every quote lies in [500, 1500], and **each side separately** lands strictly inside it.

    Per side, not in total: the two prices are drawn by two separate `pl.Series` comprehensions,
    and a mutation to one of them leaves the other producing interior values. Counting both
    together lets the surviving side vouch for the broken one -- measured, replacing the bid
    side's draw with a constant kept a combined count green.

    The "strictly inside" half is what stops the test passing for an implementation that ignored
    the draw and returned an endpoint, which is exactly what `Retailer` does and what the
    degenerate case above cannot distinguish. 12 draws per side over 4 seeds; the chance of every
    one landing on an endpoint by luck is nil.
    """
    inside = {c.TC_PRICE_PU_IN: 0, c.TC_PRICE_PU_OUT: 0}
    drawn = {c.TC_PRICE_PU_IN: set(), c.TC_PRICE_PU_OUT: set()}
    for seed in (1, 2, 3, 4):
        random.seed(seed)
        result, _ = quotes(c.MTS_ZI)

        for step, (_, _, price_in, price_out) in result.items():
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
    advance it between two runs and they do not. **HAMLET seeds it nowhere**, so in a real
    simulation the second case is the one that applies and a `zi` scenario is not reproducible
    across processes. Same shape as the clearing shuffle in #216, which `_clearing_seed` fixed;
    the strategies were not touched then. No shipped scenario selects `zi`, so this is latent.
    """
    random.seed(7)
    first, _ = quotes(c.MTS_ZI)
    random.seed(7)
    again, _ = quotes(c.MTS_ZI)

    assert first == again

    random.seed(7)
    random.random()  # advance the generator the way an unrelated caller would
    shifted, _ = quotes(c.MTS_ZI)

    assert shifted != first


# ----------------------------------------------------------------------------------------------
# Shared behaviour of `TradingBase`, which all three inherit.
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize('strategy', [c.MTS_ZI, c.MTS_LINEAR, c.MTS_RETAILER])
def test_a_timestep_with_no_net_position_produces_no_quote(strategy):
    """Step 3's setpoint is zero, so no bid and no offer are submitted for it."""
    random.seed(11)
    result, _ = quotes(strategy)

    assert set(result) == {0, 1, 2}


@pytest.mark.parametrize('strategy', [c.MTS_ZI, c.MTS_LINEAR, c.MTS_RETAILER])
def test_a_purchase_becomes_a_bid_and_a_sale_becomes_an_offer(strategy):
    """The sign convention, per strategy: `+1000 W` is `energy_in`, `-500 W` is `energy_out`.

    Asserted for all three because it lives in `_preprocess_bids_offers`, which each of them
    calls, and a reversal there would flip every agent in a simulation from buying to selling.
    """
    random.seed(11)
    result, _ = quotes(strategy)

    assert (result[0][0], result[0][1]) == (1000, 0)
    assert (result[1][0], result[1][1]) == (0, 500)
    assert (result[2][0], result[2][1]) == (2000, 0)


@pytest.mark.parametrize('strategy', [c.MTS_ZI, c.MTS_LINEAR, c.MTS_RETAILER])
def test_positions_beyond_the_trading_horizon_are_dropped(strategy):
    """A two-hour horizon admits steps 0 and 1 and excludes steps 2 and 3.

    The horizon is stated in seconds and compared against a `Timedelta`, so this brackets the
    boundary rather than testing a value far from it: step 2 sits exactly *at* two hours and is
    excluded, because the comparison is strict.
    """
    random.seed(11)
    result, _ = quotes(strategy, horizon=2 * 3600)

    assert set(result) == {0, 1}
