"""Unit -- settlement accounting against hand-computed money. This is L4.

L3 (`test_clearing_by_hand.py`) settles *who traded what with whom*. This file settles what it
then costs: balancing energy, grid fees, levies, and whether the money adds up. It drives the
real `ElectricityMarket.execute()` over `clear,settle` and asserts every figure against a number
derived on paper, per agent, by name.

**Why every invariant here carries an absolute anchor.** #241 found that `__method_pda` sold
energy nobody had offered -- 7.1 % of all locally cleared energy did not exist -- while the
conservation invariant `sum(cleared in) == sum(cleared out)` passed throughout, because the
duplication was symmetric. It held at an inflated total. Under one silent mutation a clearing of
1.8e19 Wh satisfied it. So `sum(in) == sum(out)` appears nowhere in this file on its own: every
total is also compared against something outside the pair -- the submitted book, the retailer's
named rate columns, or a quantity multiplied out by hand.

**The three invariants, and what anchors each.**

1. *Net energy is the sum of the timestep's transactions.* Anchored on the delivery timestep's
   history, which is written into the fixture rather than derived: `buyer` bought 3000 Wh at this
   clearing and had sold 1000 Wh at an earlier one, so its net is 2000 Wh and nothing but the
   arithmetic can make it so. See `test_the_fee_energy_is_the_timesteps_net_not_this_clearings`.
2. *Fees and levies are charged on net, to the right column.* Anchored on the six retailer rate
   columns, all distinct and all non-zero (`settlement_fixtures`), so every mapping is a
   different number rather than a different name for zero.
3. *The balance closes.* Anchored twice: every money column is recomputed as rate times quantity
   and must match to the euro, and the residual left after the agent-to-agent trades cancel must
   equal the named charges exactly -- no unexplained euro anywhere.

**Prices are in EURe-7/Wh**, quantities in Wh, which is what the executor uses throughout.
"""
import polars as pl
import pytest

import hamlet.constants as c
from tests.settlement_fixtures import (BALANCING_BUYS, BALANCING_SELLS, GRID_MARKET_IN,
                                       GRID_MARKET_OUT, GRID_RETAIL_IN, GRID_RETAIL_OUT,
                                       LEVIES_IN, LEVIES_OUT, RETAILER_BUYS, RETAILER_SELLS,
                                       RETAILER_ID, forecast_table, ledger, make_market, rows,
                                       settle)

# ----------------------------------------------------------------------------------------------
# The book every number below is derived from, and the derivation.
#
# Bid ladder, sorted by price descending, with its cumulative energy:
#     buyer  3000 Wh @ 3000   cumsum 3000
#     spare  2000 Wh @ 1000   cumsum 5000
# Offer ladder, ascending:
#     seller 3000 Wh @ 1200   cumsum 3000
#
# The two ladders meet at 3000 Wh and nowhere else. At that breakpoint 3000 >= 1200, so the
# segment clears: 3000 Wh. At 5000 the offer side has run out, so `spare` clears nothing and its
# whole 2000 Wh bid stays uncleared. Uniform pricing takes the midpoint of the marginal pair,
# (3000 + 1200) / 2 = 2100, so the trade is worth 2100 * 3000 = 6_300_000.
#
# None of the four submitted prices is 2100, so an implementation echoing a submitted price back
# instead of computing a clearing price fails rather than coinciding.
# ----------------------------------------------------------------------------------------------
BOOK = [('buyer', 3000, 0, 3000, RETAILER_BUYS),
        ('spare', 2000, 0, 1000, RETAILER_BUYS),
        ('seller', 0, 3000, RETAILER_SELLS, 1200)]

CLEARING_PRICE = 2100
CLEARED_ENERGY = 3000
CLEARED_MONEY = 6_300_000

#: `spare`'s bid clears nothing, so settlement buys the whole 2000 Wh as balancing energy, at the
#: retailer's *selling* price -- the agent is buying.
UNCLEARED_ENERGY = 2000
BALANCING_MONEY = BALANCING_SELLS * UNCLEARED_ENERGY  # 1600 * 2000 = 3_200_000

# ----------------------------------------------------------------------------------------------
# What this delivery timestep already carries when it settles.
#
# **The direction is the point.** `buyer` *sold* 1000 Wh at an earlier clearing of the same
# delivery timestep and buys 3000 Wh at this one, so its gross flow is 3000 in and 1000 out while
# its net is 2000 in. With a one-directional history the two would be the same number and
# `eec60f0` -- fees charged on gross -- would be unfalsifiable here.
#
# The retailer's matching row is included because a real retail trade writes both sides. It is
# what makes the money ledger closable, and the retailer is excluded from fees, so it also checks
# that exclusion against a row that actually exists.
# ----------------------------------------------------------------------------------------------
HISTORY = [('buyer', c.TT_RETAIL, 0, 1000, 0, 500_000),
           (RETAILER_ID, c.TT_RETAIL, 1000, 0, 500_000, 0)]

SOLD_EARLIER = 1000
BUYER_NET_IN = CLEARED_ENERGY - SOLD_EARLIER  # 3000 - 1000 = 2000

#: Per-agent net energy at this delivery timestep, computed by hand from the book and the history
#: above. `seller` has no history, so its net is its cleared offer; `spare` nets its balancing
#: purchase and nothing else.
NET = {'buyer': (BUYER_NET_IN, 0), 'spare': (UNCLEARED_ENERGY, 0), 'seller': (0, CLEARED_ENERGY)}

# ----------------------------------------------------------------------------------------------
# The grid fee each agent pays, derived.
#
# `__apply_grid_levies` blends the two grid rates by the share of an agent's energy that came
# from the local market: `market_rate * ratio + retail_rate * (1 - ratio)`.
#
#   buyer  -- 3000 Wh in, all of it `market`; the 1000 Wh `retail` row is on the *out* side, so
#             the in-ratio is 3000/3000 = 1 and the rate is the market one, 400.
#   spare  -- 2000 Wh in, all of it `balancing`; no `market` energy in, so the ratio is 0 and the
#             rate is the retail one, 800.
#   seller -- 3000 Wh out, all `market`; out-ratio 1, so the *feed-in* market rate, 40.
#
# The two buyers therefore pay different rates for the same 2000 Wh, which is what makes the
# blend observable at all: a mapping that ignored the ratio would give them the same number.
# ----------------------------------------------------------------------------------------------
GRID_RATE = {'buyer': GRID_MARKET_OUT, 'spare': GRID_RETAIL_OUT, 'seller': GRID_MARKET_IN}
LEVY_RATE = {'buyer': LEVIES_OUT, 'spare': LEVIES_OUT, 'seller': LEVIES_IN}

AGENTS = ('buyer', 'spare', 'seller')

#: Each agent's grid bill and levy, in EURe-7: its rate times its net quantity. The two factors
#: are independent -- the rate comes from the retailer file, the quantity from the book and the
#: history -- so multiplying them here is a hand computation rather than a restatement of what
#: the code does with them.
GRID_BILL = {agent: GRID_RATE[agent] * sum(NET[agent]) for agent in AGENTS}
LEVY_BILL = {agent: LEVY_RATE[agent] * sum(NET[agent]) for agent in AGENTS}

#: `seller` is a net exporter, so its two charges land in `price_out`. In a real tariff they
#: would be zero -- the shipped retailer files price feed-in at nothing -- and they are non-zero
#: here only so the column mapping is observable. See `settlement_fixtures`.
IMPORTERS = ('buyer', 'spare')
EXPORTERS = ('seller',)


@pytest.fixture(scope='module')
def settled():
    """`BOOK` and `HISTORY` run through the real `execute()`, once for the whole module."""
    return settle(BOOK, history=HISTORY)


@pytest.fixture(scope='module')
def book(settled):
    """`{(kind, agent): (energy_in, energy_out, price_in, price_out)}`, nulls read as 0.

    The clearing writes `None` into the direction an agent did not trade while the fee rows write
    0; both mean "no energy this way" and the arithmetic below treats them alike.
    """
    return {key: tuple(0 if value is None else value for value in values)
            for key, values in ledger(settled).items()}


# ----------------------------------------------------------------------------------------------
# Premises. Each one is something the assertions below would pass vacuously without, and two of
# them are traps this repository has already fallen into.
# ----------------------------------------------------------------------------------------------
def test_the_book_clears_the_way_this_file_assumes(book):
    """3000 Wh trades at 2100, and `spare` clears nothing -- the derivation at the top.

    Stated separately because every fee below is charged on a quantity this clearing produced. A
    book that stopped clearing would leave the fee assertions asserting about nothing.
    """
    assert book[(c.TT_MARKET, 'buyer')] == (CLEARED_ENERGY, 0, CLEARED_MONEY, 0)
    assert book[(c.TT_MARKET, 'seller')] == (0, CLEARED_ENERGY, 0, CLEARED_MONEY)
    assert (c.TT_MARKET, 'spare') not in book
    assert CLEARED_MONEY == CLEARING_PRICE * CLEARED_ENERGY


def test_the_history_runs_the_other_way_from_this_clearing():
    """`buyer`'s two flows are in opposite directions, or netting is not being tested at all.

    A history that ran the same way as the clearing would make gross and net the same number, and
    then charging fees on gross -- `eec60f0` -- would be an equivalent mutation in this file. The
    same shape defeated an earlier fixture in this repository through a "don't-care" zero.
    """
    bought_now = {agent: energy_in for agent, energy_in, _, _, _ in BOOK}
    sold_earlier = sum(energy_out for agent, kind, _, energy_out, _, _ in HISTORY
                       if agent == 'buyer')

    assert bought_now['buyer'] > 0 and sold_earlier > 0
    assert bought_now['buyer'] != sold_earlier, 'gross and net would coincide'


@pytest.mark.parametrize('rates', [
    (GRID_MARKET_IN, GRID_MARKET_OUT, GRID_RETAIL_IN, GRID_RETAIL_OUT, LEVIES_IN, LEVIES_OUT),
])
def test_every_charge_rate_in_the_fixture_is_distinct_and_non_zero(rates):
    """Six rates, six different non-zero values, so every column mapping is falsifiable.

    The shipped retailer files charge 0 on feed-in. A fixture that copied them could not tell
    `grid_retail_in` from `grid_market_in`, both being zero, and a swap between the two would be
    an equivalent mutant -- which is exactly how a placeholder value faked equivalence on !250.
    """
    assert len(set(rates)) == len(rates)
    assert all(rate > 0 for rate in rates)


def test_the_retailer_prices_buying_above_selling():
    """The convention the whole file rests on: no risk-free arbitrage against the retailer.

    `tests/README.md` calls the in/out crossing the single most error-prone thing in this
    codebase and gives this as its tell -- a test that appears to show selling paying better than
    buying is showing a convention bug, not an arbitrage.
    """
    assert RETAILER_SELLS > RETAILER_BUYS
    assert BALANCING_SELLS > BALANCING_BUYS
    assert GRID_MARKET_OUT > GRID_MARKET_IN and GRID_RETAIL_OUT > GRID_RETAIL_IN
    assert LEVIES_OUT > LEVIES_IN


# ----------------------------------------------------------------------------------------------
# Invariant 1 -- net energy is the sum of the timestep's transactions.
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize('agent', AGENTS)
@pytest.mark.parametrize('kind', [c.TT_GRID, c.TT_LEVIES])
def test_the_charged_energy_is_the_agents_net_flow(book, kind, agent):
    """Each fee row carries the agent's *net* energy for the timestep, per agent by name.

    Asserted per agent rather than as a total, because a total cannot see the right quantity
    charged to the wrong agent -- and here `buyer` and `spare` net to the same 2000 Wh by
    different routes, so an implementation that swapped them would keep every sum intact.
    """
    energy_in, energy_out, _, _ = book[(kind, agent)]

    assert (energy_in, energy_out) == NET[agent]


def test_the_fee_energy_is_the_timesteps_net_not_this_clearings(book):
    """Regression, `a860ca6`: the current timestep's market trades were left out of the netting.

    `_transactions_for_netting` concatenates three sources -- what settled at earlier clearings
    of this delivery timestep, what cleared at this one, and the balancing energy. The middle one
    was missing, so an agent's fees were computed as though it had not traded this timestep.

    Here `buyer` sold 1000 Wh earlier and bought 3000 Wh now. With the current clearing included
    it nets 2000 Wh **in**; without it, 1000 Wh **out** -- a different direction, not merely a
    different number, so the fee lands in the other column at the other rate.
    """
    energy_in, energy_out, price_in, price_out = book[(c.TT_GRID, 'buyer')]

    assert (energy_in, energy_out) == (BUYER_NET_IN, 0)
    assert (price_in, price_out) == (GRID_MARKET_OUT * BUYER_NET_IN, 0)
    # What the defect produced instead, stated so the two cannot be confused
    assert (energy_in, energy_out) != (0, SOLD_EARLIER)


def test_the_fee_energy_is_net_and_not_gross(book):
    """Regression, `eec60f0` part 1: fees were charged on gross energy, overcharging both sides.

    `buyer`'s gross flow at this timestep is 3000 Wh in *and* 1000 Wh out -- 4000 Wh of billable
    energy if both are charged, against the 2000 Wh it actually drew from the grid. The
    difference is a factor of two on its grid bill and its levy.
    """
    gross_in, gross_out = CLEARED_ENERGY, SOLD_EARLIER
    energy_in, energy_out, price_in, _ = book[(c.TT_GRID, 'buyer')]

    assert energy_in + energy_out == BUYER_NET_IN
    assert energy_in + energy_out < gross_in + gross_out
    assert price_in == GRID_MARKET_OUT * BUYER_NET_IN
    assert price_in != GRID_MARKET_OUT * gross_in


def test_only_one_direction_of_a_fee_row_is_ever_charged(book):
    """An agent cannot be a net importer and a net exporter in the same timestep."""
    for kind in (c.TT_GRID, c.TT_LEVIES):
        for agent in AGENTS:
            energy_in, energy_out, _, _ = book[(kind, agent)]
            assert energy_in == 0 or energy_out == 0


def test_the_retailer_is_not_charged_grid_fees_or_levies(book):
    """The retailer trades on this timestep -- `HISTORY` gives it a row -- and pays no fees.

    Its position is the market's supply of last resort, not a grid connection, and charging it
    would bill the tariff back to the tariff. Stated against a retailer that is genuinely
    present, so the exclusion is exercised rather than vacuous.
    """
    assert any(agent == RETAILER_ID for agent, *_ in HISTORY)
    assert not [key for key in book if key[0] in (c.TT_GRID, c.TT_LEVIES)
                and key[1] == RETAILER_ID]


# ----------------------------------------------------------------------------------------------
# Invariant 2 -- fees and levies go to the right column, at the right rate.
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize('agent', AGENTS)
def test_the_levy_rate_is_the_retailers_selling_column(settled, agent):
    """Regression, `eec60f0` part 2: the levy's two price columns were the wrong way round.

    The retailer file names its columns from the *retailer's* point of view, so `levies_price_out`
    is the retailer selling -- the agent buying -- and that is the direction a consumption tax is
    charged on. The transaction columns are from the agent's point of view, so the mapping has to
    cross: `levies_price_out` becomes the agent's `price_pu_in`.

    A swap here charges the consumption levy on feed-in and the feed-in rate on consumption, at
    1800 against 180 -- a tenfold error in whichever direction the agent actually moved energy.
    """
    charged = rates(settled, c.TT_LEVIES)[agent]

    assert charged == (LEVIES_OUT, LEVIES_IN)


@pytest.mark.parametrize('agent, rate', [('buyer', GRID_MARKET_OUT), ('spare', GRID_RETAIL_OUT)])
def test_the_grid_rate_on_consumption_is_the_forecasts_selling_column(settled, agent, rate):
    """The same crossing for grid fees, on the path that is live.

    `__apply_grid_levies` seeds the four grid-rate columns from the retailer and then overwrites
    every one of them, for every agent, from that agent's own forecast of the variable grid fee.
    So the direction that decides an agent's bill is the `forecast_dir` in that loop, and it is
    the one asserted here. See `test_the_retailer_grid_rates_never_reach_the_output` for what
    that makes of the seed.

    The two agents draw the same 2000 Wh and pay different rates, because `buyer` drew it from
    the local market and `spare` from the retailer; a mapping that lost the blend would give them
    the same number.
    """
    charged_in, _ = rates(settled, c.TT_GRID)[agent]

    assert charged_in == rate


def test_the_grid_rate_on_feed_in_is_the_forecasts_buying_column(settled):
    """The mirror direction, on the one agent that feeds in."""
    _, charged_out = rates(settled, c.TT_GRID)['seller']

    assert charged_out == GRID_MARKET_IN


@pytest.mark.parametrize('kind', [c.TT_GRID, c.TT_LEVIES])
@pytest.mark.parametrize('importer', ['buyer', 'spare'])
def test_a_charge_is_dearer_on_consumption_than_on_feed_in(settled, kind, importer):
    """`tests/README.md`'s tell, asserted on what the executor emitted rather than on the config.

    Grid fees and levies are both charged on consumption. If the rate charged to an agent drawing
    energy is not above the rate charged to one feeding it in, the in/out crossing has gone wrong
    somewhere between the retailer file and the transactions table.

    **Across two agents rather than across one agent's two columns**, and that is forced rather
    than stylistic: see `test_the_unused_direction_of_a_fee_row_is_priced_at_zero`.
    """
    charged = rates(settled, kind)[importer][0]
    paid = rates(settled, kind)['seller'][1]

    assert charged > paid > 0


def test_a_grid_rate_collapses_to_zero_where_the_agent_moved_no_energy_that_way(settled):
    """A grid rate is blended by a ratio, and the ratio is `0 / 0` in an unused direction.

    Pinned rather than fixed. `__apply_grid_levies` weights the market rate against the retail
    one by the share of the agent's energy that came from the local market. An agent with no
    energy at all in one direction divides zero by zero there, and the NaN is filled with 0 -- so
    a pure exporter's `price_pu_in` is 0 rather than the tariff it would pay if it imported.

    Harmless for the bill, because the quantity it multiplies is 0 either way. It matters because
    the per-unit columns of a `grid` row are therefore **not** a tariff table: a reader taking a
    rate off one for an agent that did not move energy that way gets zero rather than the rate.
    It is also why the assertion above compares two agents rather than one agent's two columns.

    The direction is decided on **gross** energy, not net: `buyer` nets to zero on the out side
    but did sell 1000 Wh, and so still carries a feed-in rate.
    """
    charged = rates(settled, c.TT_GRID)

    assert charged['seller'] == (0, GRID_MARKET_IN), 'the exporter has no gross import'
    assert charged['spare'] == (GRID_RETAIL_OUT, 0), 'the importer has no gross export'
    assert charged['buyer'] == (GRID_MARKET_OUT, GRID_RETAIL_IN), 'gross, not net, on both sides'


def test_the_levy_is_a_flat_rate_and_is_not_blended(settled):
    """Every agent gets the same levy pair, whatever its market share.

    Unlike the grid fee, levies carry no market-versus-retail split. Stated as "all three agree"
    rather than by restating the two values, which `test_the_levy_rate_is_the_retailers_selling_
    column` already pins: this is about the absence of a blend, not about the rate.
    """
    charged = rates(settled, c.TT_LEVIES)

    assert len({charged[agent] for agent in AGENTS}) == 1
    assert charged['buyer'] != rates(settled, c.TT_GRID)['buyer'], (
        'grid and levies now agree on every column, so this book cannot tell them apart')


def test_the_retailer_grid_rates_never_reach_the_output(settled):
    """A finding, pinned: the grid rates read off the retailer table are a dead placeholder.

    `__apply_grid_levies` writes them into four columns and the variable-fee loop then overwrites
    all four for every agent it finds -- and it finds every agent in the frame, because the frame
    is built from the transactions and every agent in it has a forecaster. So the *values* those
    four `pl.lit(retailer[...])` lines put there are never read; only the columns they create
    are.

    Measured here by giving one agent a forecast that disagrees with the retailer in both
    directions: the forecast wins on both. The consequence is that crossing or un-crossing the
    retailer-to-grid mapping in those four lines is an equivalent mutation in any scenario, which
    is worth knowing before trusting a mutation score for that function.
    """
    disagreeing = forecast_table(**{f'{c.TT_GRID}_{c.TT_MARKET}_{c.PF_OUT}': 4321,
                                    f'{c.TT_GRID}_{c.TT_MARKET}_{c.PF_IN}': 1234})
    result = settle(BOOK, history=HISTORY, forecasts={'buyer': disagreeing, 'seller': disagreeing})

    assert rates(result, c.TT_GRID)['buyer'][0] == 4321
    assert rates(result, c.TT_GRID)['seller'][1] == 1234
    # ... and the untouched agent still gets the shared default, so this is the forecast winning
    # rather than the retailer column being ignored for everyone.
    assert rates(result, c.TT_GRID)['spare'][0] == GRID_RETAIL_OUT


# ----------------------------------------------------------------------------------------------
# Invariant 3 -- the balance closes.
# ----------------------------------------------------------------------------------------------
def test_every_charge_is_its_rate_times_its_quantity(settled):
    """No unexplained euro anywhere: each money column is its per-unit column times its energy.

    This is the reconciliation half of "the balance closes" -- an agent's ledger has to be
    derivable from rates and quantities with a zero residual, on every row of every kind. It is
    not a symmetric check: the money column is compared against a product computed from two other
    columns, so a total that is internally consistent but wrong fails it.
    """
    frame = settled.market_transactions.with_columns(
        [pl.col(column).fill_null(0) for column in
         (c.TC_ENERGY_IN, c.TC_ENERGY_OUT, c.TC_PRICE_IN, c.TC_PRICE_OUT,
          c.TC_PRICE_PU_IN, c.TC_PRICE_PU_OUT)])

    residual_in = (frame[c.TC_PRICE_IN] - frame[c.TC_PRICE_PU_IN] * frame[c.TC_ENERGY_IN])
    residual_out = (frame[c.TC_PRICE_OUT] - frame[c.TC_PRICE_PU_OUT] * frame[c.TC_ENERGY_OUT])

    assert residual_in.abs().sum() == 0
    assert residual_out.abs().sum() == 0
    assert len(frame) > 0


@pytest.mark.parametrize('agent, expected', [
    ('buyer', CLEARED_MONEY + GRID_BILL['buyer'] + LEVY_BILL['buyer']),
    ('spare', BALANCING_MONEY + GRID_BILL['spare'] + LEVY_BILL['spare']),
])
def test_what_each_agent_pays_is_the_hand_computed_total(book, agent, expected):
    """Each buyer's whole bill for the timestep, to the euro, from the derivations above.

    Per agent by name and as an absolute figure -- the one form of this assertion that a defect
    crediting the right total to the wrong agent cannot satisfy.
    """
    paid = sum(book[key][2] for key in book if key[1] == agent)

    assert paid == expected


def test_what_the_seller_receives_is_the_hand_computed_total(book):
    """The mirror, including the fee it is charged on feed-in in this fixture's tariff."""
    received = sum(book[key][3] for key in book if key[1] == 'seller')

    assert received == CLEARED_MONEY + GRID_BILL['seller'] + LEVY_BILL['seller']


def test_the_agent_to_agent_trades_cancel_and_the_residual_is_exactly_the_charges(book):
    """The closure: money paid less money received equals the named charges, with nothing over.

    The local trade cancels -- what `buyer` pays, `seller` receives -- so the whole table's
    imbalance has to be accounted for, euro for euro, by the rows that move money out of the
    agents' accounts: balancing, grid and levies. A residual would be money the market created or
    destroyed without saying so.

    **The anchor is that the charges are recomputed from rates and quantities**, not read back
    out of the same sum: `sum(in) - sum(out)` on its own would be satisfied by any pair of totals
    that happened to differ by the right amount.
    """
    paid = sum(values[2] for values in book.values())
    received = sum(values[3] for values in book.values())

    leaves = BALANCING_MONEY + sum(GRID_BILL[a] + LEVY_BILL[a] for a in IMPORTERS)
    enters = sum(GRID_BILL[a] + LEVY_BILL[a] for a in EXPORTERS)

    assert paid - received == leaves - enters
    assert book[(c.TT_MARKET, 'buyer')][2] == book[(c.TT_MARKET, 'seller')][3] == CLEARED_MONEY


def test_the_local_trade_credits_the_seller_and_debits_the_buyer_and_not_the_reverse(book):
    """Direction, not just magnitude: the money moved the way the trade did.

    A sum over the table cannot tell a payment from a receipt, and the two rows of one trade
    carry the same figure -- so this is the assertion that the figure is on the right side of
    each of them.
    """
    assert book[(c.TT_MARKET, 'buyer')] == (CLEARED_ENERGY, 0, CLEARED_MONEY, 0)
    assert book[(c.TT_MARKET, 'seller')] == (0, CLEARED_ENERGY, 0, CLEARED_MONEY)


def test_balancing_energy_is_bought_at_the_retailers_selling_price(book):
    """The uncleared bid becomes a purchase from the retailer, priced the expensive way round.

    `BALANCING_SELLS` against `BALANCING_BUYS` is 1600 against 400, so a crossed mapping here
    would pay an agent to be short of energy.
    """
    energy_in, energy_out, price_in, price_out = book[(c.TT_BALANCING, 'spare')]

    assert (energy_in, energy_out) == (UNCLEARED_ENERGY, 0)
    assert (price_in, price_out) == (BALANCING_MONEY, 0)
    assert price_in > BALANCING_BUYS * UNCLEARED_ENERGY


def test_the_balancing_energy_is_the_submitted_position_less_what_cleared(book):
    """Anchored on the book, which is outside the settlement entirely.

    Settlement takes the uncleared remainder from the clearing and the clearing takes it from
    the submitted positions, so the quantity finally bought as balancing energy has to be what
    each agent asked for less what it got. Comparing it to the book rather than to the clearing's
    own uncleared table is what makes this an absolute check: an error shared by the clearing and
    the settlement -- the shape that let #241's duplication pass a conservation invariant -- is
    visible against the submitted quantity and invisible against anything derived from it.
    """
    submitted = {agent: energy_in for agent, energy_in, _, _, _ in BOOK}
    cleared = {agent: values[0] for (kind, agent), values in book.items() if kind == c.TT_MARKET}
    balanced = {agent: values[0] for (kind, agent), values in book.items()
                if kind == c.TT_BALANCING}

    for agent in submitted:
        assert balanced.get(agent, 0) == submitted[agent] - cleared.get(agent, 0), agent
    assert sum(balanced.values()) > 0, 'nothing was balanced, so this asserts about nothing'


def rates(market, kind):
    """`{agent: (price_pu_in, price_pu_out)}` for one kind of transaction row."""
    frame = market.market_transactions.with_columns(
        pl.col(c.TC_TYPE_TRANSACTION).cast(pl.Utf8), pl.col(c.TC_ID_AGENT).cast(pl.Utf8))
    frame = frame.filter(pl.col(c.TC_TYPE_TRANSACTION) == kind)
    assert len(frame) > 0, f'no {kind} rows were written, so this asserts about nothing'
    return {row[c.TC_ID_AGENT]: (row[c.TC_PRICE_PU_IN], row[c.TC_PRICE_PU_OUT])
            for row in frame.to_dicts()}
