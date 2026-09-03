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
   history, which is written into the fixture rather than derived: `buyer` bought 4000 Wh across
   this clearing and an earlier one and sold 1000 Wh at a third, so its net is 3000 Wh and
   nothing but the arithmetic can make it so.
2. *Fees and levies are charged on net, to the right column.* Anchored on the six retailer rate
   columns, all distinct and all non-zero (`settlement_fixtures`), so every mapping is a
   different number rather than a different name for zero.
3. *The balance closes.* Anchored on `test_what_each_agent_pays_is_the_hand_computed_total` and
   its two siblings, which recompute each agent's whole bill from a rate and a hand-derived
   quantity. **Not** on `test_every_charge_is_its_rate_times_its_quantity`: production computes
   `price_in` as `price_pu_in * energy_in` in so many words, so that assertion restates the code
   and can only fail on a cast or a schema slip. It is a smoke test, and its docstring says so.

**Four degrees of freedom in the book are load-bearing**, and a review panel defeated the first
version of this file through the absence of each:

* **An uncleared *offer*** (`hoarder`), not only an uncleared bid. Without one, every assertion
  touching the feed-in side of balancing multiplies by zero, and both the rate mapping and the
  concatenation that builds balancing energy can be broken silently.
* **A fractional market share, blending to a half.** `buyer` draws 3000 Wh from the local
  market and 1000 Wh from the retailer, so its grid fee is the blend evaluated at 0.75 rather
  than at an endpoint, and the two rates are chosen so that the result is 512.5 rather than a
  whole number. With every share at 0 or 1 the interpolation is never exercised; with a blend
  that lands on an integer, truncating it instead of rounding is invisible.
* **A history running the other way from the clearing.** Without it gross and net coincide and
  `eec60f0` is unfalsifiable.
* **Distinct quantities per agent**, so no two agents' bills collide -- a total credited to the
  wrong agent is what a sum cannot see.

**Three task shapes are covered**, because the accounting differs between them and only one of
them appears in a shipped run: `clear,settle` in one task, settlement as a task of its own, and
`method: None`. The section headed "the task shape every shipped run actually uses" records
which is which, and why `a860ca6` cannot be observed on any shipped scenario.

**Prices are in EURe-7/Wh**, quantities in Wh, which is what the executor uses throughout.
"""
import polars as pl
import pytest

import hamlet.constants as c
from tests.settlement_fixtures import (BALANCING_BUYS, BALANCING_SELLS, ENERGY, GRID_MARKET_IN,
                                       GRID_MARKET_OUT, GRID_RETAIL_IN, GRID_RETAIL_OUT,
                                       LEVIES_IN, LEVIES_OUT, MARKET, NAME, REGION,
                                       RETAILER_BUYS, RETAILER_SELLS, RETAILER_ID, TIMESTAMP,
                                       TIMESTEP, forecast_table, ledger, retailer_table, rows,
                                       settle)

# ----------------------------------------------------------------------------------------------
# The book every number below is derived from, and the derivation.
#
# Bid ladder, sorted by price descending, with its cumulative energy:
#     buyer   3000 Wh @ 3000   cumsum 3000
#     spare   2000 Wh @ 1000   cumsum 5000
# Offer ladder, ascending:
#     seller  3000 Wh @ 1200   cumsum 3000
#     hoarder 2500 Wh @ 4000   cumsum 5500
#
# Merged on the cumulative sums and back-filled, the ladders meet at 3000, 5000 and 5500.
#   3000 -- buyer @ 3000 against seller @ 1200: 3000 >= 1200, so 3000 Wh clears.
#   5000 -- spare @ 1000 against hoarder @ 4000 (back-filled): 1000 < 4000, uncleared.
#   5500 -- no bid beyond it at all, uncleared.
# So `spare`'s whole 2000 Wh bid and `hoarder`'s whole 2500 Wh offer go to settlement, one on
# each side, and uniform pricing takes the midpoint of the single cleared pair:
# (3000 + 1200) / 2 = 2100, worth 2100 * 3000 = 6_300_000.
#
# None of the submitted prices is 2100, so an implementation echoing a submitted price back
# instead of computing a clearing price fails rather than coinciding.
# ----------------------------------------------------------------------------------------------
BOOK = [('buyer', 3000, 0, 3000, RETAILER_BUYS),
        ('spare', 2000, 0, 1000, RETAILER_BUYS),
        ('seller', 0, 3000, RETAILER_SELLS, 1200),
        ('hoarder', 0, 2500, RETAILER_SELLS, 4000)]

CLEARING_PRICE = 2100
CLEARED_ENERGY = 3000
CLEARED_MONEY = 6_300_000

#: What settlement has to buy and sell as balancing energy: the two uncleared positions, one on
#: each side. An agent short of energy pays the retailer's *selling* price; one long is paid its
#: *buying* price. Both quantities are distinct from every other quantity in the book, so no two
#: agents' bills collide.
UNCLEARED_BID = 2000     # spare
UNCLEARED_OFFER = 2500   # hoarder
BALANCING_BOUGHT = BALANCING_SELLS * UNCLEARED_BID   # 1600 * 2000 = 3_200_000
BALANCING_SOLD = BALANCING_BUYS * UNCLEARED_OFFER    # 400 * 2500 = 1_000_000

# ----------------------------------------------------------------------------------------------
# What this delivery timestep already carries when it settles.
#
# **`buyer`'s history does two jobs.** It sold 1000 Wh at an earlier clearing of the same
# delivery step, so its gross flow is 4000 in and 1000 out while its net is 3000 in -- with a
# one-directional history the two would be the same number and `eec60f0` would be unfalsifiable.
# And it *bought* 1000 Wh from the retailer, so only 3000 of its 4000 Wh in came from the local
# market: a share of 0.75, the only fractional ratio in the fixture and the only thing that
# exercises the grid-fee blend anywhere other than at its endpoints.
#
# **`blender` exists only in the history**, and only to put a second fractional share on the
# other side of a half. It bought 3000 Wh on the market and 5000 from the retailer at earlier
# clearings of this delivery step and has nothing left uncleared, so it holds no book position and
# gets no `market` or `balancing` row -- just the two fee rows every agent in the netting gets.
# Its share is 3000/8000 = 0.375, blending to 681.25, which rounds *down*. `buyer`'s 0.75 blends
# to 512.5, which rounds *up*. One of each is needed: with only the .5 case, replacing `round()`
# with `ceil()` is an equivalent mutation, and with only a case below .5, `floor()` is.
#
# The retailer's matching row is included because a real retail trade writes both sides -- but
# note what it does and does not do here. Its money never enters the ledger the tests read: a
# seeded row reaches the market through `transactions_prev` only, and `__update_database`
# replaces `market_transactions` with this call's rows. It is there so that the fee exclusion has
# a retailer to exclude, and `test_the_retailer_is_not_charged_grid_fees_or_levies` is what that
# buys -- measured, removing the row from the filter fails two tests.
# ----------------------------------------------------------------------------------------------
HISTORY = [('buyer', c.TT_RETAIL, 0, 1000, 0, 500_000),
           ('buyer', c.TT_RETAIL, 1000, 0, 1_500_000, 0),
           (RETAILER_ID, c.TT_RETAIL, 1000, 0, 500_000, 0),
           (RETAILER_ID, c.TT_RETAIL, 0, 1000, 0, 1_500_000),
           ('blender', c.TT_MARKET, 3000, 0, 0, 0),
           ('blender', c.TT_RETAIL, 5000, 0, 0, 0)]

BLENDER_MARKET_IN = 3000
BLENDER_GROSS_IN = 8000

SOLD_EARLIER = 1000
BOUGHT_EARLIER = 1000
BUYER_GROSS_IN = CLEARED_ENERGY + BOUGHT_EARLIER      # 4000
BUYER_NET_IN = BUYER_GROSS_IN - SOLD_EARLIER          # 3000

#: Per-agent net energy at this delivery timestep, computed by hand from the book and the history.
NET = {'buyer': (BUYER_NET_IN, 0), 'spare': (UNCLEARED_BID, 0),
       'seller': (0, CLEARED_ENERGY), 'hoarder': (0, UNCLEARED_OFFER),
       'blender': (BLENDER_GROSS_IN, 0)}

# ----------------------------------------------------------------------------------------------
# The grid fee each agent pays, derived.
#
# `__apply_grid_levies` blends the two grid rates by the share of an agent's energy that came
# from the local market: `market_rate * ratio + retail_rate * (1 - ratio)`.
#
#   buyer   -- 4000 Wh in, 3000 of it `market`: share 0.75, so 400*0.75 + 850*0.25 = 512.5,
#              rounded **up** to 513.
#   blender -- 8000 Wh in, 3000 of it `market`: share 0.375, so 681.25, rounded **down** to 681.
#   spare   -- 2000 Wh in, all `balancing`: share 0, so the retail rate, 850.
#   seller  -- 3000 Wh out, all `market`: share 1, so the *feed-in* market rate, 40.
#   hoarder -- 2500 Wh out, all `balancing`: share 0, so the feed-in retail rate, 80.
#
# 513 is a hand-computed literal rather than the expression above, and
# `test_the_grid_rate_blends_the_two_rates_at_a_fractional_share` states that it is neither
# endpoint -- so an implementation that collapsed the blend to one side fails rather than
# coinciding. The blend lands on a **half**, so truncating instead of rounding fails too; see
# the note on the rates in `settlement_fixtures`.
# ----------------------------------------------------------------------------------------------
BUYER_MARKET_SHARE_IN = CLEARED_ENERGY / BUYER_GROSS_IN         # 0.75
BLENDER_MARKET_SHARE_IN = BLENDER_MARKET_IN / BLENDER_GROSS_IN  # 0.375
BUYER_GRID_RATE = 513
BLENDER_GRID_RATE = 681

GRID_RATE = {'buyer': BUYER_GRID_RATE, 'blender': BLENDER_GRID_RATE,
             'spare': GRID_RETAIL_OUT, 'seller': GRID_MARKET_IN, 'hoarder': GRID_RETAIL_IN}
LEVY_RATE = {'buyer': LEVIES_OUT, 'blender': LEVIES_OUT, 'spare': LEVIES_OUT,
             'seller': LEVIES_IN, 'hoarder': LEVIES_IN}

AGENTS = ('buyer', 'blender', 'spare', 'seller', 'hoarder')

#: Each agent's grid bill and levy, in EURe-7: its rate times its net quantity. The two factors
#: are independent -- the rate comes from the retailer file, the quantity from the book and the
#: history -- so multiplying them here is a hand computation rather than a restatement of what
#: the code does with them.
GRID_BILL = {agent: GRID_RATE[agent] * sum(NET[agent]) for agent in AGENTS}
LEVY_BILL = {agent: LEVY_RATE[agent] * sum(NET[agent]) for agent in AGENTS}

#: The two exporters' charges land in `price_out`. In a real tariff they would be zero -- the
#: shipped retailer files price feed-in at nothing -- and they are non-zero here only so the
#: column mapping is observable. See `settlement_fixtures`.
IMPORTERS = ('buyer', 'blender', 'spare')
EXPORTERS = ('seller', 'hoarder')

#: The identity every row this settlement writes must carry. `TIMESTAMP != TIMESTEP` by
#: construction, so a row stamped from the wrong one of the two is visible.
IDENTITY = {c.TC_TIMESTAMP: TIMESTAMP, c.TC_TIMESTEP: TIMESTEP, c.TC_REGION: REGION,
            c.TC_MARKET: MARKET, c.TC_NAME: NAME, c.TC_TYPE_ENERGY: ENERGY}


@pytest.fixture(scope='module')
def settled():
    """`BOOK` and `HISTORY` run through the real `execute()`, once for the whole module."""
    return settle(BOOK, history=HISTORY)


@pytest.fixture(scope='module')
def cleared_only():
    """The same book with `clear` and no `settle`, so the uncleared tables still have rows.

    `__determine_balancing_energy` ends with `bids_uncleared.clear()` -- once an uncleared
    position has become balancing energy it is not also an unmet order -- so after a
    `clear,settle` task both uncleared tables are empty. A clear-only task is the only shape in
    which they carry anything, and a `clear` task really does happen on its own: 552 of the 576
    tasks in `scenario_with_market`'s timetable are exactly that.
    """
    return settle(BOOK, history=HISTORY, actions=c.MA_CLEAR)


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
    assert (c.TT_MARKET, 'spare') not in book and (c.TT_MARKET, 'hoarder') not in book
    assert CLEARED_MONEY == CLEARING_PRICE * CLEARED_ENERGY


def test_settlement_has_a_position_on_each_side():
    """An uncleared bid **and** an uncleared offer, or half the balancing market is untested.

    A book with buyers only leaves every feed-in assertion multiplying by zero: the rate on that
    side can be read off the wrong retailer column, or the offer side can be dropped from the
    concatenation that builds balancing energy at all, and nothing notices. A review panel
    defeated the first version of this file both ways.
    """
    assert any(energy_in > 0 for _, energy_in, _, _, _ in BOOK)
    assert any(energy_out > 0 for _, _, energy_out, _, _ in BOOK)
    assert UNCLEARED_BID > 0 and UNCLEARED_OFFER > 0 and UNCLEARED_BID != UNCLEARED_OFFER


def test_two_market_shares_are_fractional_and_round_opposite_ways():
    """The grid blend is evaluated off its endpoints, and on both sides of a half.

    With every share at 0 or 1 the interpolation never runs at all. With one fractional share it
    runs, but only one of `floor` and `ceil` differs from `round` at any single value -- so a
    fixture with one is blind to the other. `buyer` blends to 512.5 (rounds up, so `floor`
    differs) and `blender` to 681.25 (rounds down, so `ceil` differs). Both survived earlier
    versions of this file, one at a time.
    """
    assert 0 < BUYER_MARKET_SHARE_IN < 1 and 0 < BLENDER_MARKET_SHARE_IN < 1
    for rate in (BUYER_GRID_RATE, BLENDER_GRID_RATE):
        assert rate not in (GRID_MARKET_OUT, GRID_RETAIL_OUT)
        assert min(GRID_MARKET_OUT, GRID_RETAIL_OUT) < rate < max(GRID_MARKET_OUT,
                                                                  GRID_RETAIL_OUT)
    # One blend sits above its floor and the other below its ceiling, which is what makes the
    # pair sensitive to the rounding mode rather than only to the interpolation.
    unrounded = {BUYER_GRID_RATE: GRID_RETAIL_OUT - 450 * BUYER_MARKET_SHARE_IN,
                 BLENDER_GRID_RATE: GRID_RETAIL_OUT - 450 * BLENDER_MARKET_SHARE_IN}
    assert any(rate > value for rate, value in unrounded.items()), 'no blend rounds up'
    assert any(rate < value for rate, value in unrounded.items()), 'no blend rounds down'


def test_no_two_agents_are_billed_the_same_amount():
    """Distinct bills, so a total credited to the wrong agent is visible rather than absorbed."""
    bills = [GRID_BILL[agent] + LEVY_BILL[agent] for agent in AGENTS]

    assert len(set(bills)) == len(bills)


def test_the_history_runs_the_other_way_from_this_clearing():
    """`buyer`'s two flows are in opposite directions, or netting is not being tested at all.

    A history that ran the same way as the clearing would make gross and net the same number, and
    then charging fees on gross -- `eec60f0` -- would be an equivalent mutation in this file. The
    same shape defeated an earlier fixture in this repository through a "don't-care" zero.
    """
    sold_earlier = sum(energy_out for agent, _, _, energy_out, _, _ in HISTORY if agent == 'buyer')
    bought_earlier = sum(energy_in for agent, _, energy_in, _, _, _ in HISTORY if agent == 'buyer')

    assert sold_earlier == SOLD_EARLIER > 0 and bought_earlier == BOUGHT_EARLIER > 0
    assert BUYER_GROSS_IN != BUYER_NET_IN, 'gross and net would coincide'


def test_every_charge_rate_in_the_fixture_is_distinct_and_non_zero():
    """Six rates, six different non-zero values, so every column mapping is falsifiable.

    The shipped retailer files charge 0 on feed-in. A fixture that copied them could not tell
    `grid_retail_in` from `grid_market_in`, both being zero, and a swap between the two would be
    an equivalent mutant -- which is exactly how a placeholder value faked equivalence on !250.
    """
    rates_ = (GRID_MARKET_IN, GRID_MARKET_OUT, GRID_RETAIL_IN, GRID_RETAIL_OUT,
              LEVIES_IN, LEVIES_OUT)

    assert len(set(rates_)) == len(rates_)
    assert all(rate > 0 for rate in rates_)


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
# Every row's identity. Nothing else in this file reads one, and two mutations lived there.
# ----------------------------------------------------------------------------------------------
def test_every_emitted_row_carries_this_clearings_identity(settled, cleared_only):
    """Timestamp, timestep, region, market, name and energy type, on every row of every kind.

    The rest of the file projects a row down to `(kind, agent, energies, money)`, which is what
    the accounting is about -- and that projection is exactly where a review panel got two
    mutations through. A fee row stamped with the clearing step instead of the delivery step, and
    a fee row written into a different market, both leave every quantity and every price correct
    and are invisible to a per-agent ledger. `get_net_energy` is keyed on `(timestep, id_agent)`
    and `_net_energy_from` filters on market and name, so either misfiling puts an agent's fees
    where nothing will find them again.

`TIMESTAMP != TIMESTEP` in the fixture, so a row stamped from the wrong column fails here
    rather than coinciding.

    **All five tables the market writes**, not only the transactions: the cleared and uncleared
    tables carry the same six columns and nothing else in this file reads one of them, so a
    region or market misfiled onto a *position* was invisible until this covered them. Every
    table asserted here is non-empty -- stated below, because four of the five would otherwise
    pass by being absent.

    The uncleared pair is read off the **clear-only** arm: settlement empties them, so on the
    `clear,settle` fixture they hold nothing. See `cleared_only`.
    """
    tables = {name: getattr(settled, name) for name in
              (c.TN_MARKET_TRANSACTIONS, c.TN_BIDS_CLEARED, c.TN_OFFERS_CLEARED)}
    tables.update({name: getattr(cleared_only, name) for name in
                   (c.TN_BIDS_UNCLEARED, c.TN_OFFERS_UNCLEARED)})

    for name, frame in tables.items():
        frame = frame.with_columns([pl.col(column).cast(pl.Utf8) for column, dtype
                                    in frame.schema.items() if dtype == pl.Categorical])
        assert len(frame) > 0, f'{name} is empty, so its identity is unasserted'
        for column, expected in IDENTITY.items():
            assert set(frame[column]) == {expected}, f"{name}.{column} is not this clearing's"


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

    Here `buyer` bought 3000 Wh at this clearing. Without it its net is the history alone: 1000
    in and 1000 out, which nets to nothing, so it would be billed for no energy at all.
    """
    energy_in, energy_out, price_in, price_out = book[(c.TT_GRID, 'buyer')]

    assert (energy_in, energy_out) == (BUYER_NET_IN, 0)
    assert (price_in, price_out) == (GRID_BILL['buyer'], 0)


def test_the_fee_energy_is_net_and_not_gross(book):
    """Regression, `eec60f0` part 1: fees were charged on gross energy, overcharging both sides.

    `buyer`'s gross flow at this timestep is 4000 Wh in *and* 1000 Wh out -- 5000 Wh of billable
    energy if both are charged, against the 3000 Wh it actually drew from the grid.
    """
    energy_in, energy_out, price_in, _ = book[(c.TT_GRID, 'buyer')]

    assert energy_in + energy_out == BUYER_NET_IN
    assert energy_in + energy_out < BUYER_GROSS_IN + SOLD_EARLIER
    assert price_in == GRID_RATE['buyer'] * BUYER_NET_IN


def test_only_one_direction_of_a_fee_row_is_ever_charged(book):
    """An agent cannot be a net importer and a net exporter in the same timestep."""
    for kind in (c.TT_GRID, c.TT_LEVIES):
        for agent in AGENTS:
            energy_in, energy_out, _, _ = book[(kind, agent)]
            assert energy_in == 0 or energy_out == 0


def test_the_retailer_is_not_charged_grid_fees_or_levies(book):
    """The retailer trades on this timestep -- `HISTORY` gives it a row -- and pays no fees.

    Its position is the market's supply of last resort, not a grid connection, and charging it
    would bill the tariff back to the tariff. `HISTORY` gives it rows on both sides.

    The exclusion is genuinely exercised -- the retailer is in `transactions_prev`, which is what
    `__apply_grid_levies` filters -- but note that the premise below reads the fixture constant,
    not the result: `settle()` returns only this call's rows, so the retailer's own `retail` row
    is not in the frame this asserts over. See `settlement_fixtures.settle`.
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


@pytest.mark.parametrize('agent', ['spare', 'hoarder'])
def test_the_balancing_rate_is_the_retailers_column_crossed(settled, agent):
    """The same crossing on balancing energy, on the two agents that carry a balancing row.

    An agent short of energy buys it at the retailer's selling price and one long is paid its
    buying price -- 1600 against 400 here. A swap makes being long of energy pay the retail rate,
    which is a risk-free arbitrage rather than a market.

    Asserted as the *pair*, not as the direction the agent happens to have traded: the unused
    column is multiplied by a zero quantity, so a test that only reads the used one cannot see a
    swap at all. That is precisely how this mutation survived the first version of this file.
    """
    assert rates(settled, c.TT_BALANCING)[agent] == (BALANCING_SELLS, BALANCING_BUYS)


@pytest.mark.parametrize('agent, rate', [('spare', GRID_RETAIL_OUT), ('hoarder', GRID_RETAIL_IN),
                                         ('seller', GRID_MARKET_IN)])
def test_the_grid_rate_is_the_forecasts_crossed_column(settled, agent, rate):
    """The same crossing for grid fees, on the path that is live, at the two endpoints.

    `__apply_grid_levies` seeds the four grid-rate columns from the retailer and then overwrites
    every one of them, for every agent, from that agent's own forecast of the variable grid fee.
    So the direction that decides an agent's bill is the `forecast_dir` in that loop, and it is
    the one asserted here. See `test_the_retailer_grid_rates_never_reach_the_output`.

    These three agents draw all of their energy from one side, so each reads one rate outright.
    `buyer` is the fractional case and has its own test.
    """
    charged_in, charged_out = rates(settled, c.TT_GRID)[agent]

    assert (charged_in if agent in IMPORTERS else charged_out) == rate


@pytest.mark.parametrize('agent, rate', [('buyer', BUYER_GRID_RATE),
                                         ('blender', BLENDER_GRID_RATE)])
def test_the_grid_rate_blends_the_two_rates_at_a_fractional_share(settled, agent, rate):
    """Two agents pay neither rate but a blend of them, rounding opposite ways.

    512.5 = 400 * 0.75 + 850 * 0.25 -> 513, and 681.25 = 400 * 0.375 + 850 * 0.625 -> 681. Both
    hand-computed and written as literals rather than as those expressions, which are the formula
    under test. Each is strictly between the endpoints, so collapsing the blend to either side
    fails; and having one on each side of a half is what makes **every** rounding mode
    observable -- `floor` moves 512.5 and leaves 681.25, `ceil` the reverse. All three survived
    earlier versions of this file.
    """
    charged_in, _ = rates(settled, c.TT_GRID)[agent]

    assert charged_in == rate
    assert charged_in not in (GRID_MARKET_OUT, GRID_RETAIL_OUT)
    assert rates(settled, c.TT_GRID)['spare'][0] == GRID_RETAIL_OUT, (
        'the endpoint importer now pays a blend too, so the contrast is gone')


@pytest.mark.parametrize('kind', [c.TT_GRID, c.TT_LEVIES])
@pytest.mark.parametrize('exporter', EXPORTERS)
@pytest.mark.parametrize('importer', IMPORTERS)
def test_a_charge_is_dearer_on_consumption_than_on_feed_in(settled, kind, importer, exporter):
    """`tests/README.md`'s tell, asserted on what the executor emitted rather than on the config.

    Grid fees and levies are both charged on consumption. If the rate charged to an agent drawing
    energy is not above the rate charged to one feeding it in, the in/out crossing has gone wrong
    somewhere between the retailer file and the transactions table.

    **Across two agents rather than across one agent's two columns**, and that is forced rather
    than stylistic: see
    `test_a_grid_rate_collapses_to_zero_where_the_agent_moved_no_energy_that_way`.
    """
    charged = rates(settled, kind)[importer][0]
    paid = rates(settled, kind)[exporter][1]

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
    assert charged['buyer'] == (BUYER_GRID_RATE, GRID_RETAIL_IN), 'gross, not net, on both sides'


def test_the_levy_is_a_flat_rate_and_is_not_blended(settled):
    """Every agent gets the same levy pair, whatever its market share.

    Unlike the grid fee, levies carry no market-versus-retail split. Stated as "all four agree"
    rather than by restating the two values, which
    `test_the_levy_rate_is_the_retailers_selling_column` already pins.
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

    Measured here by giving two agents a forecast that disagrees with the retailer in every
    direction: the forecast wins on all of them, including through the blend --
    4000 * 0.75 + 8000 * 0.25 = 5000. The consequence is that crossing or un-crossing the
    retailer-to-grid mapping in those four lines is an equivalent mutation in any scenario, which
    is worth knowing before trusting a mutation score for that function.
    """
    disagreeing = forecast_table(**{f'{c.TT_GRID}_{c.TT_MARKET}_{c.PF_OUT}': 4000,
                                    f'{c.TT_GRID}_{c.TT_RETAIL}_{c.PF_OUT}': 8000,
                                    f'{c.TT_GRID}_{c.TT_MARKET}_{c.PF_IN}': 1234})
    result = settle(BOOK, history=HISTORY,
                    forecasts={'buyer': disagreeing, 'seller': disagreeing})

    assert rates(result, c.TT_GRID)['buyer'][0] == 5000
    assert rates(result, c.TT_GRID)['seller'][1] == 1234
    # ... and the untouched agents still get the shared default, so this is the forecast winning
    # rather than the retailer column being ignored for everyone.
    assert rates(result, c.TT_GRID)['spare'][0] == GRID_RETAIL_OUT
    assert rates(result, c.TT_GRID)['hoarder'][1] == GRID_RETAIL_IN


# ----------------------------------------------------------------------------------------------
# Invariant 3 -- the balance closes.
# ----------------------------------------------------------------------------------------------
def test_every_charge_is_its_rate_times_its_quantity(settled):
    """A schema smoke test, **not** an anchor. Read this before counting it as one.

    Production computes `price_in` as `pl.col(price_pu_in) * pl.col(energy_in)` in so many words,
    so this assertion restates the code rather than checking it: it cannot see a wrong rate or a
    wrong quantity, only a cast or a schema slip that breaks the identity. Measured -- under a
    mutation that gave one agent the wrong blended rate, six tests failed and this was not among
    them.

    It earns its place because the identity is the one thing every kind of row shares, and a
    dtype change on this path would otherwise be silent. The anchor for invariant 3 is
    `test_what_each_agent_pays_is_the_hand_computed_total` and its two siblings.
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
    ('spare', BALANCING_BOUGHT + GRID_BILL['spare'] + LEVY_BILL['spare']),
])
def test_what_each_agent_pays_is_the_hand_computed_total(book, agent, expected):
    """Each buyer's whole bill for the timestep, to the euro, from the derivations above.

    Per agent by name and as an absolute figure -- the one form of this assertion that a defect
    crediting the right total to the wrong agent cannot satisfy. Measured: reversing the agent
    column of the per-agent sums leaves the closure test below green and fails this one.
    """
    paid = sum(book[key][2] for key in book if key[1] == agent)

    assert paid == expected


@pytest.mark.parametrize('agent, expected', [
    ('seller', CLEARED_MONEY + GRID_BILL['seller'] + LEVY_BILL['seller']),
    ('hoarder', BALANCING_SOLD + GRID_BILL['hoarder'] + LEVY_BILL['hoarder']),
])
def test_what_each_seller_receives_is_the_hand_computed_total(book, agent, expected):
    """The mirror, including the fee each is charged on feed-in in this fixture's tariff."""
    received = sum(book[key][3] for key in book if key[1] == agent)

    assert received == expected


def test_the_agent_to_agent_trades_cancel_and_the_residual_is_exactly_the_charges(book):
    """The closure: money paid less money received equals the named charges, with nothing over.

    The local trade cancels -- what `buyer` pays, `seller` receives -- so the whole table's
    imbalance has to be accounted for, euro for euro, by the rows that move money out of the
    agents' accounts: balancing, grid and levies. A residual would be money the market created or
    destroyed without saying so.

    **The anchor is that the charges are recomputed from rates and quantities**, not read back
    out of the same sum. And it is not on its own evidence: it stays green under a defect that
    credits every agent's total to a different agent, which is what the per-agent tests catch.
    """
    paid = sum(values[2] for values in book.values())
    received = sum(values[3] for values in book.values())

    leaves = BALANCING_BOUGHT + sum(GRID_BILL[a] + LEVY_BILL[a] for a in IMPORTERS)
    enters = BALANCING_SOLD + sum(GRID_BILL[a] + LEVY_BILL[a] for a in EXPORTERS)

    assert paid - received == leaves - enters
    assert book[(c.TT_MARKET, 'buyer')][2] == book[(c.TT_MARKET, 'seller')][3] == CLEARED_MONEY


def test_balancing_energy_is_bought_dear_and_sold_cheap(book):
    """The two uncleared positions become trades with the retailer, priced the expensive way.

    `spare` is short and pays 1600 per Wh; `hoarder` is long and is paid 400. A crossed mapping
    would pay an agent to be short of energy.
    """
    assert book[(c.TT_BALANCING, 'spare')] == (UNCLEARED_BID, 0, BALANCING_BOUGHT, 0)
    assert book[(c.TT_BALANCING, 'hoarder')] == (0, UNCLEARED_OFFER, 0, BALANCING_SOLD)
    assert BALANCING_BOUGHT / UNCLEARED_BID > BALANCING_SOLD / UNCLEARED_OFFER


def test_the_balancing_energy_is_the_submitted_position_less_what_cleared(book):
    """Anchored on the book, which is outside the settlement entirely.

    Settlement takes the uncleared remainder from the clearing and the clearing takes it from
    the submitted positions, so the quantity finally balanced has to be what each agent asked for
    less what it got. Comparing it to the book rather than to the clearing's own uncleared table
    is what makes this an absolute check: an error shared by the clearing and the settlement --
    the shape that let #241's duplication pass a conservation invariant -- is visible against the
    submitted quantity and invisible against anything derived from it.

    **Both directions**, because the first version compared only `energy_in` and the offer book
    was therefore anchored on nothing: dropping the uncleared offers from balancing energy
    altogether left it green.
    """
    submitted_in = {agent: energy_in for agent, energy_in, _, _, _ in BOOK if energy_in}
    submitted_out = {agent: energy_out for agent, _, energy_out, _, _ in BOOK if energy_out}
    cleared_in = {agent: values[0] for (kind, agent), values in book.items()
                  if kind == c.TT_MARKET}
    cleared_out = {agent: values[1] for (kind, agent), values in book.items()
                   if kind == c.TT_MARKET}
    balanced = {agent: values[:2] for (kind, agent), values in book.items()
                if kind == c.TT_BALANCING}

    for agent, submitted in submitted_in.items():
        assert balanced.get(agent, (0, 0))[0] == submitted - cleared_in.get(agent, 0), agent
    for agent, submitted in submitted_out.items():
        assert balanced.get(agent, (0, 0))[1] == submitted - cleared_out.get(agent, 0), agent
    assert sum(sum(values) for values in balanced.values()) > 0, 'nothing was balanced'


# ----------------------------------------------------------------------------------------------
# The same ledger, reached by the task shape every shipped run actually uses.
#
# **`clear,settle` in one task -- the shape above -- occurs in no shipped scenario.** Measured on
# `scenario_with_market` by instrumenting `_transactions_for_netting` through a real run: its
# timetable holds 552 `clear` tasks and 24 `settle` tasks and not one of both, and on all 24
# calls `self.transactions` was empty. It is unreachable rather than merely absent: with
# `settling: continuous`, `_create_timetable_ex_ante` appends `,settle` to the rows whose
# timestep has already arrived (`timestep <= timestamp`) and then overwrites every row with
# `timestep - timestamp < closing` -- a superset of them, for any positive `closing`, since
# `timestep <= timestamp` means the difference is at most zero. So the combination cannot
# survive, whatever `closing` is set to.
#
# Two consequences, both worth stating rather than leaving to be rediscovered:
#
# * `a860ca6` -- the current clearing missing from the netting -- **cannot be observed on any
#   shipped scenario**, because there is no clearing in flight when settlement runs; the trades
#   are already in `transactions_prev`, put there by an earlier task. Measured: reverting that
#   fix leaves the whole `tests/e2e/test_market_clearing.py` suite green, and only the unit test
#   above fails. The fix is still right, and this file is the only thing that holds it.
# * The arm above is therefore the *rare* shape. This section runs the common one: settlement
#   with the clearing already settled into history and only the uncleared remainder in the book.
#
# The two must produce the same fees, because they describe the same timestep. Asserted as
# equality against the other arm's hand-computed figures, not against the other arm's output.
# ----------------------------------------------------------------------------------------------
SETTLED_BOOK = [('spare', UNCLEARED_BID, 0, 1000, RETAILER_BUYS),
                ('hoarder', 0, UNCLEARED_OFFER, RETAILER_SELLS, 4000)]
SETTLED_HISTORY = HISTORY + [('buyer', c.TT_MARKET, CLEARED_ENERGY, 0, CLEARED_MONEY, 0),
                             ('seller', c.TT_MARKET, 0, CLEARED_ENERGY, 0, CLEARED_MONEY)]


@pytest.fixture(scope='module')
def settled_only():
    """Settlement as a task of its own, the clearing already in the table."""
    return settle(SETTLED_BOOK, history=SETTLED_HISTORY, actions=c.MA_SETTLE)


@pytest.mark.parametrize('agent', AGENTS)
@pytest.mark.parametrize('kind', [c.TT_GRID, c.TT_LEVIES])
def test_a_settle_only_task_charges_the_same_fees(settled_only, kind, agent):
    """The common task shape bills exactly what the `clear,settle` shape bills.

    Same timestep, same trades, the only difference being which task carried the clearing -- so
    every figure has to match, per agent by name. This is the arm that covers what the shipped
    scenarios run; the one above covers the arm where `a860ca6` is live.
    """
    charged = {(kind_, agent_): tuple(values) for kind_, agent_, *values in rows(settled_only)}
    energy_in, energy_out, price_in, price_out = charged[(kind, agent)]
    rate = GRID_RATE[agent] if kind == c.TT_GRID else LEVY_RATE[agent]

    assert (energy_in, energy_out) == NET[agent]
    assert (price_in, price_out) == (rate * energy_in, rate * energy_out)


def test_the_settle_only_task_really_had_no_clearing_in_flight(settled_only):
    """The premise: this arm is the one where `self.transactions` is empty at netting time.

    Without it the two arms would be the same test twice. The clearing's own rows are in the
    history here, and settlement writes only the fee and balancing rows.
    """
    written = {kind for kind, *_ in rows(settled_only)}

    assert written == {c.TT_GRID, c.TT_LEVIES, c.TT_BALANCING}
    assert c.TT_MARKET not in written, 'this arm cleared, so it is not the settle-only shape'


def test_settlement_empties_the_uncleared_tables(settled, cleared_only):
    """An uncleared position becomes balancing energy and stops being an unmet order.

    Pinned because `test_every_emitted_row_carries_this_clearings_identity` reads the uncleared
    pair off the clear-only arm and would silently cover nothing if that stopped being true.
    """
    for name in (c.TN_BIDS_UNCLEARED, c.TN_OFFERS_UNCLEARED):
        assert len(getattr(cleared_only, name)) > 0, f'{name} is empty before settlement'
        assert len(getattr(settled, name)) == 0, f'{name} survived settlement'


# ----------------------------------------------------------------------------------------------
# The same accounting under `method: None`, which is what every other shipped scenario runs.
#
# `scenario_with_market` is the only shipped scenario with a clearing method. Five set
# `method: None` -- the three other examples and both e2e fixtures -- so the settlement above
# covers the clearing the suite runs *once* and this section covers the one it runs everywhere
# else. (`config_templates/markets.yaml` sets `pda` on both of its markets, so a new user starts
# with a clearing; it is the shipped *scenarios* that do not.)
#
# The difference reaches the bill. `__method_none` types every trade `retail`, so no agent has a
# local-market share and the grid fee is the retail rate for everyone -- 850 rather than the 513
# `buyer` pays above, on the same 3000 Wh. The retailer needs a quantity here, or
# `__split_bids_offers` drops it out of the book and nothing has a counterparty.
#
#   buyer  buys 3000 Wh at the retailer's selling price, 1500  -> 4_500_000
#   seller sells 2000 Wh at the retailer's buying price, 500   -> 1_000_000
#   buyer  nets 3000 + 1000 bought - 1000 sold = 3000 Wh in
#   seller nets 2000 Wh out
# ----------------------------------------------------------------------------------------------
RETAIL_BOOK = [('buyer', 3000, 0, RETAILER_SELLS, RETAILER_BUYS),
               ('seller', 0, 2000, RETAILER_SELLS, RETAILER_BUYS)]

RETAIL_BOUGHT = RETAILER_SELLS * 3000  # 4_500_000
RETAIL_SOLD = RETAILER_BUYS * 2000     # 1_000_000
RETAIL_NET = {'buyer': (BUYER_NET_IN, 0), 'seller': (0, 2000)}


@pytest.fixture(scope='module')
def retail_only():
    """The same shape of timestep with no local clearing: everything goes to the retailer."""
    return settle(RETAIL_BOOK, history=HISTORY, method=c.MCM_NONE,
                  retailer=retailer_table(energy_in=100_000, energy_out=100_000))


def test_method_none_routes_both_sides_through_the_retailer_and_prices_them_apart(retail_only):
    """Buying costs 1500 and selling pays 500, on the same timestep, in the same table.

    `tests/README.md`'s tell, on the clearing method every shipped scenario but one uses: if the
    two ever came out the other way round the in/out crossing has gone wrong, and it would be an
    arbitrage rather than a market.
    """
    ledger_ = {(kind, agent): tuple(values) for kind, agent, *values in rows(retail_only)
               if kind == c.TT_RETAIL and agent != RETAILER_ID}

    assert ledger_[(c.TT_RETAIL, 'buyer')] == (3000, None, RETAIL_BOUGHT, None)
    assert ledger_[(c.TT_RETAIL, 'seller')] == (None, 2000, None, RETAIL_SOLD)
    assert RETAIL_BOUGHT / 3000 > RETAIL_SOLD / 2000


@pytest.mark.parametrize('agent', ['buyer', 'seller'])
def test_method_none_charges_the_retail_grid_rate_because_nothing_is_local(retail_only, agent):
    """With no local market there is no market share, so the blend is the retail rate outright.

    `buyer` draws the same 3000 Wh as in the clearing arm and pays 850 per Wh here against 513
    there -- the one number that distinguishes the two clearing methods' accounting, asserted as
    an absolute value rather than as "different".
    """
    charged = {(kind, agent_): tuple(values) for kind, agent_, *values in rows(retail_only)}
    expected_rate = GRID_RETAIL_OUT if agent == 'buyer' else GRID_RETAIL_IN
    energy_in, energy_out, price_in, price_out = charged[(c.TT_GRID, agent)]

    assert (energy_in, energy_out) == RETAIL_NET[agent]
    assert (price_in, price_out) == (expected_rate * energy_in, expected_rate * energy_out)
    if agent == 'buyer':
        assert expected_rate != BUYER_GRID_RATE, 'the two clearing methods now bill alike'


@pytest.mark.parametrize('name, id_column', [('bids_cleared', c.TC_ID_AGENT_IN),
                                             ('offers_cleared', c.TC_ID_AGENT_OUT)])
def test_the_retailers_own_row_is_stamped_with_the_clearing_not_the_delivery(retail_only, name,
                                                                            id_column):
    """The retailer's ladder row carries `(clearing time, delivery step)` like everyone else.

    `self.retailer` is filtered on `timestamp == the delivery step`, the retailer table being
    indexed by delivery time under that name, and `__create_bids_offers` used to alias that one
    column into both -- so the retailer's row was the only row in each table whose
    `(timestamp, timestep)` pair was not a clearing. The pair is the clearing key: it groups a
    clearing in `test_market_clearing.py` and namespaces the trade id in `__postprocess_trades`.

    **This is the fast-tier half of that regression, and it exists because the e2e half is not
    enough.** Before this test, reverting the fix left the whole default tier green -- 1104 of
    the current 1106 pass under the revert, and the two that do not are this test's two
    parameters. Only two `e2e` tests caught it, and `pytest.ini` deselects them by default. It
    also covers `method: None`, which the e2e test never reaches and which is what every shipped
    scenario but one runs.

    `method: None` is what puts the retailer into the cleared tables at all: it is the
    counterparty to every trade, so it holds a row on both sides. Under `pda` with the book above
    it clears nothing and the defect is unobservable, which is why this test lives here.
    """
    frame = getattr(retail_only, name).with_columns(pl.col(id_column).cast(pl.Utf8))
    retailer_rows = frame.filter(pl.col(id_column) == RETAILER_ID)

    assert len(retailer_rows) > 0, f'the retailer holds no row in {name}, so this checks nothing'
    assert set(retailer_rows[c.TC_TIMESTAMP]) == {TIMESTAMP}
    assert set(retailer_rows[c.TC_TIMESTEP]) == {TIMESTEP}
    # Both halves by value, because a mutation that swapped the two would keep every row's pair
    # inside the set of times this fixture uses.
    assert TIMESTAMP != TIMESTEP and frame.filter(
        pl.col(c.TC_TIMESTAMP) >= pl.col(c.TC_TIMESTEP)).is_empty()


def test_method_none_closes_its_money_ledger(retail_only):
    """Every euro an agent pays the retailer, the retailer receives, and vice versa.

    The retailer holds the other side of both trades here, so the `retail` rows cancel exactly
    and the residual is the fees -- recomputed from rates and quantities rather than read back
    out of the same sum.
    """
    trades = [values for kind, _, *values in rows(retail_only) if kind == c.TT_RETAIL]
    paid = sum(row[2] or 0 for row in trades)
    received = sum(row[3] or 0 for row in trades)

    assert paid == received == RETAIL_BOUGHT + RETAIL_SOLD

    fees = {(kind, agent): tuple(values) for kind, agent, *values in rows(retail_only)
            if kind in (c.TT_GRID, c.TT_LEVIES)}
    # Per agent rather than as a lump: `blender`'s market history survives into this arm, so it
    # is the one agent here that still pays a blended rate, and a total would hide which is which.
    expected = {('grid', 'buyer'): GRID_RETAIL_OUT * BUYER_NET_IN,
                ('levies', 'buyer'): LEVIES_OUT * BUYER_NET_IN,
                ('grid', 'blender'): BLENDER_GRID_RATE * BLENDER_GROSS_IN,
                ('levies', 'blender'): LEVIES_OUT * BLENDER_GROSS_IN}
    assert {key: values[2] for key, values in fees.items() if key[1] != 'seller'} == expected
    assert fees[('grid', 'seller')][3] == GRID_RETAIL_IN * 2000
    assert fees[('levies', 'seller')][3] == LEVIES_IN * 2000


def test_an_empty_timestep_settles_without_raising_and_writes_nothing():
    """No positions at all: `__apply_grid_levies` returns its two empty frames and nothing breaks.

    The branch exists because a timestep can have nothing to settle, and an exception there would
    stop a run rather than mis-bill it. Nothing else reaches it -- every other book in this file
    and in `test_clearing_by_hand.py` has positions.
    """
    result = settle([], history=[])

    assert rows(result) == []


def rates(market, kind):
    """`{agent: (price_pu_in, price_pu_out)}` for one kind of transaction row."""
    frame = market.market_transactions.with_columns(
        pl.col(c.TC_TYPE_TRANSACTION).cast(pl.Utf8), pl.col(c.TC_ID_AGENT).cast(pl.Utf8))
    frame = frame.filter(pl.col(c.TC_TYPE_TRANSACTION) == kind)
    assert len(frame) > 0, f'no {kind} rows were written, so this asserts about nothing'
    return {row[c.TC_ID_AGENT]: (row[c.TC_PRICE_PU_IN], row[c.TC_PRICE_PU_OUT])
            for row in frame.to_dicts()}
