"""End to end -- the one shipped example that actually clears a market.

**Every other scenario in the repository sets `method: None`.** Five do -- the three other
examples and both e2e fixtures -- so until this file, the market-clearing branch of
`executor/markets/electricity.py` had never executed in the suite, and neither had two of the
three trading strategies (#223). `examples/create_scenario_with_market` ships `method: pda` and
`strategy: linear`, and no test ran it (#214 closed the config half of that: its `agents.yaml`
and `agents.xlsx` now agree, but only their *text* is read).

**What this file adds over the unit tests, which is the whole reason to pay for a run.**
`tests/unit/executor/markets/test_clearing_by_hand.py` pins what the clearing arithmetic computes
from a given book, exactly, against hand-derived numbers. It cannot say whether the executor ever
reaches that code, whether the books it builds are ones that clear, or which trading strategy
produced them. Those are integration properties and this is where they are settled:

* `test_agents_traded_with_each_other` -- the local market cleared agent against agent, not
  merely agent against retailer. Without it, a run in which every position went to the retailer
  would look identical to a working local market at the level of "the path did not raise".
* `test_the_configured_linear_strategy_actually_ran` -- the answer to #223's second half. The
  substitution in `AgentBase.create_bids_offers` applies only when a market has no clearing; this
  market has one, so `Linear` should be what quotes. **The evidence is a run-time receipt, not the
  prices in the results**, and the first version of this file got that wrong: see the test's own
  docstring for the measurement that killed the price argument.
* the two conservation invariants, over a real run of hundreds of clearings rather than a handful
  of hand-written books.

**It is a run, so it is marked `e2e`.** `pytest tests/e2e/test_market_clearing.py` on its own
selects *nothing*: `pytest.ini` sets `addopts = -m "not e2e and not golden and not benchmark"`
and the deselection applies even when the file is named. Pass `-m e2e`.

**No numbers are pinned here beyond floors**, deliberately. This is not a golden master: the
scenario has no committed reference and adding one is a separate decision. What is asserted are
properties that survive a model change but not a broken clearing -- balance, uniformity of price
within a clearing, and the presence of each transaction kind. Where a floor is used it is an
absolute one with the measured value quoted beside it, so a run that collapsed to a tenth of the
trading it does today fails rather than passing on "> 0".

**A floor is one-sided, and this file cannot catch a defect that moves the numbers up.** Measured:
deleting the subtraction of already-traded energy in `_preprocess_bids_offers` triples the local
trading and every assertion here still passes. The unit tests are what catch that, by value. Read
this file as evidence that the clearing *runs and holds together*, not as evidence that its
numbers are right.
"""
from pathlib import Path

import polars as pl
import pytest
from ruamel.yaml import YAML

import hamlet.constants as c
from tests.scenario_run import REPO_ROOT, assert_strategies_honoured

EXAMPLE = REPO_ROOT / 'examples' / 'create_scenario_with_market'
SCENARIO = 'scenario_with_market'
CONFIG = EXAMPLE / SCENARIO

#: The entry point `examples/create_scenario_with_market/run.ipynb` calls. It reads `agents.xlsx`
#: rather than `agents.yaml`, so this is the file the run is actually built from -- naming the
#: other one would test a scenario the shipped notebook does not produce.
CREATOR_METHOD = 'new_scenario_from_files'

#: The market's name in `markets.yaml`, and the retailer's id. Both appear in the result tables.
MARKET_NAME = 'continuous'
RETAILER = 'retailer'

#: The retailer's two fixed prices for this scenario, in EURe-7/Wh. Read back out of
#: `markets.yaml` by `test_the_retail_band_is_what_this_file_assumes`, so the scenario cannot
#: quietly change tariff while this file goes on describing the old one.
RETAILER_BUYS = 500
RETAILER_SELLS = 1500

#: **Measured** on this scenario through `run_example` with `new_scenario_from_files` and seed
#: 20260804, on 2026-08-25: 245 cleared bids and 245 cleared offers over **79 clearings**,
#: totalling 138_059 Wh on each side; **242** agent-to-agent transaction rows carrying
#: **53_482 Wh**; 52 uncleared bids and 34 uncleared offers.
#:
#: **Floors only catch a collapse downwards, and that is a real limit rather than a quibble.**
#: Measured: deleting the subtraction of already-traded energy in `_preprocess_bids_offers`
#: triples the local trading (53_482 -> 165_969 Wh) and every assertion in this file still
#: passes. `tests/unit/executor/utilities/trading/test_strategies.py::
#: test_energy_already_traded_is_netted_out_of_the_position` is what catches that, by value.
#:
#: The floors below sit at roughly a third of each. That band is deliberate: it tolerates a model
#: change that shifts the dispatch -- #234 moved every number in `grid_golden` -- while failing a
#: clearing that has stopped matching agents to each other, which is what these tests exist to
#: notice. **They are floors, not the measurements**; read the paragraph above for the current
#: values and do not tighten a floor to one of them.
MIN_CLEARED_BIDS = 80
MIN_CLEARINGS = 25
MIN_LOCAL_TRANSACTIONS = 80
MIN_LOCAL_ENERGY_WH = 17_000


@pytest.fixture(scope='module')
def run(scenario_runs):
    """The example, run once, with a receipt of which trading strategies it constructed."""
    return scenario_runs.run(EXAMPLE, SCENARIO, creator_method=CREATOR_METHOD,
                             needs_strategies=True)


@pytest.fixture(scope='module')
def results(run):
    """The market's result directory from that run."""
    return run.results / 'markets' / c.MT_ELECTRICITY / MARKET_NAME


def table(results, name):
    """One market result table, with every Categorical cast to `Utf8`.

    The cast is not cosmetic: polars compares two Categorical columns by their *encoding* unless
    they share a string cache, so `pl.col(x) == 'retailer'` on a Categorical read back from
    Feather is a comparison this file must not make.
    """
    frame = pl.read_ipc(results / f'{name}.ft', memory_map=False)
    return frame.with_columns(
        [pl.col(column).cast(pl.Utf8) for column, dtype in frame.schema.items()
         if dtype == pl.Categorical])


# ----------------------------------------------------------------------------------------------
# Premises. Each one is something the tests below would pass vacuously without.
#
# **Marked `e2e` even though they need no run.** They cost milliseconds and would sit happily in
# the fast tier -- but `pytest.ini` deselects `e2e` by default, so an unmarked premise runs in
# exactly the invocation where its test does *not*, and `-m e2e` -- the invocation this file's
# docstring tells you to use -- would run all eight assertions and not one of the three premises
# that keep them honest. A premise that never runs beside the thing it guards is decoration.
# ----------------------------------------------------------------------------------------------
@pytest.mark.e2e
def test_this_scenario_is_the_one_that_clears():
    """`markets.yaml` selects `pda` and `uniform`, and that is why this file exists.

    Read rather than assumed: if the shipped example were switched to `method: None`, every
    assertion below would still be *satisfiable* -- by a run in which the retailer took every
    position -- and this file would quietly stop covering the clearing it was written for.
    """
    config = YAML(typ='safe').load((CONFIG / 'markets.yaml').read_text(encoding='utf-8'))
    clearing = config[MARKET_NAME]['clearing']

    assert clearing['method'] == c.MCM_PDA
    assert clearing['pricing'] == c.MP_UNIFORM
    assert clearing['type'] == c.MCT_EX_ANTE


@pytest.mark.e2e
def test_the_scenario_configures_the_linear_strategy():
    """Both files say `linear`; the workbook is the one this run is built from.

    #214 made the two agree, and `test_shipped_configs_agree_with_their_workbooks` keeps them
    that way for the backend keys. `strategy` is not among those keys, so it is checked here in
    the file that depends on it.
    """
    import pandas as pd

    sheets = pd.read_excel(CONFIG / 'agents.xlsx', sheet_name=None)
    declared = {name: set(sheet['ems/market/strategy'].dropna())
                for name, sheet in sheets.items()
                if 'ems/market/strategy' in sheet.columns}

    assert declared, 'no sheet in agents.xlsx declares a trading strategy'
    assert set().union(*declared.values()) == {c.MTS_LINEAR}


@pytest.mark.e2e
def test_the_retail_band_is_what_this_file_assumes():
    """The two retailer prices, read out of `markets.yaml` in the unit the executor uses.

    `markets.yaml` writes them in EUR/kWh as `[selling, buying]` from the agent's point of view;
    the Creator scales by `EUR_KWH_TO_EURe7_WH`. It also checks the direction, which is the half
    that would go wrong silently: `_out` must be the dearer of the two, or the agent is being
    paid more for feed-in than it pays for supply and the whole scenario is an arbitrage.

    Nothing in this file now reasons *from* the band -- the strategy receipt replaced that -- so
    this is a sanity check on the fixture rather than a premise anything depends on.
    """
    config = YAML(typ='safe').load((CONFIG / 'markets.yaml').read_text(encoding='utf-8'))
    sell, buy = config[MARKET_NAME]['pricing'][RETAILER][c.TT_ENERGY]['fixed'][c.TC_PRICE]

    assert int(sell * c.EUR_KWH_TO_EURe7_WH) == RETAILER_BUYS
    assert int(buy * c.EUR_KWH_TO_EURe7_WH) == RETAILER_SELLS
    assert RETAILER_BUYS < RETAILER_SELLS


# ----------------------------------------------------------------------------------------------
# The clearing itself.
# ----------------------------------------------------------------------------------------------
@pytest.mark.e2e
def test_the_market_cleared_at_all(results):
    """Cleared and uncleared tables both have rows, so both branches of the ladder executed.

    The uncleared half matters as much as the cleared one: a market that matched *everything*
    would never exercise `__create_market_tables`' subtraction, which is what
    `test_uncleared_is_submitted_minus_cleared` pins by value at unit level.
    """
    assert len(table(results, c.TN_BIDS_CLEARED)) >= MIN_CLEARED_BIDS
    assert len(table(results, c.TN_OFFERS_CLEARED)) >= MIN_CLEARED_BIDS
    assert len(table(results, c.TN_BIDS_UNCLEARED)) > 0
    assert len(table(results, c.TN_OFFERS_UNCLEARED)) > 0


@pytest.mark.e2e
def test_agents_traded_with_each_other(results):
    """The point of a local energy market: `market` transactions, not only `retail` ones.

    `__postprocess_trades` types a trade `retail` when either side is the retailer and `market`
    otherwise, so this counts the trades the local clearing actually matched between agents. A
    run where the retailer absorbed everything -- which is what `method: None` produces -- has
    none of these, and would satisfy every other assertion in this file.
    """
    transactions = table(results, c.TN_MARKET_TRANSACTIONS)
    local = transactions.filter(pl.col(c.TC_TYPE_TRANSACTION) == c.TT_MARKET)

    assert len(local) >= MIN_LOCAL_TRANSACTIONS
    traded = local[c.TC_ENERGY_IN].fill_null(0).sum()
    assert traded >= MIN_LOCAL_ENERGY_WH

    # Both directions are present, and they balance: a local trade writes one row for the buyer
    # and one for the seller.
    assert local[c.TC_ENERGY_IN].fill_null(0).sum() == local[c.TC_ENERGY_OUT].fill_null(0).sum()

    # And the counterparty typing is not a constant: retail trades exist too.
    assert len(transactions.filter(pl.col(c.TC_TYPE_TRANSACTION) == c.TT_RETAIL)) > 0


@pytest.mark.e2e
def test_no_local_trade_names_the_retailer(results):
    """A `market` transaction is agent-to-agent by definition, so the retailer must not appear.

    This is the assertion that fails if `__postprocess_trades` typed every trade the same way,
    which the count above would not notice.
    """
    local = (table(results, c.TN_MARKET_TRANSACTIONS)
             .filter(pl.col(c.TC_TYPE_TRANSACTION) == c.TT_MARKET))

    assert RETAILER not in set(local[c.TC_ID_AGENT])


@pytest.mark.e2e
def test_every_clearing_settled_at_a_single_price(results):
    """Uniform pricing, end to end: one price per clearing, over every clearing in the run.

    A *clearing* is one `(timestamp, timestep)` pair -- a delivery timestep is cleared repeatedly
    as the trading horizon rolls towards it, and each of those clearings has its own price, so
    grouping by `timestep` alone finds many prices and says nothing. Asserted separately for the
    two sides because they are priced by two separate calls into `self.pricing[...]`.
    """
    for name, column in ((c.TN_BIDS_CLEARED, c.TC_PRICE_PU_IN),
                         (c.TN_OFFERS_CLEARED, c.TC_PRICE_PU_OUT)):
        per_clearing = (table(results, name)
                        .group_by([c.TC_TIMESTAMP, c.TC_TIMESTEP])
                        .agg(pl.col(column).n_unique().alias('prices')))

        assert len(per_clearing) >= MIN_CLEARINGS, (
            f'{name}: too few clearings for this to mean anything')
        assert per_clearing['prices'].max() == 1, (
            f'{name}: a clearing settled at more than one price, so pricing was not uniform')


@pytest.mark.e2e
def test_the_cleared_energy_balances_at_every_timestep(results):
    """Sum of cleared energy in equals sum of cleared energy out, per delivery timestep.

    Per timestep rather than over the run, because a run total can hide an excess at one timestep
    cancelling a shortfall at another.
    """
    bought = (table(results, c.TN_BIDS_CLEARED)
              .group_by(c.TC_TIMESTEP).agg(pl.col(c.TC_ENERGY_IN).sum()))
    sold = (table(results, c.TN_OFFERS_CLEARED)
            .group_by(c.TC_TIMESTEP).agg(pl.col(c.TC_ENERGY_OUT).sum()))
    paired = bought.join(sold, on=c.TC_TIMESTEP, how='outer', coalesce=True)

    assert len(paired) == len(bought) == len(sold), 'a timestep cleared on one side only'
    assert paired.filter(pl.col(c.TC_ENERGY_IN) != pl.col(c.TC_ENERGY_OUT)).is_empty()
    assert bought[c.TC_ENERGY_IN].sum() > 0


@pytest.mark.e2e
def test_the_configured_linear_strategy_actually_ran(run):
    """#223's second half, settled by a receipt rather than by inference.

    The run records every trading-strategy class it constructed. `Linear` and nothing else means
    the configured strategy reached the market: `AgentBase.create_bids_offers` substitutes
    `Retailer` whenever a market has no clearing, so a `Retailer` in this set would mean the
    substitution fired in a scenario that does clear.

    **An earlier version of this test inferred it from prices, and was wrong.** The reasoning was
    that `Retailer` can only quote the two retailer prices, so a cleared bid priced strictly
    between them must have come from `Linear`. Two things break it. The price on a *cleared*
    position is the uniform clearing price, not the submitted one -- `__create_market_tables`
    renames `price_pu` over `price_pu_in` -- and the forecast prices the strategies quote against
    are themselves forecasts of the local market, not the flat tariff. Measured by running this
    scenario with the workbook switched to `strategy: retailer`: **50 cleared bids priced
    strictly inside the band**, and every other assertion in this file passed too. Only a
    secondary check on the number of *distinct* interior prices caught it, by one value. The
    receipt is not a stronger version of that argument; it replaces it.
    """
    assert_strategies_honoured(run.strategies, {'Linear'})


@pytest.mark.e2e
def test_uncleared_positions_are_never_the_retailers(results):
    """`__create_market_tables` drops the retailer from both uncleared tables.

    Its quantity is a capacity rather than an unmet position, and leaving it in would report the
    retailer as holding ~100 kWh of unfilled orders at every timestep.
    """
    assert RETAILER not in set(table(results, c.TN_BIDS_UNCLEARED)[c.TC_ID_AGENT_IN])
    assert RETAILER not in set(table(results, c.TN_OFFERS_UNCLEARED)[c.TC_ID_AGENT_OUT])


@pytest.mark.e2e
def test_the_settlement_stage_ran_and_charged_fees(results):
    """`clear` is only half the market's work; `settle` is what turns uncleared positions into
    balancing energy and applies grid fees and levies.

    Included because this run is the only one in the suite where settlement follows a *clearing*
    rather than a retailer-only pass, and because it is cheap: the same table is already read.
    """
    kinds = set(table(results, c.TN_MARKET_TRANSACTIONS)[c.TC_TYPE_TRANSACTION])

    assert {c.TT_MARKET, c.TT_RETAIL, c.TT_BALANCING, c.TT_GRID, c.TT_LEVIES} <= kinds
