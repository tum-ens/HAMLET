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

**The example used to ship `solver: gurobi`**, so as shipped it could not run without a
commercial licence -- which is why it had never run in CI, and which the first pipeline for this
branch discovered. It is on HiGHS now, and `test_the_clearing_example_needs_no_commercial_licence`
below is what keeps it there.

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
from hamlet.executor.utilities.database.market_db import MarketDB
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
#: 20260804, on HiGHS, on 2026-09-03: 257 cleared bids and 257 cleared offers, both over
#: **88 clearings**, carrying 126_762 Wh on each side; **274** agent-to-agent transaction rows
#: carrying **53_278 Wh**; 53 uncleared bids and 27 uncleared offers; 868 transactions in all,
#: of which 240 `retail`, 36 `balancing` and 159 each of `grid` and `levies`. Quote these only
#: for the HiGHS arm -- the same scenario on Gurobi produced different figures for every one of
#: them, so the floors have to clear the smaller side.
#:
#: **Both earlier records of these numbers were stale, for two separate reasons.** The figures
#: this block used to carry (267 / 267, 94 and 90 clearings, 139_876 Wh, 286 rows, 62_410 Wh)
#: were measured on 2026-08-26, before #241 stopped `__method_pda` trading one position's whole
#: quantity in every ladder segment it spanned -- which is why the cleared quantities came down.
#: And the two clearing counts differed because the retailer's ladder row was stamped
#: `(delivery, delivery)` rather than `(clearing, delivery)`, so it formed groups of its own.
#: Measured on this branch's parent: **93** pairs in `bids_cleared` against **89** in
#: `offers_cleared`, of which **88** were clearings; the retailer held 25 rows in `bids_cleared`
#: and 95 in `offers_cleared`. With the stamp fixed the two sides agree at 88, and
#: `test_both_sides_clear_at_the_same_set_of_clearings` is what keeps them agreeing.
#: `MIN_CLEARINGS` therefore now counts clearings rather than groups.
#:
#: **This is the only copy of these figures outside `CHANGELOG.md`**, which records them once
#: as a release note. `electricity.py` and the tests below point here rather than restating
#: them: three records of one measurement, two of them stale, is what this block replaced.
#:
#: **Floors only catch a collapse downwards, and that is a real limit rather than a quibble.**
#: Measured: deleting the subtraction of already-traded energy in `_preprocess_bids_offers`
#: triples the local trading -- measured at the time as 53_482 -> 165_969 Wh, on the Gurobi arm
#: whose defining sentence this block no longer carries; the ratio is the point, not the figures
#: -- and every assertion in this file still passes. `tests/unit/executor/utilities/trading/test_strategies.py::
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


#: What this example ships, and what the run must therefore be solved by.
#:
#: It shipped `solver: gurobi` until this branch, and the consequence had never surfaced because
#: no test ran it: **the only shipped scenario that clears a market could not run without a
#: commercial licence.** The first CI pipeline here died with "The Gurobi shared library could not
#: be loaded", having passed on a development machine that has one. The example is now on HiGHS
#: like every other shipped scenario, and `test_the_clearing_example_needs_no_commercial_licence`
#: is what keeps it there.
#:
#: The arm runs the config **as shipped** rather than switching the solver, which is the stronger
#: arrangement: a switch would keep passing if the example were flipped back to Gurobi, because it
#: matches the *key* rather than the value. Reading the receipt back fails instead. Same shape as
#: `tests/e2e/test_grid_examples.py::test_the_grid_example_needs_no_commercial_licence`, which
#: exists because a reviewer reverted two configs to Gurobi and that file still passed.
SHIPPED_BACKEND = {('poi', 'highs')}


@pytest.fixture(scope='module')
def run(scenario_runs):
    """The example, run once, with receipts of what solved it and which strategies it built."""
    return scenario_runs.run(EXAMPLE, SCENARIO, creator_method=CREATOR_METHOD,
                             needs_receipt=True, needs_strategies=True)


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
def test_the_clearing_example_needs_no_commercial_licence(run):
    """The one scenario that clears a market runs on the solver that ships with HAMLET.

    Read off the receipt the run wrote, not off the config: the config is a request, and this is
    the only thing that reports what actually solved. It is also the test that would have caught
    the state this example was in for its whole life -- `solver: gurobi` in both its files, on the
    only shipped scenario with a clearing method, unnoticed because nothing ran it.
    """
    import json

    assert run.record is not None and run.record.exists(), (
        'the run completed but wrote no backend record, so what solved it is unknown')
    used = {tuple(pair) for pair in json.loads(run.record.read_text(encoding='utf-8'))}

    assert used == SHIPPED_BACKEND, (
        f'{SCENARIO} was solved by {sorted(used)} rather than {sorted(SHIPPED_BACKEND)} as its '
        f'config specifies, so the example would need a commercial licence')


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


# ----------------------------------------------------------------------------------------------
# L4 -- the accounting invariants, over a real run.
#
# The unit layer (`tests/unit/executor/markets/test_settlement_by_hand.py`) settles what the
# arithmetic computes from a given book, exactly, against hand-derived money. What it cannot say
# is whether a run of hundreds of clearings ever produces a book where the netting matters, or
# whether the numbers stay consistent once they have been through the schema, the horizon
# truncation and the Feather round-trip. That is what these read: the tables **as written to
# disk**, reconciled against each other and against the retailer table the run priced from.
#
# **Nothing here is a bare sum-in-equals-sum-out.** #241 found a market selling energy nobody had
# offered while exactly that invariant passed, because the duplication was symmetric. Each
# assertion below either recomputes a money column from a rate and a quantity, or compares one
# table against a different one.
# ----------------------------------------------------------------------------------------------
@pytest.fixture(scope='module')
def retailer_prices(results):
    """The retailer table the market priced from, as `{column: value}` for its first row.

    Read off the run's own output rather than off `markets.yaml`: the file is what the executor
    reads, and whether the two agree is a separate question that
    `tests/unit/creator/markets/test_retailer_convention.py` settles. Constant across the run for
    this scenario, which `test_the_shipped_tariff_is_flat_and_prices_feed_in_at_zero` states
    rather than assumes.
    """
    frame = pl.read_ipc(results / 'retailer.ft', memory_map=False)
    return frame.row(0, named=True)


def net_by_agent(transactions, kinds):
    """Per `(timestep, id_agent)`, the energy netted over `kinds`, taken from the file.

    On an **unsigned** column `energy_in - energy_out` does not go negative, it wraps to ~1.8e19,
    and `clip(lower_bound=0)` then leaves that astronomical figure in place rather than zeroing
    it. `ElectricityMarket._to_net_energy` guards its own subtraction against that, on its raw
    input.

    The assertion below **states that precondition rather than guarding it**: it sits after the
    aggregation, and `fill_null(0)` demotes an unsigned column to `Int64` before the sum, so it
    holds by construction on every path that reaches it and cannot currently fail. It is here so
    that a rewrite dropping the fill -- or writing it as `pl.lit(0, pl.UInt64)`, the tidy-up the
    dtype invites -- fails loudly instead of wrapping. Guarding the raw input instead would fire
    on a `UInt64` column that the fill would have made safe, which is a false alarm rather than a
    check.
    """
    netted = (transactions
            .filter(pl.col(c.TC_TYPE_TRANSACTION).is_in(list(kinds)))
            .group_by([c.TC_TIMESTEP, c.TC_ID_AGENT])
            .agg([pl.col(c.TC_ENERGY_IN).fill_null(0).sum().alias(c.TC_ENERGY_IN),
                  pl.col(c.TC_ENERGY_OUT).fill_null(0).sum().alias(c.TC_ENERGY_OUT)])
              )
    for column in (c.TC_ENERGY_IN, c.TC_ENERGY_OUT):
        dtype = netted.schema[column]
        assert dtype.is_signed_integer() or dtype.is_float(), (
            f'{column} is {dtype}; the difference below would wrap instead of going negative')

    return netted.with_columns([
        (pl.col(c.TC_ENERGY_IN) - pl.col(c.TC_ENERGY_OUT)).clip(lower_bound=0).alias('net_in'),
        (pl.col(c.TC_ENERGY_OUT) - pl.col(c.TC_ENERGY_IN)).clip(lower_bound=0).alias('net_out')])


def charged_by_agent(transactions, kind):
    """Per `(timestep, id_agent)`, the energy one kind of fee row was charged on."""
    return (transactions
            .filter(pl.col(c.TC_TYPE_TRANSACTION) == kind)
            .group_by([c.TC_TIMESTEP, c.TC_ID_AGENT])
            .agg([pl.col(c.TC_ENERGY_IN).fill_null(0).sum().alias(c.TC_ENERGY_IN),
                  pl.col(c.TC_ENERGY_OUT).fill_null(0).sum().alias(c.TC_ENERGY_OUT)]))


@pytest.mark.e2e
def test_the_shipped_tariff_is_flat_and_prices_feed_in_at_zero(retailer_prices, results):
    """The premise the fee assertions read against, taken off the run's retailer table.

    Both charges are levied on consumption only in this scenario, so a swapped mapping does not
    reorder two numbers -- it moves every euro of grid fee and levy onto the exporting side,
    which is what `eec60f0` fixed. Stated here so that a scenario which started pricing feed-in
    fails loudly rather than making the assertions below vacuous.
    """
    frame = pl.read_ipc(results / 'retailer.ft', memory_map=False)

    for column in (f'{c.TT_GRID}_{c.TT_MARKET}', f'{c.TT_GRID}_{c.TT_RETAIL}',
                   f'{c.TT_LEVIES}_{c.TC_PRICE}'):
        assert retailer_prices[f'{column}_{c.PF_OUT}'] > 0, f'{column} is not charged at all'
        # Over the whole table rather than its first row: a tariff that priced feed-in at one
        # timestep only would satisfy a row-0 check and move every fee assertion below.
        assert frame[f'{column}_{c.PF_IN}'].max() == 0, f'{column} now prices feed-in'
        assert frame[f'{column}_{c.PF_IN}'].n_unique() == 1, f'{column} feed-in is no longer flat'
        assert frame[f'{column}_{c.PF_OUT}'].n_unique() == 1, f'{column} is no longer flat'


@pytest.mark.e2e
@pytest.mark.parametrize('kind', [c.TT_GRID, c.TT_LEVIES])
def test_the_charged_energy_is_the_net_of_the_transactions_actually_written(results, kind):
    """Invariant 1, anchored on the transactions file rather than on another derived total.

    A `grid` or `levies` row records the energy an agent is billed for at a delivery timestep.
    That figure has to be the netted sum of the agent's own trades at that timestep -- and those
    trades are in the same file, so the check does not rest on the market agreeing with itself:
    the fee rows are compared against the trade rows, per agent, per timestep, by value.

    This is the run-scale form of the first half of `eec60f0` -- fees on gross rather than net --
    which fails it on 41 of the 159 `levies` rows.

    **It is not the run-scale form of `a860ca6`, and no e2e test can be.** Reverting that fix
    leaves all of this file green, because in a shipped run `clear` and `settle` are separate
    timetable tasks: by settlement time this timestep's trades are already in
    `transactions_prev`, put there by an earlier task, and `self.transactions` is empty. See the
    section in `tests/unit/executor/markets/test_settlement_by_hand.py` headed "the task shape
    every shipped run actually uses" for the measurement and why the shape is unreachable.
    """
    transactions = table(results, c.TN_MARKET_TRANSACTIONS)
    traded = net_by_agent(transactions, MarketDB.NET_TRANSACTION_TYPES)
    charged = charged_by_agent(transactions, kind)
    # **Inner, not left.** A left join writes nulls where it found no partner; `x != null` is
    # null, which `filter` drops, so an unmatched fee row would be reported as agreeing. With an
    # inner join the row disappears instead and the length check below is what notices -- and
    # that check is only meaningful because the join is inner: `traded` comes out of a `group_by`,
    # so its keys are unique and a *left* join is always 1:1 and the length can never differ.
    paired = charged.join(traded, on=[c.TC_TIMESTEP, c.TC_ID_AGENT], how='inner')

    assert len(paired) == len(charged) > 0, 'a fee row names an agent that never traded'
    mismatched = paired.filter((pl.col(c.TC_ENERGY_IN) != pl.col('net_in'))
                               | (pl.col(c.TC_ENERGY_OUT) != pl.col('net_out')))
    assert mismatched.is_empty(), (
        f'{len(mismatched)} of {len(paired)} {kind} rows are charged on something other than '
        f'the net of that agent\'s own transactions')


@pytest.mark.e2e
def test_the_two_fee_kinds_are_charged_on_the_same_energy(results):
    """Grid fee and levy bill the same quantity, per agent per timestep.

    They are two clones of one netted frame, so a divergence means one of them was recomputed
    from something else. Cheap, and it fails on a defect that leaves each of them individually
    self-consistent.
    """
    transactions = table(results, c.TN_MARKET_TRANSACTIONS)
    grid = charged_by_agent(transactions, c.TT_GRID)
    levies = charged_by_agent(transactions, c.TT_LEVIES)
    paired = grid.join(levies, on=[c.TC_TIMESTEP, c.TC_ID_AGENT], how='inner', suffix='_levies')

    assert len(paired) == len(grid) == len(levies) > 0
    assert paired.filter(
        (pl.col(c.TC_ENERGY_IN) != pl.col(f'{c.TC_ENERGY_IN}_levies'))
        | (pl.col(c.TC_ENERGY_OUT) != pl.col(f'{c.TC_ENERGY_OUT}_levies'))).is_empty()


@pytest.mark.e2e
@pytest.mark.parametrize('kind', [c.TT_GRID, c.TT_LEVIES])
def test_every_settled_timestep_is_billed(results, kind):
    """The other direction: a trading agent-timestep without a fee row would go unnoticed above.

    `test_the_charged_energy_is_the_net_of_the_transactions_actually_written` iterates the fee
    rows, so an implementation that simply stopped writing some of them passes it.

    Two kinds of trading pair legitimately carry no fee row, and both are excluded here rather
    than tolerated by a loose bound. **The retailer** is never billed at all -- it is the supply
    of last resort, not a grid connection -- and it trades at almost every timestep, so leaving
    it in would make this assertion unsatisfiable. **Delivery steps past the last settlement**
    were traded into but never settled: the run ends first. Everything else must be billed.
    """
    transactions = table(results, c.TN_MARKET_TRANSACTIONS)
    traded = (net_by_agent(transactions, MarketDB.NET_TRANSACTION_TYPES)
              .filter(pl.col(c.TC_ID_AGENT) != RETAILER))
    charged = charged_by_agent(transactions, kind)

    assert len(traded) > 0, 'no agent but the retailer traded, so this checks nothing'

    unbilled = traded.join(charged, on=[c.TC_TIMESTEP, c.TC_ID_AGENT], how='anti')
    billed = traded.join(charged, on=[c.TC_TIMESTEP, c.TC_ID_AGENT], how='semi')

    assert len(billed) > 0 and len(unbilled) > 0, (
        'this run no longer has both billed and unbilled timesteps, so the boundary below is '
        'not where this test assumes it is')
    latest_billed = billed[c.TC_TIMESTEP].max()
    stranded = unbilled.filter(pl.col(c.TC_TIMESTEP) <= latest_billed)
    assert stranded.is_empty(), (
        f'{len(stranded)} agent-timesteps were traded but never billed, while a later one was '
        f'billed, so this is not the unsettled tail')


@pytest.mark.e2e
def test_the_netting_is_not_a_no_op_in_this_run(results):
    """The premise the reconciliation needs: some agent really does trade both ways in a timestep.

    Without it, gross and net coincide on every row and the reconciliation is satisfied by an
    implementation that does no netting at all -- the same shape as a "don't-care" fixture value.
    Asserted as a count, so a scenario change that flattened the run fails here rather than
    quietly emptying the check above.
    """
    traded = net_by_agent(table(results, c.TN_MARKET_TRANSACTIONS),
                          MarketDB.NET_TRANSACTION_TYPES)
    both_ways = traded.filter((pl.col(c.TC_ENERGY_IN) > 0) & (pl.col(c.TC_ENERGY_OUT) > 0))

    assert len(both_ways) > 0, 'no agent bought and sold in one timestep; netting is vacuous here'


@pytest.mark.e2e
def test_the_written_table_has_the_dtypes_this_file_reconciles_in(results):
    """`market_transactions.ft` comes back as `Float64`, not as the schema declares it. Pinned.

    `c.TS_MARKET_TRANSACTIONS` declares `energy_in` as `UInt64`, `price_pu_in` as `Int32` and
    `price_in` as `Int64`, and `bids_cleared.ft` from the same run does come back that way. This
    table does not, and **the difference is nullability, not the write path**: both go through
    `concat_past_data`'s `to_pandas()` round-trip, both come back carrying its undeclared
    `index`, and that round-trip floats a numeric column exactly when it contains a null.
    `market_transactions` has 267 nulls in `energy_in` alone -- every row records one direction
    -- while `bids_cleared` has none.

    So the reconciliations here are **float** equalities, exact only because the largest figure
    in the run is ~1.3e7, twenty-two binary orders below where `Float64` stops being exact on
    integers. Worth saying precisely, because the naive reading -- that `bids_cleared` is on a
    different path -- would let a change that made it nullable float it too, silently.

    Stated rather than fixed; the round-trip is not this branch's to change. The bound is
    asserted below so a run whose figures grew towards 2**53 fails here rather than losing euros
    quietly.
    """
    transactions = table(results, c.TN_MARKET_TRANSACTIONS)
    cleared = table(results, c.TN_BIDS_CLEARED)

    assert transactions.schema[c.TC_PRICE_IN] == pl.Float64
    assert cleared.schema[c.TC_PRICE_IN] == c.TS_BIDS_CLEARED[c.TC_PRICE_IN], (
        'the two tables now agree on dtype, so this note is stale')
    # The stated cause, so a future reader can tell which half moved if this ever changes
    assert transactions[c.TC_ENERGY_IN].null_count() > 0
    assert cleared[c.TC_ENERGY_IN].null_count() == 0
    for column in (c.TC_PRICE_IN, c.TC_PRICE_OUT, c.TC_ENERGY_IN, c.TC_ENERGY_OUT):
        largest = transactions[column].fill_null(0).abs().max()
        assert largest < 2 ** 53, f'{column} reaches {largest}, where Float64 stops being exact'


@pytest.mark.e2e
def test_every_money_column_is_its_rate_times_its_quantity(results):
    """Invariant 3's reconciliation half, on every row of every kind the run wrote.

    Money is compared against a product of two other columns, not against another money total, so
    a table that is internally consistent at the wrong figure fails. The comparison is exact
    rather than tolerant, which the values permit but the *dtypes* do not guarantee -- see
    `test_the_written_table_has_the_dtypes_this_file_reconciles_in`.
    """
    transactions = table(results, c.TN_MARKET_TRANSACTIONS).with_columns(
        [pl.col(column).fill_null(0) for column in
         (c.TC_ENERGY_IN, c.TC_ENERGY_OUT, c.TC_PRICE_IN, c.TC_PRICE_OUT,
          c.TC_PRICE_PU_IN, c.TC_PRICE_PU_OUT)])

    unexplained = transactions.filter(
        (pl.col(c.TC_PRICE_IN) != pl.col(c.TC_PRICE_PU_IN) * pl.col(c.TC_ENERGY_IN))
        | (pl.col(c.TC_PRICE_OUT) != pl.col(c.TC_PRICE_PU_OUT) * pl.col(c.TC_ENERGY_OUT)))

    assert len(transactions) > 0
    assert unexplained.is_empty(), (
        f'{len(unexplained)} of {len(transactions)} rows carry money that is not their per-unit '
        f'price times their energy')


@pytest.mark.e2e
def test_the_local_trades_move_money_between_agents_without_creating_any(results):
    """Invariant 3's closure half: a local trade writes a debit and a credit of the same size.

    Per clearing rather than over the run -- a run total can hide an excess at one clearing
    cancelling a shortfall at another -- and paired with the reconciliation above, which is what
    keeps it from being a bare symmetric check: each total it compares is already pinned to a
    rate times a quantity.
    """
    local = (table(results, c.TN_MARKET_TRANSACTIONS)
             .filter(pl.col(c.TC_TYPE_TRANSACTION) == c.TT_MARKET)
             .group_by([c.TC_TIMESTAMP, c.TC_TIMESTEP])
             .agg([pl.col(c.TC_PRICE_IN).fill_null(0).sum().alias('paid'),
                   pl.col(c.TC_PRICE_OUT).fill_null(0).sum().alias('received')]))

    assert len(local) >= MIN_CLEARINGS
    assert local.filter(pl.col('paid') != pl.col('received')).is_empty()
    assert local['paid'].sum() > 0


@pytest.mark.e2e
@pytest.mark.parametrize('kind, column', [(c.TT_GRID, f'{c.TT_GRID}_{c.TT_RETAIL}'),
                                          (c.TT_LEVIES, f'{c.TT_LEVIES}_{c.TC_PRICE}')])
def test_the_run_charges_fees_on_consumption_and_not_on_feed_in(results, retailer_prices,
                                                                kind, column):
    """Invariant 2 at run scale, anchored on the retailer table the run priced from.

    `tests/README.md` calls the in/out crossing the single most error-prone thing in this
    codebase, and this is its consequence in money: the retailer's `_out` rate -- the retailer
    selling, the agent buying -- has to land on the agent's `price_pu_in`, and its `_in` rate,
    which is zero here, on `price_pu_out`.

    The grid rate is bounded rather than pinned, on every row: it is a convex blend of the market
    rate and the retail one by that agent's local share, and the shipped tariff prices the market
    direction *below* the retail one (400 against 800), so the retail rate is the upper end of
    the blend. Measured, the run uses thirteen distinct grid rates between 0 and 800 and reaches
    the bound with equality rather than slack. The levy carries no blend and is pinned by value
    on every row.
    """
    charged = (table(results, c.TN_MARKET_TRANSACTIONS)
               .filter(pl.col(c.TC_TYPE_TRANSACTION) == kind)
               .with_columns([pl.col(c.TC_PRICE_IN).fill_null(0),
                              pl.col(c.TC_PRICE_OUT).fill_null(0)]))
    consumption_rate = retailer_prices[f'{column}_{c.PF_OUT}']

    assert len(charged) > 0
    assert charged[c.TC_PRICE_OUT].sum() == 0, f'{kind} was charged on feed-in'
    assert charged[c.TC_PRICE_IN].sum() > 0, f'{kind} was never charged on consumption'
    assert charged[c.TC_PRICE_PU_IN].max() <= consumption_rate, (
        f'{kind} charged more per Wh than the retailer tariff it is blended from')
    if kind == c.TT_LEVIES:
        assert set(charged[c.TC_PRICE_PU_IN]) == {consumption_rate}


@pytest.mark.e2e
@pytest.mark.parametrize('kind', [c.TT_GRID, c.TT_LEVIES])
def test_the_retailer_pays_no_grid_fee_or_levy(results, kind):
    """The retailer trades in this run and is not a grid connection, so it is excluded.

    Stated against a retailer that is genuinely present in the transactions table, so the
    exclusion is exercised rather than satisfied by its absence.
    """
    transactions = table(results, c.TN_MARKET_TRANSACTIONS)

    assert RETAILER in set(transactions.filter(
        pl.col(c.TC_TYPE_TRANSACTION) == c.TT_RETAIL)[c.TC_ID_AGENT])
    assert RETAILER not in set(transactions.filter(
        pl.col(c.TC_TYPE_TRANSACTION) == kind)[c.TC_ID_AGENT])


# ----------------------------------------------------------------------------------------------
# The retailer's ladder row, and what its timestamp was doing to the count above.
# ----------------------------------------------------------------------------------------------
@pytest.mark.e2e
def test_both_sides_clear_at_the_same_set_of_clearings(results):
    """A clearing writes bids and offers together, so the two tables agree on which pairs exist.

    They did not, until the retailer's ladder row was stamped with the clearing rather than with
    the delivery step, and its own row then sat at `(delivery, delivery)` forming a group of its
    own. `MIN_CLEARINGS` counted those pairs, so the floor it enforces was counting groups that
    were not clearings -- which is why the stamp was fixed rather than the counting: the pair is
    the clearing key everywhere, not only in this file. The figures are at the head of this file,
    beside `MIN_CLEARINGS`, and only there.
    """
    pairs = {name: set(map(tuple, table(results, name)
                           .select([c.TC_TIMESTAMP, c.TC_TIMESTEP]).unique().rows()))
             for name in (c.TN_BIDS_CLEARED, c.TN_OFFERS_CLEARED)}

    assert pairs[c.TN_BIDS_CLEARED] == pairs[c.TN_OFFERS_CLEARED]
    assert len(pairs[c.TN_BIDS_CLEARED]) >= MIN_CLEARINGS


@pytest.mark.e2e
def test_no_position_is_stamped_at_its_own_delivery_step(results):
    """A position is quoted before the step it delivers into, never at it.

    This is the tell the retailer's row used to fail: `timestamp` is when the clearing happens
    and `timestep` the step it delivers into. A row where the two are equal was stamped from the
    wrong column, and it is invisible to every assertion about quantities and prices.

    **Only the two cleared tables can carry that regression.** `__create_market_tables` drops the
    retailer from both uncleared tables outright, so those two iterations state the invariant
    generally rather than guarding this defect -- measured under the reverted stamp, both were
    already at zero. They are kept because the invariant is the same one.
    """
    for name in (c.TN_BIDS_CLEARED, c.TN_OFFERS_CLEARED, c.TN_BIDS_UNCLEARED,
                 c.TN_OFFERS_UNCLEARED):
        frame = table(results, name)
        late = frame.filter(pl.col(c.TC_TIMESTAMP) >= pl.col(c.TC_TIMESTEP))

        assert len(frame) > 0, f'{name} is empty, so this asserts about nothing'
        assert late.is_empty(), f'{name}: {len(late)} rows are stamped at or after their delivery'

