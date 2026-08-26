"""Unit -- which trading strategy `AgentBase.create_bids_offers` actually picks, and why.

`agents.yaml` lets an agent name a trading strategy; `markets.yaml` lets a market say it has no
clearing. When both happen, the market wins and the agent trades at retailer prices. That is
**correct** -- with no clearing the retailer is the only counterparty, so a price ladder aimed at
other agents has nothing to aim at, and `agents.yaml` documents `retailer` as "the default
strategy when there is no clearing".

**What was wrong is that nothing said so.** `tests/e2e/scenarios/ctsp_industry` declared
`strategy: linear` in its config *and* its workbook, and had only ever run `retailer` -- which is
how a fixture came to imply coverage of a strategy that had never executed anywhere in the suite
(#223). That fixture now says `retailer`, and this file covers the report that would have caught
it.

**And one operand was dead.** The check read `m_method.capitalize() == 'None' or m_method is
None`: the attribute lookup raises before the second operand can run, so a market written
`method: null` -- or with the value left blank -- crashed the agent stage instead of selecting
the retailer. It was latent only because PyYAML parses the shipped spelling `method: None` into
the *string* `'None'`. `test_a_market_written_as_yaml_null_selects_the_retailer` is the case that
used to raise.

The strategies themselves are priced by hand in
`tests/unit/executor/utilities/trading/test_strategies.py`; this file asks only which of them
runs. The two together are what make the chain from config to quoted price checkable.
"""
import warnings

import pytest

import hamlet.constants as c
from hamlet.executor.agents.agent_base import AgentBase, ConfiguredStrategyUnused
from tests.trading_fixtures import (FORECAST_IN, FORECAST_OUT, MARKET, NAME, OTHER_NAME,
                                    STEPS, Agent, market_data, timetable)

#: What each strategy quotes at the **third** timestep of the shared fixture, taken from the
#: hand-derivations in `test_strategies.py`. Third rather than first on purpose: `Linear`'s ladder
#: starts at the retailer's own prices, so at the first timestep the two strategies are
#: indistinguishable and this whole file would be unable to tell them apart. See
#: `test_linear_and_retailer_agree_only_at_the_nearest_step`.
LINEAR_AT_STEP_2 = (1000, 1000)
RETAILER_AT_STEP_2 = (FORECAST_OUT, FORECAST_IN)

assert LINEAR_AT_STEP_2 != RETAILER_AT_STEP_2, (
    'the two strategies quote the same thing here, so nothing below can distinguish them')


def substitutions(caught):
    """Only the strategy-substitution warnings out of everything a call raised.

    `create_bids_offers` raises nine polars deprecations on the way through, so an assertion on
    the raw list would be about those instead.
    """
    return [str(warning.message) for warning in caught
            if issubclass(warning.category, ConfiguredStrategyUnused)]


def drive(clearing_method, strategy, second_market=None):
    """Drive the real `create_bids_offers` and hand back the `AgentBase` it ran on.

    No mocking of `Trading`: the point of this file is the whole chain from the market's
    `method:` to the price an agent ends up quoting, and a recorded constructor argument would
    only show that the *request* was made.
    """
    base = object.__new__(AgentBase)
    base.agent = Agent('agent_1', strategy)
    base.timetable = timetable(clearing_method, second_market=second_market)
    base.market = {MARKET: {NAME: market_data(), OTHER_NAME: market_data(name=OTHER_NAME)}}

    base.create_bids_offers()
    return base


def run(clearing_method, strategy):
    """The quote at the distinguishing timestep, for a region with one market."""
    base = drive(clearing_method, strategy)

    by_step = {step: row for step, row in zip(STEPS, base.agent.bids_offers.to_dicts())}
    row = base.agent.bids_offers.filter(
        base.agent.bids_offers[c.TC_TIMESTEP] == STEPS[2]).to_dicts()
    assert len(row) == 1, f'expected one quote at {STEPS[2]}, got {len(row)} of {by_step}'
    return row[0][c.TC_PRICE_PU_IN], row[0][c.TC_PRICE_PU_OUT]


@pytest.mark.parametrize('method, clears', [
    (c.MCM_PDA, True),
    ('pda', True),
    ('community', True),
    (c.MCM_NONE, False),
    ('None', False),
    ('none', False),
    ('NONE', False),
    (None, False),
])
def test_clears_a_market_reads_both_spellings_of_no_clearing(method, clears):
    """`None` the object and `'None'` the string both mean no clearing; a real method does not.

    The real `None` is the case the old operand order could not reach.
    """
    assert AgentBase.clears_a_market(method) is clears


def test_a_market_that_clears_uses_the_configured_strategy():
    """`method: pda` with `strategy: linear` quotes `Linear`'s ladder, not the retailer's prices."""
    assert run(c.MCM_PDA, c.MTS_LINEAR) == LINEAR_AT_STEP_2


def test_a_market_without_clearing_substitutes_the_retailer():
    """The substitution itself, by value: `linear` is configured and `Retailer`'s prices come out."""
    with pytest.warns(ConfiguredStrategyUnused):
        quote = run(c.MCM_NONE, c.MTS_LINEAR)

    assert quote == RETAILER_AT_STEP_2


def test_a_market_written_as_yaml_null_selects_the_retailer():
    """`method: null` and a blank value both parse as a real `None`, and used to raise.

    Before #223 this reached `None.capitalize()` and the agent stage died with an
    `AttributeError` -- the `or m_method is None` guard that was meant to cover it sat on the
    wrong side of the `or`.
    """
    with pytest.warns(ConfiguredStrategyUnused):
        quote = run(None, c.MTS_LINEAR)

    assert quote == RETAILER_AT_STEP_2


def test_the_substitution_names_the_market_and_the_discarded_strategy():
    """The whole message, not three `in` checks.

    A report that merely mentions the two words could name them the wrong way round -- claiming
    the market was called `linear` and the strategy `lem` -- and three membership tests would
    pass. #223's premise was that nobody could tell which fixture was affected; a message that
    does not identify it would not have helped.
    """
    with pytest.warns(ConfiguredStrategyUnused) as caught:
        run(c.MCM_NONE, c.MTS_LINEAR)

    reported = substitutions(caught)

    assert len(reported) == 1, 'one market, so exactly one report'
    assert reported[0] == (
        f"Market '{NAME}' has no clearing (method: {c.MCM_NONE}), so the configured trading "
        f"strategy '{c.MTS_LINEAR}' cannot be used and '{c.MTS_RETAILER}' is used instead. "
        f"Set `strategy: {c.MTS_RETAILER}` for this agent, or give the market a clearing "
        f"method, so that the config says what the simulation does.")


def test_a_coherent_config_is_not_warned_about():
    """`method: None` with `strategy: retailer` is what `simple_scenario` and `grid_golden` ship.

    Nothing is being discarded there, so nothing is reported -- otherwise the report would fire
    on every agent of every shipped scenario and be tuned out immediately.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        quote = run(c.MCM_NONE, c.MTS_RETAILER)

    assert quote == RETAILER_AT_STEP_2
    assert substitutions(caught) == []


def test_a_clearing_market_is_not_warned_about_either():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        run(c.MCM_PDA, c.MTS_LINEAR)

    assert substitutions(caught) == []


def test_the_absence_checks_above_are_not_vacuous():
    """The two tests above would pass if the category were never raised at all.

    This runs the same recorder over the case that *does* substitute, so an implementation that
    stopped warning entirely fails here rather than turning both of them green. It also confirms
    the recorder sees past the nine polars deprecations the same call raises, which is what the
    dedicated category is for.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        run(c.MCM_NONE, c.MTS_LINEAR)

    assert len(substitutions(caught)) == 1
    assert len(caught) > 1, 'no unrelated warnings here, so the filtering is untested'


# ----------------------------------------------------------------------------------------------
# More than one market in a region.
#
# `create_bids_offers` reads the region's markets out of the timetable and loops over them,
# branching per market on that market's own clearing method. With a single market the loop body
# runs once, so **the loop is not tested by any of the cases above**: processing only the first
# market, or dropping `c.TC_CLEARING_METHOD` from the `unique()` subset that builds the list --
# which would collapse two markets differing only in their method into one row and give one of
# them the other's method -- are both invisible. Measured: both survive without the tests below.
# ----------------------------------------------------------------------------------------------
def test_each_market_in_a_region_gets_its_own_strategy_decision():
    """One market clears and one does not, so both branches are taken in a single call.

    The evidence is the report: exactly one market is named, and it is the one without a
    clearing. A run that processed only the first market would emit no report at all, since the
    clearing market is first.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        drive(c.MCM_PDA, c.MTS_LINEAR, second_market=c.MCM_NONE)

    reported = substitutions(caught)

    assert len(reported) == 1
    assert f"Market '{OTHER_NAME}' has no clearing" in reported[0]
    assert f"Market '{NAME}'" not in reported[0]


def test_two_markets_that_differ_only_in_their_clearing_method_are_kept_apart():
    """Reversed order, so neither position in the loop can be the one that happens to work.

    With the *first* market having no clearing and the second clearing, the report must still
    name the first and only the first. If the two markets were collapsed into one row -- which is
    what dropping the clearing method from the `unique()` subset does, since they share a region,
    type and energy -- one of them would silently run under the other's method.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        base = drive(c.MCM_NONE, c.MTS_LINEAR, second_market=c.MCM_PDA)

    reported = substitutions(caught)

    assert len(reported) == 1
    assert f"Market '{NAME}' has no clearing" in reported[0]
    assert f"Market '{OTHER_NAME}'" not in reported[0]
    assert not base.agent.bids_offers.is_empty(), 'the loop produced no quotes at all'


def test_a_second_market_makes_the_agent_quote_its_whole_timetable_twice():
    """**A stated gap, not an endorsement.** `TradingBase` does not filter the timetable.

    `_preprocess_bids_offers` builds an agent's book from `self.timetable` in full, rather than
    from the rows belonging to the market it was constructed for. In a region with two markets it
    therefore sees every timestep twice, which changes `len_table` -- and `Linear` prices its
    ladder off `len_table`, so an agent in a two-market region quotes a different ladder from the
    same agent in a one-market region.

    No shipped scenario declares two markets in one region, so this is latent. It is pinned here
    because it is the reason the tests above assert on the *reports* rather than on the prices:
    with two markets the prices are not the hand-derived ones, and a reader would otherwise
    reasonably assume they should be. Not fixed: filtering the timetable changes what every
    multi-market agent bids, which is a modelling decision. #223.
    """
    one_market = drive(c.MCM_PDA, c.MTS_LINEAR)
    two_markets = drive(c.MCM_PDA, c.MTS_LINEAR, second_market=c.MCM_PDA)

    assert len(two_markets.agent.bids_offers) == 2 * len(one_market.agent.bids_offers)

    def at_step_2(base):
        row = base.agent.bids_offers.filter(
            base.agent.bids_offers[c.TC_TIMESTEP] == STEPS[2]).to_dicts()[0]
        return row[c.TC_PRICE_PU_IN], row[c.TC_PRICE_PU_OUT]

    assert at_step_2(one_market) == LINEAR_AT_STEP_2
    assert at_step_2(two_markets) != LINEAR_AT_STEP_2
