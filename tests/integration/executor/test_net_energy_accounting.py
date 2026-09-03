"""Integration -- `MarketDB.get_net_energy` against the transactions actually in the table. L4.

This is the first of L4's three invariants: **net energy is the sum of the transactions**. It is
what every trading strategy subtracts from its setpoint to get a position, so an error here
misprices every bid in the run.

**Why the obvious test is not enough.** `get_net_energy` answers from a running cache when there
is one and recomputes from the whole table when there is not, and both routes go through
`_net_energy_from`. Comparing the two -- the differential test -- catches a fold that has gone
wrong on the incremental path and nothing else: a wrong *filter* is applied identically on both
routes, so the two agree, at the same wrong number. That is the shape #241 walked through, where
`sum(cleared in) == sum(cleared out)` held throughout a clearing in which 7.1 % of the energy did
not exist.

So every total below is anchored on a hand-computed figure derived from the fixture's rows, and
the differential test is kept only as the extra it really is --
`test_the_cached_and_recomputed_answers_agree` says so in its own docstring.

**The fixture carries `grid` and `levies` rows**, and that is the point of it. They clone the
netted transactions and hold identical energy, so a filter that admitted them would roughly
treble an agent's traded energy while leaving the cache and the recompute in perfect agreement.
"""
import datetime as dt

import polars as pl
import pytest

import hamlet.constants as c
from hamlet.executor.utilities.database.market_db import MarketDB

MARKET_TYPE = c.ET_ELECTRICITY
MARKET_NAME = 'continuous'
OTHER_MARKET = 'day_ahead'

AGENT = 'agent_a'
OTHER_AGENT = 'agent_b'

T0 = dt.datetime(2023, 1, 1, 12, tzinfo=dt.timezone.utc)
T1 = T0 + dt.timedelta(hours=1)
T2 = T0 + dt.timedelta(hours=2)

COLUMNS = (c.TC_TIMESTEP, c.TC_MARKET, c.TC_NAME, c.TC_TYPE_TRANSACTION, c.TC_ID_AGENT,
           c.TC_ENERGY_IN, c.TC_ENERGY_OUT)
SCHEMA = {column: c.TS_MARKET_TRANSACTIONS[column] for column in COLUMNS}

# ----------------------------------------------------------------------------------------------
# The table, and the netting derived from it by hand.
#
# `agent_a` at T0:  retail 900 in, retail 300 out            -> (900, 300)
# `agent_a` at T1:  market 400 in, balancing 100 in          -> (500, 0)
# `agent_b` at T0:  market 700 out                           -> (0, 700)
#
# Each of those is followed by its `grid` and `levies` rows, which carry the *net* energy of the
# timestep -- 600 in for `agent_a` at T0, 500 in at T1, 700 out for `agent_b`. Counting them
# would make `agent_a`'s T0 figure (2100, 300) instead of (900, 300).
#
# `agent_a` also trades 5000 Wh on a second market at T0. `get_net_energy` is asked about one
# market, so that row must not appear in any answer.
# ----------------------------------------------------------------------------------------------
TRADES = [
    (T0, MARKET_NAME, c.TT_RETAIL, AGENT, 900, 0),
    (T0, MARKET_NAME, c.TT_RETAIL, AGENT, 0, 300),
    (T1, MARKET_NAME, c.TT_MARKET, AGENT, 400, 0),
    (T1, MARKET_NAME, c.TT_BALANCING, AGENT, 100, 0),
    (T0, MARKET_NAME, c.TT_MARKET, OTHER_AGENT, 0, 700),
]

FEES = [
    (T0, MARKET_NAME, c.TT_GRID, AGENT, 600, 0),
    (T0, MARKET_NAME, c.TT_LEVIES, AGENT, 600, 0),
    (T1, MARKET_NAME, c.TT_GRID, AGENT, 500, 0),
    (T1, MARKET_NAME, c.TT_LEVIES, AGENT, 500, 0),
    (T0, MARKET_NAME, c.TT_GRID, OTHER_AGENT, 0, 700),
    (T0, MARKET_NAME, c.TT_LEVIES, OTHER_AGENT, 0, 700),
]

ELSEWHERE = [(T0, OTHER_MARKET, c.TT_MARKET, AGENT, 5000, 0)]

#: What `get_net_energy` must return per timestep, computed by hand from `TRADES` alone.
EXPECTED = {AGENT: {T0: (900, 300), T1: (500, 0)}, OTHER_AGENT: {T0: (0, 700)}}

#: What it would return if the fee rows were counted -- stated so the two cannot be confused, and
#: so that a fixture change which made them coincide fails `test_the_fee_rows_would_change_the_answer`.
IF_FEES_COUNTED = {AGENT: {T0: (2100, 300), T1: (1500, 0)}, OTHER_AGENT: {T0: (0, 2100)}}


def frame(rows):
    return pl.DataFrame(
        {c.TC_TIMESTEP: [row[0] for row in rows],
         c.TC_MARKET: [MARKET_TYPE] * len(rows),
         c.TC_NAME: [row[1] for row in rows],
         c.TC_TYPE_TRANSACTION: [row[2] for row in rows],
         c.TC_ID_AGENT: [row[3] for row in rows],
         c.TC_ENERGY_IN: [row[4] for row in rows],
         c.TC_ENERGY_OUT: [row[5] for row in rows]},
        schema=SCHEMA)


def database():
    """A `MarketDB` with both caches live and an empty table, built without touching disk."""
    market = object.__new__(MarketDB)
    market.market_type, market.market_name = MARKET_TYPE, MARKET_NAME
    market.market_transactions = pl.DataFrame(schema=SCHEMA)
    market._net_cache = pl.DataFrame(schema=MarketDB.NET_SCHEMA)
    market._rtc_cache = pl.DataFrame(schema=MarketDB.NET_SCHEMA)
    return market


def feed(market, rows):
    """Append `rows` the way the executor does -- whole table plus the addition, so the fold runs."""
    batch = frame(rows)
    combined = (batch if market.market_transactions.is_empty()
                else pl.concat([market.market_transactions, batch], how='vertical'))
    market.set_market_transactions(combined, new_rows=batch)


def netted(market, agent, first=T0, last=T2):
    """`{timestep: (energy_in, energy_out)}` as `get_net_energy` reports it."""
    answer = market.get_net_energy(agent, first, last)
    return {row[c.TC_TIMESTEP]: (row[c.TC_ENERGY_IN], row[c.TC_ENERGY_OUT])
            for row in answer.to_dicts()}


@pytest.fixture
def incremental():
    """The whole table, fed in four separate batches so the incremental fold does the work.

    Batched rather than written in one go on purpose: a fold that is wrong only on the
    incremental path is invisible to a test that builds the total at once, which is the same
    shape as the defect `a860ca6` fixed one layer up.
    """
    market = database()
    feed(market, TRADES[:2])
    feed(market, TRADES[2:])
    feed(market, FEES)
    feed(market, ELSEWHERE)
    return market


@pytest.fixture
def at_once():
    """The same table with no cache at all, so every answer is recomputed from it."""
    market = database()
    market.set_market_transactions(frame(TRADES + FEES + ELSEWHERE))
    return market


# ----------------------------------------------------------------------------------------------
# Premises.
# ----------------------------------------------------------------------------------------------
def test_the_fee_rows_would_change_the_answer():
    """Counting the fee rows gives a different number, or this fixture proves nothing.

    A fixture whose fee rows carried zero energy -- or none at all -- would make the type filter
    unfalsifiable, and a "don't-care" value doing exactly that is how an equivalent mutant was
    faked on !250. Stated against the two hand-computed tables rather than against the code.
    """
    assert EXPECTED != IF_FEES_COUNTED
    assert all(EXPECTED[agent][step] != IF_FEES_COUNTED[agent][step]
               for agent in EXPECTED for step in EXPECTED[agent])
    assert any(kind in {c.TT_GRID, c.TT_LEVIES} for _, _, kind, _, _, _ in FEES)


def test_the_incremental_fixture_really_folds_more_than_once(incremental):
    """Four batches, so the fold is exercised as a fold rather than as a single write."""
    assert incremental.net_cache is not None
    assert len(incremental.market_transactions) == len(TRADES) + len(FEES) + len(ELSEWHERE)


def test_the_uncached_fixture_really_has_no_cache(at_once):
    """`set_market_transactions` without `new_rows` drops the caches, so this arm recomputes."""
    assert at_once.net_cache is None


# ----------------------------------------------------------------------------------------------
# Invariant 1 -- net energy is the sum of the transactions, anchored on hand-computed figures.
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize('arm', ['incremental', 'at_once'])
@pytest.mark.parametrize('agent', [AGENT, OTHER_AGENT])
def test_the_net_energy_is_the_hand_computed_sum(request, arm, agent):
    """Both routes, both agents, against numbers derived from the rows rather than from a run.

    The absolute anchor for this invariant: `EXPECTED` is read off `TRADES` by hand at the top of
    this file, so an answer that is merely self-consistent fails it.
    """
    market = request.getfixturevalue(arm)

    assert netted(market, agent) == EXPECTED[agent]


@pytest.mark.parametrize('arm', ['incremental', 'at_once'])
def test_the_fee_rows_are_not_counted_as_traded_energy(request, arm):
    """`grid` and `levies` clone the netted energy; counting them roughly trebles it.

    **This is the defect a purely differential test would miss.** Admitting the two types widens
    `_net_energy_from`, which is the single definition both the cache and the recompute use, so
    the two routes go on agreeing exactly -- at `IF_FEES_COUNTED` instead of `EXPECTED`.
    """
    market = request.getfixturevalue(arm)

    assert netted(market, AGENT) == EXPECTED[AGENT]
    assert netted(market, AGENT) != IF_FEES_COUNTED[AGENT]


def test_the_cached_and_recomputed_answers_agree(incremental, at_once):
    """The differential check, kept for what it does cover: a fold that has gone wrong.

    **Not evidence on its own.** Both routes call `_net_energy_from`, so any error in the filter
    or the grouping is applied to both and they agree at the wrong number. It is the
    hand-computed assertions above that give this file its anchor; this one adds only the
    incremental path.
    """
    for agent in (AGENT, OTHER_AGENT):
        assert netted(incremental, agent) == netted(at_once, agent)


def test_a_second_batch_adds_to_the_running_total_rather_than_replacing_it():
    """The fold accumulates. Asserted against a hand-computed total, not against a recompute.

    Two batches at the same timestep for the same agent: 400 Wh then 250 Wh is 650 Wh. A fold
    that dropped the addition would report 400; one that replaced the running sum would report
    250. Both are wrong in a way the totals of the whole table cannot show, because the whole
    table is not what an incremental cache reads.
    """
    market = database()
    feed(market, [(T0, MARKET_NAME, c.TT_MARKET, AGENT, 400, 0)])
    feed(market, [(T0, MARKET_NAME, c.TT_MARKET, AGENT, 250, 0)])

    assert netted(market, AGENT) == {T0: (650, 0)}


def test_a_batch_that_contributes_nothing_leaves_the_running_total_alone():
    """A batch of pure fee rows folds to nothing and must not disturb what is already there."""
    market = database()
    feed(market, [(T0, MARKET_NAME, c.TT_MARKET, AGENT, 400, 0)])
    feed(market, [(T0, MARKET_NAME, c.TT_GRID, AGENT, 400, 0)])

    assert netted(market, AGENT) == {T0: (400, 0)}


def test_only_this_market_counts(incremental, at_once):
    """`agent_a` trades 5000 Wh on a second market at T0 and none of it is in the answer.

    `_net_energy_from` filters on market type *and* name; `ELSEWHERE` differs by name only, so a
    filter that dropped the name check would add 5000 Wh to T0 and nothing would raise.
    """
    for market in (incremental, at_once):
        assert netted(market, AGENT)[T0] == (900, 300)
        assert netted(market, AGENT)[T0][0] != 900 + 5000


def test_only_this_agent_counts(incremental):
    """One agent's answer holds none of the other's energy, in either direction."""
    assert netted(incremental, AGENT) == EXPECTED[AGENT]
    assert netted(incremental, OTHER_AGENT) == EXPECTED[OTHER_AGENT]
    assert T1 not in netted(incremental, OTHER_AGENT)


@pytest.mark.parametrize('first, last, expected_steps', [
    (T0, T2, [T0, T1]),
    (T0, T0, [T0]),
    (T1, T2, [T1]),
    (T2, T2, []),
])
def test_the_window_is_closed_at_both_ends(incremental, first, last, expected_steps):
    """Both bounds are inclusive, and a window with nothing in it is empty rather than an error."""
    assert sorted(netted(incremental, AGENT, first, last)) == expected_steps


def test_an_agent_that_never_traded_gets_an_empty_answer(incremental):
    """Not a zero row, and not somebody else's numbers."""
    assert netted(incremental, 'nobody') == {}


def test_dropping_the_cache_does_not_change_any_answer(incremental):
    """The cache is an optimisation, never a second source of truth.

    Replacing the table without an addition drops both caches, and every answer must survive it
    unchanged -- otherwise the cached and uncached branches have drifted and which one a caller
    gets depends on how the table was last written.
    """
    before = {agent: netted(incremental, agent) for agent in (AGENT, OTHER_AGENT)}
    incremental.set_market_transactions(incremental.market_transactions)

    assert incremental.net_cache is None
    assert {agent: netted(incremental, agent) for agent in (AGENT, OTHER_AGENT)} == before
