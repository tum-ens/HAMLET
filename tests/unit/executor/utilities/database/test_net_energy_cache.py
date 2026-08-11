"""Unit -- the cached net energy must equal the uncached net energy, always.

`MarketDB.get_net_energy` replaces a scan every agent used to do over the whole accumulated
`market_transactions` table on every timestep. Measured on design 6 (104 agents, live grid), that
scan is why the agent stage grew from 3.7 s to 10.4 s over 80 timesteps while every other stage
stayed flat.

**The reason this is a test file and not a one-line change** is that the paper branch's version of
this cache was refused during the paper-fix port, for a specific reason worth restating: it summed
`market_transactions` with **no transaction-type filter**. That table also carries `grid` and
`levies` rows which are clones of the netted transactions and hold identical energy -- measured on
the shipped example, retail 926 rows / 320,525 Wh in against grid 87 / 24,298 and levies 87 /
24,298 -- so an unfiltered sum roughly **triple-counts** traded energy for any agent paying fees.
ROADMAP item #2 recorded the condition for ever doing this: keep the type filter, and test that
cached and uncached agree.

So every test below compares the cache against the same definition computed from scratch, and
`test_fees_are_not_counted_as_traded_energy` is the one that fails if the filter is dropped.
"""
import datetime

import polars as pl
import pytest

import hamlet.constants as c
from hamlet.executor.utilities.database.market_db import MarketDB

MARKET_TYPE = c.ET_ELECTRICITY
MARKET_NAME = 'continuous'
START = datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc)


#: The real column types, not approximations of them. The first version of this file used
#: `pl.Datetime(time_zone='UTC')`, which polars defaults to microseconds while
#: `TS_MARKET_TRANSACTIONS` declares nanoseconds, and every test failed on a schema mismatch that
#: production would never hit.
COLUMNS = (c.TC_TIMESTEP, c.TC_MARKET, c.TC_NAME, c.TC_TYPE_TRANSACTION, c.TC_ID_AGENT,
           c.TC_ENERGY_IN, c.TC_ENERGY_OUT)
SCHEMA = {column: c.TS_MARKET_TRANSACTIONS[column] for column in COLUMNS}


def transactions(rows):
    """A `market_transactions`-shaped frame. `rows` are (step, agent, type, in, out)."""
    return pl.DataFrame(
        {c.TC_TIMESTEP: [START + datetime.timedelta(hours=step) for step, *_ in rows],
         c.TC_MARKET: [MARKET_TYPE] * len(rows),
         c.TC_NAME: [MARKET_NAME] * len(rows),
         c.TC_TYPE_TRANSACTION: [kind for _, _, kind, _, _ in rows],
         c.TC_ID_AGENT: [agent for _, agent, _, _, _ in rows],
         c.TC_ENERGY_IN: [energy_in for *_, energy_in, _ in rows],
         c.TC_ENERGY_OUT: [energy_out for *_, energy_out in rows]},
        schema=SCHEMA)


@pytest.fixture
def market(monkeypatch):
    """A MarketDB with no folder behind it -- only the netting logic is under test.

    **Both caches have to be initialised here**, and forgetting one is not a small mistake: an
    uninitialised cache reads as None, every getter silently falls back to recomputing from the
    full table, and the tests pass while exercising none of the code they are about. That is
    exactly what happened -- the RTC tests below were green for a while against the fallback path,
    and only a surviving mutation showed it.
    """
    monkeypatch.setattr(MarketDB, '__init__', lambda self, **kwargs: None)
    db = MarketDB()
    db.market_type = MARKET_TYPE
    db.market_name = MARKET_NAME
    db.market_transactions = pl.DataFrame()
    db._net_cache = pl.DataFrame(schema=MarketDB.NET_SCHEMA)
    db._rtc_cache = pl.DataFrame(schema=MarketDB.NET_SCHEMA)
    return db


def assert_cache_is_live(db):
    """Guard against the fixture regressing to the fallback path."""
    assert db.net_cache is not None and db.rtc_cache is not None, \
        'a cache is missing, so this test would pass against the uncached fallback'


def feed(db, batches):
    """Append each batch the way `post_markets_to_region` does, folding as it goes."""
    assert_cache_is_live(db)
    for batch in batches:
        combined = batch if db.market_transactions.is_empty() else pl.concat(
            [db.market_transactions, batch], how='vertical')
        db.set_market_transactions(combined, new_rows=batch)


def uncached(db, agent, first, last):
    """The same query with the cache taken away, which is the oracle for every test here."""
    cache, db._net_cache = db._net_cache, None
    try:
        return db.get_net_energy(agent, first, last)
    finally:
        db._net_cache = cache


def sorted_rows(frame):
    return frame.sort(c.TC_TIMESTEP).rows()


class TestCachedEqualsUncached:

    def test_a_single_batch(self, market):
        feed(market, [transactions([(0, 'a', c.TT_MARKET, 100, 0), (0, 'b', c.TT_MARKET, 0, 100)])])
        window = (START, START + datetime.timedelta(hours=10))

        assert sorted_rows(market.get_net_energy('a', *window)) == \
               sorted_rows(uncached(market, 'a', *window))

    def test_many_batches_accumulated_one_timestep_at_a_time(self, market):
        """The real access pattern: a clearing per timestep, each adding rows."""
        feed(market, [transactions([(step, 'a', c.TT_MARKET, 10 * step, 0),
                                    (step, 'b', c.TT_RETAIL, 0, 5 * step)])
                      for step in range(1, 25)])
        window = (START, START + datetime.timedelta(hours=30))

        for agent in ('a', 'b'):
            assert sorted_rows(market.get_net_energy(agent, *window)) == \
                   sorted_rows(uncached(market, agent, *window)), agent

    def test_repeated_rows_for_one_timestep_are_summed_not_replaced(self, market):
        """A timestep is cleared more than once, so its rows arrive in different batches."""
        feed(market, [transactions([(3, 'a', c.TT_MARKET, 100, 0)]),
                      transactions([(3, 'a', c.TT_RETAIL, 40, 0)]),
                      transactions([(3, 'a', c.TT_BALANCING, 0, 25)])])
        window = (START, START + datetime.timedelta(hours=10))

        assert sorted_rows(market.get_net_energy('a', *window)) == \
               sorted_rows(uncached(market, 'a', *window))
        assert market.get_net_energy('a', *window).rows() == [(START + datetime.timedelta(hours=3),
                                                               140, 25)]

    def test_an_agent_with_no_transactions_gets_nothing(self, market):
        feed(market, [transactions([(0, 'a', c.TT_MARKET, 100, 0)])])
        window = (START, START + datetime.timedelta(hours=10))

        assert market.get_net_energy('zzz', *window).is_empty()
        assert sorted_rows(market.get_net_energy('zzz', *window)) == \
               sorted_rows(uncached(market, 'zzz', *window))


class TestTheTypeFilterIsTheWholePoint:
    """`grid` and `levies` clone the netted energy. Counting them triple-counts."""

    def test_fees_are_not_counted_as_traded_energy(self, market):
        feed(market, [transactions([(1, 'a', c.TT_MARKET, 1000, 0),
                                    (1, 'a', c.TT_GRID, 1000, 0),
                                    (1, 'a', c.TT_LEVIES, 1000, 0)])])
        window = (START, START + datetime.timedelta(hours=5))

        assert market.get_net_energy('a', *window).rows() == \
               [(START + datetime.timedelta(hours=1), 1000, 0)]

    def test_the_uncached_path_filters_identically(self, market):
        feed(market, [transactions([(1, 'a', c.TT_MARKET, 1000, 0),
                                    (1, 'a', c.TT_GRID, 1000, 0),
                                    (1, 'a', c.TT_LEVIES, 1000, 0)])])
        window = (START, START + datetime.timedelta(hours=5))

        assert sorted_rows(uncached(market, 'a', *window)) == \
               [(START + datetime.timedelta(hours=1), 1000, 0)]

    def test_another_market_in_the_same_table_is_excluded(self, market):
        other = transactions([(1, 'a', c.TT_MARKET, 500, 0)]).with_columns(
            pl.lit('other').cast(c.TS_MARKET_TRANSACTIONS[c.TC_NAME]).alias(c.TC_NAME))
        feed(market, [pl.concat([transactions([(1, 'a', c.TT_MARKET, 1000, 0)]), other],
                                how='vertical')])
        window = (START, START + datetime.timedelta(hours=5))

        assert market.get_net_energy('a', *window).rows() == \
               [(START + datetime.timedelta(hours=1), 1000, 0)]


class TestTheWindow:

    def test_the_window_is_closed_at_both_ends(self, market):
        feed(market, [transactions([(step, 'a', c.TT_MARKET, 10, 0) for step in range(0, 6)])])
        first = START + datetime.timedelta(hours=2)
        last = START + datetime.timedelta(hours=4)

        steps = [row[0] for row in market.get_net_energy('a', first, last).rows()]

        assert sorted(steps) == [first, START + datetime.timedelta(hours=3), last]

    def test_it_matches_the_uncached_window(self, market):
        feed(market, [transactions([(step, 'a', c.TT_MARKET, 10, 0) for step in range(0, 6)])])
        window = (START + datetime.timedelta(hours=2), START + datetime.timedelta(hours=4))

        assert sorted_rows(market.get_net_energy('a', *window)) == \
               sorted_rows(uncached(market, 'a', *window))


class TestWithoutACache:
    """Dropping the cache must change speed, never answers."""

    def test_a_dropped_cache_still_answers_correctly(self, market):
        feed(market, [transactions([(1, 'a', c.TT_MARKET, 100, 0)])])
        market.set_market_transactions(market.market_transactions)  # no new_rows -> cache dropped
        window = (START, START + datetime.timedelta(hours=5))

        assert market._net_cache is None
        assert market.get_net_energy('a', *window).rows() == \
               [(START + datetime.timedelta(hours=1), 100, 0)]

    def test_a_dropped_cache_still_answers_the_rtc_correctly(self, market):
        feed(market, [transactions([(1, 'a', c.TT_MARKET, 100, 0)])])
        market.set_market_transactions(market.market_transactions)

        assert market._rtc_cache is None
        assert market.get_rtc_market_result('a', START + datetime.timedelta(hours=1)) == 100


def rtc_uncached(db, agent, timestep):
    """The oracle for the cache: the RTC's query written out, with no cache involved.

    Spelled here rather than called through `MarketDB` so that the cache is compared against an
    independent expression instead of against itself.
    """
    frame = db.market_transactions
    frame = frame.filter(pl.col(c.TC_ID_AGENT) == agent)
    frame = frame.filter(pl.col(c.TC_TIMESTEP) == timestep)
    frame = frame.filter(pl.col(c.TC_TYPE_TRANSACTION).is_in(list(MarketDB.NET_TRANSACTION_TYPES)))
    frame = frame.fill_null(0)
    if frame.is_empty():
        return 0
    return frame.select(pl.sum(c.TC_ENERGY_IN).cast(pl.Int64)
                        - pl.sum(c.TC_ENERGY_OUT).cast(pl.Int64)).to_series().to_list()[0]


def rtc_unfiltered(db, agent, timestep):
    """What the RTC computed *before* the type filter was added.

    Kept so the one behavioural change in this area is visible as a test rather than only as a
    commit message: this is the expression that summed `grid` and `levies` rows as traded energy.
    """
    frame = db.market_transactions
    frame = frame.filter(pl.col(c.TC_ID_AGENT) == agent)
    frame = frame.filter(pl.col(c.TC_TIMESTEP) == timestep)
    frame = frame.fill_null(0)
    return frame.select(pl.sum(c.TC_ENERGY_IN).cast(pl.Int64)
                        - pl.sum(c.TC_ENERGY_OUT).cast(pl.Int64)).to_series().to_list()[0]


class TestTheRtcResultMatchesTheCodeItReplaced:
    """The RTC's definition, which now shares the trading strategy's type filter."""

    def test_a_simple_case(self, market):
        feed(market, [transactions([(1, 'a', c.TT_MARKET, 300, 100)])])
        at = START + datetime.timedelta(hours=1)

        assert market.get_rtc_market_result('a', at) == rtc_uncached(market, 'a', at)

    def test_across_many_batches(self, market):
        feed(market, [transactions([(step, 'a', c.TT_MARKET, 10 * step, 3 * step),
                                    (step, 'b', c.TT_RETAIL, 0, 7 * step)])
                      for step in range(1, 25)])

        for step in range(1, 25):
            at = START + datetime.timedelta(hours=step)
            for agent in ('a', 'b'):
                assert market.get_rtc_market_result(agent, at) == \
                       rtc_uncached(market, agent, at), (agent, step)

    def test_rows_for_one_timestep_arriving_in_different_batches_are_summed(self, market):
        """A timestep is cleared more than once, so the fold must accumulate rather than replace.

        Added because a mutation survived without it: folding with `last()` instead of `sum()`
        passed every other test here, since each of them delivers a timestep's rows in a single
        batch and a one-row group is its own sum.
        """
        feed(market, [transactions([(3, 'a', c.TT_MARKET, 100, 0)]),
                      transactions([(3, 'a', c.TT_RETAIL, 40, 0)]),
                      transactions([(3, 'a', c.TT_BALANCING, 0, 25)])])
        at = START + datetime.timedelta(hours=3)

        assert market.get_rtc_market_result('a', at) == 115
        assert market.get_rtc_market_result('a', at) == rtc_uncached(market, 'a', at)

    def test_an_agent_with_nothing_at_that_timestep_gets_zero(self, market):
        feed(market, [transactions([(1, 'a', c.TT_MARKET, 100, 0)])])
        at = START + datetime.timedelta(hours=9)

        assert market.get_rtc_market_result('a', at) == 0
        assert market.get_rtc_market_result('a', at) == rtc_uncached(market, 'a', at)

    def test_fee_rows_are_excluded_here_as_they_are_everywhere_else(self, market):
        """The one behavioural change: the RTC no longer counts the fee clones.

        `grid` and `levies` rows carry the same energy as the transaction they are levied on, so
        summing all three trebles it. This is the case that *would* have been wrong; it does not
        arise in any run measured, because the RTC only ever asks about a timestep that has not
        been settled yet -- 96/96 calls on the shipped example and 1040/1040 on design 6 saw
        `retail` rows only. The filter makes that independent of ordering rather than reliant on it.
        """
        feed(market, [transactions([(1, 'a', c.TT_MARKET, 1000, 0),
                                    (1, 'a', c.TT_GRID, 1000, 0),
                                    (1, 'a', c.TT_LEVIES, 1000, 0)])])
        at = START + datetime.timedelta(hours=1)

        assert market.get_rtc_market_result('a', at) == 1000
        assert rtc_unfiltered(market, 'a', at) == 3000, \
            'the old expression trebled it, which is what the filter removes'
        assert market.get_net_energy('a', at, at).rows() == [(at, 1000, 0)], \
            'and it now agrees with the trading strategy'

    def test_it_is_unchanged_when_there_are_no_fee_rows(self, market):
        """Which is every case that occurs. Same answer as the code it replaces, exactly."""
        feed(market, [transactions([(step, 'a', c.TT_RETAIL, 10 * step, 4 * step)])
                      for step in range(1, 13)])

        for step in range(1, 13):
            at = START + datetime.timedelta(hours=step)
            assert market.get_rtc_market_result('a', at) == rtc_unfiltered(market, 'a', at), step
