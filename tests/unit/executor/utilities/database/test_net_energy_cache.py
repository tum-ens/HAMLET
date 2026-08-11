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
    """A MarketDB with no folder behind it -- only the netting logic is under test."""
    monkeypatch.setattr(MarketDB, '__init__', lambda self, **kwargs: None)
    db = MarketDB()
    db.market_type = MARKET_TYPE
    db.market_name = MARKET_NAME
    db.market_transactions = pl.DataFrame()
    db._net_cache = pl.DataFrame(schema=MarketDB.NET_SCHEMA)
    return db


def feed(db, batches):
    """Append each batch the way `post_markets_to_region` does, folding as it goes."""
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
