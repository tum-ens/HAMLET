"""Integration -- what the RTC actually receives as its settled energy.

`RtcBase._get_market_results` used to scan `market_transactions` inline; it now asks
`MarketDB.get_rtc_market_result`. Two claims were made when that changed, and neither was pinned
anywhere that reads the *controller* rather than the database:

1. the RTC gets the **netted** energy, `energy_in - energy_out`, for its own agent at its own
   timestep, and
2. it counts only `retail | market | balancing` -- not the `grid` and `levies` rows, which clone
   the netted transactions and would treble the number.

The docstring on `_get_market_results` claimed the opposite of (2) for a while after the filter
landed, which is how this file came to exist: the code was right and the comment was not, and
nothing failed. These tests read through the controller, so a future change to how it reaches the
database cannot quietly reintroduce either problem.
"""
import datetime

import polars as pl
import pytest

import hamlet.constants as c
from hamlet.executor.utilities.controller.rtc.rtc_base import RtcBase
from hamlet.executor.utilities.database.market_db import MarketDB

MARKET_TYPE = c.ET_ELECTRICITY
MARKET_NAME = 'continuous'
AGENT = 'agent-under-test'
AT = datetime.datetime(2023, 1, 1, 12, tzinfo=datetime.timezone.utc)

COLUMNS = (c.TC_TIMESTEP, c.TC_MARKET, c.TC_NAME, c.TC_TYPE_TRANSACTION, c.TC_ID_AGENT,
           c.TC_ENERGY_IN, c.TC_ENERGY_OUT)
SCHEMA = {column: c.TS_MARKET_TRANSACTIONS[column] for column in COLUMNS}


def transactions(rows):
    """`rows` are (offset_hours, agent, transaction type, energy_in, energy_out)."""
    return pl.DataFrame(
        {c.TC_TIMESTEP: [AT + datetime.timedelta(hours=offset) for offset, *_ in rows],
         c.TC_MARKET: [MARKET_TYPE] * len(rows),
         c.TC_NAME: [MARKET_NAME] * len(rows),
         c.TC_TYPE_TRANSACTION: [kind for _, _, kind, _, _ in rows],
         c.TC_ID_AGENT: [agent for _, agent, _, _, _ in rows],
         c.TC_ENERGY_IN: [energy_in for *_, energy_in, _ in rows],
         c.TC_ENERGY_OUT: [energy_out for *_, energy_out in rows]},
        schema=SCHEMA)


@pytest.fixture
def controller(monkeypatch):
    """An `RtcBase` reduced to the parts `_get_market_results` uses.

    Built rather than constructed for real because the question is only what this method returns
    given a market database, and a real controller needs a whole scenario behind it.
    """
    monkeypatch.setattr(MarketDB, '__init__', lambda self, **kwargs: None)
    market = MarketDB()
    market.market_type = MARKET_TYPE
    market.market_name = MARKET_NAME
    market.market_transactions = pl.DataFrame(schema=SCHEMA)
    market._net_cache = pl.DataFrame(schema=MarketDB.NET_SCHEMA)
    market._rtc_cache = pl.DataFrame(schema=MarketDB.NET_SCHEMA)

    instance = RtcBase.__new__(RtcBase)
    instance.market = {MARKET_TYPE: {MARKET_NAME: market}}
    instance.timestamp = AT

    class Agent:
        agent_id = AGENT

    instance.agent = Agent()
    return instance, market


def feed(market, rows):
    batch = transactions(rows)
    combined = batch if market.market_transactions.is_empty() else pl.concat(
        [market.market_transactions, batch], how='vertical')
    market.set_market_transactions(combined, new_rows=batch)


def test_the_rtc_gets_the_netted_energy(controller):
    instance, market = controller
    feed(market, [(0, AGENT, c.TT_RETAIL, 900, 300)])

    assert instance._get_market_results() == {MARKET_NAME: 600}


def test_fee_rows_are_not_counted(controller):
    """`grid` and `levies` clone the traded energy; counting them would treble it."""
    instance, market = controller
    feed(market, [(0, AGENT, c.TT_RETAIL, 900, 0),
                  (0, AGENT, c.TT_GRID, 900, 0),
                  (0, AGENT, c.TT_LEVIES, 900, 0)])

    assert instance._get_market_results() == {MARKET_NAME: 900}


def test_only_this_agent_counts(controller):
    instance, market = controller
    feed(market, [(0, AGENT, c.TT_RETAIL, 100, 0),
                  (0, 'someone-else', c.TT_RETAIL, 5000, 0)])

    assert instance._get_market_results() == {MARKET_NAME: 100}


def test_only_this_timestep_counts(controller):
    instance, market = controller
    feed(market, [(0, AGENT, c.TT_RETAIL, 100, 0),
                  (1, AGENT, c.TT_RETAIL, 5000, 0),
                  (-1, AGENT, c.TT_RETAIL, 7000, 0)])

    assert instance._get_market_results() == {MARKET_NAME: 100}


def test_rows_arriving_in_separate_clearings_are_summed(controller):
    instance, market = controller
    feed(market, [(0, AGENT, c.TT_RETAIL, 500, 0)])
    feed(market, [(0, AGENT, c.TT_MARKET, 0, 200)])
    feed(market, [(0, AGENT, c.TT_BALANCING, 30, 0)])

    assert instance._get_market_results() == {MARKET_NAME: 330}


def test_an_agent_that_traded_nothing_gets_zero(controller):
    """Not None, and not an empty dict -- the controller does arithmetic with this."""
    instance, market = controller
    feed(market, [(0, 'someone-else', c.TT_RETAIL, 100, 0)])

    assert instance._get_market_results() == {MARKET_NAME: 0}
