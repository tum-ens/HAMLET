"""A real `ElectricityMarket` that can run `clear,settle` over a hand-written book.

`tests/unit/executor/markets/test_clearing_by_hand.py` (L3) drives the *clearing* half of the
market. This module drives the half after it: balancing energy, grid fees and levies -- the
accounting L4 is about. It is a separate harness because settlement needs two things clearing
does not, and both are where the accounting defects lived:

* **A transaction history for the delivery timestep.** A delivery timestep is cleared repeatedly
  as the trading horizon rolls towards it, so by the time it settles an agent may already have
  bought at one clearing and sold at another. Netting only means anything when it does -- with a
  one-directional history, gross and net are the same number and `eec60f0` is unfalsifiable.
* **A forecaster per agent.** `__apply_grid_levies` overwrites the grid fee it took from the
  retailer with the agent's own forecast of it (variable grid fees), for every agent, every time.

**The rates below are deliberately distinct and non-zero in every direction**, which the shipped
retailer files are not: they charge 0 on feed-in, and a fixture that copied them could not tell a
mapping that reads `grid_retail_*` from one that reads `grid_market_*` on the selling side, both
being zero. Charging a feed-in grid fee is not what a real tariff does; it is what makes the
column mapping observable. `tests/unit/creator/markets/test_retailer_convention.py` is where the
*shipped* values are pinned.
"""
import datetime as dt
from types import SimpleNamespace

import polars as pl

import hamlet.constants as c
from hamlet.executor.markets.electricity import ElectricityMarket
from hamlet.executor.utilities.database.market_db import MarketDB
from tests.trading_fixtures import BOOK_COLUMNS

REGION, MARKET, NAME, ENERGY = 'r1', 'electricity', 'lem', 'electricity'
RETAILER_ID = 'retailer'

#: When the clearing happens, and which delivery timestep it is for. **Different values**, so a
#: row stamped with one where the other belongs is visible rather than coincidentally right.
TIMESTAMP = dt.datetime(2021, 3, 24, 0, 0, tzinfo=dt.timezone.utc)
TIMESTEP = dt.datetime(2021, 3, 24, 1, 0, tzinfo=dt.timezone.utc)

#: The retailer's energy prices, in EURe-7/Wh, in the executor's convention: `_out` is the
#: direction the agent pays for, so the retailer *sells* at `RETAILER_SELLS`.
RETAILER_BUYS = 500
RETAILER_SELLS = 1500

#: Balancing prices, same convention. An agent short of energy at settlement buys balancing at
#: `BALANCING_SELLS`; one long sells it at `BALANCING_BUYS`.
BALANCING_BUYS = 400
BALANCING_SELLS = 1600

#: Grid fees and levies, per direction, all six distinct and none zero -- see the module
#: docstring. `_OUT` is the rate the agent pays when it draws from the grid.
GRID_MARKET_OUT, GRID_MARKET_IN = 400, 40
GRID_RETAIL_OUT, GRID_RETAIL_IN = 800, 80
LEVIES_OUT, LEVIES_IN = 1800, 180

#: The columns of `market_transactions`, in schema order.
TRANSACTION_COLUMNS = list(c.TS_MARKET_TRANSACTIONS)


def book(rows_):
    """A bids-and-offers table from `(agent, energy_in, energy_out, price_pu_in, price_pu_out)`.

    Both prices are always real numbers, including on the side an agent is not trading, because
    every shipped strategy quotes both -- see `test_clearing_by_hand.book` for the mutation that
    established this.
    """
    if not rows_:
        return pl.DataFrame(schema={col: c.SCHEMA[col] for col in BOOK_COLUMNS})

    frame = pl.DataFrame(
        [{c.TC_TIMESTAMP: TIMESTAMP, c.TC_TIMESTEP: TIMESTEP, c.TC_REGION: REGION,
          c.TC_MARKET: MARKET, c.TC_NAME: NAME, c.TC_TYPE_ENERGY: ENERGY,
          c.TC_ID_AGENT: agent, c.TC_ENERGY_IN: energy_in, c.TC_ENERGY_OUT: energy_out,
          c.TC_PRICE_PU_IN: price_in, c.TC_PRICE_PU_OUT: price_out}
         for agent, energy_in, energy_out, price_in, price_out in rows_])
    return frame.select(BOOK_COLUMNS).cast({col: c.SCHEMA[col] for col in BOOK_COLUMNS})


def retailer_table(energy_in=0, energy_out=0):
    """The retailer's row for `TIMESTEP`, in the shape the Creator writes it.

    Defaults to zero quantity on both sides, which drops the retailer out of the book at
    `__split_bids_offers` and leaves a pure agent-to-agent clearing.
    """
    frame = pl.DataFrame([{
        c.TC_TIMESTAMP: TIMESTEP, c.TC_REGION: REGION, c.TC_MARKET: MARKET, c.TC_NAME: NAME,
        c.TC_ID_AGENT: RETAILER_ID, c.TC_TYPE_ENERGY: ENERGY,
        f'{c.TT_ENERGY}_{c.TC_PRICE}_{c.PF_IN}': RETAILER_BUYS,
        f'{c.TT_ENERGY}_{c.TC_PRICE}_{c.PF_OUT}': RETAILER_SELLS,
        f'{c.TT_ENERGY}_{c.TC_ENERGY}_{c.PF_IN}': energy_in,
        f'{c.TT_ENERGY}_{c.TC_ENERGY}_{c.PF_OUT}': energy_out,
        f'{c.TT_ENERGY}_{c.TC_QUALITY}': 0,
        f'{c.TT_GRID}_{c.TT_MARKET}_{c.PF_IN}': GRID_MARKET_IN,
        f'{c.TT_GRID}_{c.TT_MARKET}_{c.PF_OUT}': GRID_MARKET_OUT,
        f'{c.TT_GRID}_{c.TT_RETAIL}_{c.PF_IN}': GRID_RETAIL_IN,
        f'{c.TT_GRID}_{c.TT_RETAIL}_{c.PF_OUT}': GRID_RETAIL_OUT,
        f'{c.TT_LEVIES}_{c.TC_PRICE}_{c.PF_IN}': LEVIES_IN,
        f'{c.TT_LEVIES}_{c.TC_PRICE}_{c.PF_OUT}': LEVIES_OUT,
        f'{c.TT_BALANCING}_{c.TC_PRICE}_{c.PF_IN}': BALANCING_BUYS,
        f'{c.TT_BALANCING}_{c.TC_PRICE}_{c.PF_OUT}': BALANCING_SELLS,
        f'{c.TT_BALANCING}_{c.TC_ENERGY}_{c.PF_IN}': 1_000_000,
        f'{c.TT_BALANCING}_{c.TC_ENERGY}_{c.PF_OUT}': 1_000_000,
    }])
    return frame.cast({c.TC_TIMESTAMP: c.SCHEMA[c.TC_TIMESTAMP], c.TC_REGION: pl.Categorical,
                       c.TC_MARKET: pl.Categorical, c.TC_NAME: pl.Categorical,
                       c.TC_TYPE_ENERGY: pl.Categorical})


def forecast_table(**overrides):
    """One agent's forecast of the four grid-fee columns, for `TIMESTEP`.

    Named from the **retailer's** point of view, like every retailer column: `_out` is the rate
    the agent pays on consumption. Defaults to the retailer's own rates, so a test that does not
    care about variable fees sees the flat tariff; pass an override to make the two differ.
    """
    values = {f'{c.TT_GRID}_{c.TT_MARKET}_{c.PF_OUT}': GRID_MARKET_OUT,
              f'{c.TT_GRID}_{c.TT_MARKET}_{c.PF_IN}': GRID_MARKET_IN,
              f'{c.TT_GRID}_{c.TT_RETAIL}_{c.PF_OUT}': GRID_RETAIL_OUT,
              f'{c.TT_GRID}_{c.TT_RETAIL}_{c.PF_IN}': GRID_RETAIL_IN}
    values.update(overrides)
    return pl.DataFrame({c.TC_TIMESTAMP: [TIMESTEP],
                         **{name: [value] for name, value in values.items()}})


def transactions(rows_, timestep=TIMESTEP):
    """A `market_transactions` frame from `(agent, kind, energy_in, energy_out, price_in, price_out)`.

    Used to seed the history a delivery timestep already carries when it settles. Per-unit prices
    are left at zero: nothing in the settlement path reads them back, and the totals are what the
    money assertions are about.
    """
    if not rows_:
        return pl.DataFrame(schema=c.TS_MARKET_TRANSACTIONS)

    frame = pl.DataFrame(
        [{c.TC_TIMESTAMP: TIMESTAMP, c.TC_TIMESTEP: timestep, c.TC_REGION: REGION,
          c.TC_MARKET: MARKET, c.TC_NAME: NAME, c.TC_TYPE_ENERGY: ENERGY,
          c.TC_TYPE_TRANSACTION: kind, c.TC_ID_AGENT: agent,
          c.TC_ENERGY_IN: energy_in, c.TC_ENERGY_OUT: energy_out,
          c.TC_PRICE_PU_IN: 0, c.TC_PRICE_PU_OUT: 0,
          c.TC_PRICE_IN: price_in, c.TC_PRICE_OUT: price_out}
         for agent, kind, energy_in, energy_out, price_in, price_out in rows_])
    return frame.select(TRANSACTION_COLUMNS).cast(dict(c.TS_MARKET_TRANSACTIONS))


class SettlementDatabase:
    """The two calls the market makes on the `Database` when it clears and settles.

    A stub rather than a real `Database`, which would need a whole scenario on disk. The market's
    own `__init__` and `__apply_grid_levies` run unmodified against it, so a mutation to either
    is visible here.
    """

    def __init__(self, bids_offers, forecasts=None):
        self.bids_offers = bids_offers
        self.forecasts = forecasts or {}
        self.agents_asked = []

    def get_bids_offers(self, region, market_type, market_name, timestep):
        return self.bids_offers

    def get_agent_data(self, region, agent_id):
        self.agents_asked.append(agent_id)
        frame = self.forecasts.get(agent_id, forecast_table())
        return SimpleNamespace(forecaster=SimpleNamespace(
            train_data={f'{NAME}_{c.TT_RETAIL}': {c.K_TARGET: frame}}))


def make_market(rows_, history=None, retailer=None, forecasts=None, method=c.MCM_PDA,
                pricing=c.MP_UNIFORM, actions=f'{c.MA_CLEAR},{c.MA_SETTLE}'):
    """A real `ElectricityMarket` over a hand-written book, ready for `execute()`.

    `history` seeds `market_transactions` with what this delivery timestep already carries from
    earlier clearings -- which is what `self.transactions_prev` reads.
    """
    database = object.__new__(MarketDB)
    database.market_type, database.market_name = MARKET, NAME
    database.market_transactions = transactions(history or [])
    database.bids_cleared = pl.DataFrame(schema=c.TS_BIDS_CLEARED)
    database.bids_uncleared = pl.DataFrame(schema=c.TS_BIDS_UNCLEARED)
    database.offers_cleared = pl.DataFrame(schema=c.TS_OFFERS_CLEARED)
    database.offers_uncleared = pl.DataFrame(schema=c.TS_OFFERS_UNCLEARED)
    database.retailer = retailer_table() if retailer is None else retailer

    tasks = {c.TC_TIMESTAMP: TIMESTAMP, c.TC_TIMESTEP: TIMESTEP, c.TC_REGION: REGION,
             c.TC_MARKET: MARKET, c.TC_NAME: NAME, c.TC_TYPE_ENERGY: ENERGY,
             c.TC_ACTIONS: actions, c.TC_CLEARING_TYPE: c.MCT_EX_ANTE,
             c.TC_CLEARING_METHOD: method, c.TC_CLEARING_PRICING: pricing,
             c.TC_COUPLING: None}

    return ElectricityMarket(market=database, tasks=tasks,
                             database=SettlementDatabase(book(rows_), forecasts))


def settle(rows_, **kwargs):
    """Run a book through the real `execute()` and return the `MarketDB` it wrote.

    The market database rather than the market object, because that is what the executor keeps
    and what the result files are written from -- asserting on the market's own attributes would
    pass even if `__update_database` stopped writing them through.
    """
    return make_market(rows_, **kwargs).execute()


def rows(market):
    """`[(kind, agent, energy_in, energy_out, price_in, price_out), ...]`, ids as plain strings.

    Categoricals are cast to `Utf8` first: polars compares two Categorical columns by their
    *encoding* unless they share a string cache, and a comparison that silently answers about
    encodings once invented 369 file differences in this repository.
    """
    frame = market.market_transactions.with_columns(
        pl.col(c.TC_TYPE_TRANSACTION).cast(pl.Utf8), pl.col(c.TC_ID_AGENT).cast(pl.Utf8))
    return [(row[c.TC_TYPE_TRANSACTION], row[c.TC_ID_AGENT],
             row[c.TC_ENERGY_IN], row[c.TC_ENERGY_OUT],
             row[c.TC_PRICE_IN], row[c.TC_PRICE_OUT])
            for row in frame.to_dicts()]


def ledger(market):
    """`{(kind, agent): (energy_in, energy_out, price_in, price_out)}` for the settled table.

    **Refuses to collapse two rows for the same (kind, agent) into one.** An agent clearing
    against two counterparties really does occupy two `market` rows -- measured on a
    `scenario_with_market` run, 31 clearings have a repeated agent in `bids_cleared` -- and a
    reduction that hid that would let the market emit a second, wrong row for an agent whose last
    row happened to be right. Tests about such a book read `rows()` instead.
    """
    collected = {}
    for row in rows(market):
        key = (row[0], row[1])
        assert key not in collected, (
            f'{key} appears more than once in the transactions table, which this reduction '
            f'cannot represent; use rows() for that book')
        collected[key] = row[2:]
    return collected
