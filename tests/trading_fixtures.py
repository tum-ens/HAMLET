"""A minimal agent, timetable and market database for driving the trading strategies in process.

Shared by `tests/unit/executor/utilities/trading/test_strategies.py`, which prices each strategy
by hand, and `tests/unit/executor/agents/test_strategy_substitution.py`, which asks which
strategy `AgentBase.create_bids_offers` picks. They are two halves of one question -- *which*
strategy runs, and *what* it then quotes -- and a stub written twice would let the two halves
drift until each was answering about a different agent.

Everything here is the smallest object that satisfies what `TradingBase.__init__` and
`_preprocess_bids_offers` actually read.

**Three degrees of freedom are set deliberately, and each was found by a mutation that survived
without it.** A review panel defeated the first version of this fixture through all three:

* **The timestep is 15 minutes, not an hour.** `TradingBase` converts a setpoint in W to an
  energy in Wh by multiplying by `dt_hours`. At an hourly step that factor is 1.0, so deleting
  the multiplication changes nothing and the conversion is unfalsifiable -- while
  `config_templates/setup.yaml` ships `timestep: 900`, where the same deletion is a **fourfold
  over-bid by every agent at every timestep**.
* **`market_data(history=...)` can give the agent a trading record.** An agent's position is its
  setpoint *less what it has already traded*, and with an empty history that subtraction is a
  subtraction of zero: flipping its sign, or dropping the `fill_null(0)` that guards it, is
  invisible. Measured on a real `scenario_with_market` run, **161 of 192** constructions of a
  trading strategy saw a non-empty history, so the empty case is the rare one.
* **`timetable(...)` can carry two markets.** `AgentBase.create_bids_offers` loops over the
  markets in the timetable and branches per market on the clearing method. With one market the
  loop body runs once and neither the loop nor the `unique()` subset that feeds it is tested.
"""
import datetime as dt

import polars as pl

import hamlet.constants as c
from hamlet.executor.utilities.database.market_db import MarketDB

REGION, MARKET, NAME, ENERGY = 'r1', 'electricity', 'lem', 'electricity'

#: A second market in the same region, used only by the two-market timetable.
OTHER_NAME = 'day_ahead'

TIMESTAMP = dt.datetime(2021, 3, 24, 0, 0, tzinfo=dt.timezone.utc)

#: The timestep, in minutes. **Not 60** -- see the module docstring; the shipped default is 900 s.
STEP_MINUTES = 15
STEPS = [TIMESTAMP + dt.timedelta(minutes=STEP_MINUTES * step) for step in range(4)]

#: What the agent expects the retailer to pay it for feed-in, and to charge it. The names follow
#: the executor's convention: `_out` is the direction the agent pays for.
FORECAST_IN = 500
FORECAST_OUT = 1500

#: The agent's net position per timestep, in **W**. Positive is a purchase, negative a sale. The
#: final zero is there on purpose: `_postprocess_bids_offers` drops rows with no position, and a
#: fixture without one would let that filter be deleted unnoticed.
SETPOINTS = [1000, -500, 2000, 0]
#:
#: The energies these become at `STEP_MINUTES` are written out as literals in
#: `test_strategies.py` (`BID_0`, `OFFER_1`, `BID_2`), which is the only place they are used. They
#: deliberately do **not** live here as a second constant derived from `SETPOINTS` by the same
#: arithmetic the conversion under test performs -- that would be one fact with two homes, and
#: the derived one would check the code against itself.

#: The trading horizon, in seconds. Four steps of `STEP_MINUTES`, so one hour admits all of them.
HORIZON = 4 * STEP_MINUTES * 60

#: The columns a trading strategy emits, and therefore the columns a market receives. **Eleven,
#: not the thirteen of `c.TS_BIDS_OFFERS`.** `TradingBase.__init__` creates an empty frame with
#: that schema and `_preprocess_bids_offers` then rebuilds the frame from the timetable, so
#: `price_in` and `price_out` never exist on a real book rather than being dropped from one.
#: Verified against an instrumented `scenario_with_market` run.
#:
#: It lives here, next to the strategies that produce it, and is imported by
#: `tests/unit/executor/markets/test_clearing_by_hand.py`, whose hand-written books have to match
#: it. One definition, checked at both ends: `test_the_quoted_columns_are_the_ones_the_market_
#: reads` runs a real strategy against it, so a strategy that started leaking an internal column
#: fails there rather than reaching the market.
BOOK_COLUMNS = [c.TC_TIMESTAMP, c.TC_TIMESTEP, c.TC_REGION, c.TC_MARKET, c.TC_NAME,
                c.TC_TYPE_ENERGY, c.TC_ID_AGENT, c.TC_ENERGY_IN, c.TC_ENERGY_OUT,
                c.TC_PRICE_PU_IN, c.TC_PRICE_PU_OUT]

#: `Linear`'s two shipped shape parameters, from `config_templates/agents.yaml`.
STEPS_TO_FINAL = 1
STEPS_FROM_INIT = 0


class Agent:
    """The attributes `TradingBase` reads off an agent, and the one it writes back."""

    def __init__(self, agent_id, strategy, horizon=HORIZON, setpoints=None,
                 forecast_in=FORECAST_IN, forecast_out=FORECAST_OUT,
                 steps_to_final=STEPS_TO_FINAL, steps_from_init=STEPS_FROM_INIT, steps=None):
        steps = STEPS if steps is None else steps
        self.agent_id = agent_id
        self.account = {
            c.K_GENERAL: {'agent_id': agent_id},
            c.K_EMS: {c.K_MARKET: {'strategy': strategy, 'horizon': horizon,
                                   c.MTS_LINEAR: {'steps_to_final': steps_to_final,
                                                  'steps_from_init': steps_from_init}}},
        }
        positions = (SETPOINTS if setpoints is None else setpoints)[:len(steps)]
        # A column per market the agent could trade in: the executor names setpoint columns
        # `<market name>_<energy type>`, and an agent in a region with two markets has one for
        # each. Both carry the same positions here, which is enough for the loop in
        # `AgentBase.create_bids_offers` to have two markets to run over.
        self.setpoints = pl.DataFrame({
            c.TC_TIMESTAMP: steps,
            f'{NAME}_{ENERGY}': positions,
            f'{OTHER_NAME}_{ENERGY}': positions,
        }).cast({c.TC_TIMESTAMP: c.SCHEMA[c.TC_TIMESTAMP]})
        self.forecasts = pl.DataFrame({
            c.TC_TIMESTEP: steps,
            f'{c.TC_ENERGY}_{c.TC_PRICE}_{c.PF_IN}': [forecast_in] * len(steps),
            f'{c.TC_ENERGY}_{c.TC_PRICE}_{c.PF_OUT}': [forecast_out] * len(steps),
        }).cast({c.TC_TIMESTEP: c.SCHEMA[c.TC_TIMESTEP]})
        self.bids_offers = None


def timetable(clearing_method=c.MCM_PDA, second_market=None, steps=None):
    """One row per timestep, all issued at `TIMESTAMP`. `dt` is read off the first two rows.

    `clearing_method` is what `AgentBase.create_bids_offers` reads to decide whether the
    configured strategy applies; the strategies themselves never look at it.

    `second_market` adds a second market to the same region, with its own clearing method, so the
    loop in `create_bids_offers` runs more than once and the two branches of `clears_a_market` can
    both be taken in one call. Pass the second market's clearing method.
    """
    steps = STEPS if steps is None else steps
    markets = [(NAME, clearing_method)]
    if second_market is not None:
        markets.append((OTHER_NAME, second_market))

    columns = {
        c.TC_TIMESTAMP: [TIMESTAMP] * (len(steps) * len(markets)),
        c.TC_TIMESTEP: steps * len(markets),
        c.TC_REGION: [REGION] * (len(steps) * len(markets)),
        c.TC_MARKET: [MARKET] * (len(steps) * len(markets)),
        c.TC_NAME: [name for name, _ in markets for _ in steps],
        c.TC_TYPE_ENERGY: [ENERGY] * (len(steps) * len(markets)),
        c.TC_CLEARING_METHOD: [method for _, method in markets for _ in steps],
    }
    frame = pl.DataFrame(columns)
    # A `method: null` config yields a real None, which polars types as Null; leave that column
    # alone in that case so both spellings reach `clears_a_market` the way a timetable would.
    castable = {col: c.SCHEMA[col] for col in columns
                if col != c.TC_CLEARING_METHOD or all(m is not None for _, m in markets)}
    return frame.cast(castable)


def market_data(history=None, name=NAME):
    """A `MarketDB` holding `history`, so the agent's net position is setpoint less traded.

    The real class rather than a stub, because `TradingBase` goes through `get_net_energy` and
    that method's behaviour is part of what the strategies rely on.

    `history` is `[(step_index, energy_in, energy_out), ...]` in Wh, folded straight into the net
    cache the way `MarketDB` itself would. `None` means an agent that has not traded yet -- which
    is the *rare* case in a real run, so a test that only ever uses it is testing the rare one.

    **`_rtc_cache` is set as well as `_net_cache`,** even though nothing here reads it: `MarketDB`
    exposes both through `getattr(self, ..., None)`, so an attribute this fixture forgets fails
    *open* rather than loudly, and a future reader would silently get `None`.
    """
    database = object.__new__(MarketDB)
    database.market_type, database.market_name = MARKET, name
    database.market_transactions = pl.DataFrame(schema=c.TS_MARKET_TRANSACTIONS)
    database._net_cache = net_cache(history)
    database._rtc_cache = pl.DataFrame(schema=MarketDB.NET_SCHEMA)
    return database


def net_cache(history, agent_id='agent_1'):
    """`history` as the frame `MarketDB.get_net_energy` reads, in `MarketDB.NET_SCHEMA`."""
    if not history:
        return pl.DataFrame(schema=MarketDB.NET_SCHEMA)
    return pl.DataFrame({
        c.TC_TIMESTEP: [STEPS[step] for step, _, _ in history],
        c.TC_ID_AGENT: [agent_id] * len(history),
        c.TC_ENERGY_IN: [energy_in for _, energy_in, _ in history],
        c.TC_ENERGY_OUT: [energy_out for _, _, energy_out in history],
    }).cast(MarketDB.NET_SCHEMA)
