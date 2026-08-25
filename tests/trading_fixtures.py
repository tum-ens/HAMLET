"""A minimal agent, timetable and market database for driving the trading strategies in process.

Shared by `tests/unit/executor/utilities/trading/test_strategies.py`, which prices each strategy
by hand, and `tests/unit/executor/agents/test_strategy_substitution.py`, which asks which
strategy `AgentBase.create_bids_offers` picks. They are two halves of one question -- *which*
strategy runs, and *what* it then quotes -- and a stub written twice would let the two halves
drift until each was answering about a different agent.

Everything here is the smallest object that satisfies what `TradingBase.__init__` and
`_preprocess_bids_offers` actually read. The `MarketDB` is the real class with no transactions
recorded, so the agent's net position is exactly its setpoint.
"""
import datetime as dt

import polars as pl

import hamlet.constants as c
from hamlet.executor.utilities.database.market_db import MarketDB

REGION, MARKET, NAME, ENERGY = 'r1', 'electricity', 'lem', 'electricity'
TIMESTAMP = dt.datetime(2021, 3, 24, 0, 0, tzinfo=dt.timezone.utc)
STEPS = [TIMESTAMP + dt.timedelta(hours=hour) for hour in range(4)]

#: What the agent expects the retailer to pay it for feed-in, and to charge it. The names follow
#: the executor's convention: `_out` is the direction the agent pays for.
FORECAST_IN = 500
FORECAST_OUT = 1500

#: The agent's net position per timestep, in W. Positive is a purchase, negative a sale. The
#: timesteps are hourly, so these are also the energies in Wh. The final zero is there on
#: purpose: `_postprocess_bids_offers` drops rows with no position, and a fixture without one
#: would let that filter be deleted unnoticed.
SETPOINTS = [1000, -500, 2000, 0]

#: The trading horizon, in seconds. Four hourly steps, so 4 h admits all of them.
HORIZON = 4 * 3600

#: `Linear`'s two shipped shape parameters, from `config_templates/agents.yaml`.
STEPS_TO_FINAL = 1
STEPS_FROM_INIT = 0


class Agent:
    """The attributes `TradingBase` reads off an agent, and the one it writes back."""

    def __init__(self, agent_id, strategy, horizon=HORIZON, setpoints=None,
                 forecast_in=FORECAST_IN, forecast_out=FORECAST_OUT):
        self.account = {
            c.K_GENERAL: {'agent_id': agent_id},
            c.K_EMS: {c.K_MARKET: {'strategy': strategy, 'horizon': horizon,
                                   c.MTS_LINEAR: {'steps_to_final': STEPS_TO_FINAL,
                                                  'steps_from_init': STEPS_FROM_INIT}}},
        }
        self.setpoints = pl.DataFrame({
            c.TC_TIMESTAMP: STEPS,
            f'{NAME}_{ENERGY}': SETPOINTS if setpoints is None else setpoints,
        }).cast({c.TC_TIMESTAMP: c.SCHEMA[c.TC_TIMESTAMP]})
        self.forecasts = pl.DataFrame({
            c.TC_TIMESTEP: STEPS,
            f'{c.TC_ENERGY}_{c.TC_PRICE}_{c.PF_IN}': [forecast_in] * len(STEPS),
            f'{c.TC_ENERGY}_{c.TC_PRICE}_{c.PF_OUT}': [forecast_out] * len(STEPS),
        }).cast({c.TC_TIMESTEP: c.SCHEMA[c.TC_TIMESTEP]})
        self.bids_offers = None


def timetable(clearing_method=c.MCM_PDA):
    """One row per timestep, all issued at `TIMESTAMP`. `dt` is read off the first two rows.

    `clearing_method` is what `AgentBase.create_bids_offers` reads to decide whether the
    configured strategy applies; the strategies themselves never look at it.
    """
    columns = {
        c.TC_TIMESTAMP: [TIMESTAMP] * len(STEPS),
        c.TC_TIMESTEP: STEPS,
        c.TC_REGION: [REGION] * len(STEPS),
        c.TC_MARKET: [MARKET] * len(STEPS),
        c.TC_NAME: [NAME] * len(STEPS),
        c.TC_TYPE_ENERGY: [ENERGY] * len(STEPS),
        c.TC_CLEARING_METHOD: [clearing_method] * len(STEPS),
    }
    frame = pl.DataFrame(columns)
    # `method: null` in a config yields a real None, which polars would type as Null; keep the
    # column a string so both spellings reach `clears_a_market` the way a timetable would.
    return frame.cast({col: c.SCHEMA[col] for col in columns
                       if col != c.TC_CLEARING_METHOD or clearing_method is not None})


def market_data():
    """A `MarketDB` that has recorded no transactions, so the agent's net position is its setpoint.

    The real class rather than a stub, because `TradingBase` goes through `get_net_energy` and
    that method's empty-window behaviour is part of what the strategies rely on.
    """
    database = object.__new__(MarketDB)
    database.market_type, database.market_name = MARKET, NAME
    database.market_transactions = pl.DataFrame(schema=c.TS_MARKET_TRANSACTIONS)
    database._net_cache = pl.DataFrame(schema=MarketDB.NET_SCHEMA)
    return database
