"""L2 — MPC `Market` component physics.

Pins the buy/sell price mapping. See `tests/README.md` for the in/out convention.
"""
import pandas as pd
import pytest

import hamlet.constants as c
from hamlet.executor.utilities.controller.fbc.mpc.linopy.components import Market

# Retailer-perspective prices, in the integer units HAMLET uses (0.1 ct/kWh).
RETAIL_PRICE = 3200  # what the retailer sells at  -> the agent's BUY price
FEED_IN_PRICE = 800  # what the retailer buys at   -> the agent's SELL price

GRID_FEE_CONSUMPTION = 400  # charged on consumption only
LEVY_CONSUMPTION = 1800  # charged on consumption only


@pytest.fixture
def forecasts(timesteps):
    """A forecast frame shaped like the retailer training data the MPC actually receives.

    `energy_*` follows the retailer convention (`_out` = retailer sells = agent buys).
    `grid_*` and `levies_*` follow the agent convention (`_in` = agent consumes), matching the
    shipped `input_data/retailers/lem/{grid,levies}.csv`.
    """
    n = len(timesteps)
    return {
        f'{c.TC_ENERGY}_{c.TC_ENERGY}_{c.PF_IN}': [1_000_000] * n,
        f'{c.TC_ENERGY}_{c.TC_ENERGY}_{c.PF_OUT}': [1_000_000] * n,
        f'{c.TC_ENERGY}_{c.TC_PRICE}_{c.PF_OUT}': [RETAIL_PRICE] * n,
        f'{c.TC_ENERGY}_{c.TC_PRICE}_{c.PF_IN}': [FEED_IN_PRICE] * n,
        f'{c.TT_GRID}_{c.TT_MARKET}_{c.PF_IN}': [GRID_FEE_CONSUMPTION] * n,
        f'{c.TT_GRID}_{c.TT_MARKET}_{c.PF_OUT}': [0] * n,
        f'{c.TT_LEVIES}_{c.TC_PRICE}_{c.PF_IN}': [LEVY_CONSUMPTION] * n,
        f'{c.TT_LEVIES}_{c.TC_PRICE}_{c.PF_OUT}': [0] * n,
    }


@pytest.fixture
def market(forecasts, timesteps, delta):
    return Market('market', forecasts=forecasts, timesteps=timesteps, delta=delta)


def test_buy_price_is_the_retailers_sell_price(market, timesteps):
    """Regression: the MPC read the energy price columns the wrong way round.

    `energy_price_out` is the price at which the retailer *sells*, so it is what the agent pays
    when buying. Reading `energy_price_in` instead made buying look cheaper than selling.
    """
    assert (market.price_buy == RETAIL_PRICE).all()
    assert (market.price_sell == FEED_IN_PRICE).all()


def test_buying_is_never_cheaper_than_selling(market):
    """No risk-free arbitrage against the retailer.

    With the columns flipped the agent saw a buy price of 8 ct and a sell price of 30 ct, i.e. a
    money pump. Total cost of importing must exceed total revenue from exporting.
    """
    cost_to_buy = market.price_buy + market.grid_buy + market.levies_buy
    revenue_to_sell = market.price_sell - market.grid_sell - market.levies_sell

    assert (cost_to_buy > revenue_to_sell).all()


def test_grid_and_levy_charges_fall_on_consumption(market):
    """Grid fees and levies are charged on what the agent consumes, not on what it feeds in.

    These columns are *not* flipped: the shipped grid/levies inputs already use the agent
    convention. Pinned so the energy-price fix is not over-applied to them.
    """
    assert (market.grid_buy == GRID_FEE_CONSUMPTION).all()
    assert (market.levies_buy == LEVY_CONSUMPTION).all()
    assert (market.grid_sell == 0).all()
    assert (market.levies_sell == 0).all()


def test_price_series_are_indexed_by_timestep(market, timesteps):
    """Every price series must align with the horizon, or linopy silently broadcasts NaN."""
    for series in (market.price_buy, market.price_sell, market.grid_buy,
                   market.grid_sell, market.levies_buy, market.levies_sell):
        assert isinstance(series, pd.Series)
        assert series.index.equals(timesteps)
