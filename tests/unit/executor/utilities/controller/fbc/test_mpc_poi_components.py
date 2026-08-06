"""Unit — the PyOptInterface MPC components must construct and agree with the linopy ones.

This backend was unusable: its market component read `energy_quantity_sell` /
`energy_quantity_buy`, columns that exist nowhere in the retailer data, so constructing it
raised KeyError. Its EV component also still carried the pre-fix state of charge and `min_soc`
scheme long after the linopy one was corrected.

These tests construct the components directly. They need no solver, so they run everywhere --
which matters, because PyOptInterface's HiGHS library is usually not loadable.
"""
import datetime

import numpy as np
import polars as pl
import pytest

import hamlet.constants as c
from hamlet.executor.utilities.controller.fbc.mpc.linopy import components as linopy_components
from hamlet.executor.utilities.controller.fbc.mpc.poi import components as poi_components

MARKET = 'continuous'
EV = 'ev1'
CAPACITY = 50_000
CHARGING_POWER = 11_000

RETAIL_PRICE = 3200
FEED_IN_PRICE = 800
GRID_FEE = 400
LEVY = 1800
BUY_QUANTITY = 900_000
SELL_QUANTITY = 100_000


def market_forecasts(timesteps):
    """Retailer columns in the convention the Creator emits: `_out` is what the agent pays."""
    n = len(timesteps)
    return {
        f'{c.TC_ENERGY}_{c.TC_ENERGY}_{c.PF_OUT}': [BUY_QUANTITY] * n,
        f'{c.TC_ENERGY}_{c.TC_ENERGY}_{c.PF_IN}': [SELL_QUANTITY] * n,
        f'{c.TC_ENERGY}_{c.TC_PRICE}_{c.PF_OUT}': [RETAIL_PRICE] * n,
        f'{c.TC_ENERGY}_{c.TC_PRICE}_{c.PF_IN}': [FEED_IN_PRICE] * n,
        f'{c.TT_GRID}_{c.TT_MARKET}_{c.PF_OUT}': [GRID_FEE] * n,
        f'{c.TT_GRID}_{c.TT_MARKET}_{c.PF_IN}': [0] * n,
        f'{c.TT_LEVIES}_{c.TC_PRICE}_{c.PF_OUT}': [LEVY] * n,
        f'{c.TT_LEVIES}_{c.TC_PRICE}_{c.PF_IN}': [0] * n,
    }


def make_ev(module, timesteps, delta, soc_init, energy_consumed, availability, scheme):
    return module.Ev(
        EV,
        forecasts={f'{EV}_availability': availability,
                   f'{EV}_energy_consumed': energy_consumed},
        timesteps=timesteps, delta=delta, socs={EV: [soc_init]},
        charging_scheme=scheme,
        sizing={'capacity': CAPACITY, 'charging_home': CHARGING_POWER,
                'charging_AC': CHARGING_POWER, 'charging_DC': CHARGING_POWER,
                'charging_efficiency': 1.0, 'v2g': 0})


BACKENDS = [(linopy_components, 'linopy'), (poi_components, 'poi')]


@pytest.mark.parametrize('module, name', BACKENDS, ids=[n for _, n in BACKENDS])
class TestMarket:
    """The market component must construct, and read the same columns in both backends."""

    def test_it_constructs_at_all(self, module, name, timesteps, delta):
        """Regression: the PyOptInterface market raised KeyError on construction."""
        market = module.Market(MARKET, forecasts=market_forecasts(timesteps),
                               timesteps=timesteps, delta=delta)

        assert market is not None

    def test_buying_costs_more_than_selling(self, module, name, timesteps, delta):
        market = module.Market(MARKET, forecasts=market_forecasts(timesteps),
                               timesteps=timesteps, delta=delta)

        assert (market.price_buy > market.price_sell).all()

    def test_charges_fall_on_the_buying_direction(self, module, name, timesteps, delta):
        market = module.Market(MARKET, forecasts=market_forecasts(timesteps),
                               timesteps=timesteps, delta=delta)

        assert (market.grid_buy == GRID_FEE).all()
        assert (market.levies_buy == LEVY).all()
        assert (market.grid_sell == 0).all()
        assert (market.levies_sell == 0).all()

    def test_the_purchase_bound_comes_from_the_retailers_sale_quantity(self, module, name,
                                                                      timesteps, delta):
        """`_out` is the retailer selling, so it caps how much the agent may buy.

        Pinned with the two quantities set to different values; the shipped data has them equal,
        which is why reading the wrong one had no visible effect.
        """
        market = module.Market(MARKET, forecasts=market_forecasts(timesteps),
                               timesteps=timesteps, delta=delta)
        dt_hours = delta.total_seconds() * c.SECONDS_TO_HOURS

        assert max(market.upper) == pytest.approx(BUY_QUANTITY / dt_hours)
        assert min(market.lower) == pytest.approx(-SELL_QUANTITY / dt_hours)


@pytest.mark.parametrize('module, name', BACKENDS, ids=[n for _, n in BACKENDS])
class TestEv:
    """The EV state of charge, which the PyOptInterface backend had in its pre-fix form."""

    def test_the_starting_soc_is_capped_at_capacity(self, module, name, timesteps, delta):
        ev = make_ev(module, timesteps, delta, soc_init=CAPACITY + 10_000,
                     energy_consumed=[0, 0, 0, 0], availability=[1, 1, 1, 1],
                     scheme={'method': 'full'})

        assert ev.soc_start == CAPACITY
        assert np.max(ev.soc) <= CAPACITY

    def test_the_soc_never_goes_negative(self, module, name, timesteps, delta):
        ev = make_ev(module, timesteps, delta, soc_init=CAPACITY,
                     energy_consumed=[10_000, 10_000, 10_000, 40_000],
                     availability=[1, 1, 1, 1], scheme={'method': 'full'})

        assert np.min(ev.soc) >= 0

    def test_driving_is_clamped_to_what_the_battery_holds(self, module, name, timesteps, delta):
        ev = make_ev(module, timesteps, delta, soc_init=10_000,
                     energy_consumed=[6_000, 6_000, 6_000, 6_000],
                     availability=[0, 0, 0, 0], scheme={'method': 'full'})

        assert ev.consumption.min() >= 0
        assert ev.consumption.sum() == pytest.approx(10_000)


def test_both_backends_produce_the_same_soc_trajectory(timesteps, delta):
    """The two backends are selectable per agent, so they must not disagree."""
    kwargs = dict(soc_init=40_000, energy_consumed=[0, 15_000, 0, 0],
                  availability=[1, 0, 1, 1], scheme={'method': 'min_soc',
                                                     'min_soc': {'val': 0.8}})
    linopy_ev = make_ev(linopy_components, timesteps, delta, **kwargs)
    poi_ev = make_ev(poi_components, timesteps, delta, **kwargs)

    assert list(linopy_ev.soc) == list(poi_ev.soc)
    assert list(linopy_ev.consumption) == list(poi_ev.consumption)
    assert linopy_ev.soc_start == poi_ev.soc_start
