"""L2 — the RTC optimisation bounds are configuration, not constants.

These bounds used to be hard-coded. A value that suits one feeder silently constrains a larger
one, so they are now read from the `limits` block of the rtc controller configuration. The
defaults must reproduce the historical unbounded behaviour exactly.
"""
import datetime

import numpy as np
import polars as pl
import pytest
from linopy import Model

import hamlet.constants as c
from hamlet.executor.utilities.controller.rtc.optim.linopy.components import Market

NAME = 'market'
ENERGY_TYPE = c.ET_ELECTRICITY


def make_market(limits=None, market_result=0):
    kwargs = {'timeseries': pl.DataFrame(), 'market_result': market_result,
              'delta': datetime.timedelta(hours=1)}
    if limits is not None:
        kwargs[c.K_LIMITS] = limits
    return Market(NAME, **kwargs)


def bounds(market):
    """Build the market variables and return (lower, upper) of the power variable."""
    model = Market.define_variables(market, Model(), energy_type=ENERGY_TYPE)
    variable = model.variables[f'{NAME}_{ENERGY_TYPE}']
    return float(variable.lower), float(variable.upper)


class TestDefaults:
    """The defaults reproduce the behaviour before these became configurable."""

    def test_market_power_is_unbounded_by_default(self):
        lower, upper = bounds(make_market())

        assert lower == -np.inf
        assert upper == np.inf

    def test_balancing_power_default_is_the_historical_placeholder(self):
        assert make_market().balancing_power == 10_000_000_000

    def test_an_empty_limits_block_changes_nothing(self):
        assert make_market(limits={}).balancing_power == 10_000_000_000
        assert bounds(make_market(limits={})) == (-np.inf, np.inf)

    def test_defaults_table_covers_every_configurable_bound(self):
        """Every key `limit()` may be asked for must have a default, or it raises KeyError."""
        assert set(c.RTC_DEFAULT_LIMITS) == {
            'balancing_power', 'market_power', 'hp_power_heat', 'hp_power_electricity',
        }


class TestOverrides:
    """A configured bound must actually reach the model."""

    def test_market_power_can_be_bounded(self):
        lower, upper = bounds(make_market(limits={'market_power': 4_000_000}))

        assert (lower, upper) == (-4_000_000, 4_000_000)

    def test_balancing_power_can_be_bounded(self):
        assert make_market(limits={'balancing_power': 4_000_000}).balancing_power == 4_000_000

    def test_a_partial_limits_block_leaves_the_others_at_their_default(self):
        market = make_market(limits={'market_power': 4_000_000})

        assert market.balancing_power == 10_000_000_000

    def test_none_means_unbounded(self):
        assert bounds(make_market(limits={'market_power': None})) == (-np.inf, np.inf)


def test_deviation_variables_use_the_balancing_bound():
    """The balancing bound caps how far the RTC may deviate from the cleared market position."""
    market = make_market(limits={'balancing_power': 4_000_000})
    model = Market.define_variables(market, Model(), energy_type=ENERGY_TYPE)

    for direction in ('pos', 'neg'):
        variable = model.variables[f'{NAME}_{ENERGY_TYPE}_deviation_{direction}']
        assert float(variable.upper) == 4_000_000
