"""Unit — the PyOptInterface RTC backend must honour the same limits as the linopy one.

The two backends are selectable per agent, so a scenario that switches `framework` must get the
same model. This backend used to hard-code the bounds the linopy one reads from configuration,
so switching silently produced an unbounded model.
"""
import datetime

import numpy as np
import polars as pl
import pytest

import hamlet.constants as c
from hamlet.executor.utilities.controller.rtc.optim.poi.components import Market as PoiMarket
from hamlet.executor.utilities.controller.rtc.optim.linopy.components import Market as LinopyMarket

NAME = 'market'


def make_market(component, limits=None):
    kwargs = {'timeseries': pl.DataFrame(), 'market_result': 0,
              'delta': datetime.timedelta(hours=1)}
    if limits is not None:
        kwargs[c.K_LIMITS] = limits
    return component(NAME, **kwargs)


@pytest.mark.parametrize('component', [PoiMarket, LinopyMarket], ids=['poi', 'linopy'])
class TestBothBackendsAgree:
    """Whatever the limits say, both backends must read the same value."""

    def test_balancing_power_default(self, component):
        assert make_market(component).balancing_power == 10_000_000_000

    def test_balancing_power_override(self, component):
        assert make_market(component, {'balancing_power': 4_000_000}).balancing_power == 4_000_000

    def test_market_power_defaults_to_unbounded(self, component):
        assert make_market(component).limit('market_power') == np.inf

    def test_market_power_override(self, component):
        assert make_market(component, {'market_power': 4_000_000}).limit('market_power') == 4_000_000

    def test_heat_pump_fallbacks_default_to_unbounded(self, component):
        market = make_market(component)

        assert market.limit('hp_power_heat') == np.inf
        # the electrical fallback is negated by its caller, so the limit itself is positive
        assert market.limit('hp_power_electricity') == np.inf

    def test_a_partial_limits_block_leaves_the_others_alone(self, component):
        market = make_market(component, {'market_power': 4_000_000})

        assert market.balancing_power == 10_000_000_000

    def test_an_unknown_key_does_not_raise(self, component):
        """The default is looked up lazily, so a configured-but-untabulated key is not fatal."""
        market = make_market(component, {'market_power': 4_000_000, 'not_a_bound': 1})

        assert market.limit('market_power') == 4_000_000
