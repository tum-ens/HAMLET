"""A storage whose state of charge has drifted below zero must still be allowed to idle (#251).

`SimpleStorage` derives its discharge bound from the state of charge: `upper = round(min(P, soc *
eta / dt))`. The SoC is carried between timesteps in integer Wh and drifts by rounding; 27 of the
104 agents in a plain Design 6 January end some hour with a negative battery SoC. Because the
stored value is rounded and integer, the smallest negative that can arrive is -1 Wh -- and at an
hourly step that makes `upper` -1: the battery may not idle, it *must* charge at least 1 W. That
watt is a firm draw the grid stage cannot know about, and under a tight §14a cap it is infeasible
by exactly one watt (design 6, threshold 0, 2023-01-09 04:00, agent 3bz0WGoo6cnwd51: fixed loads
395 W, cap 395 W, battery `[-24900, -1]`).

The EV model already clamps in both backends: `self.soc = max(0, ...)`. This pins the same guard on
the storage in both backends -- the two RTC component files are meant to mirror each other, and a
clamp in one only is a backend divergence (#251). The fractional -1.2 cannot arrive at runtime; it
pins that the clamp is a floor at zero and not a rounding, which -1 alone could not (round(-1) is -1).
"""

from datetime import timedelta

import polars as pl
import pytest

from hamlet.executor.utilities.controller.rtc.optim.linopy import components as linopy_components
from hamlet.executor.utilities.controller.rtc.optim.poi import components as poi_components

NAME = 'bat'
POWER_W = 24900
CAPACITY_WH = 49800
EFFICIENCY = 0.9

BACKENDS = pytest.mark.parametrize(
    'battery_cls', [poi_components.Battery, linopy_components.Battery], ids=['poi', 'linopy'])


def battery(battery_cls, soc_wh, target_w=0):
    return battery_cls(
        NAME,
        timeseries=pl.DataFrame(),
        targets=pl.DataFrame({f'{NAME}_battery_electricity': [target_w]}),
        socs=pl.DataFrame({NAME: [soc_wh]}),
        delta=timedelta(hours=1),
        sizing={'power': POWER_W, 'capacity': CAPACITY_WH, 'efficiency': EFFICIENCY,
                'b2g': 0.0, 'g2b': 0.0, 'soc': 0.1},
    )


@BACKENDS
@pytest.mark.parametrize('soc_wh', [-1.2, -1, 0])
def test_an_empty_or_overdrawn_storage_may_idle(battery_cls, soc_wh):
    """The discharge bound of an empty storage is 0 -- never negative, which would force a charge."""
    b = battery(battery_cls, soc_wh)

    assert b.upper == 0, (
        f'SoC {soc_wh} Wh gave an upper bound of {b.upper} W; a negative bound forces the storage '
        f'to charge and is a firm draw the grid stage cannot see')
    assert b.lower == -POWER_W


@BACKENDS
def test_a_charged_storage_keeps_its_real_discharge_bound(battery_cls):
    """The clamp must not touch a healthy state of charge."""
    b = battery(battery_cls, 5000)

    assert b.upper == round(5000 * EFFICIENCY)  # 4500 W, below the 24900 W rating
    assert b.lower == -POWER_W


@BACKENDS
def test_a_target_below_an_overdrawn_bound_is_lifted_to_it(battery_cls):
    """`_limit_target` clips the MPC target into `[lower, upper]`; with a clamped upper of 0 a
    1 W charging target survives, a 1 W discharging target is clipped to 0."""
    assert battery(battery_cls, -1, target_w=-1).target == -1
    assert battery(battery_cls, -1, target_w=+1).target == 0


def test_the_heat_storage_inherits_the_clamp():
    """`HeatStorage` is a `SimpleStorage` too and takes its bounds from the same clamped SoC.

    `upper` is computed in `SimpleStorage.__init__`, so a subclass that merely re-read `self.soc`
    afterwards would change the attribute but not the bound; the `soc` assertion is what would see
    that, the `upper` assertion is what would see a subclass recomputing its bounds unclamped."""
    hs = poi_components.HeatStorage(
        NAME,
        timeseries=pl.DataFrame(),
        targets=pl.DataFrame({f'{NAME}_heat-storage_heat': [0]}),
        socs=pl.DataFrame({NAME: [-1]}),
        delta=timedelta(hours=1),
        sizing={'power': 6400, 'capacity': 12800, 'efficiency': 0.95, 'soc': 0.5},
    )

    assert hs.soc == 0
    assert hs.upper == 0
