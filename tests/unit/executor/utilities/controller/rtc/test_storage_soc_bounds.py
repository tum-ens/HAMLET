"""A storage whose state of charge has drifted below zero must still be allowed to idle.

`SimpleStorage` derives its discharge bound from the state of charge: `upper = round(min(P, soc *
eta / dt))`. The SoC is carried between timesteps in integer Wh and drifts by rounding; 27 of the
104 agents in a plain Design 6 January end some hour with a negative battery SoC. A SoC of -1.2 Wh
makes `upper` -1: the battery may not idle, it *must* charge at least 1 W. That watt is a firm draw
the grid stage cannot know about, and under a tight §14a cap it is infeasible by exactly one watt
(design 6b, 2023-01-09 04:00, agent 3bz0WGoo6cnwd51: fixed loads 395 W, cap 395 W, battery
`[-24900, -1]`).

The EV model already clamps: `self.soc = max(0, ...)`. This pins the same guard on the battery.
"""

from datetime import timedelta

import polars as pl
import pytest

from hamlet.executor.utilities.controller.rtc.optim.poi.components import Battery

NAME = 'bat'
POWER_W = 24900
CAPACITY_WH = 49800
EFFICIENCY = 0.9


def battery(soc_wh, target_w=0):
    return Battery(
        NAME,
        timeseries=pl.DataFrame(),
        targets=pl.DataFrame({f'{NAME}_battery_electricity': [target_w]}),
        socs=pl.DataFrame({NAME: [soc_wh]}),
        delta=timedelta(hours=1),
        sizing={'power': POWER_W, 'capacity': CAPACITY_WH, 'efficiency': EFFICIENCY,
                'b2g': 0.0, 'g2b': 0.0, 'soc': 0.1},
    )


@pytest.mark.parametrize('soc_wh', [-1.2, -1, -0.4, 0])
def test_an_empty_or_overdrawn_storage_may_idle(soc_wh):
    """The discharge bound of an empty storage is 0 -- never negative, which would force a charge."""
    b = battery(soc_wh)

    assert b.upper == 0, (
        f'SoC {soc_wh} Wh gave an upper bound of {b.upper} W; a negative bound forces the storage '
        f'to charge and is a firm draw the grid stage cannot see')
    assert b.lower == -POWER_W


def test_a_charged_storage_keeps_its_real_discharge_bound():
    """The clamp must not touch a healthy state of charge."""
    b = battery(5000)

    assert b.upper == round(5000 * EFFICIENCY)  # 4500 W, below the 24900 W rating
    assert b.lower == -POWER_W


def test_a_target_below_an_overdrawn_bound_is_lifted_to_it():
    """`_limit_target` clips the MPC target into `[lower, upper]`; with a clamped upper of 0 a
    1 W charging target survives, a 1 W discharging target is clipped to 0."""
    assert battery(-1.2, target_w=-1).target == -1
    assert battery(-1.2, target_w=+1).target == 0
