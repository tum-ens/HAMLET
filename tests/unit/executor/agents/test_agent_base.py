"""Regression — a controller configured as off must not run.

The EMS controller table comes from `agents.xlsx`. An empty cell there arrives as `NaN`, not
`None`, so a controller the user switched off was still constructed and executed.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hamlet import constants as c
from hamlet.executor.agents.agent_base import AgentBase


def make_agent_base(controllers):
    """An `AgentBase` with only the attributes `set_controllers` touches."""
    base = object.__new__(AgentBase)
    base.agent = MagicMock()
    base.agent.account = {c.K_EMS: {c.C_CONTROLLER: controllers}}
    base.timetable = MagicMock()
    base.market = MagicMock()
    base.grid_commands = {}
    return base


def run_and_count_constructed(controllers):
    """Return how many controllers `set_controllers` actually built."""
    base = make_agent_base(controllers)

    with patch('hamlet.executor.agents.agent_base.Controller') as controller_cls:
        base.set_controllers()

    return controller_cls.call_count


@pytest.mark.parametrize('disabled', [None, np.nan, float('nan')])
def test_disabled_controller_is_skipped(disabled):
    """Regression: only `None` was treated as "off", so a NaN from Excel ran the controller."""
    assert run_and_count_constructed({'fbc': {'method': disabled}}) == 0


def test_enabled_controller_still_runs():
    """The guard must not swallow a real configuration."""
    assert run_and_count_constructed({'fbc': {'method': 'mpc'}}) == 1


def test_only_the_disabled_controller_is_skipped():
    """A mixed table runs exactly the controllers that are configured."""
    controllers = {
        'fbc': {'method': 'mpc'},
        'rtc': {'method': np.nan},
        'other': {'method': 'optimization'},
    }

    assert run_and_count_constructed(controllers) == 2
