"""What `EnWG14a.__fee_horizon_steps` accepts, and what it refuses (#211).

Its own docstring says why it refuses a horizon rather than casting one. These tests are the
other half: that each refusal actually fires, that the accepting path hands `pd.date_range` an
`int` -- pandas warns on a non-integer `periods` and has said it will raise -- and that the
guard is not a landmine for the configs that are fine.
`tests/integration/executor/test_variable_grid_fee_config_horizons.py` checks the shipped
configs against it.
"""
import warnings

import pandapower as pp
import pandas as pd
import pytest

import hamlet.constants as c
from hamlet.executor.utilities.grid_restrictions.enwg_14a import EnWG14a


#: Raised in place of the horizon's power flows, to show the guard let the call through. Any
#: exception the method does not catch would do; `LoadflowNotConverged` is the one it *does*
#: catch, so it must not be that one.
class _GotPastTheGuard(Exception):
    pass


class _DatabaseStub:
    def __init__(self, timestep):
        self._timestep = timestep

    def get_general_data(self):
        return {c.K_GENERAL: {'time': {'timestep': self._timestep}}}


def _restriction(horizon_seconds, timestep_seconds):
    restriction = object.__new__(EnWG14a)
    restriction.grid = pp.create_empty_network()
    restriction.database = _DatabaseStub(timestep_seconds)
    restriction.timestamp = pd.Timestamp('2021-03-24 00:00:00')
    restriction.restriction_config = {
        'variable_grid_fees': {'active': True, 'horizon': horizon_seconds, 'update': 3600,
                               'grid_fee_base': 0.0937}}
    return restriction


def _calculate(horizon_seconds, timestep_seconds):
    """Run the fee calculation up to its first power flow, with pandas' own complaint fatal.

    `pd.date_range` gets the step count a few lines after the guard. Promoting its non-integer
    `periods` warning to an error is what catches a change that keeps the guard but drops the
    cast -- the guard alone proves the quotient is whole, not that an `int` is what pandas sees.

    Driven through `__calculate_variable_grid_fees` rather than by calling the guard directly, so
    that a guard which stops being *reached* fails these tests too.
    """
    restriction = _restriction(horizon_seconds, timestep_seconds)

    def _no_power_flows(grid, horizon):
        raise _GotPastTheGuard(horizon)

    restriction._EnWG14a__run_horizon_power_flows = _no_power_flows
    with warnings.catch_warnings():
        warnings.simplefilter('always')
        warnings.filterwarnings('error', message="Non-integer 'periods'")
        return restriction._EnWG14a__calculate_variable_grid_fees()


#: Horizons that are not whole multiples of their timestep: shorter than a day, an hourly run
#: with a 90-minute horizon, a quarter-hourly run off by a single minute, and the two floats that
#: would raise out of an `int()` inside the condition instead of reaching the refusal.
NON_DIVISIBLE = [(50_000, 3600), (5_400, 3600), (86_340, 900),
                 (float('inf'), 3600), (float('nan'), 3600)]

#: Horizons that *do* divide evenly and are still unusable -- the half a divisibility test alone
#: admits. `-3600 % 3600 == 0`, and `pd.date_range` accepts `periods=-1` and `periods=0` by
#: returning an empty index, so each of these used to run to completion writing empty fees for
#: every bus rather than failing. One step short of the minimum is included so the boundary is
#: asserted rather than assumed.
DIVISIBLE_BUT_UNUSABLE = [(-3600, 3600), (0, 3600), (3600, 3600), (7200, 3600), (1800, 900)]


@pytest.mark.parametrize('horizon, timestep', NON_DIVISIBLE + DIVISIBLE_BUT_UNUSABLE)
def test_a_horizon_the_fee_calculation_cannot_use_is_refused(horizon, timestep):
    """Both halves of the guard, and the reason #211 is not closed by a cast.

    A horizon that does not divide has no step count at all; one that divides into fewer than
    `MIN_FEE_HORIZON_STEPS` has a count the calculation cannot use. Only the first is what #211
    reported -- the second is what a guard against the first fails to catch, and is the shape
    that made #234's fix recur one level down.
    """
    with pytest.raises(ValueError) as raised:
        _calculate(horizon, timestep)

    message = str(raised.value)
    assert str(horizon) in message and str(timestep) in message, (
        f'the error must name both values so the config can be fixed without reading the '
        f'source, but it said: {message}')
    assert 'grids.yaml' in message and 'setup.yaml' in message, (
        f'the error must name the files the two values live in, but it said: {message}')


@pytest.mark.parametrize('timestep', [0, -900, None])
def test_a_timestep_that_cannot_be_divided_by_is_refused_before_the_division(timestep):
    """`timestep: 0` used to be a `ZeroDivisionError` raised from inside §14a, naming nothing."""
    with pytest.raises(ValueError, match='timestep'):
        _calculate(86_400, timestep)


@pytest.mark.parametrize('lead', [1, 2, 3, 4])
def test_the_minimum_horizon_follows_the_unreachable_lead(lead, monkeypatch):
    """Change how many leading timesteps are dropped and the minimum must move with it.

    `MIN_FEE_HORIZON_STEPS` is `FEE_HORIZON_LEAD_STEPS + 1` because the lead is sliced off and at
    least one timestep has to survive. Writing the minimum out as a literal would have left a
    guard that went on accepting a two-step horizon after someone changed the lead to three --
    silently, since an empty fee series raises nothing. This asserts the relationship rather than
    either number: at `lead` timesteps the fee series is empty and the horizon is refused; at
    `lead + 1` exactly one timestep survives and it is accepted.
    """
    monkeypatch.setattr(EnWG14a, 'FEE_HORIZON_LEAD_STEPS', lead)
    monkeypatch.setattr(EnWG14a, 'MIN_FEE_HORIZON_STEPS', lead + 1)
    timestep = 3600

    with pytest.raises(ValueError, match='at least'):
        _calculate(lead * timestep, timestep)

    with pytest.raises(_GotPastTheGuard) as raised:
        _calculate((lead + 1) * timestep, timestep)
    assert raised.value.args[0] == lead + 1


def test_the_minimum_is_derived_from_the_lead_and_not_written_out():
    """The relationship the test above exercises, asserted on the class itself.

    Without this, someone could satisfy the parametrised test by monkeypatching both constants --
    which it does -- while the shipped pair had drifted apart.
    """
    assert EnWG14a.MIN_FEE_HORIZON_STEPS == EnWG14a.FEE_HORIZON_LEAD_STEPS + 1


@pytest.mark.parametrize('horizon, timestep', [(86_400, 3600), (86_400, 900), (10_800, 3600)])
def test_a_usable_horizon_is_let_through_as_a_whole_step_count(horizon, timestep):
    """The other half of the guard: it must not fire on the configs that are fine.

    A guard nothing reaches is indistinguishable from no guard, so this asserts the call gets as
    far as the horizon's power flows *and* that the step count handed to them is an `int` -- the
    quantity `pd.date_range` is given a few lines earlier. `10_800 / 3600` is the minimum the
    guard accepts, so the boundary is pinned from both sides.
    """
    with pytest.raises(_GotPastTheGuard) as raised:
        _calculate(horizon, timestep)

    steps = raised.value.args[0]
    assert steps == horizon // timestep
    assert isinstance(steps, int) and not isinstance(steps, bool)
