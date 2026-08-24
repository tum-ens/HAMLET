"""Is the §14a variable-grid-fee horizon a whole number of timesteps, and what if it is not?

`__calculate_variable_grid_fees` divides the configured horizon (seconds) by the simulation
timestep (seconds) and uses the quotient as a *count*: the number of periods on the fee
timestamp index, and the number of power flows to run over the horizon. Python's true division
makes that quotient a float even when it is integral, and pandas used to accept a fractional
`periods` by truncating it -- silently shortening the fee horizon -- with a `FutureWarning`
saying it will raise instead (#211).

Casting to `int` would have removed the warning and pinned the truncation for every future
config. Whether §14a's fee horizon should round down, round up, or be refused is a modelling
question about the Festlegung, not a formatting one, so the implementation refuses instead.

These tests cover both halves of that: the guard fires on a horizon that does not divide, and
every config shipped in this repository divides -- so the guard is a guard and not a landmine.
"""
import subprocess
import warnings
from pathlib import Path

import pandapower as pp
import pandas as pd
import pytest
import yaml

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

    `pd.date_range` gets the step count two lines after the guard. Promoting its non-integer
    `periods` warning to an error is what catches a change that keeps the guard but drops the
    cast -- the guard alone proves the quotient is whole, not that an `int` is what pandas sees.
    """
    restriction = _restriction(horizon_seconds, timestep_seconds)

    def _no_power_flows(grid, horizon):
        raise _GotPastTheGuard(horizon)

    restriction._EnWG14a__run_horizon_power_flows = _no_power_flows
    with warnings.catch_warnings():
        warnings.simplefilter('always')
        warnings.filterwarnings('error', message="Non-integer 'periods'")
        return restriction._EnWG14a__calculate_variable_grid_fees()


#: Horizons that are not whole multiples of their timestep, each for a different reason: a
#: horizon shorter than a day, an hourly run with a 90-minute horizon, and a quarter-hourly run
#: whose horizon is off by a single minute.
NON_DIVISIBLE = [(50_000, 3600), (5_400, 3600), (86_340, 900)]


@pytest.mark.parametrize('horizon, timestep', NON_DIVISIBLE)
def test_a_horizon_that_is_not_a_whole_number_of_timesteps_is_refused(horizon, timestep):
    """The guard, and the reason #211 is not closed by a cast."""
    with pytest.raises(ValueError) as raised:
        _calculate(horizon, timestep)

    message = str(raised.value)
    assert str(horizon) in message and str(timestep) in message, (
        f'the error must name both values so the config can be fixed without reading the '
        f'source, but it said: {message}')


@pytest.mark.parametrize('horizon, timestep', [(86_400, 3600), (86_400, 900), (3_600, 900)])
def test_a_horizon_that_divides_evenly_is_let_through_as_a_whole_step_count(horizon, timestep):
    """The other half of the guard: it must not fire on the configs that are fine.

    A guard nothing reaches is indistinguishable from no guard, so this asserts the call gets as
    far as the horizon's power flows *and* that the step count handed to them is an `int` -- the
    quantity `pd.date_range` is given two lines earlier.
    """
    with pytest.raises(_GotPastTheGuard) as raised:
        _calculate(horizon, timestep)

    steps = raised.value.args[0]
    assert steps == horizon // timestep
    assert isinstance(steps, int) and not isinstance(steps, bool)


def _repo_root():
    return Path(__file__).resolve().parents[5]


def _tracked_grids_yaml():
    """Every `grids.yaml` git tracks, newest-first order irrelevant.

    Asked of git rather than of the filesystem on purpose: a run leaves generated copies of the
    scenario configs under the gitignored `scenarios/` tree, and globbing would sweep those too
    -- making the parametrisation depend on whether anyone had run a scenario in this checkout.
    A `git ls-files` that fails returns nothing, which `test_the_config_sweep_found_something_to
    _check` turns into a failure rather than a silently empty sweep.
    """
    try:
        listed = subprocess.run(['git', 'ls-files', '*grids.yaml'], cwd=_repo_root(),
                                capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return sorted(line for line in listed.splitlines() if line.strip())


def _variable_grid_fee_configs():
    """Every shipped `grids.yaml` that switches the variable grid fee on, with its timestep.

    Yields `(label, horizon_seconds, timestep_seconds)`. The horizon lives in `grids.yaml` under
    a per-energy-type key and the timestep in the sibling `setup.yaml`, so the two have to be
    read together -- which is the whole point: divisibility is a property of the *pair*, and
    neither file can be checked on its own.
    """
    for tracked in _tracked_grids_yaml():
        grids = _repo_root() / tracked
        setup = grids.parent / 'setup.yaml'
        if not setup.exists():
            continue
        timestep = (yaml.safe_load(setup.read_text(encoding='utf-8')) or {}) \
            .get('time', {}).get('timestep')
        for energy_type, section in (yaml.safe_load(grids.read_text(encoding='utf-8')) or {}).items():
            fees = ((section or {}).get('restrictions', {}) or {}) \
                .get('enwg_14a', {}).get('variable_grid_fees', {})
            if fees.get('active') and timestep:
                label = f'{grids.parent.relative_to(_repo_root()).as_posix()} [{energy_type}]'
                yield label, fees['horizon'], timestep


CONFIGS = list(_variable_grid_fee_configs())


def test_the_config_sweep_found_something_to_check():
    """`rglob` returning nothing would make the sweep below vacuously green."""
    assert CONFIGS, 'no shipped config with an active variable grid fee was found'


@pytest.mark.parametrize('label, horizon, timestep', CONFIGS,
                         ids=[label for label, _, _ in CONFIGS])
def test_every_shipped_config_has_a_whole_number_of_fee_timesteps(label, horizon, timestep):
    """No config in the repository trips the guard, so #211 could not move a number."""
    assert horizon % timestep == 0, (
        f'{label}: fee horizon {horizon} s is not a whole number of {timestep} s timesteps '
        f'({horizon / timestep} steps), so §14a would refuse to run it')
