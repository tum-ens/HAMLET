"""Does every config this repository ships survive §14a's fee-horizon guard (#211)?

The guard refuses a variable grid fee horizon that is not a usable whole number of simulation
timesteps. That is a property of a *pair* of files -- the horizon lives in `grids.yaml` under a
per-energy-type key, the timestep in the sibling `setup.yaml` -- so neither can be checked on its
own, and a guard no shipped config passes would be a landmine rather than a guard. What the guard
accepts and refuses is pinned in
`tests/unit/executor/utilities/grid_restrictions/test_variable_grid_fee_horizon.py`.

Here rather than in `unit/` because it reads files, which `tests/README.md` makes the criterion.
"""
import subprocess
from pathlib import Path

import pytest
import yaml

from hamlet.executor.utilities.grid_restrictions.enwg_14a import EnWG14a


def _repo_root():
    return Path(__file__).resolve().parents[3]


def _tracked_grids_yaml():
    """Every `grids.yaml` git tracks.

    Asked of git rather than of the filesystem on purpose: a run leaves generated copies of the
    scenario configs under the gitignored `scenarios/` tree, and globbing would sweep those too --
    making the parametrisation depend on whether anyone had run a scenario in this checkout. A
    `git ls-files` that fails returns nothing, which `test_the_config_sweep_found_something_to_
    check` turns into a failure rather than a silently empty sweep.
    """
    try:
        listed = subprocess.run(['git', 'ls-files', '*grids.yaml'], cwd=_repo_root(),
                                capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return sorted(line for line in listed.splitlines() if line.strip())


def _variable_grid_fee_configs():
    """Every tracked `grids.yaml` that switches the variable grid fee on, with its timestep."""
    for tracked in _tracked_grids_yaml():
        grids = _repo_root() / tracked
        setup = grids.parent / 'setup.yaml'
        if not setup.exists():
            continue
        timestep = (yaml.safe_load(setup.read_text(encoding='utf-8')) or {}) \
            .get('time', {}).get('timestep')
        for energy_type, section in (yaml.safe_load(grids.read_text(encoding='utf-8'))
                                     or {}).items():
            fees = ((section or {}).get('restrictions', {}) or {}) \
                .get('enwg_14a', {}).get('variable_grid_fees', {})
            if fees.get('active') and timestep:
                label = f'{grids.parent.relative_to(_repo_root()).as_posix()} [{energy_type}]'
                yield label, fees['horizon'], timestep


CONFIGS = list(_variable_grid_fee_configs())


def test_the_config_sweep_found_something_to_check():
    """A `git ls-files` that found nothing would make the sweep below vacuously green."""
    assert CONFIGS, 'no shipped config with an active variable grid fee was found'


@pytest.mark.parametrize('label, horizon, timestep', CONFIGS,
                         ids=[label for label, _, _ in CONFIGS])
def test_every_shipped_config_survives_the_fee_horizon_guard(label, horizon, timestep):
    """Asked of the guard itself, not of a copy of its arithmetic.

    Re-deriving `horizon % timestep == 0` here would go on passing a guard that had been weakened
    into something else, which is the case this file exists to notice.
    """
    steps = EnWG14a._EnWG14a__fee_horizon_steps(horizon, timestep)
    assert steps == horizon // timestep, f'{label}: {steps} steps from {horizon} s / {timestep} s'
