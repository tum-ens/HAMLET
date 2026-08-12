"""End-to-end — §14a EnWG actually fires, and the agents actually obey it.

The golden master pins the *numbers* `grid_golden` produces. This file pins the thing that makes
those numbers worth pinning: that the restriction fired at all. A grid scenario in which nothing
ever overloads exercises the happy path and nothing else, and would keep passing while every
restriction mechanism underneath it rotted.

Both mechanisms are asserted separately, because they are independent code paths and one is far
easier to reach than the other:

* **Indirect** — variable grid fees, computed in `enwg_14a` and applied outside the solver in
  `agent_base.py`. Reached on every timestep once switched on; never causes a re-simulation.
* **Direct** — power control, which caps an agent through `apply_grid_commands` in the RTC. This
  is the only thing that can set `grid_ok = False`, so it is the only reason the
  `while not grid_ok` loop in `executor/setup.py` ever iterates. It is also the path that had
  never executed for anyone: it read a `hp_min_control` column nothing wrote, and no shipped
  example enables it.

A single "§14a ran" assertion would let a run where only the fees moved stand in for one where a
device was curtailed. These are different claims and they fail separately.

The assertions read the grid's own results rather than the agents' setpoints, deliberately: agent
setpoint tables only retain the forecast horizon, so by the end of a run the timestep a command
was issued at is no longer in them. `res_bus` is kept for every timestep, and in this fixture each
agent sits alone on its bus, so bus power *is* agent power as the power flow saw it.
"""
import json
import shutil
from pathlib import Path

import pytest

from tests.scenario_run import REPO_ROOT, run_example

SCENARIO = 'grid_golden'
CONFIG_ROOT = REPO_ROOT / 'tests' / 'e2e' / 'scenarios'


@pytest.fixture(scope='module')
def grid_results(tmp_path_factory):
    """Run the fixture once and hand back its results directory."""
    base = tmp_path_factory.mktemp('e2e_14a')
    try:
        run_example(base, CONFIG_ROOT, SCENARIO, creator_method='new_scenario_from_files')
        yield base / 'results' / SCENARIO
    finally:
        shutil.rmtree(base, ignore_errors=True)


def read_csv(results, name):
    import pandas as pd

    path = results / 'grids' / 'electricity' / name
    assert path.exists(), f'{name} was not written, so that part of the grid stage did not run'
    return pd.read_csv(path)


@pytest.mark.e2e
def test_the_feeder_actually_overloads(grid_results):
    """The premise of everything below. Without an overload, §14a has nothing to do.

    Stated as its own test so that a fixture which quietly stops overloading -- a changed profile,
    a re-sized transformer, a solver that shifts charging -- fails *here*, naming the cause,
    rather than showing up as a puzzling absence of commands further down.
    """
    trafo = read_csv(grid_results, 'res_trafo.csv')
    overloaded = trafo[trafo['loading_percent'] > 100]

    assert not overloaded.empty, (
        f'the transformer never exceeded 100 % loading (peak '
        f'{trafo["loading_percent"].max():.1f} %), so no restriction could be triggered and every '
        f'assertion in this file would pass vacuously')


@pytest.mark.e2e
def test_variable_grid_fees_vary(grid_results):
    """The indirect mechanism: fees must actually differ across time.

    A constant fee column would mean the calculation ran and produced the base rate everywhere,
    which is what a broken loading signal looks like.
    """
    fees = read_csv(grid_results, 'res_variable_grid_fee.csv')
    numeric = fees.select_dtypes('number').drop(columns=[fees.columns[0]], errors='ignore')

    assert not numeric.empty, 'the variable grid fee table carries no numeric columns'
    spread = (numeric.max() - numeric.min()).max()
    assert spread > 0, (
        'every variable grid fee is identical, so the fee is not varying with grid loading')


@pytest.mark.e2e
def test_direct_power_control_issued_a_command(grid_results):
    """The direct mechanism fired, which also means the `while not grid_ok` loop iterated.

    `EnWG14a.execute` starts from `grid_ok = True` and only `__calculate_direct_power_control` can
    set it False, so a command existing at all is proof the timestep was re-simulated: the command
    is written on one pass and the loop runs again because of it.
    """
    commands = read_csv(grid_results, 'res_direct_power_control.csv')

    assert not commands.empty, (
        'no direct power control command was issued, so the `while not grid_ok` loop never '
        'iterated and the restriction was never applied')
    assert (commands['control_result'] > 0).all(), (
        f'a command capped an agent at zero or below: {commands["control_result"].tolist()}')


@pytest.mark.e2e
def test_the_curtailment_command_was_respected(grid_results):
    """The one that matters: the agent drew no more than the grid operator allowed.

    Everything above can hold while the cap is silently discarded -- which is exactly what
    happened for `framework: poi` until the backend grew its own `apply_grid_commands`, because
    the base class's was a no-op. The grid stage cannot notice that by itself: it re-simulates,
    gets the same answer, and converges on an uncapped grid.

    Read from `res_bus` at the agent's own bus, taking the last row for the timestep -- that is
    the converged iteration, the one the run kept.
    """
    commands = read_csv(grid_results, 'res_direct_power_control.csv')
    bus = read_csv(grid_results, 'res_bus.csv')
    bus_column = bus.columns[0]

    agent_bus = {path.name: json.loads((path / 'account.json').read_text(encoding='utf-8'))
                 ['general']['bus']
                 for type_dir in (grid_results / 'agents').iterdir() if type_dir.is_dir()
                 for path in type_dir.iterdir() if path.is_dir()}

    checked = 0
    for _, command in commands.iterrows():
        rows = bus[(bus[bus_column] == agent_bus[command['id_agent']])
                   & (bus['timestamp'] == command['timestamp'])]
        assert not rows.empty, (
            f'no power flow result at bus {agent_bus[command["id_agent"]]} for '
            f'{command["timestamp"]}, so the command cannot be checked against anything')

        drawn_w = rows['p_mw'].iloc[-1] * 1e6
        cap_w = command['control_result']
        assert drawn_w <= cap_w + 1, (
            f'{command["id_agent"]} was capped at {cap_w:.0f} W at {command["timestamp"]} but drew '
            f'{drawn_w:.0f} W, so the grid operator\'s command was accepted and then ignored')
        checked += 1

    # A loop over an empty table asserts nothing. The previous test already fails on that, but
    # this one must not be able to pass by iterating zero times.
    assert checked > 0, 'no command was checked'
