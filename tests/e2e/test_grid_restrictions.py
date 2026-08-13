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
from pathlib import Path

import pytest

from tests.scenario_run import REPO_ROOT

SCENARIO = 'grid_golden'

#: Where the §14a fee lands once `agent_base` has applied it: the agent's own forecast, which is
#: what it then optimises against. `res_variable_grid_fee.csv` does *not* hold fees -- see
#: `test_the_variable_grid_fee_reached_the_agents`.
FEE_COLUMN = 'grid_market_out'
CONFIG_ROOT = REPO_ROOT / 'tests' / 'e2e' / 'scenarios'


@pytest.fixture(scope='module')
def grid_results(scenario_runs):
    """Run the fixture once and hand back its results directory.

    This request is byte-identical to `test_golden_master`'s for `grid_golden`, and going
    through `scenario_runs` is what lets the two share one run instead of paying 158 s twice.
    They only ever co-exist in a session that selects both markers, which no CI job does --
    see `tests/scenario_cache.py` for why the saving is local-only.
    """
    return scenario_runs.run(CONFIG_ROOT, SCENARIO,
                             creator_method='new_scenario_from_files').results


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
def test_the_variable_grid_fee_reached_the_agents(grid_results):
    """The indirect mechanism, read where it lands rather than where it is computed.

    **Not** from `res_variable_grid_fee.csv`, despite the name. `enwg_14a` writes
    `combined_loading_for_bus` under that key -- per-unit loading, roughly 0.1 to 2.3 -- while the
    fees themselves go only into `restriction_commands` and are never saved. Asserting that that
    table varies is asserting that the *loading* varies, which is what the overload test above
    already says, and it holds whether or not a single agent ever sees a fee.

    That is not hypothetical. Severing the mechanism entirely -- an early `return` in
    `agent_base.apply_grid_commands`, the only place a variable fee reaches an agent -- leaves the
    grid-side table completely unchanged, and a review panel demonstrated all four tests in this
    file passing against it.

    The fee lands in each agent's forecast, which is the whole point of an *indirect* control: it
    changes what the agent optimises against. With the mechanism live it varies over the horizon
    and rises above the flat base rate; with it severed the column is the shipped constant.
    """
    import polars as pl

    spreads = {}
    for type_dir in sorted((grid_results / 'agents').iterdir()):
        if not type_dir.is_dir():
            continue
        for agent_dir in sorted(type_dir.iterdir()):
            forecasts = pl.read_ipc(agent_dir / 'forecasts.ft', memory_map=False)
            if FEE_COLUMN not in forecasts.columns:
                continue
            column = forecasts[FEE_COLUMN]
            spreads[agent_dir.name] = float(column.max()) - float(column.min())

    assert spreads, (
        f'no agent forecast carries a {FEE_COLUMN!r} column, so the grid fee an agent optimises '
        f'against cannot be read and this test asserts nothing')
    assert max(spreads.values()) > 0, (
        f'every agent sees a constant grid fee across the whole horizon ({spreads}), so the '
        f'variable grid fee never reached them -- the indirect mechanism is not connected')


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

    Read from `res_bus` at the agent's own bus, taking the **last** row for the timestep. Not the
    converged one: `grids.yaml` sets `max_iteration: 3`, and at both restricted timesteps
    `executor/setup.py` forces `grid_ok = True` on the iteration cap rather than reaching a fixed
    point (`res_trafo.csv` carries 4 rows at each of them). The last row is the state the run
    actually kept, which is what the assertion is about.

    The cap is recomputed from each pass's power flow, so at the end of a converging sequence it
    sits exactly on the power drawn -- the margins here are 0 W, and the `+ 1` below is what
    absorbs the rounding. Under a backend that ignores the cap the sequence does not converge at
    all: the draw stays at its uncontrolled value while the cap keeps asking for less, and the
    gap is hundreds of watts.
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
