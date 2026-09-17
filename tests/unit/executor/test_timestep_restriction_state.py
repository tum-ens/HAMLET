"""A §14a command must not outlive the timestep it was computed for.

`current_direct_power_control` is an instruction for one timestep, derived from that timestep's power
flow. `EnWG14a` clears it on the paths it controls, but `executor/setup.py` ends a timestep itself
once `num_iteration` passes `max_iteration`, and on that path the restriction is never called again.
The command therefore survived, and the next timestep's agents read `restriction_commands` *before*
the grid stage, so they dispatched under a cap computed for an hour they were never assessed for.

On design 6a that path is taken by **36.8 %** of timesteps, and the run died at 666/744 with
`TerminationStatusCode.INFEASIBLE` after an agent's cap fell across successive hours to 452 W (#247).

**Asserting on the converged path passes on the broken code**, because `EnWG14a` already clears
there. Every case below drives a timestep that ends on the cap.
"""

from types import SimpleNamespace

import polars as pl
import pytest

from hamlet import constants as c
from hamlet.executor.setup import Executor
from hamlet.executor.utilities.database.database import Database
from hamlet.executor.utilities.database.grid_db import ElectricityGridDB, HeatGridDB

MAX_ITERATION = 3

#: What a restriction leaves behind for one timestep, and what must survive it.
COMMAND_KEY = 'current_direct_power_control'
GUARANTEE_KEY = 'direct_power_control_guarantee'
FEE_KEY = 'current_variable_grid_fees'


def grid_db_with_state():
    """A GridDB carrying only `restriction_commands`, populated as a controlled timestep leaves it."""
    grid_db = ElectricityGridDB.__new__(ElectricityGridDB)
    grid_db.restriction_commands = {
        COMMAND_KEY: {'agent_one': {'ems': 4586}},
        GUARANTEE_KEY: {'timestamp': '2023-01-28 15:00:00+00:00', 'agents': {'agent_one': 4586}},
        FEE_KEY: {'agent_one': 0.31},
    }
    return grid_db


def test_clearing_drops_the_command_and_the_guarantee():
    grid_db = grid_db_with_state()

    grid_db.clear_timestep_restriction_state()

    assert COMMAND_KEY not in grid_db.restriction_commands
    assert GUARANTEE_KEY not in grid_db.restriction_commands


def test_clearing_leaves_the_variable_grid_fee_alone():
    """The fee is the one piece of restriction state that *must* cross the boundary.

    It is computed in a timestep's grid stage and read in the **next** timestep's agent stage,
    before the grid runs, so surviving the boundary is the only way it reaches an agent at all.
    Clearing it here makes variable grid fees a near-total no-op: measured, a fee reached 16 of 112
    agent executions instead of 116 of 120.
    """
    grid_db = grid_db_with_state()

    grid_db.clear_timestep_restriction_state()

    assert grid_db.restriction_commands.get(FEE_KEY) == {'agent_one': 0.31}, (
        'the variable grid fee was dropped with the per-timestep state; it has to survive the '
        'boundary or it never reaches an agent')


def test_clearing_is_safe_when_a_restriction_left_nothing():
    grid_db = ElectricityGridDB.__new__(ElectricityGridDB)
    grid_db.restriction_commands = {}

    grid_db.clear_timestep_restriction_state()

    assert grid_db.restriction_commands == {}


def test_a_guarantee_is_dropped_even_with_no_command_beside_it():
    """Clearing must not be conditional on there being a command to clear.

    A timestep can leave an anchored guarantee behind without a surviving command, and a guard that
    returned early on an empty command would strand it.
    """
    grid_db = ElectricityGridDB.__new__(ElectricityGridDB)
    grid_db.restriction_commands = {GUARANTEE_KEY: {'timestamp': 'x', 'agents': {'a': 1}}}

    grid_db.clear_timestep_restriction_state()

    assert GUARANTEE_KEY not in grid_db.restriction_commands


def test_ending_a_timestep_reaches_every_grid():
    """A scenario can carry more than one grid, and a command left on any of them is imposed."""
    database = Database.__new__(Database)
    # A second *kind* of grid, not a second electricity one: defining the method on a subclass
    # rather than on `GridDB` would otherwise go unnoticed.
    heat = HeatGridDB.__new__(HeatGridDB)
    heat.restriction_commands = dict(grid_db_with_state().restriction_commands)
    grids = {'electricity': grid_db_with_state(), 'heat': heat}
    database._Database__grids = grids

    database.end_timestep()

    for grid in grids.values():
        assert COMMAND_KEY not in grid.restriction_commands


def timetable(timestamps):
    """One row per (timestamp, region), which is what `execute` partitions on."""
    return pl.DataFrame({
        c.TC_TIMESTAMP: list(timestamps),
        c.TC_REGION: ['a_region'] * len(timestamps),
    })


class RecordingDatabase:
    """Records the order of `end_timestep` calls against the passes that preceded them."""

    def __init__(self, log):
        self.log = log

    def end_timestep(self):
        self.log.append('end_timestep')


def executor_over(timestamps, grid_ok_after_cap=True):
    """An `Executor` stubbed down to the timestep loop, with the real iteration-cap behaviour.

    `__execute_grids` is where the cap lives in production, so the stub reproduces exactly that:
    it refuses to converge and returns True only once `num_iteration` passes `max_iteration`. A
    stub that simply returned False would hang, and one that returned True immediately would test
    the path `EnWG14a` already handles.
    """
    log = []
    executor = Executor.__new__(Executor)
    executor.timetable = timetable(timestamps)
    executor.type = 'sim'
    executor.max_iteration = MAX_ITERATION
    executor.checkpoint_interval = None  # the stub database has no agents to checkpoint
    executor.rollback_possible = True  # the stub grid refuses to converge, so a copy is taken
    executor.database = RecordingDatabase(log)
    executor.pbar = SimpleNamespace(reset=lambda **kw: None, set_description=lambda *a, **kw: None,
                                    update=lambda *a: None)

    executor._Executor__execute_agents = lambda tasks: log.append('agents')
    executor._Executor__execute_markets = lambda tasks: log.append('markets')

    def execute_grids(tasklist, initial_db, num_iteration):
        log.append(f'grids:{num_iteration}')
        # Read the executor's own cap rather than a module constant, so this cannot drift from the
        # value under test -- a stub carrying its own copy asserts only about itself.
        return num_iteration > executor.max_iteration if grid_ok_after_cap else True

    executor._Executor__execute_grids = execute_grids
    return executor, log


def test_the_executor_ends_a_timestep_that_stopped_on_the_iteration_cap():
    """The regression. This is the path `EnWG14a` never hears about."""
    executor, log = executor_over(['2023-01-28 15:00:00+00:00'])

    executor.execute()

    # Four passes, because the cap is 3 and the loop stops when `num_iteration` exceeds it.
    assert [entry for entry in log if entry.startswith('grids')] == [
        'grids:1', 'grids:2', 'grids:3', 'grids:4']
    assert log.count('end_timestep') == 1, (
        'the timestep ended on the iteration cap without dropping its restriction state, so its '
        'command is imposed on the next timestep')
    assert log[-1] == 'end_timestep', 'state must be dropped after the last pass, not during'


def test_every_timestep_is_ended_even_when_none_of_them_converge():
    """Two capped timesteps in a row: the second must not start under the first one's command."""
    executor, log = executor_over(['2023-01-28 15:00:00+00:00', '2023-01-28 16:00:00+00:00'])

    executor.execute()

    assert log.count('end_timestep') == 2

    # And it happens between them, not only at the end of the run.
    first_end = log.index('end_timestep')
    assert 'agents' in log[first_end:], 'the next timestep ran before its predecessor was ended'


def test_a_timestep_that_converges_is_ended_too():
    """Not the regression -- `EnWG14a` already clears here -- but the paths must not diverge."""
    executor, log = executor_over(['2023-01-28 15:00:00+00:00'], grid_ok_after_cap=False)

    executor.execute()

    assert [entry for entry in log if entry.startswith('grids')] == ['grids:1']
    assert log.count('end_timestep') == 1
