"""On a forced iteration exit, the published §14a command must be the one the kept dispatch obeyed.

Within a timestep each pass replaces the same-timestamp `res_direct_power_control` frame with its
own, so only the last pass's command survives. On a converged exit that is right -- the last
command equals the one before it. On an exit forced by `max_iteration` it is not: the executor
stops the loop *after* a command was computed but *before* any agent ran under it, so the surviving
frame records a command that was never in force, while the dispatch that is kept obeyed the pass
before. Measured on `grid_golden`, 4586 W was published against a 4678 W draw that had obeyed a
4678 W command (#248).

The fix keeps the frame a pass supersedes and lets the executor restore it when it forces the exit.

**A converged exit must not restore anything** -- there the last frame is correct -- and a test that
only checks the forced path would pass on an implementation that restored unconditionally. Both
paths are below.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from hamlet import constants as c
from hamlet.executor import setup as executor_module
from hamlet.executor.setup import Executor
from hamlet.executor.utilities.database.database import Database
from hamlet.executor.utilities.database.grid_db import ElectricityGridDB
from hamlet.executor.utilities.grid_restrictions.enwg_14a import EnWG14a

TS = '2023-01-28 15:00:00+00:00'
STASH = ElectricityGridDB.SUPERSEDED_COMMAND_FRAME_KEY


def frame(command_w):
    return pd.DataFrame([[TS, 'agent_one', 'ems', command_w]],
                        columns=[c.TC_TIMESTAMP, c.TC_ID_AGENT, c.TC_ID_PLANT, 'control_result'])


def grid_db_after_passes(published_w, superseded_w=None):
    """A GridDB as it stands after the last pass wrote `published_w`, having replaced `superseded_w`."""
    grid_db = ElectricityGridDB.__new__(ElectricityGridDB)
    grid_db.results = {'res_direct_power_control': [frame(published_w)]}
    grid_db.restriction_commands = {}
    if superseded_w is not None:
        grid_db.restriction_commands[STASH] = frame(superseded_w)
    return grid_db


def published(grid_db):
    return int(grid_db.results['res_direct_power_control'][-1]['control_result'].iloc[0])


def test_restoring_publishes_the_superseded_command():
    """The regression, at the GridDB: 4678 W was obeyed, 4586 W was published."""
    grid_db = grid_db_after_passes(published_w=4586, superseded_w=4678)

    grid_db.restore_superseded_command_frame()

    assert published(grid_db) == 4678
    # Replaced, not appended: a second same-timestamp frame would publish two commands for one hour.
    assert len(grid_db.results['res_direct_power_control']) == 1
    assert STASH not in grid_db.restriction_commands, 'the stash must be consumed, not left to leak'


def test_restoring_with_nothing_superseded_leaves_the_frame_alone():
    """A final pass that issued no command never replaced anything; the standing frame is right."""
    grid_db = grid_db_after_passes(published_w=4678)

    grid_db.restore_superseded_command_frame()

    assert published(grid_db) == 4678


def test_restoring_is_safe_before_any_command_exists():
    grid_db = ElectricityGridDB.__new__(ElectricityGridDB)
    grid_db.results = {}
    grid_db.restriction_commands = {STASH: frame(4678)}

    grid_db.restore_superseded_command_frame()

    assert 'res_direct_power_control' not in grid_db.results


def test_the_stash_is_dropped_with_the_rest_of_the_timestep_state():
    """It is per-timestep state; surviving into the next timestep would restore the wrong frame."""
    grid_db = grid_db_after_passes(published_w=4586, superseded_w=4678)

    grid_db.clear_timestep_restriction_state()

    assert STASH not in grid_db.restriction_commands


def test_the_restriction_stashes_the_frame_it_replaces():
    """`__calculate_direct_power_control` must keep what it deletes, or there is nothing to restore."""
    grid_db = ElectricityGridDB.__new__(ElectricityGridDB)
    grid_db.results = {'res_direct_power_control': [frame(4678)]}
    grid_db.restriction_commands = {'current_direct_power_control': {'agent_one': {'ems': 4586}}}

    enwg = EnWG14a.__new__(EnWG14a)
    enwg.grid_db = grid_db
    enwg.timestamp = TS
    # The overload/bus machinery is not under test: patch the driver to only run the frame write.
    enwg.grid = SimpleNamespace(res_trafo=pd.DataFrame({'p_lv_mw': [], 'loading_percent': []}))

    # Drive the frame-write block directly through its public entry with no transformers to walk.
    enwg._EnWG14a__calculate_direct_power_control()

    assert published(grid_db) == 4586, 'the new pass\'s command is what stands'
    assert int(grid_db.restriction_commands[STASH]['control_result'].iloc[0]) == 4678, (
        'the frame it replaced must be kept for the executor to restore')


class RecordingDatabase:
    """Stands in for `Database` inside `__execute_grids`, recording what the executor asks of it."""

    def __init__(self, log):
        self.log = log

    def get_grid_data(self):
        return {'electricity': SimpleNamespace()}

    def post_grids(self, grid_type, grid, path_results=None):
        self.log.append(f'post:{grid_type}')

    def restore_superseded_commands(self):
        self.log.append('restore')


def execute_grids(monkeypatch, num_iteration, grid_reports_ok):
    """Run the real `__execute_grids` with `Grid` stubbed to report the given convergence."""
    log = []
    executor = Executor.__new__(Executor)
    executor.max_iteration = 3
    executor.database = RecordingDatabase(log)
    executor.path_results = 'unused'

    class StubGrid:
        def __init__(self, **kwargs):
            pass

        def execute(self):
            return SimpleNamespace(), grid_reports_ok

    monkeypatch.setattr(executor_module, 'Grid', StubGrid)
    # A pass that is not ok rolls the database back to `initial_db`, so it has to be a database.
    grid_ok = executor._Executor__execute_grids(tasklist=None, initial_db=RecordingDatabase(log),
                                                num_iteration=num_iteration)
    return grid_ok, log


def test_the_executor_restores_on_the_forced_exit(monkeypatch):
    """Pass 4 with `max_iteration: 3`, grid still not ok: forced out, and the frame is restored."""
    grid_ok, log = execute_grids(monkeypatch, num_iteration=4, grid_reports_ok=False)

    assert grid_ok is True, 'the cap forces the exit'
    assert log == ['post:electricity', 'restore'], (
        'restore must follow the grid being posted, so it acts on the frame that was just written')


def test_the_executor_does_not_restore_when_the_timestep_converged(monkeypatch):
    """On convergence the last frame is the obeyed one; restoring would publish a stale command."""
    grid_ok, log = execute_grids(monkeypatch, num_iteration=2, grid_reports_ok=True)

    assert grid_ok is True
    assert 'restore' not in log


def test_a_timestep_that_converges_exactly_on_the_cap_pass_is_not_restored(monkeypatch):
    """Pass 4 with `max_iteration: 3`, but the grid reports ok: converged, not forced.

    The cap and convergence are independent, and they can coincide. Here the last frame *is* the
    obeyed one, and restoring would publish the pass before it. A test that only checks pass 4
    with the grid still failing cannot see this cell; a review lens wrote it and it failed.
    """
    grid_ok, log = execute_grids(monkeypatch, num_iteration=4, grid_reports_ok=True)

    assert grid_ok is True
    assert 'restore' not in log, 'the grid converged on its own; the last frame is the obeyed one'


def test_a_pass_that_writes_no_command_leaves_no_stale_stash_to_restore():
    """The defect the fix could reproduce: a stash left over from an earlier pass.

    Passes 1-3 issue 9000, 8000, 7000 W. Pass 4's agents obey 7000, the overload clears, and the
    restriction writes nothing -- but the stash still holds 8000 from pass 3's supersession. A
    forced exit restoring that publishes a command two passes old. Reproduced with real objects
    before the per-pass clearing was added.
    """
    grid_db = ElectricityGridDB.__new__(ElectricityGridDB)
    grid_db.results = {'res_direct_power_control': [frame(7000)]}
    grid_db.restriction_commands = {STASH: frame(8000), 'current_direct_power_control': {}}

    enwg = EnWG14a.__new__(EnWG14a)
    enwg.grid_db = grid_db
    enwg.timestamp = TS
    enwg.restriction_config = {'variable_grid_fees': {'active': False},
                               'direct_power_control': {'active': True}}
    # No overload this pass, so `execute` takes the branch that writes no command.
    enwg._EnWG14a__check_overloads = lambda: False
    enwg.grid = SimpleNamespace(res_trafo=pd.DataFrame({'p_lv_mw': [], 'loading_percent': []}))

    enwg.execute()
    grid_db.restore_superseded_command_frame()

    assert published(grid_db) == 7000, 'a pass that replaced nothing must leave nothing to restore'


def test_the_executor_does_not_restore_mid_loop(monkeypatch):
    """A pass that is going to be re-run must not touch the frame; the next pass replaces it anyway."""
    grid_ok, log = execute_grids(monkeypatch, num_iteration=2, grid_reports_ok=False)

    assert grid_ok is False
    assert 'restore' not in log
