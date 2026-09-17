"""Agent results are written to the results folder every few timesteps, not only at the end.

`AgentDB` holds every result table in memory and `save_database` wrote them once, when the run
was over, so a crash lost all of them (#249). Two things are pinned here and each has a way of
being quietly wrong. **When** the checkpoint runs: after the timestep is closed, outside the
`while not grid_ok` loop -- `Executor.execute` says why. And **what** it writes: the same files,
at the same paths, with the same contents, that the final save would write -- asserted on values
read back from disk, not on the files existing, because the scenario copy already puts a
plausible `agents/` tree in the results folder before the first timestep runs.
"""

import os
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from hamlet import constants as c
from hamlet import functions as f
from hamlet.executor.setup import Executor
from hamlet.executor.utilities.database import agent_db as agent_db_module
from hamlet.executor.utilities.database.agent_db import AgentDB
from hamlet.executor.utilities.database.database import Database
from hamlet.executor.utilities.database.region_db import RegionDB

TIMESTAMPS = [f'2023-01-28 {hour:02d}:00:00+00:00' for hour in range(5)]
TABLES = ('meters', 'timeseries', 'socs', 'setpoints', 'forecasts')


# --- the executor loop -------------------------------------------------------------------------

class RecordingDatabase:
    """Records `end_timestep` and `save_agents` in the order the executor calls them."""

    def __init__(self, log):
        self.log = log
        self.timesteps_ended = 0

    def end_timestep(self):
        self.timesteps_ended += 1
        self.log.append('end_timestep')

    def save_agents(self, path):
        # Tag the call with how many timesteps had been closed when it happened, and where it was
        # told to write, so a test can assert the schedule and the destination as values.
        self.log.append(('save_agents', self.timesteps_ended, path))


def timetable(timestamps):
    return pl.DataFrame({c.TC_TIMESTAMP: list(timestamps),
                         c.TC_REGION: ['a_region'] * len(timestamps)})


def executor_over(timestamps, checkpoint_interval, max_iteration=1, converge=True):
    """An `Executor` stubbed down to the timestep loop, as in test_timestep_restriction_state."""
    log = []
    executor = Executor.__new__(Executor)
    executor.timetable = timetable(timestamps)
    executor.type = 'sim'
    executor.max_iteration = max_iteration
    executor.checkpoint_interval = checkpoint_interval
    executor.path_results = os.path.join('some', 'results', 'run_name')
    executor.database = RecordingDatabase(log)
    executor.pbar = SimpleNamespace(reset=lambda **kw: None, set_description=lambda *a, **kw: None,
                                    update=lambda *a: None)
    executor._Executor__execute_agents = lambda tasks: log.append('agents')
    executor._Executor__execute_markets = lambda tasks: log.append('markets')

    def execute_grids(tasklist, initial_db, num_iteration):
        log.append(f'grids:{num_iteration}')
        return True if converge else num_iteration > executor.max_iteration

    executor._Executor__execute_grids = execute_grids
    return executor, log


def checkpoints(log):
    return [entry for entry in log if isinstance(entry, tuple)]


def test_agents_are_checkpointed_every_interval_timesteps():
    executor, log = executor_over(TIMESTAMPS, checkpoint_interval=2)

    executor.execute()

    assert [step for _, step, _ in checkpoints(log)] == [2, 4], (
        'five timesteps at an interval of two checkpoint after the second and the fourth; the '
        'fifth is left to the final save')


def test_the_checkpoint_writes_to_the_results_root_the_final_save_uses():
    """`cleanup` passes `dirname(path_results)`; the checkpoint has to pass the same thing."""
    executor, log = executor_over(TIMESTAMPS, checkpoint_interval=1)

    executor.execute()

    assert {path for _, _, path in checkpoints(log)} == {os.path.join('some', 'results')}


def test_a_checkpoint_comes_after_the_timestep_is_ended():
    executor, log = executor_over(TIMESTAMPS, checkpoint_interval=1)

    executor.execute()

    assert checkpoints(log), 'no checkpoint happened, so there is no ordering to check'
    for index, entry in enumerate(log):
        if isinstance(entry, tuple):
            assert log[index - 1] == 'end_timestep', (
                f'checkpoint at log index {index} was not preceded by end_timestep: {log[index - 1]}')


def test_a_checkpoint_waits_for_the_rollback_loop_to_finish():
    """With a grid that never converges, the cap forces four passes; only the last one stands.

    A checkpoint inside the loop would write pass 1's dispatch, which pass 2 then rolls back.
    """
    executor, log = executor_over(TIMESTAMPS[:1], checkpoint_interval=1, max_iteration=3,
                                  converge=False)

    executor.execute()

    assert log == ['agents', 'markets', 'grids:1',
                   'agents', 'markets', 'grids:2',
                   'agents', 'markets', 'grids:3',
                   'agents', 'markets', 'grids:4',
                   'end_timestep', ('save_agents', 1, os.path.join('some', 'results'))]


@pytest.mark.parametrize('interval', [None, 0])
def test_no_interval_means_no_checkpoint(interval):
    executor, log = executor_over(TIMESTAMPS, checkpoint_interval=interval)

    executor.execute()

    assert checkpoints(log) == []
    assert log.count('end_timestep') == 5, 'switching checkpoints off must not skip timesteps'


def test_an_interval_longer_than_the_run_checkpoints_nothing():
    executor, log = executor_over(TIMESTAMPS, checkpoint_interval=6)

    executor.execute()

    assert checkpoints(log) == []


# --- the constructor -------------------------------------------------------------------------

@pytest.mark.parametrize('bad', [-1, 1.5, True, '24'])
def test_the_constructor_refuses_an_interval_that_would_silently_mean_never(bad, tmp_path):
    with pytest.raises(ValueError, match='checkpoint_interval'):
        Executor(str(tmp_path / 'scenario'), checkpoint_interval=bad)


@pytest.mark.parametrize('given, stored', [(None, None), (0, None), (7, 7), (np.int64(7), 7)])
def test_the_constructor_stores_the_interval(given, stored, tmp_path):
    """A numpy integer -- what a value read out of a config table arrives as -- is an interval too."""
    stored_value = Executor(str(tmp_path / 'scenario'), checkpoint_interval=given).checkpoint_interval
    assert stored_value == stored
    assert type(stored_value) is type(stored)


def test_the_default_is_on(tmp_path):
    assert Executor(str(tmp_path / 'scenario')).checkpoint_interval == Executor.DEFAULT_CHECKPOINT_INTERVAL
    assert Executor.DEFAULT_CHECKPOINT_INTERVAL > 0


# --- what a checkpoint writes ----------------------------------------------------------------

def table(value):
    return pl.DataFrame({c.TC_TIMESTAMP: [1, 2, 3], 'reading': [value, value + 1, value + 2]})


def agent(agent_type, agent_id, value):
    """An AgentDB carrying only its result tables, each filled with a distinct value."""
    db = AgentDB(path=os.path.join('unused', agent_id), agent_type=agent_type, agent_id=agent_id)
    for offset, name in enumerate(TABLES):
        setattr(db, name, table(value + 10 * offset))
    return db


class RecordingMarket:
    """Stands in for a MarketDB; a checkpoint must not reach it."""

    def __init__(self):
        self.saved = 0

    def save_market(self, path, save_all=False):
        self.saved += 1


def database_with(tmp_path, agents, run_name='a_run', region='a_region'):
    database = Database(os.path.join(str(tmp_path), 'scenario_folder'), name=run_name)
    region_db = RegionDB(os.path.join(str(tmp_path), 'scenario_folder'))
    for db in agents:
        region_db.agents.setdefault(db.agent_type, {})[db.agent_id] = db
    region_db.markets = {'electricity': {'lem': RecordingMarket()}}
    database._Database__regions = {region: region_db}
    database._Database__grids = {}
    return database, region_db


def read(path):
    return f.load_file(path, df='polars', method='eager')


def test_a_checkpoint_writes_every_table_of_every_agent_with_its_current_contents(tmp_path):
    agents = [agent('sfh', 'agent_one', 1), agent('sfh', 'agent_two', 100), agent('ctsp', 'agent_three', 1000)]
    database, _ = database_with(tmp_path, agents)

    database.save_agents(str(tmp_path / 'results'))

    for db in agents:
        for name in TABLES:
            written = read(str(tmp_path / 'results' / 'a_run' / 'agents' / db.agent_type / db.agent_id
                              / f'{name}.ft'))
            assert written.equals(getattr(db, name)), f'{db.agent_id}/{name} differs from memory'


def test_a_checkpoint_lands_where_the_final_save_will_overwrite_it(tmp_path):
    """Same paths, so a crashed run's agent tables are those of its last checkpoint."""
    db = agent('sfh', 'agent_one', 1)
    database, _ = database_with(tmp_path, [db])

    database.save_agents(str(tmp_path / 'results'))
    checkpointed = {name: read(str(tmp_path / 'results' / 'a_run' / 'agents' / 'sfh' / 'agent_one' / f'{name}.ft'))
                    for name in TABLES}

    # Simulate more timesteps, then the end of the run.
    for offset, name in enumerate(TABLES):
        setattr(db, name, table(500 + 10 * offset))
    database.save_database(str(tmp_path / 'results'))

    for name in TABLES:
        path = str(tmp_path / 'results' / 'a_run' / 'agents' / 'sfh' / 'agent_one' / f'{name}.ft')
        assert read(path).equals(getattr(db, name)), f'final save did not overwrite {name}'
        assert not read(path).equals(checkpointed[name])


def test_a_checkpoint_does_not_touch_the_markets(tmp_path):
    """Market rows are drained per timestep already; the retained window is written once, at the end."""
    database, region_db = database_with(tmp_path, [agent('sfh', 'agent_one', 1)])

    database.save_agents(str(tmp_path / 'results'))

    assert region_db.markets['electricity']['lem'].saved == 0
    database.save_database(str(tmp_path / 'results'))
    assert region_db.markets['electricity']['lem'].saved == 1


def test_a_checkpoint_keeps_the_run_name_out_of_sub_regions(tmp_path):
    """The root region writes under the run; every later region stays a sibling of it (#243)."""
    root = agent('sfh', 'root_agent', 1)
    sub = agent('sfh', 'sub_agent', 2)
    database, _ = database_with(tmp_path, [root], run_name='named_run', region='root_region')
    sub_region = RegionDB(os.path.join(str(tmp_path), 'scenario_folder', 'sub_region'))
    sub_region.agents = {'sfh': {'sub_agent': sub}}
    sub_region.markets = {}
    database._Database__regions['sub_region'] = sub_region

    database.save_agents(str(tmp_path / 'results'))

    assert read(str(tmp_path / 'results' / 'named_run' / 'agents' / 'sfh' / 'root_agent' / 'meters.ft')).equals(
        root.meters)
    assert read(str(tmp_path / 'results' / 'sub_region' / 'agents' / 'sfh' / 'sub_agent' / 'meters.ft')).equals(
        sub.meters)


# --- a crash during a checkpoint -----------------------------------------------------------

def test_a_crash_mid_write_leaves_the_previous_checkpoint_of_that_table_intact(tmp_path, monkeypatch):
    """Each table is staged and renamed over the old file, so a file is old or new, never truncated.

    A checkpoint overwrites the one before it. Written in place, a crash during the write would
    destroy the previous checkpoint too -- exactly the interval the feature promises to keep.
    """
    db = agent('sfh', 'agent_one', 1)
    target = tmp_path / 'agent_one'
    db.save_agent(str(target))
    first = {name: read(str(target / f'{name}.ft')) for name in TABLES}

    for offset, name in enumerate(TABLES):
        setattr(db, name, table(500 + 10 * offset))

    real_save_file = agent_db_module.f.save_file

    def crash_while_writing_socs(path, data, **kwargs):
        # Keyed on the table being written, not on the file name, so a change to the staging
        # convention cannot quietly turn this into a test that never crashes.
        if data is db.socs:
            # Half a file: what the process leaves behind when it dies mid-write.
            with open(path, 'wb') as handle:
                handle.write(b'ARROW1\x00\x00garbage')
            raise OSError('simulated crash during write')
        return real_save_file(path, data, **kwargs)

    monkeypatch.setattr(agent_db_module.f, 'save_file', crash_while_writing_socs)

    with pytest.raises(OSError, match='simulated crash'):
        db.save_agent(str(target))

    # The table being written when the crash hit still reads back as the checkpoint before.
    assert read(str(target / 'socs.ft')).equals(first['socs'])
    # The tables written before it carry the new checkpoint; the ones after it the old one.
    assert read(str(target / 'meters.ft')).equals(db.meters)
    assert read(str(target / 'timeseries.ft')).equals(db.timeseries)
    assert read(str(target / 'setpoints.ft')).equals(first['setpoints'])
    assert read(str(target / 'forecasts.ft')).equals(first['forecasts'])


def test_the_swap_is_a_single_rename_and_never_removes_the_target_first(tmp_path, monkeypatch):
    """A `remove` then `rename` is not atomic: a crash between the two leaves no file at all.

    Only `os.replace` may touch the target. With every other way of moving or deleting a file
    made to raise, the save has to still complete and land the new contents.
    """
    db = agent('sfh', 'agent_one', 1)
    db.save_agent(str(tmp_path / 'agent_one'))
    for offset, name in enumerate(TABLES):
        setattr(db, name, table(500 + 10 * offset))

    def forbidden(*args, **kwargs):
        raise AssertionError(f'the save tried to move or delete a file with {args}')

    for function in ('remove', 'unlink', 'rename', 'renames'):
        monkeypatch.setattr(agent_db_module.os, function, forbidden)

    db.save_agent(str(tmp_path / 'agent_one'))

    for name in TABLES:
        assert read(str(tmp_path / 'agent_one' / f'{name}.ft')).equals(getattr(db, name))


class RefusingReplace:
    """`os.replace` as Windows behaves while a scanner holds the file: refuses N times, then works."""

    def __init__(self, refusals):
        self.refusals = refusals
        self.calls = 0
        self.real_replace = os.replace  # bound before the monkeypatch swaps it out

    def __call__(self, staging, target):
        self.calls += 1
        if self.calls <= self.refusals:
            raise PermissionError(5, 'Access is denied', staging)
        return self.real_replace(staging, target)


# Absolute, not derived from `AgentDB.REPLACE_ATTEMPTS`: a count taken from the constant under test
# passes whatever the constant is, so `REPLACE_ATTEMPTS = 1` -- no retry at all -- stayed green
# until a review panel tried it. Two refusals must be absorbed; three must not.
ABSORBED_REFUSALS = 2
FATAL_REFUSALS = 3


def test_a_transient_windows_refusal_of_the_rename_is_retried(tmp_path, monkeypatch):
    db = agent('sfh', 'agent_one', 1)
    refusing = RefusingReplace(refusals=ABSORBED_REFUSALS)
    monkeypatch.setattr(agent_db_module.os, 'replace', refusing)
    monkeypatch.setattr(AgentDB, 'REPLACE_RETRY_SECONDS', 0)

    db.save_agent(str(tmp_path / 'agent_one'))

    # Every table's rename went through, after the refusals were absorbed on the first one.
    assert refusing.calls == len(TABLES) + refusing.refusals
    for name in TABLES:
        assert read(str(tmp_path / 'agent_one' / f'{name}.ft')).equals(getattr(db, name))


def test_a_persistent_refusal_of_the_rename_still_fails_the_save(tmp_path, monkeypatch):
    """Retrying is for a scanner's lock, not for a file that is genuinely held open."""
    db = agent('sfh', 'agent_one', 1)
    refusing = RefusingReplace(refusals=FATAL_REFUSALS)
    monkeypatch.setattr(agent_db_module.os, 'replace', refusing)
    monkeypatch.setattr(AgentDB, 'REPLACE_RETRY_SECONDS', 0)

    with pytest.raises(PermissionError):
        db.save_agent(str(tmp_path / 'agent_one'))

    assert refusing.calls == FATAL_REFUSALS, 'gave up earlier or later than the budget'


def test_a_completed_save_leaves_no_staging_file_behind(tmp_path):
    db = agent('sfh', 'agent_one', 1)

    db.save_agent(str(tmp_path / 'agent_one'))

    assert sorted(os.listdir(tmp_path / 'agent_one')) == sorted(f'{name}.ft' for name in TABLES)
