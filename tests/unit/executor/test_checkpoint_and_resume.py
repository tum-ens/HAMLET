"""The checkpoint file, and what the executor does with it on resume and pause.

The acceptance test is `tests/e2e/test_resume_matches_golden.py`: a run stopped and resumed
matches the uninterrupted golden fingerprint. That is what says the *state* is complete. What is
pinned here is the mechanics the acceptance test cannot separate: the file is written atomically
and refused on a format mismatch; both RNGs come back where they were (neither golden scenario
draws from one, so the e2e cannot see this); the grid CSVs are cut back to the checkpoint's length
and no further; the timetable resumes strictly after the checkpoint with the checkpoint cadence
intact; and `pause` checkpoints, stops, and keeps `cleanup` off the folder it is leaving behind.
"""
import json
import os
import pickle
import random

import numpy as np
import polars as pl
import pytest

from hamlet import constants as c
from hamlet import functions as f
from hamlet.executor.setup import Executor
from hamlet.executor.utilities import checkpoint
from tests.unit.executor.executor_stub import bare_executor

TIMESTAMPS = [f'2023-01-28 {hour:02d}:00:00+00:00' for hour in range(6)]


class FakeDatabase:
    """Picklable, with a value the round trip can be checked against."""

    def __init__(self, marker):
        self.marker = marker
        self.tables = {'meters': pl.DataFrame({'a': [1, 2, 3]})}


def grid_csv(results, key, lines):
    folder = results / 'grids' / 'electricity'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{key}.csv'
    path.write_bytes(b''.join(f'{i},row{i}\n'.encode() for i in range(lines)))
    return path


# --- the file -------------------------------------------------------------------------------

def test_write_then_read_returns_the_state_that_was_written(tmp_path):
    grid_csv(tmp_path, 'res_trafo', 5)
    grid_csv(tmp_path, 'res_bus', 3)
    database = FakeDatabase('as of step 8')

    checkpoint.write(str(tmp_path), database, step=8, last_timestamp='2023-01-28 07:00:00+00:00')
    state = checkpoint.read(str(tmp_path))

    assert state['step'] == 8
    assert state['last_timestamp'] == '2023-01-28 07:00:00+00:00'
    assert state['database'].marker == 'as of step 8'
    assert state['database'].tables['meters'].equals(database.tables['meters'])
    assert state['grid_csv_lines'] == {'grids/electricity/res_bus.csv': 3, 'grids/electricity/res_trafo.csv': 5}
    assert state['checkpoint_format_version'] == c.CHECKPOINT_FORMAT_VERSION


def test_the_manifest_beside_it_says_the_same_without_the_graph(tmp_path):
    checkpoint.write(str(tmp_path), FakeDatabase('x'), step=3, last_timestamp='2023-01-28 02:00:00+00:00')

    manifest = json.loads((tmp_path / 'checkpoint' / 'manifest.json').read_text(encoding='utf-8'))

    assert manifest['step'] == 3
    assert manifest['last_timestamp'] == '2023-01-28 02:00:00+00:00'
    assert 'database' not in manifest and 'rng' not in manifest


def test_a_checkpoint_of_another_format_is_refused_not_read(tmp_path):
    checkpoint.write(str(tmp_path), FakeDatabase('x'), step=1, last_timestamp='t')
    target = tmp_path / 'checkpoint' / 'state.pkl'
    state = pickle.loads(target.read_bytes())
    state['checkpoint_format_version'] = c.CHECKPOINT_FORMAT_VERSION + 1
    target.write_bytes(pickle.dumps(state))

    with pytest.raises(checkpoint.CheckpointFormatError, match=str(c.CHECKPOINT_FORMAT_VERSION + 1)):
        checkpoint.read(str(tmp_path))


def test_no_checkpoint_is_a_named_refusal(tmp_path):
    with pytest.raises(FileNotFoundError, match='checkpoint_interval'):
        checkpoint.read(str(tmp_path))


def test_a_crash_during_the_write_leaves_the_previous_checkpoint_readable(tmp_path, monkeypatch):
    checkpoint.write(str(tmp_path), FakeDatabase('first'), step=8, last_timestamp='t8')

    def crash_mid_dump(obj, handle, **kwargs):
        handle.write(b'\x80\x05garbage')
        raise OSError('simulated crash during write')

    monkeypatch.setattr(checkpoint.pickle, 'dump', crash_mid_dump)
    with pytest.raises(OSError, match='simulated crash'):
        checkpoint.write(str(tmp_path), FakeDatabase('second'), step=16, last_timestamp='t16')

    state = checkpoint.read(str(tmp_path))
    assert state['database'].marker == 'first' and state['step'] == 8


def test_only_replace_may_touch_the_state_file(tmp_path, monkeypatch):
    """A `remove` then `rename` is not atomic: a crash between them leaves no checkpoint at all."""
    checkpoint.write(str(tmp_path), FakeDatabase('first'), step=8, last_timestamp='t8')

    def forbidden(*args, **kwargs):
        raise AssertionError(f'the write tried to move or delete a file with {args}')

    for function in ('remove', 'unlink', 'rename', 'renames'):
        monkeypatch.setattr(checkpoint.os, function, forbidden)

    checkpoint.write(str(tmp_path), FakeDatabase('second'), step=16, last_timestamp='t16')

    assert checkpoint.read(str(tmp_path))['database'].marker == 'second'


# --- the RNGs -------------------------------------------------------------------------------

def test_both_generators_resume_where_the_checkpointed_process_left_them(tmp_path):
    random.seed(20260917)
    np.random.seed(20260917)
    random.random()
    np.random.random()
    checkpoint.write(str(tmp_path), FakeDatabase('x'), step=1, last_timestamp='t')
    continued = (random.random(), float(np.random.random()))

    # Another process, at a different point in both sequences.
    random.seed(1)
    np.random.seed(1)
    checkpoint.restore_rng(checkpoint.read(str(tmp_path)))

    assert (random.random(), float(np.random.random())) == continued


# --- the grid CSVs --------------------------------------------------------------------------

def test_csvs_are_cut_back_to_the_checkpointed_length(tmp_path):
    path = grid_csv(tmp_path, 'res_trafo', 5)
    checkpoint.write(str(tmp_path), FakeDatabase('x'), step=4, last_timestamp='t4')
    # Three more timesteps flushed before the crash.
    path.write_bytes(path.read_bytes() + b'5,row5\n6,row6\n7,row7\n')

    removed = checkpoint.truncate_grid_csvs(str(tmp_path), checkpoint.read(str(tmp_path)))

    assert path.read_bytes() == b''.join(f'{i},row{i}\n'.encode() for i in range(5))
    assert removed == {'grids/electricity/res_trafo.csv': 3}


def test_a_csv_that_did_not_grow_is_left_alone_and_one_the_checkpoint_never_saw_too(tmp_path):
    same = grid_csv(tmp_path, 'res_bus', 4)
    checkpoint.write(str(tmp_path), FakeDatabase('x'), step=4, last_timestamp='t4')
    later = grid_csv(tmp_path, 'res_line', 2)
    before = same.read_bytes(), later.read_bytes()

    removed = checkpoint.truncate_grid_csvs(str(tmp_path), checkpoint.read(str(tmp_path)))

    assert (same.read_bytes(), later.read_bytes()) == before
    assert removed == {'grids/electricity/res_bus.csv': 0}


def test_a_csv_shorter_than_recorded_means_the_folder_was_edited(tmp_path):
    path = grid_csv(tmp_path, 'res_trafo', 5)
    checkpoint.write(str(tmp_path), FakeDatabase('x'), step=4, last_timestamp='t4')
    path.write_bytes(b'0,row0\n')

    with pytest.raises(RuntimeError, match='recorded 5'):
        checkpoint.truncate_grid_csvs(str(tmp_path), checkpoint.read(str(tmp_path)))


# --- the executor: resume -------------------------------------------------------------------

class LoggingDatabase:
    def __init__(self, log):
        self.log = log

    def end_timestep(self):
        self.log.append('end')


def resumed_executor(timestamps, last_timestamp, resumed_from_step, checkpoint_interval):
    log = []
    executor = bare_executor(timestamps, database=LoggingDatabase(log), checkpoint_interval=checkpoint_interval,
                             resumed_from_step=resumed_from_step, _Executor__last_timestamp=last_timestamp)
    executor._Executor__execute_agents = lambda tasks: log.append(('agents', str(tasks[c.TC_TIMESTAMP][0])))
    executor._Executor__execute_markets = lambda tasks: None
    executor._Executor__execute_grids = lambda tasklist, initial_db, num_iteration: True
    executor._Executor__checkpoint = lambda step, last: log.append(('checkpoint', step, str(last)))
    return executor, log


def agents_run(log):
    return [entry[1] for entry in log if isinstance(entry, tuple) and entry[0] == 'agents']


def checkpoints(log):
    return [(entry[1], entry[2]) for entry in log if isinstance(entry, tuple) and entry[0] == 'checkpoint']


def test_a_resumed_run_starts_strictly_after_the_checkpointed_timestep():
    executor, log = resumed_executor(TIMESTAMPS, last_timestamp=TIMESTAMPS[2], resumed_from_step=3,
                                     checkpoint_interval=None)

    executor.execute()

    assert agents_run(log) == TIMESTAMPS[3:], (
        'the checkpointed timestep must not run again, and nothing after it may be skipped')


def test_a_resumed_run_keeps_the_checkpoint_cadence_of_the_uninterrupted_run():
    """Resumed from step 3 at an interval of 2, the next checkpoint is step 4, not step 5."""
    executor, log = resumed_executor(TIMESTAMPS, last_timestamp=TIMESTAMPS[2], resumed_from_step=3,
                                     checkpoint_interval=2)

    executor.execute()

    assert checkpoints(log) == [
        (4, TIMESTAMPS[3]), (6, TIMESTAMPS[5])]


def test_a_fresh_run_still_starts_at_the_first_timestep():
    executor, log = resumed_executor(TIMESTAMPS, last_timestamp=None, resumed_from_step=None,
                                     checkpoint_interval=None)

    executor.execute()

    assert agents_run(log) == TIMESTAMPS


def test_a_checkpoint_records_the_step_and_the_timestep_it_closed():
    executor, log = resumed_executor(TIMESTAMPS, last_timestamp=None, resumed_from_step=None,
                                     checkpoint_interval=3)

    executor.execute()

    assert checkpoints(log) == [
        (3, TIMESTAMPS[2]), (6, TIMESTAMPS[5])]


# --- the executor: pause --------------------------------------------------------------------

def test_pause_checkpoints_and_stops_after_the_timestep_in_progress():
    executor, log = resumed_executor(TIMESTAMPS, last_timestamp=None, resumed_from_step=None,
                                     checkpoint_interval=None)
    original_agents = executor._Executor__execute_agents

    def agents_then_pause(tasks):
        original_agents(tasks)
        if str(tasks[c.TC_TIMESTAMP][0]) == TIMESTAMPS[1]:
            executor.pause()

    executor._Executor__execute_agents = agents_then_pause

    executor.execute()

    assert executor.paused is True
    assert agents_run(log) == TIMESTAMPS[:2], 'stopped after the timestep, not before'
    assert log[-1] == ('checkpoint', 2, TIMESTAMPS[1]), 'the pause must end with a checkpoint of the step it closed'
    assert log[-2] == 'end', 'the timestep in progress is closed before the checkpoint'


def test_a_paused_run_skips_cleanup_and_a_finished_one_does_not():
    calls = []
    for paused in (True, False):
        executor = bare_executor()
        executor.setup = lambda: calls.append('setup')
        executor.execute = lambda paused=paused: setattr(executor, 'paused', paused) or calls.append('execute')
        executor.cleanup = lambda: calls.append('cleanup')

        executor.run()

    assert calls == ['setup', 'execute', 'setup', 'execute', 'cleanup']


# --- the executor: setup on resume ----------------------------------------------------------

def results_folder(tmp_path, name='a_run'):
    """What a checkpointed run leaves behind, down to what `__prepare_scenario` reads."""
    folder = tmp_path / 'results' / name
    (folder / 'general').mkdir(parents=True)
    (folder / 'config').mkdir()
    f.save_file(str(folder / 'general' / 'general.json'),
                {c.K_SCENARIO_FORMAT_VERSION: c.SCENARIO_FORMAT_VERSION, 'structure': {}})
    (folder / 'config' / 'setup.yaml').write_text('paths:\n  results: ../results\n', encoding='utf-8')
    pl.DataFrame({c.TC_TIMESTAMP: TIMESTAMPS, c.TC_REGION: ['r'] * len(TIMESTAMPS)}).write_ipc(
        str(folder / 'general' / 'timetable.ft'))
    return folder


class CheckpointedDatabase:
    def get_general_data(self):
        return {c.K_GRID: {c.G_ELECTRICITY: {'restrictions': {'max_iteration': 7}}}}


def test_resuming_reads_the_results_folder_in_place_and_copies_nothing(tmp_path, monkeypatch):
    folder = results_folder(tmp_path)
    grid_csv(folder, 'res_trafo', 2)
    random.seed(20260917)
    checkpoint.write(str(folder), CheckpointedDatabase(), step=8, last_timestamp=TIMESTAMPS[3])
    continued = random.random()
    grid_csv(folder, 'res_trafo', 4)  # two timesteps flushed after the checkpoint
    monkeypatch.setattr(f, 'copy_folder', lambda *a, **k: pytest.fail('a resume must not copy or delete'))
    random.seed(1)

    executor = Executor(str(folder), resume=True)
    executor.setup()

    assert executor.path_results == str(folder)
    assert isinstance(executor.database, CheckpointedDatabase)
    assert executor.resumed_from_step == 8
    assert executor.max_iteration == 7, 'derived from the checkpointed database, as a fresh run derives it'
    assert (folder / 'grids' / 'electricity' / 'res_trafo.csv').read_bytes() == b'0,row0\n1,row1\n'
    assert random.random() == continued, 'the RNG was not restored as part of setup'


def test_resuming_a_folder_without_a_checkpoint_is_refused_before_anything_is_touched(tmp_path):
    folder = results_folder(tmp_path)

    with pytest.raises(FileNotFoundError, match='nothing to resume'):
        Executor(str(folder), resume=True).setup()


def test_a_fresh_run_still_copies_the_scenario(tmp_path, monkeypatch):
    folder = results_folder(tmp_path)
    copied = []
    monkeypatch.setattr(f, 'copy_folder', lambda src, dst, **k: copied.append((src, dst)))
    executor = Executor(str(folder))

    executor._Executor__prepare_scenario()

    assert copied == [(str(folder), os.path.join('../results', 'a_run'))]
