"""A run that dies keeps every agent result up to its last checkpoint, and nothing after it.

The unit tests in `tests/unit/executor/test_agent_checkpoint.py` pin the schedule and the files;
this is the promise itself, on the real chain -- Creator, Executor, a crash -- because the promise
is what #249 was filed for: design 6a's January run died at step 666 of 744 and lost every agent
result, and no test in this tree would have noticed that it did.

Two Executors run on one created scenario. The first completes and is the oracle. The second is
told to checkpoint every 8 timesteps and is killed inside the 10th, by raising from
`Database.end_timestep`. Its results folder must then hold, for every agent, exactly the rows the
complete run wrote for the first 8 timesteps -- and for the rows after that, exactly what the
scenario shipped. The second half is what pins the *interval*: a run that saved every timestep
would carry rows 9 and 10 as well and pass the first half.

Rows are compared as values against the complete run, never against zero. The scenario's meters
start at zero, so "rows after the checkpoint are zero" would pass a checkpoint that wrote nothing
at all.
"""
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

from hamlet import constants as c
from tests.scenario_run import REPO_ROOT, SEED, prepare_config

EXAMPLE = REPO_ROOT / 'examples' / 'create_simple_scenario'
SCENARIO_NAME = 'simple_scenario'
INTERVAL = 8
CRASH_AT_STEP = 10
COMPLETE, CRASHED = 'complete', 'crashed'

# The complete run first, then the crashing one over the same scenario folder. The crash is a
# raise from `Database.end_timestep`, which the executor calls once per timestep after the grid
# loop -- i.e. after the timestep's results are in the database and before any checkpoint of it.
RUNNER = """
import random
import numpy as np
random.seed({seed})
np.random.seed({seed})
from hamlet import Creator, Executor
from hamlet.executor.utilities.database.database import Database

Creator(path=r"{config_dir}").new_scenario_from_configs()
Executor(r"{scenario}", name="{complete}", num_workers=1).run()

_ended = 0
_real_end_timestep = Database.end_timestep

def _end_timestep(self):
    global _ended
    _ended += 1
    if _ended == {crash_at}:
        raise RuntimeError("simulated crash")
    return _real_end_timestep(self)

Database.end_timestep = _end_timestep
try:
    Executor(r"{scenario}", name="{crashed}", num_workers=1, checkpoint_interval={interval}).run()
except RuntimeError as error:
    print("CRASHED_AT_STEP", _ended, error)
else:
    print("DID_NOT_CRASH")
"""


@pytest.fixture(scope='module')
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp('checkpoint')
    config = prepare_config(base, EXAMPLE, SCENARIO_NAME)
    scenario = base / 'scenarios' / SCENARIO_NAME
    script = RUNNER.format(config_dir=config.as_posix(), seed=SEED, scenario=scenario.as_posix(),
                           complete=COMPLETE, crashed=CRASHED, crash_at=CRASH_AT_STEP,
                           interval=INTERVAL)
    completed = subprocess.run(
        [sys.executable, '-c', script], capture_output=True, text=True, encoding='utf-8',
        errors='replace', timeout=3600,
        env={**os.environ, 'MPLBACKEND': 'Agg', 'PYTHONIOENCODING': 'utf-8', 'PYTHONHASHSEED': '0'})
    assert f'CRASHED_AT_STEP {CRASH_AT_STEP} simulated crash' in completed.stdout, (
        completed.stdout[-2000:] + completed.stderr[-4000:])
    return base


def agent_dirs(root):
    """Every `agents/<type>/<id>` directory under a scenario or results folder, relative to it."""
    return sorted(path.relative_to(root) for path in Path(root).glob('agents/*/*') if path.is_dir())


def meters(base, folder, agent_dir):
    return pl.read_ipc(base / folder / agent_dir / 'meters.ft', memory_map=False)


def split_at_checkpoint(frame):
    """Rows the checkpoint should carry, and rows it must not, by timestamp.

    A timestep at time t writes the meter row for t + dt (`rtc_base.update_meters`), so after
    INTERVAL timesteps the rows up to start + INTERVAL * dt are written and the rest are untouched.
    """
    start = frame[c.TC_TIMESTAMP][0]
    step = frame[c.TC_TIMESTAMP][1] - start
    boundary = start + INTERVAL * step
    return (frame.filter(pl.col(c.TC_TIMESTAMP) <= boundary),
            frame.filter(pl.col(c.TC_TIMESTAMP) > boundary))


@pytest.mark.e2e
def test_the_crashed_run_kept_exactly_the_checkpointed_timesteps(runs):
    scenario = runs / 'scenarios' / SCENARIO_NAME
    results = runs / 'results'
    agents = agent_dirs(scenario)
    assert agents, 'the Creator produced no agents, so there is nothing to compare'
    assert agent_dirs(results / CRASHED) == agents, 'the crashed run does not carry every agent'

    for agent_dir in agents:
        initial = meters(runs / 'scenarios', SCENARIO_NAME, agent_dir)
        complete = meters(results, COMPLETE, agent_dir)
        crashed = meters(results, CRASHED, agent_dir)
        assert len(initial) == len(complete) == len(crashed) > CRASH_AT_STEP + 1, agent_dir

        kept, dropped = split_at_checkpoint(crashed)
        complete_kept, complete_dropped = split_at_checkpoint(complete)
        initial_kept, initial_dropped = split_at_checkpoint(initial)

        # Non-vacuity: the complete run moved every row it wrote away from the shipped values, on
        # both sides of the boundary. Otherwise "equals the complete run" and "equals the
        # scenario" could both be satisfied by a checkpoint that wrote nothing.
        assert not complete_kept[1:].equals(initial_kept[1:]), (
            f'{agent_dir}: the complete run left the first {INTERVAL} meter rows at their shipped '
            f'values, so nothing here can tell a checkpoint from no checkpoint')
        assert not complete_dropped.equals(initial_dropped), (
            f'{agent_dir}: the complete run left the rows after the checkpoint at their shipped '
            f'values, so nothing here can tell "checkpointed at 8" from "saved every timestep"')

        assert kept.equals(complete_kept), (
            f'{agent_dir}: the first {INTERVAL} timesteps in the crashed run differ from the '
            f'complete run -- the checkpoint did not write what the run had computed')
        assert dropped.equals(initial_dropped), (
            f'{agent_dir}: rows after the checkpoint carry results; either the crash at step '
            f'{CRASH_AT_STEP} did not lose them or the interval was not {INTERVAL}')


@pytest.mark.e2e
def test_the_checkpoint_wrote_every_agent_table(runs):
    """All five tables, each the checkpointed version rather than the scenario's copy.

    Asserted on the files' contents differing from the shipped ones, not on their existing: the
    executor copies the scenario into the results folder before the first timestep, so every file
    exists before anything is checkpointed.
    """
    scenario = runs / 'scenarios' / SCENARIO_NAME
    results = runs / 'results'
    changed = {name: 0 for name in ('meters', 'socs', 'timeseries', 'setpoints', 'forecasts')}
    for agent_dir in agent_dirs(scenario):
        for name in changed:
            shipped = pl.read_ipc(scenario / agent_dir / f'{name}.ft', memory_map=False)
            written = pl.read_ipc(results / CRASHED / agent_dir / f'{name}.ft', memory_map=False)
            complete = pl.read_ipc(results / COMPLETE / agent_dir / f'{name}.ft', memory_map=False)
            assert written.columns == complete.columns, (agent_dir, name)
            if not written.equals(shipped):
                changed[name] += 1
    # `timeseries` is an input the run never rewrites, so it is expected to match the scenario and
    # is asserted only through its columns above. The other four must have moved for at least one
    # agent each, or the checkpoint did not write them.
    for name in ('meters', 'socs', 'setpoints', 'forecasts'):
        assert changed[name] > 0, f'no agent has a checkpointed {name} table'
