"""A run stopped at step N and resumed produces the same golden fingerprint as one that never stopped.

This is the acceptance test for resuming (#249, part d), and it is the whole point: whatever state
a checkpoint failed to capture -- a market's retained window, a §14a guarantee anchored to a
timestep, a forecaster's history, an RNG -- shows up here as a moved number, without anyone having
to enumerate it correctly in advance. Without this criterion "resume" silently means
"approximately continue", which is worse than chunking.

The oracle is the committed golden reference, so this runs in the `golden` job and on the same
platform the reference was measured on. Each golden scenario is run once: checkpointing every 8
timesteps, killed inside timestep 10 by a raise from `Database.end_timestep`, then resumed from the
results folder. Two receipts keep the comparison honest: the resumed run must report that it
started from step 8 and executed exactly the remaining timesteps -- a resume that quietly restarts
from the scenario copy would also match the reference.
"""
import json
import os
import subprocess
import sys

import pytest

from tests.e2e.test_golden_master import RELATIVE_TOLERANCE, SCENARIOS
from tests.scenario_run import REPO_ROOT, SEED, compare, fingerprint, prepare_config

INTERVAL = 8
CRASH_AT_STEP = 10
RUN_NAME = 'resumed'

RUNNER = """
import json, random
import numpy as np
random.seed({seed})
np.random.seed({seed})
from hamlet import Creator, Executor
from hamlet.executor.utilities.database.database import Database

Creator(path=r"{config_dir}").{creator_method}()

_ended = 0
_real_end_timestep = Database.end_timestep

def _crashing_end_timestep(self):
    global _ended
    _ended += 1
    if _ended == {crash_at}:
        raise RuntimeError("simulated crash")
    return _real_end_timestep(self)

Database.end_timestep = _crashing_end_timestep
try:
    Executor(r"{scenario}", name="{run_name}", num_workers=1, checkpoint_interval={interval}).run()
except RuntimeError as error:
    print("CRASHED_AT_STEP", _ended, error)

# A fresh process would not carry the patch; this one has to put it back.
Database.end_timestep = _real_end_timestep
_resumed = 0

def _counting_end_timestep(self):
    global _resumed
    _resumed += 1
    return _real_end_timestep(self)

Database.end_timestep = _counting_end_timestep
executor = Executor(r"{results}", resume=True, num_workers=1, checkpoint_interval={interval})
executor.run()
print("RESUMED_FROM_STEP", executor.resumed_from_step)
print("RESUMED_STEPS", _resumed)
import polars as pl
print("TOTAL_STEPS", pl.read_ipc(r"{results}/general/timetable.ft")["timestamp"].n_unique())
"""


@pytest.fixture(scope='module', params=SCENARIOS, ids=lambda pinned: pinned.name)
def scenario(request):
    return request.param


@pytest.fixture(scope='module')
def resumed(scenario, tmp_path_factory):
    base = tmp_path_factory.mktemp('resume')
    config = prepare_config(base, scenario.config_dir, scenario.name)
    results = base / 'results' / RUN_NAME
    script = RUNNER.format(config_dir=config.as_posix(), seed=SEED, creator_method=scenario.creator_method,
                           scenario=(base / 'scenarios' / scenario.name).as_posix(), run_name=RUN_NAME,
                           results=results.as_posix(), crash_at=CRASH_AT_STEP, interval=INTERVAL)
    completed = subprocess.run(
        [sys.executable, '-c', script], capture_output=True, text=True, encoding='utf-8',
        errors='replace', timeout=3600,
        env={**os.environ, 'MPLBACKEND': 'Agg', 'PYTHONIOENCODING': 'utf-8', 'PYTHONHASHSEED': '0'})
    assert f'CRASHED_AT_STEP {CRASH_AT_STEP} simulated crash' in completed.stdout, (
        completed.stdout[-2000:] + completed.stderr[-4000:])
    assert 'RESUMED_STEPS' in completed.stdout, completed.stdout[-2000:] + completed.stderr[-4000:]
    return completed.stdout, results


def receipt(stdout, key):
    return int(next(line for line in stdout.splitlines() if line.startswith(key)).split()[1])


@pytest.mark.golden
def test_the_resumed_run_started_from_the_checkpoint_and_ran_only_the_rest(scenario, resumed):
    stdout, _ = resumed
    total = receipt(stdout, 'TOTAL_STEPS')
    assert total > CRASH_AT_STEP, 'the scenario is too short for a crash after the first checkpoint'
    assert receipt(stdout, 'RESUMED_FROM_STEP') == INTERVAL, (
        f'the resume did not start from the checkpoint at step {INTERVAL}')
    assert receipt(stdout, 'RESUMED_STEPS') == total - INTERVAL, (
        f'a resume from step {INTERVAL} of {total} runs the remaining {total - INTERVAL}; a restart '
        f'from the scenario copy would run all {total}')


@pytest.mark.golden
def test_the_resumed_run_matches_the_golden_reference(scenario, resumed):
    _, results = resumed
    expected = json.loads(scenario.reference.read_text(encoding='utf-8'))

    differences = compare(fingerprint(results), expected, 'resumed', 'reference', RELATIVE_TOLERANCE)

    assert not differences, (
        'a run stopped at step 8 and resumed differs from the uninterrupted reference -- some state '
        'did not survive the checkpoint:\n  ' + '\n  '.join(differences))
