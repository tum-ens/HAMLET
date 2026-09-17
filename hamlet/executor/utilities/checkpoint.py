"""A run's state at a timestep boundary, written so that a later process can continue from it.

What is captured is decided by one criterion and not by a list: a run stopped at step N and
resumed must produce the same golden fingerprint as one that never stopped
(`tests/e2e/test_resume_matches_golden.py`). That is why the state is the live `Database` object
graph, pickled whole -- agent tables, every market's retained window and net-energy caches, every
grid's `restriction_commands` and unsettled results, every forecaster's history and fitted models
-- rather than an enumeration of the pieces someone thought mattered. The executor already deep-
copies that same graph every timestep it can roll back, so it is known to survive a copy.

Two things live outside the graph and are captured beside it. The **RNG states**: a random-forest
forecaster fits from `numpy.random` and the `zi` trading strategy draws from `random`, so a resumed
process must draw the same numbers a continuous one would have. And the **grid CSVs' lengths**:
the grid drain appends settled rows to `results/<run>/grids/<type>/<key>.csv` every timestep, so
a run that died after the checkpoint has rows on disk the resumed run will write again -- the
checkpoint records how long each file was, and resume truncates to that. Market fragments need
nothing: `MarketDB.save_and_drop_past_records` names each by its horizon threshold and a re-run
step rewrites it with identical rows.

The pickle is written to a sibling file and renamed into place, so a crash during the write leaves
the previous checkpoint intact; the manifest beside it is for people and is never read back.
"""
import json
import os
import pickle
import random
import time
from datetime import datetime, timezone

import numpy as np

from hamlet import constants as c

#: Where a run keeps its checkpoint, under its results folder.
DIRECTORY = 'checkpoint'
STATE_FILE = 'state.pkl'
MANIFEST_FILE = 'manifest.json'


class CheckpointFormatError(RuntimeError):
    """The checkpoint was written by a HAMLET whose checkpoint format this one does not read."""


def write(path_results: str, database, step: int, last_timestamp) -> str:
    """Write the run's state after `step` timesteps, the last of which was `last_timestamp`.

    Returns the checkpoint directory. Called by the executor after `end_timestep`, outside the
    grid loop, so the graph is the one the next timestep starts from.
    """
    folder = os.path.join(path_results, DIRECTORY)
    os.makedirs(folder, exist_ok=True)
    state = {
        'checkpoint_format_version': c.CHECKPOINT_FORMAT_VERSION,
        'step': step,
        'last_timestamp': last_timestamp,
        'written_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'rng': {'random': random.getstate(), 'numpy': np.random.get_state()},
        'grid_csv_lines': _grid_csv_lines(path_results),
        'database': database,
    }
    target = os.path.join(folder, STATE_FILE)
    staging = target + '.tmp'
    with open(staging, 'wb') as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
    _replace(staging, target)

    manifest = {key: value for key, value in state.items() if key not in ('database', 'rng')}
    manifest['last_timestamp'] = str(last_timestamp)
    with open(os.path.join(folder, MANIFEST_FILE), 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2)
    return folder


def read(path_results: str) -> dict:
    """The state `write` put under `path_results`, with its format checked before anything else."""
    target = os.path.join(path_results, DIRECTORY, STATE_FILE)
    if not os.path.exists(target):
        raise FileNotFoundError(
            f'no checkpoint under {path_results}: nothing to resume from. A run writes one every '
            f'`checkpoint_interval` timesteps; a run with checkpointing off cannot be resumed.')
    with open(target, 'rb') as handle:
        state = pickle.load(handle)
    found = state.get('checkpoint_format_version')
    if found != c.CHECKPOINT_FORMAT_VERSION:
        raise CheckpointFormatError(
            f'{target} carries checkpoint format {found!r}; this HAMLET reads '
            f'{c.CHECKPOINT_FORMAT_VERSION}. Resume it with the HAMLET that wrote it, or restart the run.')
    return state


def restore_rng(state: dict) -> None:
    """Put both generators where the checkpointed process left them."""
    random.setstate(state['rng']['random'])
    np.random.set_state(state['rng']['numpy'])


def truncate_grid_csvs(path_results: str, state: dict) -> dict:
    """Cut every grid CSV back to the length recorded at the checkpoint.

    Rows past that length were appended by timesteps the resumed run is about to execute again.
    A file the checkpoint did not know is left alone: its key is not in the grid's
    `flushed_results`, so the first append of the resumed run rewrites it from scratch. A file
    shorter than recorded is an error -- something other than this run has edited the results.

    Returns {relative path: lines removed}, for the caller to report.
    """
    removed = {}
    for relative, lines in state['grid_csv_lines'].items():
        file_path = os.path.join(path_results, relative)
        with open(file_path, 'rb') as handle:
            content = handle.read().splitlines(keepends=True)
        if len(content) < lines:
            raise RuntimeError(
                f'{file_path} has {len(content)} lines but the checkpoint recorded {lines}; the '
                f'results folder was edited after the checkpoint and cannot be resumed from.')
        if len(content) > lines:
            with open(file_path, 'wb') as handle:
                handle.writelines(content[:lines])
        removed[relative] = len(content) - lines
    return removed


def _grid_csv_lines(path_results: str) -> dict:
    """Line count of every CSV under `grids/`, keyed by its path relative to the results folder."""
    counts = {}
    grids = os.path.join(path_results, 'grids')
    for root, _, files in os.walk(grids):
        for name in sorted(files):
            if not name.endswith('.csv'):
                continue
            file_path = os.path.join(root, name)
            with open(file_path, 'rb') as handle:
                counts[os.path.relpath(file_path, path_results).replace(os.sep, '/')] = (
                    sum(1 for _ in handle))
    return counts


def _replace(staging: str, target: str, attempts: int = 3, wait: float = 0.1) -> None:
    """`os.replace` with the Windows retry `AgentDB._replace` explains."""
    for attempt in range(1, attempts + 1):
        try:
            os.replace(staging, target)
            return
        except PermissionError:
            if attempt == attempts:
                raise
            time.sleep(wait)
