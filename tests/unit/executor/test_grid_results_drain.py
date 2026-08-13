"""Draining grid results must not change the file the run produces.

`GridDB.results` accumulated one DataFrame per key per timestep and was written once at the end
with a single `pd.concat`. It is now written out incrementally and dropped from memory, because
the executor deepcopies the whole database every timestep and that list was 100% of the deepcopy's
growth -- linear per step, so quadratic over a run.

The output is a committed artefact: the grid golden master fingerprints these CSVs. So the test
that matters is not "the drain runs" but **"the CSV is byte-identical to the one the undrained
code produced"**, and that is what every case here asserts, by building both and comparing bytes.

The retry case is the subtle one. A timestep can be simulated twice -- `while not grid_ok` rolls
the database back and runs it again, and `enwg_14a` rewrites the last entry of its keys when that
happens -- so the newest timestep's frames must stay in memory where they can still be replaced.
Draining them would write a row that the re-run then writes again.
"""

import pandas as pd
import pytest

from hamlet import constants as c
from hamlet.executor.utilities.database.grid_db import ElectricityGridDB


def make_grid_db(tmp_path):
    """A GridDB with only the state the drain touches, so the test needs no scenario."""
    grid_db = ElectricityGridDB.__new__(ElectricityGridDB)
    grid_db.grid_type = 'electricity'
    grid_db.results = {}
    grid_db.flushed_results = set()
    return grid_db


def frame(timestamp, value):
    """Shaped like what `Grid._write_result_to_grid_db` appends: timestamp inserted at column 0."""
    result = pd.DataFrame({'p_mw': [value, value + 0.5], 'q_mvar': [0.1, 0.2]}, index=[0, 1])
    result.insert(0, c.TC_TIMESTAMP, str(timestamp))
    return result


def undrained_bytes(tmp_path, frames):
    """What the previous implementation wrote: one concat, one `to_csv`."""
    path = tmp_path / 'reference.csv'
    pd.concat(frames).to_csv(path)
    return path.read_bytes()


@pytest.mark.parametrize('steps', [2, 3, 7])
def test_drained_file_matches_the_undrained_one(tmp_path, steps):
    frames = [frame(f'2023-01-01 0{i}:00:00+00:00', i) for i in range(steps)]

    grid_db = make_grid_db(tmp_path)
    grid_db.results['res_bus'] = []
    for one in frames:
        grid_db.results['res_bus'].append(one)
        grid_db.save_and_drop_past_results(str(tmp_path))

    # `save_grid` writes whatever is still held; call the same helper it uses.
    grid_db._append_results(str(tmp_path), 'res_bus', grid_db.results['res_bus'])

    written = (tmp_path / 'electricity' / 'res_bus.csv').read_bytes()
    assert written == undrained_bytes(tmp_path, frames)


def test_the_newest_timestep_is_kept_in_memory(tmp_path):
    """A timestep that can still be re-run must not have been written yet."""
    grid_db = make_grid_db(tmp_path)
    grid_db.results['res_bus'] = [frame('2023-01-01 00:00:00+00:00', 0),
                                  frame('2023-01-01 01:00:00+00:00', 1)]

    grid_db.save_and_drop_past_results(str(tmp_path))

    held = grid_db.results['res_bus']
    assert len(held) == 1
    assert str(held[0][c.TC_TIMESTAMP].iloc[0]) == '2023-01-01 01:00:00+00:00'


def test_a_rewritten_last_entry_is_not_double_counted(tmp_path):
    """`enwg_14a` deletes and re-appends the newest entry on a re-run; the file must show it once.

    This is the case the drain could plausibly corrupt, so it is spelled out rather than trusted:
    two frames arrive for the same timestep, the second replacing the first.
    """
    grid_db = make_grid_db(tmp_path)
    grid_db.results['res_bus'] = [frame('2023-01-01 00:00:00+00:00', 0),
                                  frame('2023-01-01 01:00:00+00:00', 1)]
    grid_db.save_and_drop_past_results(str(tmp_path))

    # the re-run: drop the newest entry and append the corrected one
    del grid_db.results['res_bus'][-1]
    corrected = frame('2023-01-01 01:00:00+00:00', 99)
    grid_db.results['res_bus'].append(corrected)
    grid_db.save_and_drop_past_results(str(tmp_path))
    grid_db._append_results(str(tmp_path), 'res_bus', grid_db.results['res_bus'])

    written = pd.read_csv(tmp_path / 'electricity' / 'res_bus.csv')
    hour_one = written[written[c.TC_TIMESTAMP] == '2023-01-01 01:00:00+00:00']
    assert len(hour_one) == 2, 'the corrected frame should appear once, with its two rows'
    assert set(hour_one['p_mw']) == {99, 99.5}, 'the superseded values must not be in the file'


def test_nothing_is_written_before_anything_settles(tmp_path):
    """One timestep in flight is not settled, so there is nothing to write yet."""
    grid_db = make_grid_db(tmp_path)
    grid_db.results['res_bus'] = [frame('2023-01-01 00:00:00+00:00', 0)]

    grid_db.save_and_drop_past_results(str(tmp_path))

    assert grid_db.results['res_bus'], 'the only frame present must be kept'
    assert not (tmp_path / 'electricity' / 'res_bus.csv').exists()


def test_every_key_is_drained_independently(tmp_path):
    """Seven keys accumulate on an electricity grid; each has its own file and its own state."""
    grid_db = make_grid_db(tmp_path)
    for key in ('res_bus', 'res_line', 'res_trafo'):
        grid_db.results[key] = [frame('2023-01-01 00:00:00+00:00', 0),
                                frame('2023-01-01 01:00:00+00:00', 1)]

    grid_db.save_and_drop_past_results(str(tmp_path))

    for key in ('res_bus', 'res_line', 'res_trafo'):
        assert (tmp_path / 'electricity' / f'{key}.csv').exists()
        assert len(grid_db.results[key]) == 1
        assert key in grid_db.flushed_results
