"""Two runs of one scenario must not write into the same results directory.

`Executor(name=...)` names a run, and `Database` used to key the results directory on the first
**region** instead. For every shipped scenario those are the same string -- a run is named after
its scenario folder, and the one region is named after it too -- so the two were
indistinguishable until an experiment ran the same scenario twice under two names. Then both runs
resolved to one directory and the second overwrote the first (#243).

It failed silently, and that is what makes the assertions here specific. The scenario copy and the
per-timestep agent saves *do* honour the name, so after the collision both runs still had a
populated results folder with a plausible tree in it. Asserting that the directory exists, or that
a CSV is non-empty, passes on the broken code. Only the **contents** distinguish them, so every
case below asserts which timestamps a run's own file holds.
"""

import os

import pandas as pd
import pytest

from hamlet import constants as c
from hamlet.executor.setup import Executor
from hamlet.executor.utilities.database.database import Database
from hamlet.executor.utilities.database.grid_db import ElectricityGridDB

#: The region both runs share. Deliberately *not* equal to either run name -- that equality is
#: exactly the coincidence that hid the bug.
REGION = 'a_region'


class RecordingRegion:
    """Stands in for RegionDB, recording only where it was told to write."""

    def __init__(self):
        self.saved_to = None

    def save_region(self, path):
        self.saved_to = path


class RecordingGrid:
    """Stands in for a GridDB, recording only where `save_grid` was told to write."""

    def __init__(self):
        self.saved_to = None

    def save_grid(self, path):
        self.saved_to = path


def make_grid_db():
    """A GridDB carrying only the state the drain touches, as in test_grid_results_drain."""
    grid_db = ElectricityGridDB.__new__(ElectricityGridDB)
    grid_db.grid_type = 'electricity'
    grid_db.results = {}
    grid_db.flushed_results = set()
    return grid_db


def frame(timestamp, value):
    """Shaped like what `Grid._write_result_to_grid_db` appends."""
    result = pd.DataFrame({'p_mw': [value]}, index=[0])
    result.insert(0, c.TC_TIMESTAMP, str(timestamp))
    return result


def run_one_arm(root, name, timestamps):
    """Drive one run's grid drain through `post_grids`, as the executor does per timestep."""
    database = Database(os.path.join(str(root), 'scenario_folder'), name=name)
    # Both arms carry the *same* region. Without it the pre-fix code raises IndexError on an empty
    # region dict, which reddens the test without ever producing the collision it is about.
    database._Database__regions = {REGION: RecordingRegion()}
    grid_db = make_grid_db()
    grid_db.results['res_trafo'] = []

    path_results = os.path.join(str(root), name)
    for index, timestamp in enumerate(timestamps):
        grid_db.results['res_trafo'].append(frame(timestamp, index))
        database.post_grids(grid_type='electricity', grid=grid_db, path_results=path_results)

    # `save_grid` flushes whatever is still held at the end of a run; the drain deliberately keeps
    # the newest timestep because it can still be re-run.
    grid_db._append_results(
        database._Database__grid_results_path(str(root)), 'res_trafo', grid_db.results['res_trafo'])
    return path_results


def written_timestamps(root, name):
    path = os.path.join(str(root), name, 'grids', 'electricity', 'res_trafo.csv')
    assert os.path.exists(path), (
        f"{name} has no grid results at its own path {path}. Both runs resolved to one directory, "
        f"and whichever ran second owns whatever is on disk.")
    return list(pd.read_csv(path)[c.TC_TIMESTAMP].astype(str))


ARM_A = ['2023-01-01 00:00:00+00:00', '2023-01-01 01:00:00+00:00', '2023-01-01 02:00:00+00:00']
ARM_B = ['2023-06-01 00:00:00+00:00', '2023-06-01 01:00:00+00:00']


def test_two_runs_of_one_scenario_keep_their_own_grid_results(tmp_path):
    """The failing case: arm B used to truncate arm A's file and write its own rows into it."""
    run_one_arm(tmp_path, 'arm_a', ARM_A)
    run_one_arm(tmp_path, 'arm_b', ARM_B)

    # Absolute values on both sides. A count, or "arm A's file is non-empty", survives the
    # collision -- arm B leaves three rows of its own behind.
    assert written_timestamps(tmp_path, 'arm_a') == ARM_A
    assert written_timestamps(tmp_path, 'arm_b') == ARM_B


def test_the_two_runs_write_to_different_directories(tmp_path):
    """States the mechanism directly, so a failure says *why* rather than only *that*."""
    database_a = Database(os.path.join(str(tmp_path), 'scenario_folder'), name='arm_a')
    database_b = Database(os.path.join(str(tmp_path), 'scenario_folder'), name='arm_b')
    database_a._Database__regions = {REGION: RecordingRegion()}
    database_b._Database__regions = {REGION: RecordingRegion()}

    path_a = database_a._Database__grid_results_path(str(tmp_path))
    path_b = database_b._Database__grid_results_path(str(tmp_path))

    assert path_a == os.path.join(str(tmp_path), 'arm_a', 'grids')
    assert path_b == os.path.join(str(tmp_path), 'arm_b', 'grids')


def test_region_data_also_follows_the_run_name(tmp_path):
    """`save_database` collided too -- it is not only the grid half (the issue understated this)."""
    saved = {}
    for name in ('arm_a', 'arm_b'):
        database = Database(os.path.join(str(tmp_path), 'scenario_folder'), name=name)
        region = RecordingRegion()
        database._Database__regions = {REGION: region}
        database.save_database(str(tmp_path))
        saved[name] = region.saved_to

    assert saved['arm_a'] == os.path.join(str(tmp_path), 'arm_a')
    assert saved['arm_b'] == os.path.join(str(tmp_path), 'arm_b')


def test_the_root_region_takes_the_run_directory_and_later_ones_do_not_move(tmp_path):
    """The root region's output is the run's output; the rest stay exactly where they were.

    Nesting later regions under the run was tried and reverted: the market drain writes dropped
    rows per timestep to `<path_results>/markets/...` with no region in the path, while
    `concat_past_data` reads them back from `<region_save>/markets/...`. Moving a region separates
    the two, and the market table silently ends up holding only the in-flight tail.
    """
    database = Database(os.path.join(str(tmp_path), 'scenario_folder'), name='arm_a')
    root, second = RecordingRegion(), RecordingRegion()
    database._Database__regions = {'scenario_folder': root, 'south': second}
    database.save_database(str(tmp_path))

    assert root.saved_to == os.path.join(str(tmp_path), 'arm_a')
    # A sibling of the run directory, which is where it landed before this change.
    assert second.saved_to == os.path.join(str(tmp_path), 'south')


def test_the_region_and_the_grid_halves_land_under_one_run_directory(tmp_path):
    """The whole point of sharing a construction: the two cannot drift apart.

    `save_database` must put the grid results beside the region's, not somewhere else. Without a
    grid registered its loop body never runs, so this registers one -- the line is otherwise
    executed without ever producing a path.
    """
    database = Database(os.path.join(str(tmp_path), 'scenario_folder'), name='arm_a')
    region = RecordingRegion()
    database._Database__regions = {'scenario_folder': region}
    grid = RecordingGrid()
    database._Database__grids = {'electricity': grid}

    database.save_database(str(tmp_path))

    assert region.saved_to == os.path.join(str(tmp_path), 'arm_a')
    assert grid.saved_to == os.path.join(str(tmp_path), 'arm_a', 'grids')
    assert os.path.dirname(grid.saved_to) == region.saved_to


def test_a_region_named_after_the_run_still_collapses_into_it(tmp_path):
    """The shipped configuration, which every other case here deliberately avoids.

    `REGION` is unequal to the run name everywhere else, because that inequality is what makes the
    pre-fix bug visible. But it means the *actual* shipped shape -- one region named after the
    scenario folder, which is also the default run name -- goes unasserted, and an implementation
    that nested exactly that case would move every existing results folder down one level while
    every other test here stayed green.
    """
    database = Database(os.path.join(str(tmp_path), 'six_centralized'))
    region = RecordingRegion()
    database._Database__regions = {'six_centralized': region}
    database.save_database(str(tmp_path))

    assert region.saved_to == os.path.join(str(tmp_path), 'six_centralized')


def test_the_executor_hands_its_name_to_the_database(tmp_path):
    """The one line that connects the fix to the product.

    Everything else here builds a `Database` directly and passes the name itself, so dropping
    `name=self.name` at the `Database(...)` construction restores #243 in full with every tier
    green -- no call site in the repository passes `Executor(name=...)`, so the default masks it.
    """
    executor = Executor(os.path.join(str(tmp_path), 'six_centralized'), name='arm_b',
                        num_workers=1)

    assert executor.name == 'arm_b'
    assert executor.database._Database__run_name == 'arm_b'
    assert executor.database._Database__grid_results_path(str(tmp_path)) == os.path.join(
        str(tmp_path), 'arm_b', 'grids')


def test_default_name_is_the_scenario_folder_so_existing_layouts_do_not_move(tmp_path):
    """A Database built without a name must write exactly where it always did."""
    database = Database(os.path.join(str(tmp_path), 'six_centralized'))
    database._Database__regions = {REGION: RecordingRegion()}
    assert database._Database__grid_results_path(str(tmp_path)) == os.path.join(
        str(tmp_path), 'six_centralized', 'grids')
