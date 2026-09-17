"""The per-timestep database copy is taken only when a restriction can reject a pass.

`Executor.execute` deep-copies the whole database at the top of every timestep so that a pass a
grid restriction rejects can be rolled back. The copy is every agent's full-length tables, so it
scales with scenario length and not with anything the timestep does; measured on design 6
(2160 steps, 104 agents) it is 28 ms of a 3.9 s step and never grows, and a full-year scenario
copies ~575 MB per step. Only `EnWG14a` can reject a pass, and only with direct power control on,
so every other run took the copy for nothing (#249, part c).

Three things are pinned. The copy is skipped exactly when `rollback_possible` is False. The flag is
derived from the same config the grid stage reads. And a rejection with no copy to roll back to is
an error -- a restriction whose declaration is wrong must not continue from the rejected state.
"""

from types import SimpleNamespace

import polars as pl
import pytest

import hamlet.executor.setup as executor_module
from hamlet import constants as c
from hamlet.executor.setup import Executor
from hamlet.executor.utilities.grid_restrictions.enwg_14a import EnWG14a
from hamlet.executor.utilities.grid_restrictions.grid_restriction import GridRestriction
from hamlet.executor.utilities.grid_restrictions.grid_restriction_base import GridRestrictionBase

TIMESTAMPS = [f'2023-01-28 {hour:02d}:00:00+00:00' for hour in range(3)]


# --- the copy -------------------------------------------------------------------------------

class CountingDatabase:
    """Counts how often it is deep-copied; the executor copies nothing else."""

    def __init__(self):
        self.copies = 0

    def __deepcopy__(self, memo):
        self.copies += 1
        return self

    def end_timestep(self):
        pass


def executor_over(timestamps, rollback_possible, converge=True, max_iteration=1):
    executor = Executor.__new__(Executor)
    executor.timetable = pl.DataFrame({c.TC_TIMESTAMP: list(timestamps),
                                       c.TC_REGION: ['a_region'] * len(timestamps)})
    executor.type = 'sim'
    executor.max_iteration = max_iteration
    executor.checkpoint_interval = None
    executor.rollback_possible = rollback_possible
    executor.path_results = 'unused'
    executor.database = CountingDatabase()
    executor.pbar = SimpleNamespace(reset=lambda **kw: None, set_description=lambda *a, **kw: None,
                                    update=lambda *a: None)
    executor._Executor__execute_agents = lambda tasks: None
    executor._Executor__execute_markets = lambda tasks: None
    seen = []

    def execute_grids(tasklist, initial_db, num_iteration):
        seen.append(initial_db)
        return True if converge else num_iteration > executor.max_iteration

    executor._Executor__execute_grids = execute_grids
    return executor, seen


def test_the_database_is_copied_once_per_timestep_when_a_pass_can_be_rejected():
    executor, seen = executor_over(TIMESTAMPS, rollback_possible=True)

    executor.execute()

    assert executor.database.copies == len(TIMESTAMPS)
    assert all(initial_db is executor.database for initial_db in seen)


def test_the_database_is_not_copied_when_no_pass_can_be_rejected():
    executor, seen = executor_over(TIMESTAMPS, rollback_possible=False)

    executor.execute()

    assert executor.database.copies == 0
    assert seen == [None] * len(TIMESTAMPS), 'the grid stage must be told there is nothing to roll back to'


def test_a_rejected_pass_still_rolls_back_when_the_copy_was_taken():
    """The existing behaviour, kept: with a copy, a refusing grid is retried up to the cap."""
    executor, seen = executor_over(TIMESTAMPS[:1], rollback_possible=True, converge=False, max_iteration=2)

    executor.execute()

    assert len(seen) == 3, 'cap 2 means passes 1, 2 and the forced third'


# --- the fail-closed path -------------------------------------------------------------------

class RejectingGrid:
    """A grid whose restriction rejects every pass."""

    def __init__(self, *args, **kwargs):
        pass

    def execute(self):
        return object(), False


class AcceptingGrid:
    def __init__(self, *args, **kwargs):
        pass

    def execute(self):
        return object(), True


class OneGridDatabase:
    def __init__(self):
        self.posted = 0

    def get_grid_data(self):
        return {c.G_ELECTRICITY: object()}

    def post_grids(self, grid_type, grid, path_results=None):
        self.posted += 1

    def restore_superseded_commands(self):
        pass


def executor_for_grids(monkeypatch, grid_class):
    monkeypatch.setattr(executor_module, 'Grid', grid_class)
    executor = Executor.__new__(Executor)
    executor.database = OneGridDatabase()
    executor.max_iteration = 3
    executor.path_results = 'unused'
    executor.pbar = SimpleNamespace(set_description=lambda *a, **kw: None)
    return executor


def test_a_rejection_with_no_copy_is_an_error_not_a_continuation(monkeypatch):
    """A restriction that said it could not reject, then did, must stop the run.

    The alternative -- carrying on from the rejected dispatch -- is exactly the silent wrong
    answer the copy existed to prevent, and would never surface in the results.
    """
    executor = executor_for_grids(monkeypatch, RejectingGrid)

    with pytest.raises(RuntimeError, match='can_reject_a_pass'):
        executor._Executor__execute_grids(tasklist=pl.DataFrame(), initial_db=None, num_iteration=1)


def test_an_accepted_pass_needs_no_copy(monkeypatch):
    executor = executor_for_grids(monkeypatch, AcceptingGrid)

    grid_ok = executor._Executor__execute_grids(tasklist=pl.DataFrame(), initial_db=None, num_iteration=1)

    assert grid_ok is True
    assert executor.database.posted == 1


def test_a_rejection_past_the_cap_needs_no_copy_either(monkeypatch):
    """Past `max_iteration` the executor keeps the pass rather than rolling it back, copy or not."""
    executor = executor_for_grids(monkeypatch, RejectingGrid)

    grid_ok = executor._Executor__execute_grids(tasklist=pl.DataFrame(), initial_db=None, num_iteration=4)

    assert grid_ok is True


# --- the declaration ------------------------------------------------------------------------

def restrictions(apply, direct_power_control_active):
    return {
        'apply': list(apply),
        'max_iteration': 3,
        'enwg_14a': {'variable_grid_fees': {'active': True},
                     'direct_power_control': {'active': direct_power_control_active}},
    }


class ConfigDatabase:
    def __init__(self, grids, restrictions_config):
        self.grids = grids
        self.restrictions_config = restrictions_config

    def get_grid_data(self):
        return self.grids

    def get_general_data(self):
        return {c.K_GRID: {c.G_ELECTRICITY: {'restrictions': self.restrictions_config}}}


@pytest.mark.parametrize('grids, apply, dpc_active, expected', [
    ({}, ['enwg_14a'], True, False),                       # no electricity grid registered at all
    ({c.G_ELECTRICITY: object()}, [], True, False),        # grid, no restriction applied
    ({c.G_ELECTRICITY: object()}, ['enwg_14a'], False, False),  # §14a with fees only
    ({c.G_ELECTRICITY: object()}, ['enwg_14a'], True, True),    # §14a with direct power control
])
def test_rollback_is_possible_exactly_when_an_applied_restriction_can_reject(grids, apply, dpc_active, expected):
    executor = Executor.__new__(Executor)
    executor.database = ConfigDatabase(grids, restrictions(apply, dpc_active))

    assert executor._Executor__rollback_possible() is expected


def test_the_declaration_matches_what_enwg_14a_execute_does():
    """`execute` sets `grid_ok` only under direct power control; the declaration must say the same."""
    assert EnWG14a.can_reject_a_pass({'direct_power_control': {'active': True}}) is True
    assert EnWG14a.can_reject_a_pass({'direct_power_control': {'active': False}}) is False


def test_a_restriction_that_does_not_declare_cannot_reject():
    """The base answers False, so a new restriction that forgets the override is *refused* on a
    rejection (the test above) rather than silently continued -- the safe default."""
    assert GridRestrictionBase.can_reject_a_pass({'anything': True}) is False


@pytest.mark.parametrize('dpc_active', [True, False])
def test_the_registry_delegates_to_the_restriction_class(dpc_active):
    config = restrictions(['enwg_14a'], dpc_active)['enwg_14a']
    assert GridRestriction.can_reject_a_pass('enwg_14a', config) is EnWG14a.can_reject_a_pass(config)


def test_an_unknown_restriction_type_is_refused_by_name():
    with pytest.raises(KeyError):
        GridRestriction.can_reject_a_pass('not_a_restriction', {})


class SetupDatabase(ConfigDatabase):
    """What `__setup_database` touches, and nothing else."""

    def setup_database(self, structure):
        self.structure = structure


def test_setup_derives_the_flag_rather_than_leaving_the_constructor_default():
    """`__init__` defaults to True (copy, the safe side); setup must replace it with the answer."""
    executor = Executor.__new__(Executor)
    executor.structure = {}
    executor.rollback_possible = True
    executor.database = SetupDatabase({}, restrictions(['enwg_14a'], True))

    executor._Executor__setup_database()

    assert executor.rollback_possible is False, 'no electricity grid is registered, so no pass can be rejected'
    assert executor.max_iteration == 3
