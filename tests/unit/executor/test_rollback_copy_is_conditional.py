"""The per-timestep database copy is taken only when a restriction can reject a pass.

`Executor.execute` deep-copies the whole database at the top of every timestep so that a pass a
grid restriction rejects can be rolled back. The copy is every agent's full-length tables, so it
scales with scenario length and not with anything the timestep does; measured on design 6
(2160 steps, 104 agents) it is 28 ms of a 3.9 s step and never grows, and a full-year scenario
copies ~575 MB per step. Only `EnWG14a` can reject a pass, and only with direct power control on,
so every other run took the copy for nothing (#249, part c).

Four things are pinned. The copy is skipped exactly when `rollback_possible` is False, and taken
once per timestep rather than once per pass. The flag is derived from the same config the grid
stage reads. `EnWG14a.execute` agrees with its own declaration -- the declaration is a claim about
`execute`, and a claim about a method is worth nothing unless the method is run against it. And a
rejection with no copy to roll back to is an error, per `Executor.__execute_grids`.
"""

from types import SimpleNamespace

import pandas as pd
import polars as pl
import pytest

import hamlet.executor.setup as executor_module
from hamlet import constants as c
from hamlet.executor.setup import Executor
from hamlet.executor.utilities.database.grid_db import ElectricityGridDB
from hamlet.executor.utilities.grid_restrictions.enwg_14a import EnWG14a
from hamlet.executor.utilities.grid_restrictions.grid_restriction import GridRestriction
from hamlet.executor.utilities.grid_restrictions.grid_restriction_base import GridRestrictionBase
from tests.unit.executor.executor_stub import bare_executor

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
    executor = bare_executor(timestamps, database=CountingDatabase(), max_iteration=max_iteration,
                             rollback_possible=rollback_possible)
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


def test_a_rejected_pass_is_retried_and_the_copy_is_taken_once_not_once_per_pass():
    """With a copy, a refusing grid is retried up to the cap; the copy is still one per timestep.

    Moving the copy inside the `while not grid_ok` loop keeps every other test here green, because
    a converging step has one pass; it is the cost this change is about, so it is pinned where the
    passes are several.
    """
    executor, seen = executor_over(TIMESTAMPS[:1], rollback_possible=True, converge=False, max_iteration=2)

    executor.execute()

    assert len(seen) == 3, 'cap 2 means passes 1, 2 and the forced third'
    assert executor.database.copies == 1


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


class Snapshot:
    """A database whose deep copy is distinguishable from itself."""

    def __init__(self, tag):
        self.tag = tag

    def __deepcopy__(self, memo):
        return Snapshot(self.tag + '-copy')

    def get_grid_data(self):
        return {c.G_ELECTRICITY: object()}

    def post_grids(self, grid_type, grid, path_results=None):
        pass

    def restore_superseded_commands(self):
        pass


def test_a_rejected_pass_replaces_the_database_with_a_copy_of_the_snapshot(monkeypatch):
    """The rollback the copy exists for. Aliasing the snapshot instead of copying it would let
    the second pass mutate what the third rolls back to; not rolling back at all keeps the
    rejected dispatch. Neither was pinned before this file."""
    executor = executor_for_grids(monkeypatch, RejectingGrid)
    executor.database = Snapshot('live')
    snapshot = Snapshot('initial')

    executor._Executor__execute_grids(tasklist=pl.DataFrame(), initial_db=snapshot, num_iteration=1)

    assert executor.database is not snapshot, 'the snapshot must survive for the next pass'
    assert executor.database.tag == 'initial-copy', 'the rejected pass must be discarded'


def test_a_rejection_with_no_copy_is_an_error_not_a_continuation(monkeypatch):
    """A restriction that said it could not reject, then did, must stop the run."""
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
    ({}, ['enwg_14a'], True, False),                       # no grid registered at all
    ({c.G_HEAT: object()}, ['enwg_14a'], True, False),     # a grid, but not the one that runs restrictions
    ({c.G_ELECTRICITY: object()}, [], True, False),        # grid, no restriction applied
    ({c.G_ELECTRICITY: object()}, ['enwg_14a'], False, False),  # §14a with fees only
    ({c.G_ELECTRICITY: object()}, ['enwg_14a'], True, True),    # §14a with direct power control
])
def test_rollback_is_possible_exactly_when_an_applied_restriction_can_reject(grids, apply, dpc_active, expected):
    executor = Executor.__new__(Executor)
    executor.database = ConfigDatabase(grids, restrictions(apply, dpc_active))

    assert executor._Executor__rollback_possible() is expected


def enwg_14a_with(dpc_active, overloaded):
    """An `EnWG14a` stubbed down to the branch structure of `execute`, as test_published_command does.

    `__calculate_direct_power_control` answers "rejected" whenever it is reached, so the only
    question left is whether `execute` reaches it.
    """
    grid_db = ElectricityGridDB.__new__(ElectricityGridDB)
    grid_db.results = {}
    grid_db.restriction_commands = {}
    enwg = EnWG14a.__new__(EnWG14a)
    enwg.grid_db = grid_db
    enwg.timestamp = pd.Timestamp('2023-01-28 00:00:00+00:00')
    enwg.restriction_config = {'variable_grid_fees': {'active': False},
                               'direct_power_control': {'active': dpc_active}}
    enwg._EnWG14a__check_overloads = lambda: overloaded
    enwg._EnWG14a__calculate_direct_power_control = lambda: False
    enwg.grid = SimpleNamespace(res_trafo=pd.DataFrame({'p_lv_mw': [], 'loading_percent': []}))
    return enwg


@pytest.mark.parametrize('overloaded', [False, True])
def test_enwg_14a_never_rejects_when_its_declaration_says_it_cannot(overloaded):
    """With direct power control off, an overloaded grid is reported, not sent back.

    A `grid_ok = False` on that branch would make every grid + fees-only run raise at timestep 1,
    and neither golden scenario has that shape -- one has no grid, the other has control on.
    """
    enwg = enwg_14a_with(dpc_active=False, overloaded=overloaded)
    assert EnWG14a.can_reject_a_pass(enwg.restriction_config) is False

    _, grid_ok = enwg.execute()

    assert grid_ok is True


def test_enwg_14a_rejects_exactly_when_its_declaration_says_it_can():
    enwg = enwg_14a_with(dpc_active=True, overloaded=True)
    assert EnWG14a.can_reject_a_pass(enwg.restriction_config) is True

    _, grid_ok = enwg.execute()

    assert grid_ok is False


def test_a_restriction_that_does_not_declare_cannot_reject():
    """The base answers False, so a new restriction that forgets the override is *refused* on a
    rejection (the test above) rather than silently continued -- the safe default."""
    assert GridRestrictionBase.can_reject_a_pass({'anything': True}) is False


@pytest.mark.parametrize('dpc_active', [True, False])
def test_the_registry_delegates_to_the_restriction_class(dpc_active):
    config = restrictions(['enwg_14a'], dpc_active)['enwg_14a']
    assert GridRestriction.can_reject_a_pass('enwg_14a', config) is EnWG14a.can_reject_a_pass(config)


def test_an_unknown_restriction_type_is_refused_by_name():
    # A complete config, so a fallback to a known type could answer it -- the refusal has to be
    # about the name.
    with pytest.raises(KeyError, match='not_a_restriction'):
        GridRestriction.can_reject_a_pass('not_a_restriction', restrictions(['enwg_14a'], True)['enwg_14a'])


def test_each_applied_restriction_is_asked_with_its_own_sub_config(monkeypatch):
    """With one type in the registry, reading `restrictions['enwg_14a']` for every name would pass."""
    class Other:
        @staticmethod
        def can_reject_a_pass(config):
            return config['flag']

    monkeypatch.setitem(GridRestriction.TYPES, 'other', Other)
    executor = Executor.__new__(Executor)
    config = restrictions(['other'], False)
    config['other'] = {'flag': True}
    executor.database = ConfigDatabase({c.G_ELECTRICITY: object()}, config)

    assert executor._Executor__rollback_possible() is True


class SetupDatabase(ConfigDatabase):
    """What `__setup_database` touches, and nothing else."""

    def setup_database(self, structure):
        self.structure = structure


def test_setup_derives_the_flag_rather_than_leaving_the_constructor_default():
    """`__init__` defaults to True (copy, the safe side); setup must replace it with the answer."""
    executor = bare_executor(database=SetupDatabase({}, restrictions(['enwg_14a'], True)), structure={})

    executor._Executor__setup_database()

    assert executor.rollback_possible is False, 'no electricity grid is registered, so no pass can be rejected'
    assert executor.max_iteration == 3
