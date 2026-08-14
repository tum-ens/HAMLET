"""Unit -- `num_workers` above 1 is refused, not quietly ignored.

The multiprocessing path is gone (ROADMAP section 6.3). It was not merely unused: measured on the
Mac mini against `develop`, `num_workers=2` fails at the first timestep of every shipped example
it can reach, because `agent_pool.get_grid_restriction_commands` lists a `grids/` directory that a
grid-less scenario does not have, `agent_pool.task`'s bare `except` turns that into a `None`, and
the parent unpacks it. The two grid-bearing examples never reach the executor at all.

`num_workers` stays in the signature because it is public API -- `run.py`, all four example
notebooks and the tests pass it -- but a caller who asks for eight workers now finds out. Running
serial without saying so is the same class of quiet wrong answer as accepting a timed-out solve.

These construct the Executor and let it fail in `__init__`, which is deliberate: the check has to
come before any scenario is read, or the refusal costs the caller a scenario load first.
"""
import pytest

from hamlet.executor.setup import Executor


@pytest.mark.parametrize('requested', [2, 4, 8, 16])
def test_more_than_one_worker_is_refused(tmp_path, requested):
    with pytest.raises(ValueError, match='one process'):
        Executor(str(tmp_path / 'scenario'), num_workers=requested)


def test_the_message_names_the_number_that_was_asked_for(tmp_path):
    """So the caller can find the argument, rather than hunting for which default did this."""
    with pytest.raises(ValueError, match=r'num_workers=8\b'):
        Executor(str(tmp_path / 'scenario'), num_workers=8)


def test_zero_is_refused_too(tmp_path):
    """It used to mean "pick a count for me" via `psutil.cpu_count`, so it is not a serial run."""
    with pytest.raises(ValueError):
        Executor(str(tmp_path / 'scenario'), num_workers=0)


@pytest.mark.parametrize('requested', [1, None])
def test_one_worker_and_the_default_are_accepted(tmp_path, requested):
    """Constructing must not raise: every caller in the tree passes one of these two."""
    scenario = tmp_path / 'scenario'
    scenario.mkdir()
    executor = Executor(str(scenario), num_workers=requested)

    assert executor.name == 'scenario'
