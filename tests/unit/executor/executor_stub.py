"""An `Executor` built without `__init__`, carrying every attribute the timestep loop reads.

Four test modules stub the loop this way, and each new loop attribute -- `checkpoint_interval`,
`rollback_possible`, the resume and pause state -- broke every stub that existed at the time. One
place to add the next one to.
"""
from types import SimpleNamespace

import polars as pl

from hamlet import constants as c
from hamlet.executor.setup import Executor


def timetable(timestamps, region='a_region'):
    """One row per (timestamp, region), which is what `execute` partitions on."""
    return pl.DataFrame({c.TC_TIMESTAMP: list(timestamps), c.TC_REGION: [region] * len(timestamps)})


def bare_executor(timestamps=(), database=None, max_iteration=1, checkpoint_interval=None,
                  path_results='unused', **overrides):
    """The loop's inputs at their most inert; `overrides` set anything else by attribute name.

    Name-mangled attributes are set by their mangled name (`_Executor__last_timestamp`).
    """
    executor = Executor.__new__(Executor)
    executor.timetable = timetable(timestamps)
    executor.type = 'sim'
    executor.max_iteration = max_iteration
    executor.checkpoint_interval = checkpoint_interval
    executor.rollback_possible = True  # a stub grid may refuse to converge
    executor.path_results = path_results
    executor.database = database
    executor.resume = False
    executor.resumed_from_step = None
    executor._Executor__last_timestamp = None
    executor._Executor__pause_requested = False
    executor.paused = False
    executor.pbar = SimpleNamespace(reset=lambda **kw: None, set_description=lambda *a, **kw: None,
                                    update=lambda *a: None)
    for name, value in overrides.items():
        setattr(executor, name, value)
    return executor
