"""Regression — dropping out-of-horizon market records must not change what is kept.

`save_and_drop_past_records` was rewritten to slice a sorted frame instead of scanning it
twice with complementary filters. These tests pin the behaviour that must not change:
every row is either written out or retained, exactly once.
"""
import datetime

import polars as pl
import pytest

import hamlet.constants as c
from hamlet.executor.utilities.database.market_db import MarketDB

HORIZON_SECONDS = 3600
START = datetime.datetime(2021, 3, 24, 12, tzinfo=datetime.timezone.utc)


def table(offsets_seconds):
    """A market table with one row per offset from START.

    Carries both time columns because `concat_past_data` sorts by them.
    """
    stamps = [START + datetime.timedelta(seconds=s) for s in offsets_seconds]
    return pl.DataFrame({
        c.TC_TIMESTAMP: stamps,
        c.TC_TIMESTEP: stamps,
        c.TC_ID_AGENT: [f'agent{i}' for i in range(len(offsets_seconds))],
    })


@pytest.fixture
def market_db(tmp_path):
    """A `MarketDB` carrying a single table, with no files loaded from disk."""
    db = object.__new__(MarketDB)
    db.market_type = 'electricity'
    db.market_name = 'continuous'
    db.market_config = {'clearing': {'timing': {'horizon': [0, HORIZON_SECONDS]}}}
    db.files = [(f'{c.TN_MARKET_TRANSACTIONS}.ft', None)]
    db.market_transactions = pl.DataFrame()
    return db


def run(db, tmp_path, now):
    db.save_and_drop_past_records(now, str(tmp_path))
    return db.market_transactions


def test_rows_before_the_horizon_are_dropped(market_db, tmp_path):
    """Anything older than the horizon leaves the in-memory table."""
    # now = START + 2 h, so the horizon starts at START + 1 h
    market_db.market_transactions = table([0, 1800, 3600, 5400, 7200])

    kept = run(market_db, tmp_path, START + datetime.timedelta(seconds=7200))

    assert kept[c.TC_TIMESTEP].to_list() == [
        START + datetime.timedelta(seconds=s) for s in (3600, 5400, 7200)
    ]


def test_nothing_is_lost_or_duplicated(market_db, tmp_path):
    """Every row is either written to disk or kept, exactly once."""
    offsets = [0, 1800, 3600, 5400, 7200]
    market_db.market_transactions = table(offsets)

    kept = run(market_db, tmp_path, START + datetime.timedelta(seconds=7200))
    written = pl.concat([pl.read_ipc(p) for p in tmp_path.rglob('*.ft')])

    assert len(kept) + len(written) == len(offsets)
    assert set(kept[c.TC_ID_AGENT]) | set(written[c.TC_ID_AGENT]) == set(table(offsets)[c.TC_ID_AGENT])


def test_no_folder_is_created_when_everything_is_in_horizon(market_db, tmp_path):
    """The output folder is only created when there is something to write.

    It used to be created for every table on every timestep whether or not anything was
    dropped, and `create_folder` sleeps 10 ms on each call.
    """
    market_db.market_transactions = table([3600, 5400, 7200])

    kept = run(market_db, tmp_path, START + datetime.timedelta(seconds=7200))

    assert len(kept) == 3
    assert list(tmp_path.rglob('*.ft')) == []
    assert list(tmp_path.iterdir()) == []


def test_empty_table_is_a_no_op(market_db, tmp_path):
    """An empty table must not create folders or raise."""
    market_db.market_transactions = table([0]).clear()

    kept = run(market_db, tmp_path, START + datetime.timedelta(seconds=7200))

    assert kept.is_empty()
    assert list(tmp_path.rglob('*.ft')) == []


def test_concat_past_data_tolerates_a_missing_folder(market_db, tmp_path):
    """Nothing dropped means no folder, and concatenating must still work.

    The folder used to be created unconditionally as a side effect of dropping records, and
    `concat_past_data` relied on that: deferring the creation made it raise FileNotFoundError
    at the end of a run in which no table ever left the horizon.
    """
    market_db.market_save = str(tmp_path)
    market_db.market_transactions = table([3600, 5400, 7200])

    run(market_db, tmp_path, START + datetime.timedelta(seconds=7200))
    market_db.concat_past_data()

    written = list(tmp_path.glob(f'{c.TN_MARKET_TRANSACTIONS}.ft'))
    assert len(written) == 1


def test_unsorted_table_still_splits_correctly(market_db, tmp_path):
    """The slicing path assumes sortedness; an unsorted table must fall back to filtering."""
    market_db.market_transactions = table([7200, 0, 5400, 1800, 3600])

    kept = run(market_db, tmp_path, START + datetime.timedelta(seconds=7200))

    assert sorted(kept[c.TC_TIMESTEP].to_list()) == [
        START + datetime.timedelta(seconds=s) for s in (3600, 5400, 7200)
    ]
