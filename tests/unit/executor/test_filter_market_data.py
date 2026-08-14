"""`Database.filter_market_data` must answer exactly what it answered before it got fast.

The function was rewritten because it was ~11 % of a design 6 timestep: for a datetime value it
materialised a full-length literal column onto the table, compared two columns row-wise, and
dropped the column again, once per call, 24 calls a timestep. The replacement casts the literal to
the column's dtype instead.

That is a pure performance change, so the test that matters is a **differential** one: the
previous implementation is kept here verbatim as `reference_filter`, and every case asserts the
two agree frame-for-frame. A rewrite that is faster and returns different rows is not a rewrite,
and the only thing that can catch that is running both.

The cases cover what the live caller does (three columns ANDed, one of them a tz-aware datetime)
plus the paths the caller does not reach but the signature allows -- OR mode, several values in
one column, no match, and an empty filter -- because a public static method's contract is its
signature, not its current caller.
"""

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from hamlet.executor.utilities.database.database import Database


def reference_filter(market, by: list[str], values: list[list], inclusive: bool = False):
    """The implementation as it stood at develop@77743782, kept to compare against.

    Copied unchanged apart from the name. If this ever needs editing to keep the test passing,
    the rewrite has changed behaviour and the test has done its job.
    """
    filters = {}
    new_columns_count = 0
    new_columns = []

    for i in range(len(by)):
        filters[by[i]] = False

        for value in values[i]:
            if isinstance(value, datetime):
                datetime_index = market.select(by[i])
                dtype = datetime_index.dtypes[0]
                time_unit = dtype.time_unit
                time_zone = dtype.time_zone

                column_name = 'new_columns_' + str(new_columns_count)
                market = market.with_columns(pl.lit(value)
                                             .alias(column_name)
                                             .cast(pl.Datetime(time_unit=time_unit,
                                                               time_zone=time_zone)))

                new_columns_count += 1
                new_columns.append(column_name)

                filters[by[i]] = filters[by[i]] | (pl.col(by[i]) == pl.col(column_name))
            else:
                filters[by[i]] = filters[by[i]] | (pl.col(by[i]) == value)

    if inclusive:
        predicate = True
        for column in filters.keys():
            predicate = predicate & filters[column]
    else:
        predicate = False
        for column in filters.keys():
            predicate = predicate | (filters[column])

    filtered_market = market.filter(predicate)

    if new_columns:
        filtered_market = filtered_market.drop(new_columns)

    return filtered_market


BASE = datetime(2023, 1, 1, tzinfo=timezone.utc)
#: The same instant without a timezone. `get_bids_offers` passes aware datetimes
#: pulled off the timetable, so this is the signature's contract rather than the
#: caller's habit -- and it is the only input that distinguishes the two
#: implementations.
NAIVE = datetime(2023, 1, 1)


@pytest.fixture
def market():
    """A table shaped like the one `get_bids_offers` builds: several markets, several timesteps.

    The timestamp column is `Datetime('ns', 'UTC')` because that is what the scenario's frames
    carry, and the time unit is precisely what the removed temporary column existed to reconcile.
    """
    rows = []
    for market_type, name in (('electricity', 'lem_continuous'), ('electricity', 'lem_daily'),
                              ('heat', 'lhm_continuous')):
        for step in range(6):
            for agent in ('a', 'b'):
                rows.append({'market': market_type, 'name': name,
                             'timestep': BASE + timedelta(hours=step),
                             'agent_id': agent, 'energy': step * 10})
    return pl.DataFrame(rows).with_columns(
        pl.col('timestep').cast(pl.Datetime(time_unit='ns', time_zone='UTC')))


CASES = [
    pytest.param(['market'], [['electricity']], True, id='one-column-string'),
    pytest.param(['timestep'], [[BASE + timedelta(hours=2)]], True, id='one-column-datetime'),
    pytest.param(['market', 'name', 'timestep'],
                 [['electricity'], ['lem_continuous'], [BASE + timedelta(hours=3)]], True,
                 id='what-get_bids_offers-asks'),
    pytest.param(['market', 'name'], [['electricity'], ['lhm_continuous']], True,
                 id='inclusive-with-no-match'),
    pytest.param(['market', 'name'], [['electricity'], ['lhm_continuous']], False,
                 id='exclusive-or-mode'),
    pytest.param(['name'], [['lem_continuous', 'lem_daily']], True, id='several-values-one-column'),
    pytest.param(['timestep'], [[BASE, BASE + timedelta(hours=5)]], True,
                 id='several-datetimes-one-column'),
    pytest.param(['market', 'timestep'],
                 [['heat'], [BASE + timedelta(hours=1)]], False, id='exclusive-mixed-types'),
    pytest.param(['timestep'], [[BASE + timedelta(hours=99)]], True, id='datetime-with-no-match'),
    # The cases below are the ones that make the dtype cast load-bearing. Without it, polars
    # cannot find a supertype for `datetime[ns, UTC]` and a naive `datetime[us]` literal and
    # panics out of Rust -- so a rewrite that drops the cast fails here and nowhere else.
    # Mutation testing found that gap: every other case passes with the cast removed.
    pytest.param(['timestep'], [[NAIVE + timedelta(hours=2)]], True,
                 id='naive-datetime-against-aware-column'),
    pytest.param(['market', 'timestep'], [['electricity'], [NAIVE + timedelta(hours=2)]], True,
                 id='naive-datetime-with-another-column'),
    pytest.param(['timestep'], [[NAIVE, BASE + timedelta(hours=5)]], True,
                 id='naive-and-aware-in-one-column'),
]


@pytest.mark.parametrize('by, values, inclusive', CASES)
def test_matches_the_previous_implementation(market, by, values, inclusive):
    expected = reference_filter(market, by=by, values=values, inclusive=inclusive)
    actual = Database.filter_market_data(market=market, by=by, values=values, inclusive=inclusive)
    assert actual.equals(expected), (
        f'{len(actual)} rows against {len(expected)} for by={by} inclusive={inclusive}')


def test_leaves_the_table_untouched(market):
    """The old implementation added and dropped columns on the way through; neither may leak.

    It rebound `market` to the frame carrying the temporary column, so a caller holding the
    original was unaffected -- but only by luck of polars being immutable. Pinning it means the
    rewrite cannot regress into mutating its argument.
    """
    before = market.clone()
    Database.filter_market_data(market=market, by=['market'], values=[['electricity']],
                                inclusive=True)
    assert market.equals(before)
    assert market.columns == before.columns


def test_returns_no_temporary_columns(market):
    """No `new_columns_0` may survive into the result, under either implementation."""
    result = Database.filter_market_data(
        market=market, by=['timestep'], values=[[BASE + timedelta(hours=2)]], inclusive=True)
    assert result.columns == market.columns


def test_datetime_filter_actually_selects(market):
    """A guard against both implementations agreeing on nothing.

    Every case above compares two implementations, so a bug that made both return zero rows would
    pass all of them. This asserts the interesting case selects the rows it should.
    """
    wanted = BASE + timedelta(hours=2)
    result = Database.filter_market_data(market=market, by=['timestep'], values=[[wanted]],
                                         inclusive=True)
    assert len(result) == 6  # three market/name pairs x two agents
    assert result['timestep'].unique().to_list() == [wanted]
