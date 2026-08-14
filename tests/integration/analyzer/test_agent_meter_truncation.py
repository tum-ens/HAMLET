"""Integration -- where the agent meter series is cut, and where its time axis comes from (#230).

The meters table is allocated for the whole forecast horizon, so the rows past the simulated end
are all-zero padding and have to be dropped. The rule used to be
`max_index = meters.abs().idxmax().max()` -- the row at which some meter *peaked*, which is the
last recorded row only while a meter is still rising at the end.

**Every agent of every shipped scenario owns a continuously-rising load, so the old rule returned
the right answer for all 13 of them.** That is why this lives here and not in
`tests/e2e/test_analyzer_processors.py`: the committed reference did not move when #230 was fixed
and cannot move if it is reverted, so the e2e layer can never pin this. An agent owning only PV
and a battery is the case that breaks it -- both meters flatten after sunset -- and on real
`grid_golden` readings restricted to those two plants the old rule returned 19 of 25 timesteps,
dropping the entire evening battery discharge.
"""
import pandas as pd
import pytest

import hamlet.constants as c
from hamlet.analyzer.agents.agent_data_processor import AgentDataProcessor

#: 8 recorded rows then 4 of horizon padding. `pv` is cumulative generation that stops at row 4
#: and `battery` a cumulative net that peaks at row 3, so **no meter is rising at row 7** -- which
#: is exactly the shape `idxmax` gets wrong. Row 0 is the opening reading and is zero for both.
PV = [0, 0, 10, 20, 30, 30, 30, 30, 0, 0, 0, 0]
BATTERY = [0, 5, 3, 8, 2, 2, 2, 2, 0, 0, 0, 0]
RECORDED_ROWS = 8


def write_agent(root, agent, columns, rows=len(PV), start='2021-03-19 00:00'):
    folder = root / 'agents' / 'sfh' / agent
    folder.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({c.TC_TIMESTAMP: pd.date_range(start, periods=rows, freq='h', tz='UTC'),
                          **columns})
    frame.to_feather(folder / 'meters.ft')


@pytest.fixture
def processor(tmp_path):
    def build():
        return AgentDataProcessor(path={'run': str(tmp_path)}, config={'grids': {}})

    return build, tmp_path


def test_the_series_runs_to_the_last_recorded_row_not_to_the_last_peak(processor):
    """The #230 regression. Reverting to `idxmax` returns 5 rows here instead of 8."""
    build, root = processor
    write_agent(root, 'producer', {'p1_pv_electricity': PV, 'p2_battery_electricity': BATTERY})

    electricity = build().process_all_meters_data()['run']['electricity']

    assert len(electricity) == RECORDED_ROWS, (
        f'the meter series was cut to {len(electricity)} rows. Both meters flatten before the run '
        f'ends, so a rule based on where a meter peaks stops at row 4 and silently drops the last '
        f'{RECORDED_ROWS - 5} recorded timesteps')
    assert electricity.index[-1] == pd.Timestamp('2021-03-19 07:00', tz='UTC')


def test_the_horizon_padding_is_still_dropped(processor):
    """The other half: the padding must go, or every plot ends in a flat run of zeros.

    Without it the first padding row also reads as one enormous negative flow, because the meters
    are cumulative and the padding is zero.
    """
    build, root = processor
    write_agent(root, 'producer', {'p1_pv_electricity': PV, 'p2_battery_electricity': BATTERY})

    electricity = build().process_all_meters_data()['run']['electricity']

    assert len(electricity) < len(PV), 'the horizon padding was not dropped'
    assert electricity['total'].min() > -1, (
        'a large negative spike survived, so a padding row was differenced against a real reading')


def test_an_agent_that_stops_early_does_not_shorten_the_others(processor):
    """The scenario's length is the longest agent's, and a short agent contributes zeros."""
    build, root = processor
    write_agent(root, 'producer', {'p1_pv_electricity': PV, 'p2_battery_electricity': BATTERY})
    short = [0, 4, 6] + [0] * (len(PV) - 3)
    write_agent(root, 'short', {'p3_pv_electricity': short})

    electricity = build().process_all_meters_data()['run']['electricity']

    assert len(electricity) == RECORDED_ROWS


def test_the_time_axis_is_not_taken_from_whichever_agent_is_read_last(processor):
    """`timestamps` used to leak out of the per-agent loop, so the index came from `os.listdir`.

    Two agents whose meters disagree on the time axis is a broken results tree; before #230 it
    silently produced a plot indexed by one agent and filled by all of them.
    """
    build, root = processor
    write_agent(root, 'a_first', {'p1_pv_electricity': PV})
    write_agent(root, 'b_second', {'p2_pv_electricity': PV}, start='2022-01-01 00:00')

    with pytest.raises(ValueError, match='no single time axis'):
        build().process_all_meters_data()


def test_a_scenario_whose_meters_never_move_is_refused(processor):
    """Nothing to plot is reported, not returned as an empty frame nobody checks."""
    build, root = processor
    write_agent(root, 'idle', {'p1_pv_electricity': [0] * len(PV)})

    with pytest.raises(ValueError, match='nothing to plot'):
        build().process_all_meters_data()
