"""Regression — EV time series must be resampled by column meaning, not averaged.

An EV series holds an availability flag and the energy consumed by driving. The generic
resampler averages when aggregating and interpolates when splitting, which is right for a
power series and wrong for both of these.
"""
import pandas as pd
import pytest

from hamlet import constants as c
from hamlet.creator.agents.agents import Agents

PLANT_ID = 'ev1'
AVAILABILITY = f'{PLANT_ID}_availability'
ENERGY = f'{PLANT_ID}_energy_consumed'

QUARTER_HOURLY = pd.Timedelta(minutes=15)
HOURLY = pd.Timedelta(hours=1)


@pytest.fixture
def resampler():
    """`_resample_timeseries` bound to an `Agents` with only its plant registry populated."""
    agents = object.__new__(Agents)
    agents.plants = {
        c.P_EV: {'resample_ts': Agents._Agents__resample_timeseries_ev},
        c.P_BATTERY: {},
    }
    return agents


@pytest.fixture
def quarter_hourly_trip():
    """One hour of quarter-hourly data: the car leaves for the middle two quarters."""
    index = pd.date_range('2021-03-24', periods=4, freq='15min', tz='UTC')
    return pd.DataFrame(
        {AVAILABILITY: [1, 0, 0, 1], ENERGY: [0, 2000, 3000, 0]},
        index=index.astype('int64') // 10 ** 9,
    )


def test_driving_energy_is_summed_not_averaged(resampler, quarter_hourly_trip):
    """Regression: aggregating averaged the trip energy, deleting most of the consumption.

    The car used 5 kWh over the hour. Averaging the four quarter-hours reports 1.25 kWh, so
    the EV silently arrives with far more charge than it should.
    """
    out = resampler._resample_timeseries(quarter_hourly_trip, HOURLY, {'type': c.P_EV})

    assert out[ENERGY].iloc[0] == 5000


def test_availability_stays_a_flag(resampler, quarter_hourly_trip):
    """Regression: averaging the flag produced a fractional availability.

    The controller multiplies the charging power limit by this value, so 0.5 silently halves
    the charger's rating instead of meaning "home for part of the hour".
    """
    out = resampler._resample_timeseries(quarter_hourly_trip, HOURLY, {'type': c.P_EV})

    assert out[AVAILABILITY].isin([0, 1]).all()
    assert out[AVAILABILITY].iloc[0] == 1


def test_splitting_carries_values_forward(resampler):
    """Interpolating between an unavailable and an available step invents fractional states."""
    index = pd.date_range('2021-03-24', periods=2, freq='1h', tz='UTC')
    hourly = pd.DataFrame(
        {AVAILABILITY: [0, 1], ENERGY: [4000, 0]},
        index=index.astype('int64') // 10 ** 9,
    )

    out = resampler._resample_timeseries(hourly, QUARTER_HOURLY, {'type': c.P_EV})

    assert out[AVAILABILITY].isin([0, 1]).all()


def test_the_generic_resampler_would_mangle_ev_data(resampler, quarter_hourly_trip):
    """Documents the defect being fixed, since the old signature cannot be called any more.

    Routing the same EV data through the generic (mean) path loses 3.75 of the 5 kWh consumed.
    Worse, the availability flag averages to 0.5 and is then cast back to the input's integer
    dtype, so it truncates to 0: the car reads as never available and can never charge.
    """
    out = resampler._resample_timeseries(quarter_hourly_trip, HOURLY, {'type': c.P_BATTERY})

    assert out[ENERGY].iloc[0] == 1250
    assert out[AVAILABILITY].iloc[0] == 0


def test_other_plant_types_keep_the_generic_resampler(resampler):
    """The dispatch must only divert plant types that registered their own function."""
    index = pd.date_range('2021-03-24', periods=4, freq='15min', tz='UTC')
    power = pd.DataFrame({'p': [100, 200, 300, 400]}, index=index.astype('int64') // 10 ** 9)

    out = resampler._resample_timeseries(power, HOURLY, {'type': c.P_BATTERY})

    assert out['p'].iloc[0] == 250  # mean, not sum
