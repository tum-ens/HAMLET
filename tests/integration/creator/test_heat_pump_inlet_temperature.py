"""The inlet temperature each heat pump is simulated at, per shipped spec file.

**Why this is a test rather than a line of code someone reads.** `agents.py:858` used to read the
air/brine discriminator from `specs['type']`, which is the *plant* type and is the literal `'hp'`
in every spec file HAMLET ships. The comparison at `:887` therefore never matched, and **every**
air-source heat pump in every scenario was simulated on soil inlet temperature -- warmer than
winter air by 15 K or more, so its COP never fell and its winter electrical draw never rose
(#234). Nothing failed, because the COP series is an *input* to every downstream number: the
golden master pinned the output of the wrong series, and regenerating it would have re-pinned the
corrected one.

So the assertion has to be about the series itself, and it is made two ways on purpose:

- `test_the_air_unit_is_simulated_at_ambient_temperature` and its brine counterpart pin the
  series **exactly**, against an hplib call the test makes itself. This is the direct statement of
  the fix: the air spec selects `t_amb`, the ground and water specs select `calc_brine_temp`.
- `test_the_air_unit_collapses_in_the_cold_and_the_ground_unit_does_not` pins the **physics**,
  against no reimplementation at all. `calc_brine_temp` has a floor near 3 degC over this weather
  file, so the brine path cannot produce a low COP however cold the air gets; the air path must.
  An exact-match test alone can be satisfied by a test and an implementation that mirror the same
  wrong idea, and that has happened here before.

Deliberately over the **shipped spec files** rather than over whatever a seeded scenario happens
to draw. `grid_golden` draws three air units and one brine unit today, but that is an RNG outcome:
a reseeding that drew four brine units would leave the air path untested and the file green.
"""
import os

import numpy as np
import pandas as pd
import pytest
from hplib import hplib

import hamlet.constants as c
from hamlet.creator.agents.agents import Agents
from hamlet.functions import load_file
from tests.scenario_run import REPO_ROOT

#: The spec files HAMLET ships for single-family homes, and the inlet temperature each must be
#: simulated at. `hp_type` is the discriminator; `type` is `'hp'` for all three, which is the
#: defect. Only `Outdoor Air/Water` takes ambient air.
SHIPPED_SPECS = {
    'hp_Bosch Thermotechnik_air.json': ('Outdoor Air/Water', 'ambient'),
    'hp_DAIKIN Europe_ground.json': ('Brine/Water', 'brine'),
    'hp_ait-deutschland_water.json': ('Water/Water', 'brine'),
}

#: Any config folder with a `setup.yaml` naming the weather file will do -- the method under test
#: reads only `self.input_path` and `self.setup['location']['weather']`.
FIXTURE = REPO_ROOT / 'tests' / 'e2e' / 'scenarios' / 'grid_golden'
SPEC_DIR = REPO_ROOT / 'input_data' / 'agents' / c.A_SFH / c.P_HP


@pytest.fixture(scope='module')
def creator(tmp_path_factory):
    """A Creator pointed at the shipped input data, for the timeseries method alone.

    Constructed directly rather than through a scenario run: the method under test draws no random
    numbers and depends on no agent, so making it depend on a seeded draw would only add a second
    reason for this file to fail.
    """
    return Agents(config_path=str(FIXTURE), input_path=str(REPO_ROOT / 'input_data'),
                  scenario_path=str(tmp_path_factory.mktemp('hp_inlet')), config_root=str(FIXTURE))


@pytest.fixture(scope='module')
def ambient_temperature(creator):
    """The weather file's ambient temperature in degC, read the way the Creator reads it."""
    weather = load_file(os.path.join(str(REPO_ROOT / 'input_data'), 'general', 'weather',
                                     creator.setup['location']['weather']))
    weather = weather[weather[c.TC_TIMESTAMP] == weather[c.TC_TIMESTEP]]
    weather.index = pd.DatetimeIndex(pd.to_datetime(weather[c.TC_TIMESTAMP], unit='s', utc=True))
    return weather[c.TC_TEMPERATURE] + c.KELVIN_TO_CELSIUS


def simulate(spec, t_in, t_amb):
    """hplib's heating-mode COP for one spec at a given inlet series, in the method's units.

    Mirrors `agents.py`'s supply temperature and transfer loss (40 degC flow, 5 K) and its
    `COP * 100`, rounded to int, so the comparison is against the number that actually reaches
    the scenario rather than a float the scenario never sees.
    """
    system = hplib.HeatPump(hplib.get_parameters(model=spec['model']))
    result = system.simulate(t_in_primary=np.array(t_in), t_in_secondary=40 - 5,
                             t_amb=np.array(t_amb), mode=1)
    return pd.Series(result['COP'] * c.COP_TO_COP100).round().astype(int).to_numpy()


def series_for(creator, filename):
    """The heat-pump timeseries the Creator builds from one shipped spec file."""
    spec = load_file(str(SPEC_DIR / filename))
    # Name-mangled: the method is private to `Agents` and has no public caller that takes a spec
    # directly. `plant` is accepted and unused (`agents.py:844`).
    built = creator._Agents__timeseries_from_specs_hp(specs=spec, plant={})
    return spec, built[f'{c.S_COP}_{c.P_HEAT}'].to_numpy()


@pytest.mark.parametrize('filename', [name for name, (_, inlet) in SHIPPED_SPECS.items()
                                      if inlet == 'ambient'])
def test_the_air_unit_is_simulated_at_ambient_temperature(creator, ambient_temperature, filename):
    """An `Outdoor Air/Water` spec must draw its source heat from the air, not from the soil."""
    spec, built = series_for(creator, filename)
    assert spec['hp_type'] == SHIPPED_SPECS[filename][0], (
        f'{filename} no longer declares {SHIPPED_SPECS[filename][0]!r}, so this test is pointed '
        f'at the wrong unit')

    ambient = simulate(spec, ambient_temperature, ambient_temperature)
    brine = simulate(spec, Agents.calc_brine_temp(ambient_temperature), ambient_temperature)

    assert not np.array_equal(ambient, brine), (
        'the two inlet temperatures produce the same COP series, so this test cannot tell them '
        'apart and asserts nothing')
    np.testing.assert_array_equal(built, ambient)


@pytest.mark.parametrize('filename', [name for name, (_, inlet) in SHIPPED_SPECS.items()
                                      if inlet == 'brine'])
def test_the_ground_and_water_units_are_simulated_at_brine_temperature(creator,
                                                                       ambient_temperature,
                                                                       filename):
    """Everything that is not an air unit keeps the soil inlet -- the fix must not reach further.

    This is the half that says the change is contained. `Water/Water` taking `calc_brine_temp` is
    itself questionable (groundwater is not soil, #234), but that is a modelling question and
    this pins today's behaviour so a change to it has to be made deliberately.
    """
    spec, built = series_for(creator, filename)
    assert spec['hp_type'] == SHIPPED_SPECS[filename][0], (
        f'{filename} no longer declares {SHIPPED_SPECS[filename][0]!r}')

    brine = simulate(spec, Agents.calc_brine_temp(ambient_temperature), ambient_temperature)
    np.testing.assert_array_equal(built, brine)


def test_the_air_unit_collapses_in_the_cold_and_the_ground_unit_does_not(creator,
                                                                        ambient_temperature):
    """At the coldest hour the air unit must be far worse than the ground unit.

    No hplib call of the test's own: this reads the two series the Creator built and compares
    them. `calc_brine_temp` bottoms out at about 3 degC over this weather file, so the brine path
    has a COP floor no air temperature can push it below; an air unit at -12 degC must fall well
    under it. Before #234 was fixed the air unit sat *above* 4.0 here, which is what this catches.
    """
    coldest = int(np.argmin(ambient_temperature.to_numpy()))
    assert ambient_temperature.iloc[coldest] < -10, (
        f'the weather file no longer gets colder than {ambient_temperature.iloc[coldest]:.1f} degC,'
        f' so it cannot separate an air inlet from a soil inlet and this test asserts nothing')

    _, air = series_for(creator, 'hp_Bosch Thermotechnik_air.json')
    _, ground = series_for(creator, 'hp_DAIKIN Europe_ground.json')

    assert air[coldest] < 300, (
        f'the air unit shows COP {air[coldest] / c.COP_TO_COP100:.2f} at '
        f'{ambient_temperature.iloc[coldest]:.1f} degC ambient, which no air-source unit reaches; '
        f'it is being simulated on soil temperature (#234)')
    assert ground[coldest] > 400, (
        f'the ground unit shows COP {ground[coldest] / c.COP_TO_COP100:.2f} at the coldest hour, '
        f'so it is no longer on the soil inlet either and the comparison means nothing')
