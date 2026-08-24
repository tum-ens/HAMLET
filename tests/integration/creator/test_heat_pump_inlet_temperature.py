"""The inlet temperature each heat pump is simulated at, per shipped spec file.

**Why this is a test rather than a line of code someone reads.** `agents.py:858` used to read the
air/brine discriminator from `specs['type']`, which is the *plant* type and is the literal `'hp'`
in every spec file HAMLET ships. The comparison at `:887` therefore never matched, and **every**
air-source heat pump in every scenario was simulated on soil inlet temperature -- over the shipped
weather file, 3.7 K warmer than the air on average below 7 degC and 15.5 K warmer at the coldest
hour -- so its COP never fell as far as it should and its winter electrical draw never rose
(#234). Nothing failed, because the COP series is an *input* to every downstream number: the
golden master pinned the output of the wrong series, and regenerating it would have re-pinned the
corrected one.

So the assertion has to be about the series itself, and it is made two ways on purpose:

- `test_the_air_unit_is_simulated_at_ambient_temperature` and its brine counterpart pin the
  series **exactly**, against an hplib call the test makes itself. This is the direct statement of
  the fix: the air spec selects `t_amb`, the ground and water specs select `calc_brine_temp`.
- `test_the_air_unit_collapses_in_the_cold` pins the **physics**, on one spec against itself.
  `calc_brine_temp` floors at 3.16 degC over this weather file, so the soil path cannot produce a
  low COP however cold the air gets; the air path must. An exact-match test alone can be satisfied
  by a test and an implementation that mirror the same wrong idea, and that has happened here
  before.

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

#: Every agent type that ships heat-pump specs. `SPEC_DIR` above is only the first of them, and
#: `test_the_shipped_specs_are_the_ones_this_file_tests` is what stops the other two drifting.
SPEC_DIRS = tuple(REPO_ROOT / 'input_data' / 'agents' / agent_type / c.P_HP
                  for agent_type in (c.A_SFH, c.A_CTSP, c.A_INDUSTRY))

#: COP * 100 separating the two inlets at the coldest hour. Measured on the shipped Bosch unit and
#: this weather file: 141 on air, 459 on soil. Both sides are asserted, so this is not a threshold
#: taken on trust -- see `test_the_air_unit_collapses_in_the_cold`.
COLD_COP100 = 300


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
    """hplib's output for one spec at a given inlet series, reduced the way `agents.py` does.

    **All six columns, not just the COP.** The method makes two `system.simulate` calls, one per
    mode, each taking `t_in_primary` separately (`agents.py:894` and `:897`), and writes six
    columns from them. Asserting only `COP100_heat` would leave a change that reverted the DHW
    call alone -- half the defect, in the same method, four lines apart -- entirely green. And the
    electrical power is the quantity #234 is actually *about*: the COP is what moved, the draw is
    what it cost.

    Mirrors `agents.py`'s supply temperatures and transfer loss (40 degC flow for heating, 55 for
    DHW, 5 K loss) and its rounding to int, so the comparison is against the numbers that reach
    the scenario rather than floats the scenario never sees.
    """
    system = hplib.HeatPump(hplib.get_parameters(model=spec['model']))
    columns = {}
    for mode, supply in ((c.P_HEAT, 40), (c.P_DHW, 55)):
        result = system.simulate(t_in_primary=np.array(t_in), t_in_secondary=supply - 5,
                                 t_amb=np.array(t_amb), mode=1)
        columns[f'{c.S_POWER}_{c.ET_ELECTRICITY}_{mode}'] = result['P_el']
        columns[f'{c.S_POWER}_{c.ET_HEAT}_{mode}'] = result['P_th']
        columns[f'{c.S_COP}_{mode}'] = result['COP'] * c.COP_TO_COP100
    return pd.DataFrame(columns).round().astype(int)


def series_for(creator, filename):
    """The heat-pump timeseries the Creator builds from one shipped spec file."""
    spec = load_file(str(SPEC_DIR / filename))
    # Name-mangled: the method is private to `Agents` and has no public caller that takes a spec
    # directly. `plant` is accepted and unused (`agents.py:844`).
    built = creator._Agents__timeseries_from_specs_hp(specs=spec, plant={})

    # An empty frame compares equal to an empty frame, so every assertion below would hold
    # vacuously if the weather filter in `ambient_temperature` ever matched nothing. Checked once,
    # here, rather than in each test.
    assert len(built) > 0, (
        f'{filename} produced an empty timeseries, so every comparison against it is between two '
        f'empty frames and passes without asserting anything')
    return spec, built


def assert_matches(built, expected, other, filename, inlet):
    """`built` is `expected` and is not `other` -- the second half is what makes the first mean
    something.

    Comparing a built series against a reference the test computed is only an assertion if the two
    candidate references differ. Where a model's hplib fit happens to be insensitive to
    `t_in_primary` over this weather file, the equality holds whichever branch the code took, and
    the test reports success for a defect. So both references are computed and their difference is
    asserted before the equality is.
    """
    differing = [column for column in expected.columns
                 if not expected[column].equals(other[column])]
    assert differing, (
        f'for {filename} the ambient and brine inlets produce identical values in every column, '
        f'so this test cannot tell the two apart and asserts nothing about which was selected')

    for column in expected.columns:
        np.testing.assert_array_equal(
            built[column].to_numpy(), expected[column].to_numpy(),
            err_msg=f'{filename}: {column} does not match the {inlet} inlet')


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
    assert_matches(built, ambient, brine, filename, 'ambient')


@pytest.mark.parametrize('filename', [name for name, (_, inlet) in SHIPPED_SPECS.items()
                                      if inlet == 'brine'])
def test_the_ground_and_water_units_are_simulated_at_brine_temperature(creator,
                                                                       ambient_temperature,
                                                                       filename):
    """Everything that is not an air unit keeps the soil inlet -- the fix must not reach further.

    This is the half that says the change is contained. `Water/Water` taking `calc_brine_temp` is
    itself questionable (groundwater is not soil) -- that is a separate modelling question, **#236**,
    and this pins today's behaviour so a change to it has to be made deliberately.
    """
    spec, built = series_for(creator, filename)
    assert spec['hp_type'] == SHIPPED_SPECS[filename][0], (
        f'{filename} no longer declares {SHIPPED_SPECS[filename][0]!r}')

    ambient = simulate(spec, ambient_temperature, ambient_temperature)
    brine = simulate(spec, Agents.calc_brine_temp(ambient_temperature), ambient_temperature)
    assert_matches(built, brine, ambient, filename, 'brine')


def test_the_air_unit_collapses_in_the_cold(creator, ambient_temperature):
    """At the coldest hour the air unit's COP must be near 1, and the soil inlet cannot give that.

    **A paired assertion on one spec, not a comparison of two models.** Comparing the Bosch air
    unit against the DAIKIN ground unit would be green under any inlet, because those two models
    differ from each other anyway; it would assert that ground beats air, which is not the claim.
    So this takes the *same* spec and computes what it would have produced on the soil inlet, and
    asserts the built series is on the far side of a threshold the soil inlet cannot reach.

    That makes the threshold discriminating by construction rather than by a comment. Measured on
    this weather file: the coldest hour is -12.32 degC, `calc_brine_temp` maps it to 3.18 degC, and
    the Bosch unit gives COP 1.41 on air against 4.59 on soil. `calc_brine_temp` floors at
    3.16 degC over the whole file, so no air temperature can push the soil path low; 3.00 sits
    between the two with room for an hplib refit to move either.

    This arm is the reason the file does not rest on the exact-match tests alone: those recompute
    an hplib call, and a test and an implementation can mirror the same wrong idea. The only thing
    this one takes from `agents.py` is `calc_brine_temp`, the function whose output the fix stops
    using for air units.
    """
    coldest = int(np.argmin(ambient_temperature.to_numpy()))
    assert ambient_temperature.iloc[coldest] < -10, (
        f'the weather file no longer gets colder than {ambient_temperature.iloc[coldest]:.1f} degC,'
        f' so it cannot separate an air inlet from a soil inlet and this test asserts nothing')

    filename = next(name for name, (_, inlet) in SHIPPED_SPECS.items() if inlet == 'ambient')
    spec, built = series_for(creator, filename)
    assert spec['hp_type'] == SHIPPED_SPECS[filename][0], (
        f'{filename} no longer declares {SHIPPED_SPECS[filename][0]!r}, so this test is pointed '
        f'at the wrong unit')

    column = f'{c.S_COP}_{c.P_HEAT}'
    on_soil = simulate(spec, Agents.calc_brine_temp(ambient_temperature),
                       ambient_temperature)[column].to_numpy()

    assert on_soil[coldest] > COLD_COP100, (
        f'the same unit on the soil inlet shows COP {on_soil[coldest] / c.COP_TO_COP100:.2f} at '
        f'the coldest hour, below the {COLD_COP100 / c.COP_TO_COP100:.2f} this test uses to '
        f'separate the two inlets -- so passing the assertion below would say nothing')
    assert built[column].to_numpy()[coldest] < COLD_COP100, (
        f'the air unit shows COP {built[column].to_numpy()[coldest] / c.COP_TO_COP100:.2f} at '
        f'{ambient_temperature.iloc[coldest]:.1f} degC ambient, which no air-source unit reaches; '
        f'it is being simulated on soil temperature (#234)')


def test_the_shipped_specs_are_the_ones_this_file_tests():
    """`SHIPPED_SPECS` is a hand-written literal; this is what stops it drifting from the tree.

    Two ways the tests above could quietly stop covering the defect. A **fourth** spec file added
    to `input_data/` would simply not appear in the literal, so a new air unit could ship with no
    assertion on it at all. And `SPEC_DIR` names only the `sfh` directory, while the same three
    files also ship under `ctsp` and `industry` -- nine files, of which the tests above read
    three. An `hp_type` edited in one of the other six would leave this file green while those
    agents ran on the wrong inlet.

    Unmarked, so it runs in the default tier: it reads tracked files and needs no scenario.
    """
    for directory in SPEC_DIRS:
        found = {path.name for path in directory.glob('hp_*.json')}
        assert found == set(SHIPPED_SPECS), (
            f'{directory.relative_to(REPO_ROOT)} ships {sorted(found)}, but SHIPPED_SPECS names '
            f'{sorted(SHIPPED_SPECS)}. A spec this file does not name is a heat pump nothing here '
            f'asserts an inlet temperature for')

        for filename, (hp_type, _) in SHIPPED_SPECS.items():
            declared = load_file(str(directory / filename))['hp_type']
            assert declared == hp_type, (
                f'{directory.relative_to(REPO_ROOT) / filename} declares hp_type {declared!r}, '
                f'but the copy this file tests declares {hp_type!r}; the two would take different '
                f'branches and only one of them is covered')


def test_an_unknown_heat_pump_type_is_refused(creator):
    """The discriminator fails **closed**, which is what stops #234 recurring one level down.

    The branch in `agents.py` tests one literal and sends everything else to the soil model, so
    before `HP_TYPES` an unrecognised `hp_type` reproduced #234 exactly -- an air unit on soil
    inlet, per spec file, silently. `config_templates/agents.yaml` invites users to name their own
    spec file, so this is a reachable state and not a hypothetical.

    Asserted on the value #234 actually read. `'hp'` is what `specs['type']` holds in every
    shipped file, so this is the exact input that produced the defect, and it now raises.
    """
    spec = dict(load_file(str(SPEC_DIR / 'hp_Bosch Thermotechnik_air.json')))
    spec['hp_type'] = 'hp'

    with pytest.raises(ValueError, match='hp_type'):
        creator._Agents__timeseries_from_specs_hp(specs=spec, plant={})


def test_a_missing_key_is_reported_as_a_missing_key(creator):
    """A spec file missing a required key must say so, not blame the plant type.

    `__make_timeseries` used to call the specs function inside `except KeyError`, so every
    `KeyError` raised *within* it -- a spec missing `model`, or `hp_type` -- surfaced as
    "Time series creation from spec file not available for plant type hp", which is both false and
    a dead end for whoever has to fix the file. Only the registry lookup can raise that.
    """
    spec = dict(load_file(str(SPEC_DIR / 'hp_Bosch Thermotechnik_air.json')))
    del spec['hp_type']

    with pytest.raises(KeyError) as raised:
        creator._Agents__timeseries_from_specs_hp(specs=spec, plant={})
    assert 'hp_type' in str(raised.value), (
        f'a spec missing hp_type raised {raised.value!r}, which does not name the missing key')
