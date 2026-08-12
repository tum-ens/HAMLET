"""Sizing an inflexible load from a grid file's MW demand, for every agent class that does it.

#212: `Ctsp._inflexible_load_grid` sized the load as `(df['demand'] * 1e6).astype('Int64')`, which
raises `TypeError` whenever `demand * 1e6` is not exactly representable in float64 -- 349 of the
10 000 three-decimal values between 0 and 10, `1.001` among them. `industry` and `sfh` had both
been changed to handle it and `ctsp` was the copy the change missed.

**The three classes are tested together, from one table, on purpose.** The defect existed because
three near-identical implementations drifted and nothing compared them. A test that only pinned
`ctsp` would leave the next divergence just as invisible, so `PARAMETRISED` names all three and
every case runs against each. Where they legitimately differ -- `industry` floors where `ctsp` and
`sfh` round -- that is stated as data in `ROUNDING`, so the divergence is recorded rather than
averaged away. See #213, which is the issue for collapsing them.

The classes are driven through a stub `load` frame rather than a generated scenario. That is not
only for speed: `_inflexible_load_grid` is reached from `new_scenario_from_grids` alone, and no
config in the repository is built that way -- `tests/e2e/scenarios/ctsp_industry/` declares both
types but is built from configs and from files, so it never reaches this method. A scenario-level
test would have to author a grid file to reach one line. What the method needs is `self.load`, `self.df` and a `fcast`
config, and those are what it is given.
"""
import numpy as np
import pandas as pd
import pytest

import hamlet.constants as c
from hamlet.creator.agents.ctsp import Ctsp
from hamlet.creator.agents.industry import Industry
from hamlet.creator.agents.sfh import Sfh

KEY = c.P_INFLEXIBLE_LOAD

#: The three classes that size an inflexible load from a grid file's `demand` column, and how each
#: one turns a fractional watt into an integer. `industry` differs and that difference is recorded
#: here rather than asserted away -- it is one of the divergences #213 has to decide.
ROUNDING = {'ctsp': (Ctsp, round), 'industry': (Industry, np.floor), 'sfh': (Sfh, round)}
PARAMETRISED = pytest.mark.parametrize('name', sorted(ROUNDING), ids=sorted(ROUNDING))

#: A value from the failing set. `1.001` MW is a 1 MW load stated to three decimals, which is the
#: precision the shipped grid file writes for an inflexible load (`demand:2.455`). `1.001 * 1e6`
#: is 1000999.9999999999 in float64, and that is what the old expression could not cast.
INEXACT = 1.001


def size_demand(agent_class, demands):
    """Run `_inflexible_load_grid` over `demands` and return the sizing column it wrote.

    Only the collaborators the method actually reads are supplied. `file` is non-null so that
    `_pick_files` -- which would need an input directory on disk -- is not reached; picking files
    is a different concern with its own test
    (`test_file_selection_is_order_independent.py`).
    """
    agent = agent_class.__new__(agent_class)  # __init__ wants a config tree; the method wants none
    agent.load = pd.DataFrame({'load_type': [KEY] * len(demands),
                               'demand': list(demands),
                               'file': [f'load_{i}.csv' for i in range(len(demands))]})
    agent.df = pd.DataFrame(index=range(len(demands)))

    agent._inflexible_load_grid(key=KEY, config={'fcast': {}})

    return agent.df[f'{KEY}/sizing/demand_0']


@PARAMETRISED
def test_an_inexact_megawatt_value_does_not_raise(name):
    """The regression. Against the unfixed `ctsp` this raises `TypeError`.

    Stated as "does not raise" separately from the value it produces, because the defect was an
    abort rather than a wrong number: scenario creation stopped, and a test that only checked the
    value would report the same failure for two very different causes.
    """
    agent_class, _ = ROUNDING[name]

    sized = size_demand(agent_class, [INEXACT])

    assert len(sized) == 1


@PARAMETRISED
def test_an_inexact_megawatt_value_is_sized_to_the_watt(name):
    """And the value is the one the class's own rounding rule gives, not merely *a* number.

    `1.001 MW` is 1 001 000 W under `round` and 1 000 999 W under `floor`. Asserting the exact
    integer is what makes this test able to fail if someone "fixes" the crash by coercing to NaN,
    which would pass the test above.
    """
    agent_class, rounding = ROUNDING[name]

    sized = size_demand(agent_class, [INEXACT])

    assert sized.tolist() == [int(rounding(INEXACT * 1e6))]


@PARAMETRISED
def test_a_value_that_was_always_exact_is_unchanged(name):
    """The fix must not move sizings that never had the problem.

    Whole-megawatt and half-megawatt values are exact in float64, so `astype`, `round` and `floor`
    all agreed on them before. If this moves, the fix changed generated scenarios for inputs that
    were never broken, which is a different and worse change than the one #212 asked for.
    """
    agent_class, _ = ROUNDING[name]

    sized = size_demand(agent_class, [1.0, 0.5, 2.25])

    assert sized.tolist() == [1_000_000, 500_000, 2_250_000]


@PARAMETRISED
def test_every_value_in_a_mixed_column_is_sized(name):
    """One inexact value must not take the whole column with it.

    The old expression cast the column, so a single bad row aborted every agent's sizing. Sizing
    each row independently is the property that makes the failure local even if it comes back, and
    a single-row test cannot see it.
    """
    agent_class, rounding = ROUNDING[name]
    demands = [1.0, INEXACT, 2.455, 0.0385]

    sized = size_demand(agent_class, demands)

    assert sized.tolist() == [int(rounding(value * 1e6)) for value in demands]


@PARAMETRISED
def test_the_written_column_holds_integers(name):
    """`Int64`, because the scenario format says these are whole watts.

    A fix that produced floats would satisfy every assertion above -- `1001000.0 == 1001000` --
    while changing the dtype the Executor reads and the format fingerprint pins.
    """
    agent_class, _ = ROUNDING[name]

    sized = size_demand(agent_class, [INEXACT])

    assert sized.dtype == 'Int64'


def test_the_three_classes_agree_on_a_value_that_is_exact():
    """Where nothing is being rounded, all three must produce the same number.

    This is the guard for a dedupe (#213): whatever `ctsp.py` and `industry.py` are collapsed into
    has to keep this true. It is deliberately about exact values only -- the classes genuinely
    disagree on inexact ones, and `ROUNDING` above is where that is recorded.
    """
    sized = {name: size_demand(agent_class, [1.0, 0.5]).tolist()
             for name, (agent_class, _) in ROUNDING.items()}

    assert len(set(map(tuple, sized.values()))) == 1, sized


# ---------------------------------------------------------------------------------------------
# The same defect, at every other site that has it.
#
# #212 was filed against one line. It is one line *per device group*: `_pv_grid`, `_wind_grid`,
# `_fixed_gen_grid` and `_battery_grid` each size a plant as `(index.map(df['power']) * 1e6)
# .astype('Int64')` in `ctsp` and `industry`, and `sfh` rounds all four. Fixing only the demand
# column would have left four identical crashes per class behind -- which is this repository's
# recurring shape: the fix for a silent failure contains the same failure one level down.
# ---------------------------------------------------------------------------------------------

#: The grid-path sizing methods that turn a MW column into integer watts, and the column each
#: writes. `_inflexible_load_grid` reads `demand`; the rest read `power`.
POWER_METHODS = ('_pv_grid', '_wind_grid', '_fixed_gen_grid', '_battery_grid')


#: Every column these four methods read off the sgen/battery frame, gathered from the source rather
#: than discovered one `KeyError` at a time. A stub that is missing one fails for a reason that has
#: nothing to do with what is under test, which is how a test ends up asserting its own scaffolding.
GRID_PLANT_COLUMNS = {
    'file': 'plant.csv', 'file_add': 'plant_add.csv', 'orientation': 0, 'angle': 30, 'height': 100,
    'capacity': 1.0, 'efficiency': 0.9, 'soc': 0.5, 'charging_home': 0.0, 'charging_ac': 0.011,
    'charging_dc': 0.05, 'v2g': False, 'v2h': False, 'g2b': True, 'b2g': False,
}


def size_power(agent_class, method_name, powers, key='pv'):
    """Run one of the `power`-sizing grid methods and return the sizing column it wrote.

    These methods read `self.sgen` (or `self.battery`), filter it by `plant_type`, index it by
    `owner`, and then read a further dozen columns off the same frame -- so the stub supplies the
    whole set (`GRID_PLANT_COLUMNS`) rather than the two the sizing line itself needs. Only the
    frame is stubbed; the method under test runs unmodified.
    """
    agent = agent_class.__new__(agent_class)
    agent.n_digits = 3
    index = list(range(len(powers)))
    data = {'owner': index, 'plant_type': [key] * len(powers), 'sgen_type': [key] * len(powers),
            'load_type': [key] * len(powers), 'power': list(powers)}
    data.update({name: [value] * len(powers) for name, value in GRID_PLANT_COLUMNS.items()})
    frame = pd.DataFrame(data)
    agent.sgen = frame
    agent.battery = frame
    agent.load = frame
    agent.df = pd.DataFrame(index=index)

    config = {'fcast': {}, 'quality': 1,
              'sizing': {'controllable': [False], 'efficiency': [0.9], 'g2b': [True],
                         'b2g': [False], 'soc': [0.5], 'capacity': [1.0]}}
    getattr(agent, method_name)(key=key, config=config)

    return agent.df[f'{key}/sizing/power_0']


@pytest.mark.parametrize('name', sorted(ROUNDING), ids=sorted(ROUNDING))
def test_every_grid_sizing_method_of_this_class_is_covered_here(name):
    """`POWER_METHODS` is the set of methods with this shape, not a subset someone remembered.

    An allowlist that is not compared against the class silently exempts the next method to grow
    the same line -- and the whole point of #212 is that a line was copied into places nobody was
    checking. So the class is asked which of these it defines.
    """
    agent_class, _ = ROUNDING[name]

    defined = [method for method in POWER_METHODS if hasattr(agent_class, method)]

    assert defined == list(POWER_METHODS), (
        f'{name} defines {defined} of the grid sizing methods, not all of {list(POWER_METHODS)}; '
        f'if one was renamed or removed, update POWER_METHODS deliberately')


def test_no_creator_class_sizes_a_megawatt_column_with_a_bare_cast():
    """The completeness guard: every class that does this must be in `ROUNDING`.

    `ROUNDING` names three classes. Nothing tied it to the tree, so dropping an entry -- or adding
    a fourth agent class with the same line -- left every assertion in this file silently
    inapplicable to it. Demonstrated: removing `sfh` from `ROUNDING` and reverting `sfh.py`'s fix
    left the whole suite green.

    `AgentBase` itself raises `NotImplementedError` for these, so the set compared against is the
    subclasses that actually override `_inflexible_load_grid`.
    """
    from hamlet.creator.agents.agent_base import AgentBase

    overriders = {cls.__name__.lower() for cls in AgentBase.__subclasses__()
                  if cls.__dict__.get('_inflexible_load_grid') is not None}

    assert overriders == set(ROUNDING), (
        f'these classes override _inflexible_load_grid: {sorted(overriders)}, but ROUNDING names '
        f'{sorted(ROUNDING)}. A class not listed here is not covered by any assertion in this '
        f'file, which is how #212 survived in ctsp')


@pytest.mark.parametrize('name', sorted(ROUNDING), ids=sorted(ROUNDING))
@pytest.mark.parametrize('method', POWER_METHODS)
def test_an_inexact_megawatt_power_does_not_raise(name, method):
    """The four sites #212 did not name. Against the unfixed code these raise `TypeError`."""
    agent_class, _ = ROUNDING[name]

    sized = size_power(agent_class, method, [INEXACT])

    assert len(sized) == 1


@pytest.mark.parametrize('name', sorted(ROUNDING), ids=sorted(ROUNDING))
@pytest.mark.parametrize('method', POWER_METHODS)
def test_an_inexact_megawatt_power_is_sized_to_the_watt(name, method):
    """And to the watt. All three classes round `power`, unlike `demand`, where `industry` floors."""
    agent_class, _ = ROUNDING[name]

    sized = size_power(agent_class, method, [INEXACT])

    assert sized.tolist() == [round(INEXACT * 1e6)]
