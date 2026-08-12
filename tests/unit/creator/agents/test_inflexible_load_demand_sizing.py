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
only for speed: `_inflexible_load_grid` is reached from `new_scenario_from_grids`, and no shipped
config declares a `ctsp` or `industry` agent at all, so a scenario-level test would have to author
a grid file to reach one line. What the method needs is `self.load`, `self.df` and a `fcast`
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
#: precision the shipped grid file writes (`demand:2.455`, `demand:0.0385`). `1.001 * 1e6` is
#: 1000999.9999999999 in float64, and that is what the old expression could not cast.
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
