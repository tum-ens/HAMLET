"""`direct_power_control.reduction` decides where a §14a EMS command lands: required | minimum | ramp.

Under `required` the executor's `while not grid_ok` loop re-simulates the timestep after every pass
and the command approaches the overload asymptotically, so where it *stops* is decided by
`max_iteration` -- a simulation setting -- rather than by the policy being modelled (#245). The
other two strategies stop where the policy says: `minimum` on the §14a floor in one pass, `ramp`
one rung further down a ladder per pass until the overload clears or the floor is reached.

Every value here is absolute and written out, on the same fixture `test_enwg_14a_cap_floor.py`
uses: 1000 W inflexible plus two 11000 W EVs, so 23000 W uncurtailed and a 9400 W floor. The three
strategies are told apart on **one input** -- loading 1.5, where `required` takes exactly a third --
because a strategy that silently fell back to `required` produces 15334 W there and neither of the
others can.

**What each test is sensitive to, and what it deliberately is not.** The default is pinned by the
existing cap-floor tests, which name no strategy and expect `required`'s numbers; nothing here
repeats that. `ramp`'s step is measured from the *anchored* uncurtailed draw and not from the draw
of the pass being simulated -- `test_a_ramp_step_is_measured_from_the_anchor` is the one case that
separates the two, and it needs a dispatch that lands *below* the cap, which is what an EMS free to
distribute a setpoint does. The over-generation side is not covered, for the reason
`test_enwg_14a_cap_floor.py` gives: nothing reaches it and it carries its own sign defect (#244).
"""

from types import SimpleNamespace

import polars as pl
import pytest

from hamlet import constants as c
from hamlet.executor.utilities.database.grid_db import ElectricityGridDB
from hamlet.executor.utilities.grid_restrictions.enwg_14a import EnWG14a
from tests.unit.executor.test_enwg_14a_cap_floor import (
    AGENT, AGENT_TWO, BUS, GUARANTEE_W, SUB_BUDGET_COMMAND_W, THRESHOLD_W, UNCURTAILED_W, grid_with,
    loads_at, restriction, two_agent_loads)
from tests.unit.executor.test_enwg_14a_cap_floor import control as command_bus

#: A mild overload: `required` takes a third of the draw here, so it lands well above the floor
#: and the other two strategies are visibly *not* it.
MILD_LOADING = 1.5

#: The ladder `ramp_steps: 3` puts between 23000 W and 9400 W: a 13600 W span in three rungs of
#: 4533.33 W, truncated toward the uncurtailed end by `int()`. Written out so a change in how the
#: rung is rounded fails here by name.
RAMP_STEPS = 3
RUNG_1_W = 18467
RUNG_2_W = 13934


def configured(enwg, **keys):
    """The cap-floor fixture with extra `direct_power_control` keys, as `grids.yaml` would give them."""
    enwg.restriction_config['direct_power_control'].update(keys)
    return enwg


def control(enwg, combined_loading):
    """One pass: the overload signal `__calculate_direct_power_control` gives, then the command.

    The cap-floor fixture's `control` calls `__control_via_ems` alone, which is one *command*. A
    pass is one overload signal followed by a command per overloaded bus, and the ramp counts
    rungs per signal -- see `test_a_ramp_takes_one_rung_per_pass_however_many_buses_command_it`.
    """
    if enwg.restriction_config['direct_power_control'].get('reduction') == 'ramp':
        enwg._EnWG14a__begin_ramp_pass()
    return command_bus(enwg, combined_loading)


# ----------------------------------------------------------------------------------------------
# required
# ----------------------------------------------------------------------------------------------


def test_naming_required_is_the_same_as_naming_nothing():
    """The key is additive: a scenario that spells the default out gets what one without it got."""
    implicit = restriction(grid_with(loads_at([11000, 11000])))
    explicit = configured(restriction(grid_with(loads_at([11000, 11000]))), reduction='required')

    assert control(implicit, MILD_LOADING)[AGENT]['ems'] == SUB_BUDGET_COMMAND_W
    assert control(explicit, MILD_LOADING)[AGENT]['ems'] == SUB_BUDGET_COMMAND_W


# ----------------------------------------------------------------------------------------------
# minimum
# ----------------------------------------------------------------------------------------------


def test_minimum_lands_on_the_floor_whatever_the_overload():
    """One pass, straight to the §14a floor -- 5934 W deeper than the overload needed."""
    enwg = configured(restriction(grid_with(loads_at([11000, 11000]))), reduction='minimum')

    result = control(enwg, MILD_LOADING)[AGENT]['ems']

    assert result == GUARANTEE_W, (
        f'minimum landed at {result} W; the floor is {GUARANTEE_W} W and required would have '
        f'landed at {SUB_BUDGET_COMMAND_W} W')


def test_minimum_holds_the_floor_on_the_second_pass():
    """The agent obeyed by zeroing one EV and leaving the other above `threshold`.

    The remaining EV is still in the filter, so a strategy that re-took its headroom from the
    current draw would command 5200 W; the anchored floor keeps it at 9400 W and, because the
    command stops moving, the timestep converges.
    """
    enwg = configured(restriction(grid_with(loads_at([11000, 11000]))), reduction='minimum')
    control(enwg, MILD_LOADING)

    enwg.grid = grid_with(loads_at([8400, 0]))
    second = control(enwg, MILD_LOADING)[AGENT]['ems']

    assert second == GUARANTEE_W


def test_minimum_takes_each_agent_to_its_own_floor():
    """Two agents on one bus: `required` would split the bus reduction pro rata, `minimum` does not."""
    enwg = configured(restriction(grid_with(two_agent_loads(11000, 11000))), reduction='minimum')

    result = control(enwg, MILD_LOADING)

    # 1000 W + one EV at 4200 W; 9000 W + one EV at 4200 W. The floor is rounded once and
    # upward since #250, so it is 13200 exactly, not the 13199 the truncated anchor produced.
    assert result[AGENT]['ems'] == 5200
    assert result[AGENT_TWO]['ems'] == 13200


def test_minimum_rounds_a_fractional_floor_up_never_down():
    """Pins the floor's rounding direction (#250), which integer-watt cases cannot.

    For those `round(firm_w, 3)` alone repairs the float noise and `ceil`/`floor` agree; under
    `required` the command's own arithmetic also lands above the floor, so the clamp never
    engages. Under `minimum` the floor *is* the command. A heat pump's `hp_min_control` comes from
    a formula and can be fractional, making the floor fractional: 1000 + 5000 - (5000 - 1500.5) =
    2500.5 W. It is a bound the agent must be able to meet, so it rounds up; `floor` gives 2500
    and authorises 0.5 W the agent does not have.
    """
    loads = [
        {'bus': BUS, 'load_type': c.P_INFLEXIBLE_LOAD, 'p_mw': 1000 * c.WH_TO_MWH,
         c.TC_ID_AGENT: AGENT, 'hp_min_control': 0.0},
        {'bus': BUS, 'load_type': c.P_HP, 'p_mw': 5000 * c.WH_TO_MWH,
         c.TC_ID_AGENT: AGENT, 'hp_min_control': 1500.5 * c.WH_TO_MWH},
    ]
    enwg = configured(restriction(grid_with(loads)), reduction='minimum', threshold=0)

    cap = control(enwg, 3.0)[AGENT]['ems']

    assert cap == 2501, f'a 2500.5 W floor became {cap} W; a floor rounds up'


# ----------------------------------------------------------------------------------------------
# ramp
# ----------------------------------------------------------------------------------------------


def ramp(loads, steps=RAMP_STEPS):
    return configured(restriction(grid_with(loads)), reduction='ramp', ramp_steps=steps)


def test_a_ramp_descends_one_rung_per_pass_and_stops_on_the_floor():
    """Each pass that still sees an overload takes the next rung; the floor is the last one.

    The fourth pass is what makes the ladder *stop*: its arithmetic goes below the floor and the
    clamp holds it there, so the command equals the previous one and the timestep converges on its
    own rather than on `max_iteration`.
    """
    enwg = ramp(loads_at([11000, 11000]))

    rungs = []
    for _ in range(RAMP_STEPS + 1):
        rungs.append(control(enwg, 3.0)[AGENT]['ems'])
        # The agent obeys exactly, on one EV.
        enwg.grid = grid_with(loads_at([rungs[-1] - 1000, 0]))

    assert rungs == [RUNG_1_W, RUNG_2_W, GUARANTEE_W, GUARANTEE_W]


def test_a_ramp_step_does_not_know_the_size_of_the_overload():
    """The rung is the same at loading 1.5 and 3.0. That is the whole point of the strategy --
    it needs an overload signal, not a state estimate -- and it is what separates it from
    `required`, whose first pass differs by 5934 W between the two loadings."""
    mild = control(ramp(loads_at([11000, 11000])), MILD_LOADING)[AGENT]['ems']
    severe = control(ramp(loads_at([11000, 11000])), 3.0)[AGENT]['ems']

    assert mild == severe == RUNG_1_W
    assert RUNG_1_W > SUB_BUDGET_COMMAND_W, 'the first rung must sit above where required lands'


def test_a_ramp_step_is_measured_from_the_anchor():
    """The rung is a point on a ladder fixed at the first pass, not a decrement from the last draw.

    The agent was capped at 18467 W and dispatched 15000 W -- below the cap, as an EMS may. A step
    of the same 4533 W taken from that draw would command 10467 W; the ladder says 13934 W.
    """
    enwg = ramp(loads_at([11000, 11000]))
    control(enwg, 3.0)

    enwg.grid = grid_with(loads_at([8000, 6000]))
    second = control(enwg, 3.0)[AGENT]['ems']

    assert second == RUNG_2_W, (
        f'the second rung landed at {second} W; 10467 W would mean the step was taken from the '
        f'15000 W the agent happened to draw rather than from the anchored 23000 W')


def test_a_ramp_takes_one_rung_per_pass_however_many_buses_command_it():
    """One rung per overload signal, not per command.

    `__calculate_direct_power_control` visits every bus of a zone once per transformer in it, so
    a zone with two transformers commands each agent twice per pass. A step counted per command
    would take two rungs per pass there and halve the ladder. Here the second command of the
    same pass must land on the same rung as the first.
    """
    enwg = ramp(loads_at([11000, 11000]))
    enwg._EnWG14a__begin_ramp_pass()
    first = command_bus(enwg, 3.0)[AGENT]['ems']
    second = command_bus(enwg, 3.0)[AGENT]['ems']

    assert first == second == RUNG_1_W, (
        f'two commands in one pass landed on {first} W and {second} W; a rung per command would '
        f'put the second on {RUNG_2_W} W')

    enwg._EnWG14a__begin_ramp_pass()
    assert command_bus(enwg, 3.0)[AGENT]['ems'] == RUNG_2_W


def test_each_agent_descends_its_own_ladder():
    """Two agents on one bus: each rung is that agent's anchor less its own span over its own step.

    Every other ramp case here has one agent, so a step counter shared across the bus, a ladder
    anchored on the bus draw, or a span taken from a neighbour's guarantee would all be invisible
    in them. A review lens built those three mutants and every one survived until this test.
    """
    enwg = ramp(two_agent_loads(11000, 11000))

    first = control(enwg, 3.0)
    # agent_one: 12000 W down a 6800 W span (floor 5200); agent_two: 19999 W down a 6799 W span
    # -- its 0.009 + 0.011 MW sum truncates to 19999, but since #250 its floor is 13200 exactly,
    # so the ladder is one watt shorter than the command it starts from. One rung each.
    assert first[AGENT]['ems'] == 9734
    assert first[AGENT_TWO]['ems'] == 17733

    enwg.grid = grid_with(two_agent_loads(8734, 8733))  # both obey exactly, on their EV
    second = control(enwg, 3.0)
    assert second[AGENT]['ems'] == 7467
    assert second[AGENT_TWO]['ems'] == 15467, (
        f"agent_two's second rung is {second[AGENT_TWO]['ems']} W; a span taken from agent_one's "
        f"guarantee would give 13200 W, a bus-anchored ladder something else again")


def test_the_overload_signal_is_what_advances_the_pass():
    """`__calculate_direct_power_control` advances the counter; no command does.

    The `control` helper above advances it itself, so nothing else in this file can tell whether
    the production method is wired to -- deleting that one call left every unit test green until
    this one. Driven with no transformer to walk, as the #248 tests do, so only the signal runs.
    """
    import pandas as pd

    enwg = ramp(loads_at([11000, 11000]))
    enwg.grid_db.results = {}
    enwg.grid = SimpleNamespace(res_trafo=pd.DataFrame({'p_lv_mw': [], 'loading_percent': []}))

    enwg._EnWG14a__calculate_direct_power_control()
    enwg._EnWG14a__calculate_direct_power_control()

    assert enwg.grid_db.restriction_commands[ElectricityGridDB.RAMP_STATE_KEY]['passes'] == 2


def test_a_one_step_ramp_is_the_floor():
    assert control(ramp(loads_at([11000, 11000]), steps=1), MILD_LOADING)[AGENT]['ems'] == GUARANTEE_W


def test_the_step_count_restarts_with_the_timestep():
    """Two rungs down at 01:00; the first command at 02:00 must be the first rung again."""
    enwg = ramp(loads_at([11000, 11000]))
    control(enwg, 3.0)
    enwg.grid = grid_with(loads_at([RUNG_1_W - 1000, 0]))
    assert control(enwg, 3.0)[AGENT]['ems'] == RUNG_2_W

    enwg.timestamp = '2023-01-08 02:00:00+00:00'
    enwg.grid = grid_with(loads_at([11000, 11000]))

    assert control(enwg, 3.0)[AGENT]['ems'] == RUNG_1_W


def test_the_ramp_state_is_per_timestep_state():
    """Where the counter lives is a contract, not a detail.

    The executor clears `GridDB.TIMESTEP_RESTRICTION_KEYS` at the end of every timestep, and
    #249's resume is expected to checkpoint exactly those keys. A counter kept under any other key
    would survive a forced exit into the next timestep and would not survive a pause.
    """
    enwg = ramp(loads_at([11000, 11000]))
    control(enwg, 3.0)

    state = enwg.grid_db.restriction_commands[ElectricityGridDB.RAMP_STATE_KEY]
    assert state['passes'] == 1
    assert state['agents'] == {AGENT: {'from': UNCURTAILED_W, 'step': 1, 'pass': 1}}
    assert ElectricityGridDB.RAMP_STATE_KEY in ElectricityGridDB.TIMESTEP_RESTRICTION_KEYS


# ----------------------------------------------------------------------------------------------
# The configuration is refused rather than guessed at.
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize('keys, names', [
    ({'reduction': 'gentle'}, ('gentle', 'required', 'minimum', 'ramp')),
    ({'reduction': None}, ('None', 'required', 'minimum', 'ramp')),
    ({'reduction': 'ramp'}, ('ramp_steps', 'None', 'max_iteration')),
    ({'reduction': 'ramp', 'ramp_steps': 0}, ('ramp_steps = 0',)),
    ({'reduction': 'ramp', 'ramp_steps': -2}, ('ramp_steps = -2',)),
    ({'reduction': 'ramp', 'ramp_steps': 2.5}, ('ramp_steps = 2.5',)),
    ({'reduction': 'ramp', 'ramp_steps': '3'}, ("ramp_steps = '3'",)),
    ({'reduction': 'ramp', 'ramp_steps': True}, ('ramp_steps = True',)),
    ({'reduction': 'minimum', 'method': 'individual'}, ('minimum', 'individual', 'method: ems')),
    ({'reduction': 'ramp', 'ramp_steps': 3, 'method': 'individual'}, ('ramp', 'individual')),
    ({'reduction': 'minimum', 'method': 'emss'}, ('minimum', 'emss', 'method: ems')),
])
def test_a_bad_strategy_is_refused_by_name(keys, names):
    """The message has to name the key and the value, or the next person reads the source.

    `reduction: null` is refused rather than read as the default: a key that is present and empty
    is a decision nobody made. The `emss` case is what keeps the method check a whitelist -- a
    strategy is defined for `ems` and nothing else, not merely undefined for `individual`.
    """
    enwg = configured(restriction(grid_with(loads_at([11000, 11000]))), **keys)

    with pytest.raises(ValueError) as refusal:
        control(enwg, 3.0)

    for name in names:
        assert name in str(refusal.value), f'{name!r} missing from: {refusal.value}'


def test_required_is_allowed_with_the_individual_method():
    """Only a *non-default* strategy is undefined for `individual`; the default must keep loading.

    `config_templates/grids.yaml` and every shipped example select `individual` and spell out
    `reduction: required`; a scenario written before the key existed names nothing. Both take this
    path, and refusing it would refuse every existing scenario.
    `tests/integration/executor/test_reduction_strategy_config.py` checks the tracked files.
    """
    enwg = configured(restriction(grid_with(loads_at([11000, 11000]))), method='individual')

    assert enwg._EnWG14a__reduction_strategy() == ('required', None)


def constructed(direct_power_control):
    """`EnWG14a` built through its real `__init__`, with the least a restriction needs around it."""
    tasks = pl.DataFrame({c.TC_TIMESTAMP: ['2023-01-08 01:00:00+00:00']})
    general = {c.K_GRID: {c.G_ELECTRICITY: {'powerflow': 'dc'}}}
    database = SimpleNamespace(get_general_data=lambda: general)
    return EnWG14a(grid_db=SimpleNamespace(restriction_commands={}), grid=None, tasks=tasks,
                   database=database, variable_grid_fees={'active': False},
                   direct_power_control=direct_power_control)


def test_a_bad_strategy_is_refused_when_the_restriction_is_built():
    """On the first timestep, not on the first overload -- which may be hours of simulation in."""
    with pytest.raises(ValueError, match='ramp_steps'):
        constructed({'active': True, 'method': 'ems', 'threshold': THRESHOLD_W,
                     'reduction': 'ramp'})


def test_an_inactive_block_is_not_validated():
    """Switching direct power control off must not start refusing whatever was left beside it."""
    constructed({'active': False, 'method': 'individual', 'threshold': THRESHOLD_W,
                 'reduction': 'ramp'})
