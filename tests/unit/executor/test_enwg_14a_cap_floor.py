"""The §14a EMS command may never take a connection below its guaranteed minimum power.

`__control_via_ems` issues an absolute cap, `total_p_agent + reduced_p_agent`, and the executor's
`while not grid_ok` loop calls it again for the same timestep. `total_p_agent` is re-read from the
grid each pass, so by the second pass it already carries the first pass's curtailment, and a share
is subtracted from it again. `threshold` bounds what each *device* may give up; nothing bounded what
the connection as a whole was left with, so the command compounded downward through the §14a
Mindestleistung. On design 6a it took 53 % more than §14a permits on 68.8 % of the controlled
agent-timesteps and reached 339 W on one connection -- below that agent's own inflexible load, so
its optimisation had no feasible point and the run died (#242).

**A single pass cannot show any of this.** The first pass is correct by construction: the reduction
is capped at `sum(reducible)` and each agent's share divides by that same sum, so one pass can never
breach. The defect needs the loop, and the loop needs the budget to run out -- a bus whose overload
is smaller than what §14a permits takes less than the budget, and is covered separately below.

The second pass also has to be given a dispatch that keeps a device **above** `threshold`. That is
what an EMS does: Ziffer 4.5.2 Satz 6 lets it deploy the granted setpoint as it sees fit, so it may
zero one device and leave another running rather than lowering both to their floors. Had every
device been left at its floor it would drop out of the `p_mw > threshold` filter, `reducible` would
be zero, and the command would converge on its own -- passing on the broken code.

**Only the over-consumption branch is covered.** The clamp takes the other side for
over-generation (`min` rather than `max`), and mutating that direction is caught by nothing here.
That is deliberate rather than an oversight: no shipped scenario reaches the branch, so a test
would pin behaviour nothing runs -- and the branch has a sign problem of its own that predates
this change (#244). `power_reduction_at_bus = min(sum(reducible), p_mw_to_be_reduced)` takes the
second term whenever the bus exports, because it is negative there, so the command asks an
over-generating bus to generate *more*. Pinning the clamp on top of that would fix the wrong
behaviour in place.
"""

from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from hamlet import constants as c
from hamlet.executor.utilities.grid_restrictions.enwg_14a import EnWG14a

#: §14a's flat minimum, in watts, as the scenarios configure it.
THRESHOLD_W = 4200

AGENT = 'agent_one'
BUS = 0

#: Absolute expectations, written out rather than recomputed from `THRESHOLD_W`, so that changing
#: the guarantee changes what the test demands.
UNCURTAILED_W = 23000        # 1000 W inflexible + two EVs at 11000 W
GUARANTEE_W = 9400           # 1000 W inflexible + both EVs held at their 4200 W floors
BREACHING_CAP_W = 5199       # what the second pass produced before this was fixed
SUB_BUDGET_COMMAND_W = 15334 # loading 1.5: a third of the 23000 W draw taken


def grid_with(loads):
    """A stand-in for the pandapower net, carrying only what `__control_via_ems` reads."""
    load = pd.DataFrame(loads)
    sgen = pd.DataFrame({'bus': [], 'plant_type': [], 'p_mw': [], c.TC_ID_AGENT: []})
    return SimpleNamespace(
        load=load,
        sgen=sgen,
        res_bus=pd.DataFrame({'p_mw': [load['p_mw'].sum()]}, index=[BUS]),
    )


def loads_at(ev_powers_w, inflexible_w=1000):
    """One inflexible load plus one EV row per given power."""
    rows = [{'bus': BUS, 'load_type': c.P_INFLEXIBLE_LOAD, 'p_mw': inflexible_w * c.WH_TO_MWH,
             c.TC_ID_AGENT: AGENT, 'hp_min_control': 0.0}]
    rows += [{'bus': BUS, 'load_type': c.P_EV, 'p_mw': power * c.WH_TO_MWH,
              c.TC_ID_AGENT: AGENT, 'hp_min_control': 0.0} for power in ev_powers_w]
    return rows


def restriction(grid, timestamp='2023-01-08 01:00:00+00:00'):
    """An `EnWG14a` carrying only the state this path touches, so the test needs no scenario."""
    enwg = EnWG14a.__new__(EnWG14a)
    enwg.grid = grid
    enwg.timestamp = timestamp
    enwg.grid_db = SimpleNamespace(restriction_commands={'current_direct_power_control': {}})
    enwg.restriction_config = {'direct_power_control': {'threshold': THRESHOLD_W, 'method': 'ems'}}
    return enwg


def control(enwg, combined_loading):
    """One pass of EMS control over an overloaded bus, as the executor's loop drives it."""
    total_p_at_bus = enwg.grid.res_bus.loc[BUS, 'p_mw']
    return enwg._EnWG14a__control_via_ems(
        bus=BUS, trafo_power=-abs(total_p_at_bus) - 1, combined_loading=combined_loading)


def test_the_first_pass_reduces_to_exactly_the_guarantee():
    """With the overload larger than §14a permits, one pass spends the whole budget and stops."""
    enwg = restriction(grid_with(loads_at([11000, 11000])))

    result = control(enwg, combined_loading=3.0)

    assert result[AGENT]['ems'] == GUARANTEE_W


def test_a_second_pass_does_not_curtail_below_the_guarantee():
    """The regression. Before the fix the second pass returned 5200 W -- 4200 W below the floor.

    The dispatch handed to the second pass is the agent obeying the first: total power down to the
    cap, with one EV zeroed and the other left above `threshold`, which is what an EMS free to
    distribute the setpoint does.
    """
    enwg = restriction(grid_with(loads_at([11000, 11000])))
    first = control(enwg, combined_loading=3.0)[AGENT]['ems']

    # The agent complied: 1000 W inflexible + 8400 W on one EV + 0 W on the other == 9400 W.
    enwg.grid = grid_with(loads_at([8400, 0]))
    second = control(enwg, combined_loading=3.0)[AGENT]['ems']

    assert first == GUARANTEE_W
    assert second == GUARANTEE_W, (
        f'the second pass curtailed to {second} W, below the {GUARANTEE_W} W §14a guarantee; '
        f'before #242 it produced {BREACHING_CAP_W} W')
    assert second >= THRESHOLD_W


def test_the_command_stops_moving_so_the_timestep_converges():
    """Termination is a consequence of the floor, not of `max_iteration`.

    `__calculate_direct_power_control` sets `grid_ok` by comparing the command dict with the
    previous pass's. A command that stops changing is what lets a timestep settle; before the fix it
    moved every pass, and every controlled timestep ran the iteration budget out.
    """
    enwg = restriction(grid_with(loads_at([11000, 11000])))
    control(enwg, combined_loading=3.0)
    after_first = deepcopy(enwg.grid_db.restriction_commands['current_direct_power_control'])

    enwg.grid = grid_with(loads_at([8400, 0]))
    control(enwg, combined_loading=3.0)
    after_second = enwg.grid_db.restriction_commands['current_direct_power_control']

    assert after_second == after_first


def test_an_overload_smaller_than_the_budget_still_takes_only_what_is_needed():
    """The floor must not become a reduction of its own where §14a was not the binding limit.

    Here the bus needs far less than the devices could give, so the command sits well above the
    guarantee and the clamp has to leave it alone. Without this, clamping unconditionally would
    over-curtail every mild overload while every other case here still passed.
    """
    enwg = restriction(grid_with(loads_at([11000, 11000])))

    # Deliberately far over, because at a mild overload the two plausible readings of
    # `p_mw_to_be_reduced` -- `total * (1 - 1/loading)` and `total * (loading - 1)` -- land within a
    # few tens of watts of each other, and an approximate assertion cannot tell them apart. At
    # loading 1.5 they differ by thousands.
    result = control(enwg, combined_loading=1.5)[AGENT]['ems']

    assert GUARANTEE_W < result < UNCURTAILED_W
    # Exact: 23000 W drawn, a third of it asked for. No tolerance, because the value is an integer
    # computed deterministically and any slack here hides a different formula.
    assert result == SUB_BUDGET_COMMAND_W


def test_a_settled_timestep_does_not_bound_the_next_one():
    """A stale anchor is worse than none, and several paths end a timestep without clearing one.

    `execute` empties the commands when a later pass finds no overload left -- the success case,
    which is exactly when an anchor would survive -- and `executor/setup.py` forces `grid_ok` once
    `num_iteration` passes `max_iteration`, which the restriction never hears about. An anchor
    carried across is a bound taken against a different dispatch: it authorises curtailment below
    the next timestep's real floor. Here the next timestep draws far more, so its own floor is
    28400 W and the previous timestep's 9400 W would licence a 19 kW breach.
    """
    enwg = restriction(grid_with(loads_at([11000, 11000])), timestamp='2023-01-08 01:00:00+00:00')
    control(enwg, combined_loading=3.0)
    assert enwg.grid_db.restriction_commands['direct_power_control_guarantee']['agents']

    # A new timestep. Nothing cleared anything -- the reset has to come from the timestamp.
    enwg.timestamp = '2023-01-08 02:00:00+00:00'
    enwg.grid = grid_with(loads_at([11000, 11000], inflexible_w=20000))

    first = control(enwg, combined_loading=3.0)[AGENT]['ems']
    enwg.grid = grid_with(loads_at([8400, 0], inflexible_w=20000))
    second = control(enwg, combined_loading=3.0)[AGENT]['ems']

    # 20000 W inflexible + both EVs at their 4200 W floors. Until #250 this read 28399: the bound
    # was the difference of two truncated terms, chosen to equal the deepest command pass 1's own
    # arithmetic produces. That is exactly how it came to sit 1 W under an agent's inflexible load.
    # The floor is now rounded once and upward, and lifts a truncated command to the true floor.
    assert first == 28400
    assert second == 28400, (
        f'the second pass of the new timestep reached {second} W; a guarantee anchored in the '
        f'previous timestep would have allowed about 24200 W')


def test_a_pump_below_its_own_floor_does_not_block_curtailment_the_law_allows():
    """The bound is what pass 1 computes, not a re-derivation from device floors.

    `hp_reduction` is `max(0.0, hp_power - hp_floor)`, so a pump drawing less than its
    `hp_min_control` contributes nothing to `reducible` while its draw still counts in
    `total_p_agent`. Summing device floors instead would put the bound at 13200 W -- above the
    10200 W pass 1 legitimately produces -- and block 3000 W of curtailment §14a permits, on the
    agent's EV, which has nothing to do with the pump.
    """
    loads = loads_at([11000])
    loads.append({'bus': BUS, 'load_type': c.P_HP, 'p_mw': 5000 * c.WH_TO_MWH,
                  c.TC_ID_AGENT: AGENT, 'hp_min_control': 8000 * c.WH_TO_MWH})
    enwg = restriction(grid_with(loads))

    result = control(enwg, combined_loading=3.0)[AGENT]['ems']

    # 17000 W drawn, 6800 W of it reducible on the EV alone.
    assert result == 10200


def test_the_guarantee_is_forgotten_between_timesteps():
    """Anchoring is per timestep: a stale anchor would bound a timestep by the previous dispatch."""
    enwg = restriction(grid_with(loads_at([11000, 11000])))
    control(enwg, combined_loading=3.0)
    assert enwg.grid_db.restriction_commands['direct_power_control_guarantee']['agents']

    # A new timestep, which is what resets the anchor.
    enwg.timestamp = '2023-01-08 02:00:00+00:00'

    # A later timestep with a different device *count*, which is what makes this discriminating:
    # two 6000 W EVs would leave the guarantee at 9400 W again and a stale anchor would be
    # indistinguishable from a fresh one. One EV puts it at 5200 W.
    enwg.grid = grid_with(loads_at([6000]))
    result = control(enwg, combined_loading=3.0)[AGENT]['ems']

    assert result == 5200, (
        f'got {result} W; the previous timestep anchored {GUARANTEE_W} W, so a guarantee carried '
        f'over would show up here')

AGENT_TWO = 'agent_two'


def two_agent_loads(first_ev_w, second_ev_w):
    """Two agents on one bus with different non-controllable draws, so their floors differ."""
    return [
        # `agent_two` first, deliberately: it has the higher floor, so a per-bus anchor would be
        # written from it and would then hold `agent_one` up at 13200 W instead of its own 5200 W.
        # In the other order the stale anchor is lower, `max` ignores it, and the mix-up is
        # invisible.
        {'bus': BUS, 'load_type': c.P_INFLEXIBLE_LOAD, 'p_mw': 9000 * c.WH_TO_MWH,
         c.TC_ID_AGENT: AGENT_TWO, 'hp_min_control': 0.0},
        {'bus': BUS, 'load_type': c.P_EV, 'p_mw': second_ev_w * c.WH_TO_MWH,
         c.TC_ID_AGENT: AGENT_TWO, 'hp_min_control': 0.0},
        {'bus': BUS, 'load_type': c.P_INFLEXIBLE_LOAD, 'p_mw': 1000 * c.WH_TO_MWH,
         c.TC_ID_AGENT: AGENT, 'hp_min_control': 0.0},
        {'bus': BUS, 'load_type': c.P_EV, 'p_mw': first_ev_w * c.WH_TO_MWH,
         c.TC_ID_AGENT: AGENT, 'hp_min_control': 0.0},
    ]


def test_each_agent_is_bounded_by_its_own_floor_not_a_neighbour_s():
    """Two agents on one bus have different guarantees, and the anchor is per agent.

    Keyed by bus instead, the first agent computed would bound the second: here that would hold
    `agent_two` to 5200 W when its own §14a floor is 13200 W, an 8000 W breach caused entirely by a
    neighbour. Every other case in this file has a single agent, so the key is unobservable in them
    -- the anchor is written and read inside one call and never has to be looked up for anyone else.
    """
    enwg = restriction(grid_with(two_agent_loads(11000, 11000)))

    first = control(enwg, combined_loading=3.0)
    # 13200 exactly: 9000 W inflexible + one EV at its 4200 W floor. Until #250 this read 13199,
    # the floor having inherited the command's truncation (see the 28400 case above).
    assert first[AGENT]['ems'] == 5200        # 1000 W inflexible + one EV at its 4200 W floor
    assert first[AGENT_TWO]['ems'] == 13200   # 9000 W inflexible + one EV at its floor

    # Both obey, each keeping its EV above `threshold` so the devices stay in the filter.
    enwg.grid = grid_with(two_agent_loads(4300, 4300))
    second = control(enwg, combined_loading=3.0)

    # Pass 2's own computation lands on the floor and the clamp leaves it there. The bound is a
    # floor, not a setpoint.
    assert second[AGENT]['ems'] == 5200
    assert second[AGENT_TWO]['ems'] == 13200, (
        f"agent_two was held to {second[AGENT_TWO]['ems']} W; 5200 W would mean it inherited "
        f"agent_one's floor")


# --- The floor must never sit below the agent's firm draw (#250) ---------------------------------
#
# `int()` truncates toward zero, and the guarantee was the *difference of two truncated terms*:
# `int(total_mw * 1e6) - int(reducible_mw * 1e6)`. 28 % of integer-watt values do not survive
# `w * WH_TO_MWH * MWH_TO_WH`, and once the agent's loads are summed in MW first the total can lose
# a watt that the reducible term keeps -- leaving the guarantee 1 W below the inflexible load it is
# supposed to protect. With `threshold: 4200` the guarantee sits far above that load and the watt is
# invisible; with `threshold: 0` the guarantee *is* the inflexible load, and the RTC -- whose
# inflexible load is a fixed variable -- is infeasible by exactly that watt. Design 6b crashed at
# 2023-01-09 05:00 on agent e2W83ae1O1eyEW2: cap 1206 W, firm draw 1207 W.

def restriction_at(grid, threshold_w):
    enwg = restriction(grid)
    enwg.restriction_config['direct_power_control']['threshold'] = threshold_w
    return enwg


# 100, 107 and 112 lose the watt with an 11000 W EV; 1207 does not, and is the control.
@pytest.mark.parametrize('inflexible_w', [100, 107, 112, 1207])
def test_the_floor_is_never_below_the_inflexible_load(inflexible_w):
    """At threshold 0 the whole EV is reducible, so the guarantee is the inflexible draw exactly.

    Three of the values lose a watt when summed with the EV in MW and converted back -- the bug's
    precondition -- so a truncated guarantee lands at `inflexible_w - 1`. The fourth survives the
    round trip and pins that the fix does not over-guarantee either.
    """
    enwg = restriction_at(grid_with(loads_at([11000], inflexible_w=inflexible_w)), threshold_w=0)

    # An overload far beyond the reducible budget, so `required` spends all of it and the
    # command lands on the guarantee itself -- the situation at 2023-01-09 05:00.
    cap = control(enwg, combined_loading=1e9)[AGENT]['ems']

    assert cap >= inflexible_w, (
        f'cap {cap} W is below the {inflexible_w} W inflexible load the agent cannot shed; '
        f'the RTC is infeasible by {inflexible_w - cap} W')
    assert cap == inflexible_w
