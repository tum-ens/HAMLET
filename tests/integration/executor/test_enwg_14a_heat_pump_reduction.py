"""How much §14a takes off a heat pump, in both direct-power-control methods.

**Nothing has ever asserted this.** `grid_golden` is the only scenario with an active electricity
grid and `enwg_14a` in `restrictions.apply`, and its four heat pumps draw at most 1.9 kW against
a 4.2 kW threshold, so they are filtered out before either method's heat-pump arithmetic runs --
measured over an instrumented run, zero dispatches into `__control_via_ems` saw a heat pump above
the threshold, and the `individual` method's heat-pump loop body executed zero times (#209, #232).
The e2e file `tests/e2e/test_grid_restrictions.py` pins *which* device is curtailed first; this
one pins *how much* comes off a heat pump, which no scenario can currently reach.

The rule, BK6-22-300 Anlage 1 Ziffer 4.5.1 (Direktansteuerung, HAMLET's `individual`) and 4.5.2
(Steuerung mittels EMS, HAMLET's `ems`): the operator may reduce a heat pump down to its
Mindestleistung and no further. That floor is 4,2 kW, or 40 % of the **Netzanschlussleistung**
where that exceeds 11 kW. `grid_db._set_heat_pump_minimum_power` computes it per device and writes
it to the load table as `hp_min_control`; both methods here read that column and nothing else.

**Both methods used to test the heat pump's *instantaneous* draw against 11 kW instead** and
apply a 0.6 factor to it. **Three of the cases below distinguish that reading; the rest pin
behaviour it happened to get right**, and the file says which is which rather than claiming all
of them discriminate. The discriminating shape is a heat pump drawing 12 kW from a connection
rated under 11 kW: it is guaranteed 4.2 kW, but the old code offered only `0.6 x 12 kW = 7.2 kW`
of reduction and left it at 4.8 kW. Everywhere the draw does not exceed the connection power,
`0.6 x draw` and `draw - 0.4 x connection` collapse to the same clamped answer -- which is the
real reason the defect was invisible, and why the assertions here are on the command each plant
receives rather than on any total.

The methods are driven directly rather than through a scenario: they need `self.grid`,
`self.grid_db.restriction_commands` and `self.restriction_config` and nothing else, and building
a scenario that reaches them means changing `grid_golden`'s heat demand, which is what
`test_the_feeder_actually_overloads` exists to catch.
"""
import pandas as pd
import pandapower as pp
import pytest

import hamlet.constants as c
from hamlet.executor.utilities.grid_restrictions.enwg_14a import EnWG14a

THRESHOLD_W = 4200
AGENT = 'agent_a'
BUS = 1


class _GridDBStub:
    """Only the attribute both methods touch."""

    def __init__(self):
        self.restriction_commands = {'current_direct_power_control': {}}


def restriction_for(loads, sgens=(), method='individual', threshold_w=THRESHOLD_W):
    """An `EnWG14a` over a one-bus network carrying `loads` and `sgens`.

    `loads` is a list of (plant_id, load_type, p_w, hp_min_w) with an optional fifth element
    naming the owning agent, and `sgens` a list of (plant_id, plant_type, p_w), where a charging
    battery draws power and so has a *negative* `p_w` — the sign convention the sgen table uses
    and the control methods rely on.

    `__init__` is bypassed on purpose: it wants a task table and the whole `Database`, neither of
    which either control method reads.
    """
    net = pp.create_empty_network()
    pp.create_bus(net, vn_kv=0.4, name='slack')
    pp.create_bus(net, vn_kv=0.4, name='agent')
    pp.create_ext_grid(net, bus=0)

    for plant_id, load_type, p_w, hp_min_w, *owner in loads:
        index = pp.create_load(net, bus=BUS, p_mw=p_w * c.WH_TO_MWH, name=plant_id)
        net.load.loc[index, 'load_type'] = load_type
        net.load.loc[index, c.TC_ID_AGENT] = owner[0] if owner else AGENT
        net.load.loc[index, c.TC_ID_PLANT] = plant_id
        net.load.loc[index, 'hp_min_control'] = hp_min_w * c.WH_TO_MWH

    # A PV is always present so the battery filters in both methods have a table with their columns
    # to select on. It is not a battery, so it never takes part; an empty table would raise on
    # `plant_type` instead of selecting nothing.
    for plant_id, plant_type, p_w in (('p_pv', c.P_PV, 0.0), *sgens):
        index = pp.create_sgen(net, bus=BUS, p_mw=p_w * c.WH_TO_MWH, name=plant_id)
        net.sgen.loc[index, 'plant_type'] = plant_type
        net.sgen.loc[index, c.TC_ID_AGENT] = AGENT
        net.sgen.loc[index, c.TC_ID_PLANT] = plant_id

    # Written rather than solved: the methods read one cell of it, and a power flow here would only
    # add a way for this fixture to fail for reasons that are not the thing under test. A charging
    # battery is negative generation, so subtracting the sgen sum adds its draw to the bus.
    net.res_bus = pd.DataFrame(
        {'p_mw': [0.0, net.load['p_mw'].sum() - net.sgen['p_mw'].sum()]}, index=[0, BUS])

    restriction = object.__new__(EnWG14a)
    restriction.grid = net
    restriction.grid_db = _GridDBStub()
    restriction.restriction_config = {
        'direct_power_control': {'active': True, 'method': method, 'threshold': threshold_w}}
    return restriction


def commands(restriction, method, loading):
    """Run one control pass and hand back its command dict.

    `loading` is the bus's combined loading; the reduction budget the method works with is
    `total_p_at_bus * (1 - 1 / loading)`, so a large value means "budget is not the binding
    constraint" and a small one makes it binding.
    """
    call = (restriction._EnWG14a__individual_device_control if method == 'individual'
            else restriction._EnWG14a__control_via_ems)
    return call(bus=BUS, trafo_power=-1.0, combined_loading=loading)


#: Budget far above any headroom in these fixtures, so the floor is what decides the answer.
UNBOUNDED = 100.0


def test_the_individual_method_reduces_a_heat_pump_to_its_minimum():
    """12 kW drawn, 4.2 kW guaranteed: the command is 4.2 kW, so 7.8 kW comes off.

    Reading the 11 kW limit off the instantaneous 12 kW instead offers `0.6 x 12 kW` and leaves
    the pump at 4.8 kW, which is above the Mindestleistung and so under-curtails.
    """
    restriction = restriction_for([('p_hp', c.P_HP, 12_000, THRESHOLD_W)])

    target = commands(restriction, 'individual', UNBOUNDED)

    assert target[AGENT]['p_hp'] == -THRESHOLD_W


def test_the_ems_method_reduces_a_heat_pump_to_its_minimum():
    """The same case through the EMS method, which sets one net figure for the agent."""
    restriction = restriction_for([('p_hp', c.P_HP, 12_000, THRESHOLD_W)], method='ems')

    target = commands(restriction, 'ems', UNBOUNDED)

    assert target[AGENT]['ems'] == THRESHOLD_W


@pytest.mark.parametrize('method, expected_key, sign', [('individual', 'p_hp', -1), ('ems', 'ems', 1)])
def test_the_forty_percent_floor_is_respected(method, expected_key, sign):
    """A 20 kW connection guarantees 8 kW, so a pump drawing 10 kW gives up only 2 kW.

    This is the branch the flat threshold does not cover, and it is the one the regulation writes
    in terms of the Netzanschlussleistung. `hp_min_control` carries the 8 kW; neither method here
    recomputes it.
    """
    restriction = restriction_for([('p_hp', c.P_HP, 10_000, 8_000)], method=method)

    target = commands(restriction, method, UNBOUNDED)

    assert target[AGENT][expected_key] == sign * 8_000


@pytest.mark.parametrize('draw_w, floor_w, why', [
    (8_000, 8_000, 'exactly at its floor'),
    (5_000, 8_000, 'below its floor, which a connection over 11 kW makes reachable'),
])
def test_the_individual_method_does_not_control_a_pump_with_nothing_to_give(draw_w, floor_w, why):
    """It is not commanded at all -- not commanded to what it is already drawing.

    The device filter admits a device only if it draws above **its own** Mindestleistung, so a
    pump at or below its floor never reaches the loop. That distinction is not cosmetic:
    `apply_grid_commands` pins the RTC's target variable to whatever command arrives, overwriting
    what the agent's own optimiser wanted, so a command equal to the current draw is still an
    intervention. Before the filter these pumps produced a *negative* reduction -- a command to
    consume more, during an overload.
    """
    restriction = restriction_for([('p_hp', c.P_HP, draw_w, floor_w)])

    target = commands(restriction, 'individual', UNBOUNDED)

    assert 'p_hp' not in target.get(AGENT, {}), why


@pytest.mark.parametrize('draw_w, floor_w', [(8_000, 8_000), (5_000, 8_000)])
def test_the_ems_method_leaves_an_agent_with_nothing_to_give_at_its_own_draw(draw_w, floor_w):
    """The EMS twin, where the answer is a setpoint rather than an absence.

    `ems` keeps the flat-threshold filter on purpose -- see `__control_via_ems` -- so the pump
    does reach the arithmetic, and the aggregate floor is what stops it being pushed up. The
    agent is left at exactly what it draws.
    """
    restriction = restriction_for([('p_hp', c.P_HP, draw_w, floor_w)], method='ems')

    target = commands(restriction, 'ems', UNBOUNDED)

    assert target[AGENT]['ems'] == draw_w


def test_the_two_methods_clamp_at_different_granularities():
    """`individual` clamps per device, `ems` clamps the agent's total -- deliberately.

    The two control modes guarantee different things. Direktansteuerung (Ziffer 4.5.1) binds each
    device, so a pump above its own floor still owes its headroom no matter what its neighbour is
    doing. EMS control (Ziffer 4.5.2) grants one total and Satz 6 lets the Betreiber deploy it
    `nach eigener Massgabe`, so an agent already below its combined floor owes nothing at all.

    On the input below -- one pump 7.8 kW above its floor, one 15 kW below a 20 kW floor it earns
    from a 50 kW connection -- that difference is 7.8 kW of curtailment, which makes this the case
    where writing one clamp for both methods would go unnoticed. Neither answer is a compromise
    and neither is the other's bug.
    """
    loads = [('p_hp', c.P_HP, 12_000, THRESHOLD_W), ('p_under', c.P_HP, 5_000, 20_000)]

    per_device = commands(restriction_for(loads), 'individual', UNBOUNDED)
    aggregate = commands(restriction_for(loads, method='ems'), 'ems', UNBOUNDED)

    assert per_device[AGENT] == {'p_hp': -THRESHOLD_W}, (
        'the pump with headroom gives it up; the one below its floor is not controlled at all')
    assert aggregate[AGENT]['ems'] == 12_000 + 5_000, (
        'the agent draws 17 kW against a combined floor of 24.2 kW, so it owes nothing')


def test_an_under_floor_agent_does_not_take_a_negative_share_of_the_ems_reduction():
    """The EMS form of the case above, which needs two agents to be observable.

    With one agent the negative reducible total is caught by the zero guard on the division and
    the command comes out right by accident. With two, the total stays positive and the negative
    share is applied: the under-floor agent's `reducible / total` is negative, so multiplying it
    by a negative bus reduction hands it a *positive* correction and its command comes back
    **above** its own draw, while the other agent absorbs the difference.
    """
    other = 'agent_b'
    restriction = restriction_for([('p_hp', c.P_HP, 12_000, THRESHOLD_W),
                                   ('p_hp_b', c.P_HP, 5_000, 8_000, other)], method='ems')

    target = commands(restriction, 'ems', UNBOUNDED)

    # Only the first assertion discriminates: unfixed, the other agent's share works out to the
    # same 4200 W because the smaller denominator cancels the smaller bus reduction exactly. It
    # is kept as the consistency half of the pair -- the bus total has to land somewhere -- and
    # labelled so it is not mistaken for coverage it does not provide.
    assert target[other]['ems'] == 5_000, 'the under-floor agent is left exactly as it was'
    assert target[AGENT]['ems'] == THRESHOLD_W, 'the other agent still goes to its own floor'


def test_a_heat_pump_below_its_minimum_does_not_credit_the_budget_to_the_next_device():
    """The negative reduction above would also be added to the running total.

    `p_mw_14a_reduced` accumulates each device's reduction, so a negative one *increases* the
    budget still to be found and the next device in the order is asked for more than the overload
    needs. The filter now keeps the under-floor pump out of the loop entirely, so it can no longer
    contribute anything of either sign -- and this pins the consequence where it would show:
    the EV ahead gives its 2.8 kW, and the last pump gives the genuine 2 kW remainder rather than
    a remainder inflated by the 3 kW the under-floor pump used to hand back.
    """
    restriction = restriction_for([('p_ev', c.P_EV, 7_000, 0.0),
                                   ('p_under', c.P_HP, 5_000, 8_000),
                                   ('p_hp', c.P_HP, 12_000, THRESHOLD_W)])

    # total 24 kW; budget = 24 kW * (1 - 1/1.25) = 4.8 kW
    target = commands(restriction, 'individual', 1.25)

    assert target[AGENT]['p_ev'] == -THRESHOLD_W
    assert 'p_under' not in target[AGENT], 'nothing to give, so not controlled'
    assert target[AGENT]['p_hp'] == -(12_000 - 2_000), 'only the 2 kW the EV left, not 5 kW'


def test_the_reduction_stops_at_the_budget_rather_than_the_floor():
    """When the overload needs less than the headroom, only the overload is taken.

    Without this the two tests above would also pass an implementation that always reduces every
    heat pump to its floor regardless of how much reduction the bus actually needs.
    """
    # 12 kW drawn, loading 1.25 -> budget = 12 kW * (1 - 1/1.25) = 2.4 kW, well under the 7.8 kW
    # of headroom above the floor.
    restriction = restriction_for([('p_hp', c.P_HP, 12_000, THRESHOLD_W)])

    target = commands(restriction, 'individual', 1.25)

    assert target[AGENT]['p_hp'] == -(12_000 - 2_400)


def test_a_charging_battery_is_curtailed_alongside_the_heat_pump():
    """The battery arm of the same pass, which no heat-pump case reaches on its own.

    Batteries come first in the documented order and are curtailed to the flat threshold — they
    have no `hp_min_control` and the §14a heat-pump rule does not apply to them. Here the budget
    covers both, so both end at their own floor and the order does not decide the answer; the
    ordering claim is `test_the_ev_is_still_taken_before_the_heat_pump`'s and the e2e file's.
    """
    restriction = restriction_for([('p_hp', c.P_HP, 12_000, THRESHOLD_W)],
                                  sgens=[('p_battery', c.P_BATTERY, -10_000)])

    target = commands(restriction, 'individual', UNBOUNDED)

    assert target[AGENT]['p_battery'] == -THRESHOLD_W, 'charging is reduced to the flat threshold'
    assert target[AGENT]['p_hp'] == -THRESHOLD_W


def test_the_ems_reduction_stops_at_the_budget_rather_than_the_floor():
    """The EMS twin of the budget test, which the `individual` case does not cover for it.

    The two methods cap against the budget in different places -- `individual` inside each
    device's `min`, `ems` once at the bus in `power_reduction_at_bus` -- so a test of one says
    nothing about the other. Removing the EMS cap entirely leaves every other test in this file
    green.
    """
    restriction = restriction_for([('p_hp', c.P_HP, 12_000, THRESHOLD_W)], method='ems')

    # budget = 12 kW * (1 - 1/2) = 6 kW, against 7.8 kW of headroom above the floor. A loading
    # of exactly 2 keeps the budget representable; at 1.25 the `int()` truncation in the command
    # writes 9601 rather than 9600, which is real but is not what this test is about.
    target = commands(restriction, 'ems', 2.0)

    assert target[AGENT]['ems'] == 12_000 - 6_000, 'the budget binds, so the floor is not reached'


def test_the_ems_method_sums_every_heat_pump_the_agent_owns():
    """Two pumps with different floors, so the aggregate cannot be faked by one of them.

    `__control_via_ems` reduces an agent to a single net figure built from `.sum()` over its
    devices. With one pump per agent -- which is every other case in this file -- `.sum()`,
    `.max()` and "first row only" are indistinguishable, and all three pass. That is this
    repository's order-invariant-statistic trap in its single-element form, so the aggregate needs
    a case where the three disagree.

    Headroom is 12 - 4.2 = 7.8 kW on one and 10 - 8 = 2 kW on the other, so the sum is 9.8 kW and
    every wrong aggregate gives a different command: taking the larger floor alone leaves 8 kW,
    the larger draw alone leaves 22 kW, and the first row alone leaves 14.2 kW.

    The command is a setpoint, not a reduction, so with the budget unbounded the agent is left at
    exactly the sum of its two floors -- which is the EMS form of "reduced to the Mindestleistung
    and no further".
    """
    restriction = restriction_for([('p_hp', c.P_HP, 12_000, THRESHOLD_W),
                                   ('p_hp_big', c.P_HP, 10_000, 8_000)], method='ems')

    target = commands(restriction, 'ems', UNBOUNDED)

    assert target[AGENT]['ems'] == THRESHOLD_W + 8_000


def test_a_heat_pump_is_untouched_once_the_budget_is_already_met():
    """The heat-pump loop's guard, on its false arm.

    Every other test here reaches the heat pump with budget left, so the guard is only ever seen
    taking its true arm and replacing it with `if True:` changes nothing. Here the EV alone
    covers the whole overload, and the pump behind it must receive no command at all -- not a
    command equal to its current draw, which is what an unguarded body would write.
    """
    restriction = restriction_for([('p_ev', c.P_EV, 7_000, 0.0),
                                   ('p_hp', c.P_HP, 12_000, THRESHOLD_W)])

    # total 19 kW; budget = 19 kW * (1 - 1/1.0526) = 0.95 kW, well inside the EV's 2.8 kW headroom
    target = commands(restriction, 'individual', 1.0526315789473684)

    assert 'p_hp' not in target.get(AGENT, {}), 'the overload was already resolved by the EV'


def test_the_ev_is_still_taken_before_the_heat_pump():
    """The device order is unchanged by any of this, and the heat pump is still last.

    Guards the change against a rewrite that reaches the heat pump earlier: the EV has 2.8 kW of
    headroom and the budget is 3.8 kW, so the EV goes to the floor first and only the remaining
    1 kW comes off the heat pump.
    """
    restriction = restriction_for([('p_ev', c.P_EV, 7_000, 0.0),
                                   ('p_hp', c.P_HP, 12_000, THRESHOLD_W)])

    # total 19 kW; budget = 19 kW * (1 - 1/1.25) = 3.8 kW
    target = commands(restriction, 'individual', 1.25)

    assert target[AGENT]['p_ev'] == -THRESHOLD_W, 'the EV gives up its whole headroom first'
    assert target[AGENT]['p_hp'] == -(12_000 - 1_000), 'the heat pump gives up only the remainder'
