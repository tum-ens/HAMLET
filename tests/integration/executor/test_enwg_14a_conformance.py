"""Does §14a do what BK6-22-300 says, across every input we can reach?

The sibling files pin *specific* numbers the implementation produces. This one asks a different
question: for any bus we can construct, does the result satisfy the guarantees the Festlegung
states? Each test names the Ziffer it comes from and is written from that text, not from the
implementation -- so a change that keeps every pinned number and still breaks the regulation
fails here.

Source: BNetzA BK6-22-300 vom 27.11.2023, **Anlage 1** ("Festlegung zur Durchfuehrung der
netzorientierten Steuerung ... nach § 14a EnWG"). The four guarantees exercised below:

* **Ziffer 4.1** -- the operator may *reduce* the netzwirksamer Leistungsbezug. It is a power to
  reduce; nothing in the Festlegung lets a Netzbetreiber raise a device's consumption.
* **Ziffer 4.3 Satz 1** -- the reduction happens `im notwendigen Umfang ... solange wie nach
  Intensitaet und Dauer ... erforderlich`. More than the overload requires is not authorised.
* **Ziffer 4.5** -- the Betreiber keeps `einen Anspruch auf einen mindestens zu gewaehrenden
  netzwirksamen Leistungsbezug (Mindestleistung)` even while being controlled. This is the
  guarantee that must never be breached.
* **Ziffer 4.5.1** -- what that Mindestleistung is under Direktansteuerung: 4,2 kW (Satz 1), or
  the Netzanschlussleistung times a scaling factor where it exceeds 11 kW (Satz 2), the factor
  presumed appropriate at 0,4 (Satz 3).

**Where HAMLET knowingly departs from the text, that is asserted too rather than left implicit** --
see `test_the_ems_total_is_the_sum_of_the_device_minima_not_ziffer_4_5_2s_formula` and
`test_over_generation_control_does_nothing_at_all`. A conformance file that only tests the parts
that conform is a file that reads as conformance and is not.
"""
import pandas as pd
import pandapower as pp
import pytest

import hamlet.constants as c
from hamlet.executor.utilities.grid_restrictions.enwg_14a import EnWG14a

#: §14a's Mindestleistung for a directly controlled device, Ziffer 4.5.1 Satz 1.
MINDESTLEISTUNG_W = 4200

#: Ziffer 4.5.1 Satz 2: the connection power above which the scaling rule replaces the flat figure.
SCALING_LIMIT_W = 11_000

#: Ziffer 4.5.1 Satz 3: the scaling factor whose appropriateness the Festlegung presumes.
SCALING_FACTOR = 0.4

BUS = 1


class _GridDBStub:
    def __init__(self):
        self.restriction_commands = {'current_direct_power_control': {}}


def floor_for(connection_w, threshold_w=MINDESTLEISTUNG_W):
    """The Mindestleistung the Festlegung grants a heat pump, straight from Ziffer 4.5.1.

    Written out here rather than imported from `hamlet.constants` on purpose: importing the
    implementation's own constants would make this file agree with the code by construction, which
    is the failure mode the whole file exists to avoid.
    """
    return (SCALING_FACTOR * connection_w if connection_w > SCALING_LIMIT_W else threshold_w)


def build(devices, method, threshold_w=MINDESTLEISTUNG_W):
    """A restriction over one bus. `devices` is a list of dicts, see `DEVICE_SETS`."""
    net = pp.create_empty_network()
    pp.create_bus(net, vn_kv=0.4, name='slack')
    pp.create_bus(net, vn_kv=0.4, name='agent')
    pp.create_ext_grid(net, bus=0)

    for device in devices:
        if device['kind'] == c.P_BATTERY:
            index = pp.create_sgen(net, bus=BUS, p_mw=device['p_w'] * c.WH_TO_MWH,
                                   name=device['id'])
            net.sgen.loc[index, 'plant_type'] = c.P_BATTERY
            net.sgen.loc[index, c.TC_ID_AGENT] = device['agent']
            net.sgen.loc[index, c.TC_ID_PLANT] = device['id']
            continue
        index = pp.create_load(net, bus=BUS, p_mw=device['p_w'] * c.WH_TO_MWH, name=device['id'])
        net.load.loc[index, 'load_type'] = device['kind']
        net.load.loc[index, c.TC_ID_AGENT] = device['agent']
        net.load.loc[index, c.TC_ID_PLANT] = device['id']
        net.load.loc[index, 'hp_min_control'] = (
            floor_for(device['connection_w'], threshold_w) * c.WH_TO_MWH
            if device['kind'] == c.P_HP else 0.0)

    if net.sgen.empty:
        index = pp.create_sgen(net, bus=BUS, p_mw=0.0, name='pv')
        net.sgen.loc[index, 'plant_type'] = c.P_PV
        net.sgen.loc[index, c.TC_ID_AGENT] = devices[0]['agent']
        net.sgen.loc[index, c.TC_ID_PLANT] = 'pv'

    net.res_bus = pd.DataFrame(
        {'p_mw': [0.0, net.load['p_mw'].sum() - net.sgen['p_mw'].sum()]}, index=[0, BUS])

    restriction = object.__new__(EnWG14a)
    restriction.grid = net
    restriction.grid_db = _GridDBStub()
    restriction.restriction_config = {
        'direct_power_control': {'active': True, 'method': method, 'threshold': threshold_w}}
    return restriction


def control(restriction, method, loading):
    call = (restriction._EnWG14a__individual_device_control if method == 'individual'
            else restriction._EnWG14a__control_via_ems)
    return call(bus=BUS, trafo_power=-1.0, combined_loading=loading)


def hp(plant_id, p_w, connection_w, agent='agent_a'):
    return {'id': plant_id, 'kind': c.P_HP, 'p_w': p_w, 'connection_w': connection_w,
            'agent': agent}


def ev(plant_id, p_w, agent='agent_a'):
    return {'id': plant_id, 'kind': c.P_EV, 'p_w': p_w, 'connection_w': 0, 'agent': agent}


def battery(plant_id, p_w, agent='agent_a'):
    return {'id': plant_id, 'kind': c.P_BATTERY, 'p_w': p_w, 'connection_w': 0, 'agent': agent}


#: A sweep over the shapes the two methods can meet, not a list of interesting cases. It mixes
#: connection powers either side of 11 kW, draws above and below each device's own floor, single-
#: and multi-device agents, one and two agents on a bus, and a battery. Every invariant below runs
#: against all of them at three loadings, so the guarantees are asserted over roughly a hundred
#: distinct control passes rather than over the handful anyone thought to write down.
DEVICE_SETS = {
    'one small pump': [hp('hp1', 8_000, 6_000)],
    'one large pump above its floor': [hp('hp1', 20_000, 30_000)],
    'one large pump below its floor': [hp('hp1', 8_000, 30_000)],
    'pump exactly at its floor': [hp('hp1', 12_000, 30_000)],
    'two pumps, one under its floor': [hp('hp1', 12_000, 6_000), hp('hp2', 5_000, 50_000)],
    'pump behind an ev': [ev('ev1', 9_000), hp('hp1', 12_000, 6_000)],
    'pump behind an ev and a battery': [battery('b1', -10_000), ev('ev1', 9_000),
                                        hp('hp1', 12_000, 6_000)],
    'two agents': [hp('hp1', 12_000, 6_000), hp('hp2', 9_000, 40_000, agent='agent_b')],
    'two agents, one with nothing to give': [hp('hp1', 12_000, 6_000),
                                             hp('hp2', 5_000, 50_000, agent='agent_b')],
    'everything at its floor': [hp('hp1', MINDESTLEISTUNG_W, 6_000), ev('ev1', MINDESTLEISTUNG_W)],
}

#: Loadings spanning "barely overloaded" to "budget far exceeds anything available", so the
#: binding constraint differs across the sweep instead of being the floor every time.
LOADINGS = (1.05, 1.5, 100.0)

CASES = [(name, devices, loading) for name, devices in DEVICE_SETS.items() for loading in LOADINGS]
IDS = [f'{name} @ loading {loading}' for name, _, loading in CASES]


def draws(devices):
    """Each device's current draw from the grid, in watts, sign-normalised to consumption."""
    return {d['id']: abs(d['p_w']) for d in devices}


def commanded_draw(target, devices, method):
    """What each *device* is told to draw, in watts, or None where nothing was commanded.

    The two methods speak different dialects: `individual` names each plant and writes consumption
    negative, `ems` writes one positive net figure per agent. Both are normalised to "watts this
    device (or agent) is now allowed to draw" so one invariant can be written over both.
    """
    if method == 'individual':
        return {d['id']: (-target[d['agent']][d['id']] if d['id'] in target.get(d['agent'], {})
                          else None) for d in devices}
    return {d['agent']: target[d['agent']]['ems'] for d in devices if d['agent'] in target}


@pytest.mark.parametrize('method', ['individual', 'ems'])
@pytest.mark.parametrize('name, devices, loading', CASES, ids=IDS)
def test_no_device_is_ever_taken_below_its_mindestleistung(method, name, devices, loading):
    """Ziffer 4.5: the Betreiber keeps a claim to the Mindestleistung throughout.

    The strongest single guarantee in the Festlegung, and the one a wrong threshold, a wrong
    quantity or a wrong sign all break. For `individual` the claim is per device; for `ems` it is
    per agent over the sum of its devices' minima, because Ziffer 4.5.2 grants one total and
    Satz 6 lets the Betreiber spend it as it likes.
    """
    target = control(build(devices, method), method, loading)
    allowed = commanded_draw(target, devices, method)

    if method == 'individual':
        for device in devices:
            if device['kind'] != c.P_HP or allowed[device['id']] is None:
                continue
            guaranteed = floor_for(device['connection_w'])
            assert allowed[device['id']] >= guaranteed - 1, (
                f"{device['id']} cut to {allowed[device['id']]} W below its {guaranteed} W floor")
        return

    for agent, commanded in allowed.items():
        owned = [d for d in devices if d['agent'] == agent]
        guaranteed = sum(floor_for(d['connection_w']) for d in owned if d['kind'] == c.P_HP)
        drawn = sum(draws(owned).values())
        # Only meaningful where the agent was drawing above its guarantee to begin with; an agent
        # already below it is covered by the next test, which forbids moving it at all.
        if drawn >= guaranteed:
            assert commanded >= guaranteed - 1, (
                f'{agent} cut to {commanded} W below its {guaranteed} W combined floor')


@pytest.mark.parametrize('method', ['individual', 'ems'])
@pytest.mark.parametrize('name, devices, loading', CASES, ids=IDS)
def test_no_device_is_ever_told_to_consume_more_than_it_already_does(method, name, devices,
                                                                    loading):
    """Ziffer 4.1: the power is to *reduce* the netzwirksamer Leistungsbezug.

    Nothing in the Festlegung authorises raising a device's consumption, and a §14a command that
    raises one is not a curtailment that came out small -- it is an intervention in the opposite
    direction, during an overload, which the RTC pins rather than treats as a bound.
    """
    target = control(build(devices, method), method, loading)
    allowed = commanded_draw(target, devices, method)

    if method == 'individual':
        for device in devices:
            if allowed[device['id']] is None:
                continue
            assert allowed[device['id']] <= draws(devices)[device['id']] + 1, (
                f"{device['id']} draws {draws(devices)[device['id']]} W and was told "
                f"{allowed[device['id']]} W")
        return

    for agent, commanded in allowed.items():
        drawn = sum(draws([d for d in devices if d['agent'] == agent]).values())
        assert commanded <= drawn + 1, f'{agent} draws {drawn} W and was told {commanded} W'


@pytest.mark.parametrize('method', ['individual', 'ems'])
@pytest.mark.parametrize('name, devices, loading', CASES, ids=IDS)
def test_no_more_is_taken_than_the_overload_requires(method, name, devices, loading):
    """Ziffer 4.3 Satz 1: the reduction happens `im notwendigen Umfang`.

    The budget is `total_p_at_bus * (1 - 1/loading)`, the power that has to come off to bring the
    bus back to 100 %. Taking more than that is not authorised, however much headroom the devices
    happen to have. A tolerance of one watt per device absorbs the `int()` truncation the
    commands are written with.
    """
    restriction = build(devices, method)
    total_p_w = restriction.grid.res_bus.loc[BUS, 'p_mw'] / c.WH_TO_MWH
    budget_w = total_p_w * (1 - 1 / loading)

    target = control(restriction, method, loading)
    allowed = commanded_draw(target, devices, method)

    if method == 'individual':
        taken = sum(draws(devices)[d['id']] - allowed[d['id']]
                    for d in devices if allowed[d['id']] is not None)
    else:
        taken = sum(sum(draws([d for d in devices if d['agent'] == agent]).values()) - commanded
                    for agent, commanded in allowed.items())

    assert taken <= budget_w + len(devices), (
        f'took {taken:.1f} W for an overload of {budget_w:.1f} W')


@pytest.mark.parametrize('connection_w, expected_w', [
    (4_300, MINDESTLEISTUNG_W),
    (11_000, MINDESTLEISTUNG_W),
    (11_001, SCALING_FACTOR * 11_001),
    (30_000, SCALING_FACTOR * 30_000),
])
def test_the_floor_a_heat_pump_is_reduced_to_is_ziffer_4_5_1s(connection_w, expected_w):
    """Ziffer 4.5.1 Satz 1-3, read off the command rather than off `hp_min_control`.

    `test_heat_pump_minimum_power.py` pins the column; this pins that the control actually stops
    there. `11_000` is the boundary the Festlegung words as `ueber 11 kW`, so equality is below it.
    """
    devices = [hp('hp1', int(connection_w * 0.9) + MINDESTLEISTUNG_W, connection_w)]

    target = control(build(devices, 'individual'), 'individual', 100.0)

    assert -target['agent_a']['hp1'] == pytest.approx(expected_w, abs=1)


def test_a_device_already_at_its_floor_is_left_alone_entirely():
    """Ziffer 4.5 again, at the boundary: nothing to give means no command, not a command of zero.

    A command equal to the current draw is not a no-op downstream -- `apply_grid_commands` pins
    the RTC's target variable to it, overwriting what the agent's own optimiser wanted. So "not
    controlled" and "controlled to exactly what you are doing" are different outcomes, and a
    device with no headroom is entitled to the first.
    """
    devices = [hp('hp1', 8_000, 30_000)]  # floor 12 kW, drawing 8 kW

    target = control(build(devices, 'individual'), 'individual', 100.0)

    assert 'hp1' not in target.get('agent_a', {})


def test_the_ems_total_is_the_sum_of_the_device_minima_not_ziffer_4_5_2s_formula():
    """A knowing departure from the text, asserted so that it cannot be mistaken for conformance.

    Ziffer 4.5.2 Satz 3 grants an EMS-controlled installation
    `Max(0,4 x P_Summe WP; 0,4 x P_Summe Klima) + (n_steuVE - 1) x GZF x 4,2 kW` -- one Sockel plus
    a flat 4,2 kW for each further steuerbare Verbrauchseinrichtung, under the tabulated
    Gleichzeitigkeitsfaktor (0,8 at two devices). HAMLET grants the plain sum of the per-device
    Ziffer 4.5.1 floors instead. The two agree for a single device and diverge beyond one, in
    either direction: higher than the Festlegung for several small pumps, lower for several large
    ones.

    Two pumps of 6 kW connection: HAMLET grants 4,2 + 4,2 = 8,4 kW where Ziffer 4.5.2 grants
    4,2 + 1 x 0,8 x 4,2 = 7,56 kW. The gap is small here and is not always; either way it is a
    modelling simplification that predates #209 and is recorded rather than silently carried.
    """
    devices = [hp('hp1', 12_000, 6_000), hp('hp2', 12_000, 6_000)]

    target = control(build(devices, 'ems'), 'ems', 100.0)

    hamlet_total = 2 * MINDESTLEISTUNG_W
    ziffer_4_5_2 = MINDESTLEISTUNG_W + 1 * 0.8 * MINDESTLEISTUNG_W

    assert target['agent_a']['ems'] == pytest.approx(hamlet_total, abs=1)
    assert hamlet_total != pytest.approx(ziffer_4_5_2, abs=1), (
        'if these ever coincide, this test has stopped saying anything')


@pytest.mark.parametrize('method', ['individual', 'ems'])
def test_over_generation_control_does_nothing_at_all(method):
    """The other knowing gap: §14a never curtails over-generation, in either method.

    Ziffer 4.1 covers `strom- oder spannungsbedingte` hazards in both directions, and both methods
    carry an over-generation branch. Neither works. `p_mw_to_be_reduced` is
    `total_p_at_bus * (1 - 1/loading)` and `total_p_at_bus` is negative when the bus exports, so
    the budget is negative: `__individual_device_control` guards every device with
    `p_mw_14a_reduced < p_mw_to_be_reduced`, i.e. `0 < negative`, which is never true, and
    `__control_via_ems` takes `min(positive reducible, negative budget)` and so always returns the
    budget, making its reducible cap dead code.

    Pinned as a characterisation test, not an xfail: it states what the code does today, and the
    day someone fixes the branch this goes red and has to be rewritten deliberately. An xfail
    would go quietly green instead.
    """
    devices = [battery('b1', 10_000), ev('ev1', -7_000)]
    restriction = build(devices, method)
    net = restriction.grid
    net.res_bus = pd.DataFrame(
        {'p_mw': [0.0, net.load['p_mw'].sum() - net.sgen['p_mw'].sum()]}, index=[0, BUS])

    call = (restriction._EnWG14a__individual_device_control if method == 'individual'
            else restriction._EnWG14a__control_via_ems)
    target = call(bus=BUS, trafo_power=1.0, combined_loading=2.0)

    if method == 'individual':
        assert target == {}, 'the over-generation branch issues no command whatsoever'
    else:
        # The EMS branch does write a setpoint, but one derived from the negative budget rather
        # than from what the devices could actually give back.
        assert target['agent_a']['ems'] < -sum(abs(d['p_w']) for d in devices), (
            'the setpoint is slacker than the status quo, so it curtails nothing')
