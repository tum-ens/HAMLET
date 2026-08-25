"""Do §14a's device filters select exactly the rows they name, by label and not by position?

`__individual_device_control` and `__control_via_ems` both narrow `grid.load` and `grid.sgen`
down to the devices §14a may control. Until #210 every one of those filters was written as a
chained subscript -- `frame[mask_a][mask_b]`, with `mask_b` computed against the *unfiltered*
frame and handed to the *filtered* one. pandas made that work by reindexing `mask_b` by label,
warning each time, and has said it will raise instead in a future release. The filters are now
written as `frame[mask_a & mask_b]`.

That rewrite is only correct if the reindex was doing nothing but a label-aligned subset. These
tests pin that, on frames built so a *positional* rewrite -- the obvious wrong way to remove a
reindex -- gives a different answer:

* the element tables carry a gappy, non-monotonic index, so label and position disagree;
* every row that must not be selected belongs to `agent_decoy`, an agent with no §14a device at
  the bus under test, so any row wrongly selected shows up as an agent that should not be there;
* the same control pass is run again with the decoy rows deleted and again with the rows
  shuffled, and must return the identical command dict.

Which of them is the pin against the chained form coming back is not obvious, and is stated on
the test that is.
"""
import warnings
from copy import deepcopy

import pandas as pd
import pandapower as pp
import pytest

import hamlet.constants as c
from hamlet.executor.utilities.grid_restrictions.enwg_14a import EnWG14a

#: The bus §14a is asked to control. Everything at any other bus is out of scope by definition.
BUS = 1

#: A second bus, used only to hold rows the filters must not reach.
OTHER_BUS = 2

#: Mindestleistung per device, in watts; the threshold the restriction config carries.
THRESHOLD_W = 4200

#: The agent every row that must *not* be selected belongs to. It owns no controllable device at
#: `BUS`, so a correct filter never names it in the result -- which turns "the wrong row was
#: selected" into a single readable assertion instead of an arithmetic comparison.
DECOY = 'agent_decoy'

#: Deliberately gappy and out of order. With a default RangeIndex a positional rewrite of the
#: masks would pass every test in this file.
LOAD_INDEX = [7, 3, 9, 1, 5]
SGEN_INDEX = [4, 2, 8]


class _GridDBStub:
    def __init__(self):
        self.restriction_commands = {'current_direct_power_control': {}}


def _net(loads, sgens, p_mw_at_bus):
    """A net whose element tables are exactly `loads`/`sgens`, on a non-monotonic index."""
    net = pp.create_empty_network()
    pp.create_bus(net, vn_kv=0.4, name='slack')
    pp.create_bus(net, vn_kv=0.4, name='controlled')
    pp.create_bus(net, vn_kv=0.4, name='elsewhere')
    pp.create_ext_grid(net, bus=0)

    net['load'] = pd.DataFrame(loads, index=LOAD_INDEX[:len(loads)])
    net['sgen'] = pd.DataFrame(sgens, index=SGEN_INDEX[:len(sgens)])
    net['res_bus'] = pd.DataFrame({'p_mw': [0.0, p_mw_at_bus, 0.0]}, index=[0, BUS, OTHER_BUS])
    return net


def _load(bus, load_type, p_w, agent, plant, connection_floor_w=THRESHOLD_W):
    return {'bus': bus, 'load_type': load_type, 'p_mw': p_w * c.WH_TO_MWH,
            c.TC_ID_AGENT: agent, c.TC_ID_PLANT: plant,
            'hp_min_control': (connection_floor_w * c.WH_TO_MWH
                               if load_type == c.P_HP else 0.0)}


def _sgen(bus, plant_type, p_w, agent, plant):
    return {'bus': bus, 'plant_type': plant_type, 'p_mw': p_w * c.WH_TO_MWH,
            c.TC_ID_AGENT: agent, c.TC_ID_PLANT: plant}


def _restriction(net, method):
    restriction = object.__new__(EnWG14a)
    restriction.grid = net
    restriction.grid_db = _GridDBStub()
    restriction.restriction_config = {
        'direct_power_control': {'active': True, 'method': method, 'threshold': THRESHOLD_W}}
    return restriction


def _control(net, method, trafo_power, loading=1.5):
    """One control pass over a copy of `net`, so a caller can reuse the net it built."""
    restriction = _restriction(deepcopy(net), method)
    call = (restriction._EnWG14a__individual_device_control if method == 'individual'
            else restriction._EnWG14a__control_via_ems)
    return call(bus=BUS, trafo_power=trafo_power, combined_loading=loading)


def _control_without_reindex_warning(net, method, trafo_power, loading=1.5):
    """The same call, with pandas' boolean-key reindex promoted to an error -- that one message
    only, since the rest of the run's warnings are not this test's business.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('always')
        warnings.filterwarnings('error', message='Boolean Series key will be reindexed')
        return _control(net, method, trafo_power, loading)


def _without_decoys(net):
    """The same net with every `agent_decoy` row deleted, index order otherwise preserved."""
    trimmed = deepcopy(net)
    trimmed['load'] = trimmed['load'][trimmed['load'][c.TC_ID_AGENT] != DECOY]
    trimmed['sgen'] = trimmed['sgen'][trimmed['sgen'][c.TC_ID_AGENT] != DECOY]
    return trimmed


def _shuffled(net):
    """The same rows in a different order. Label-based selection cannot notice; position can."""
    shuffled = deepcopy(net)
    shuffled['load'] = shuffled['load'].iloc[::-1]
    shuffled['sgen'] = shuffled['sgen'].iloc[::-1]
    return shuffled


def _p_at_bus(loads, sgens):
    """Net active power at `BUS`, the quantity `res_bus` would carry after a power flow."""
    return (sum(row['p_mw'] for row in loads if row['bus'] == BUS) -
            sum(row['p_mw'] for row in sgens if row['bus'] == BUS))


# --- the two nets -------------------------------------------------------------------------------
# Each carries one row per way a filter can go wrong: a controllable device at the wrong bus, and
# an uncontrollable device type at the right bus -- the latter on both `load` and `sgen`, since
# the two methods filter both tables and the per-agent reductions inside `__control_via_ems` apply
# the type mask a second time.

def _over_consumption_net():
    """Bus 1 draws power; the transformer imports. `trafo_power < 0 < p_at_bus`."""
    loads = [
        _load(OTHER_BUS, c.P_EV, 11_000, DECOY, 'ev_elsewhere'),           # right type, wrong bus
        _load(BUS, c.P_HP, 9_000, 'agent_a', 'hp_a'),                      # selected
        _load(BUS, c.P_INFLEXIBLE_LOAD, 12_000, DECOY, 'inflexible'),      # right bus, wrong type
        _load(BUS, c.P_EV, 9_000, 'agent_a', 'ev_a'),                      # selected
        _load(BUS, c.P_EV, 8_000, 'agent_b', 'ev_b'),                      # selected
    ]
    sgens = [
        _sgen(OTHER_BUS, c.P_BATTERY, -10_000, DECOY, 'bat_elsewhere'),    # right type, wrong bus
        _sgen(BUS, c.P_BATTERY, -10_000, 'agent_a', 'bat_a'),              # selected (charging)
        _sgen(BUS, c.P_PV, -6_000, DECOY, 'pv_decoy'),                     # right bus, wrong type
    ]
    return _net(loads, sgens, _p_at_bus(loads, sgens))


def _over_generation_net():
    """Bus 1 feeds in; the transformer exports. `trafo_power > 0 > p_at_bus`."""
    loads = [
        _load(OTHER_BUS, c.P_EV, -11_000, DECOY, 'ev_elsewhere'),          # right type, wrong bus
        _load(BUS, c.P_EV, -9_000, 'agent_a', 'ev_a'),                     # selected (feeding in)
        _load(BUS, c.P_INFLEXIBLE_LOAD, -12_000, DECOY, 'inflexible'),     # right bus, wrong type
        _load(BUS, c.P_EV, -8_000, 'agent_b', 'ev_b'),                     # selected (feeding in)
    ]
    sgens = [
        _sgen(OTHER_BUS, c.P_BATTERY, 10_000, DECOY, 'bat_elsewhere'),     # right type, wrong bus
        _sgen(BUS, c.P_BATTERY, 10_000, 'agent_a', 'bat_a'),               # selected (discharging)
        _sgen(BUS, c.P_PV, 20_000, DECOY, 'pv_decoy'),                     # right bus, wrong type
    ]
    return _net(loads, sgens, _p_at_bus(loads, sgens))


#: Both branches of both methods. `trafo_power` picks the branch; its magnitude is never read.
DIRECTIONS = [('over-consumption', _over_consumption_net, -1.0),
              ('over-generation', _over_generation_net, 1.0)]

#: Exactly what each case must command, device by device and watt by watt.
#:
#: Asserting the *keys* is not enough, and asserting a shape is not enough: a filter that names
#: the wrong device class -- `c.P_EV` where `c.P_HP` belongs, `plant_type` where `load_type`
#: does -- usually leaves the same agents in the result and changes only what they were told.
#: Four such mutations survived a version of this file that checked keys alone, one of them
#: removing heat pumps from §14a individual control altogether.
#:
#: These are the numbers today's implementation produces; they are here to be *stable*, not to
#: be independently derived. Ziffer 4.5's floors and the priority order are asserted from the
#: Festlegung's own text in `tests/integration/executor/test_enwg_14a_conformance.py`, so a
#: number that moves here without moving there is a selection change -- which is the point.
#:
#: The over-consumption budget deliberately does **not** run out part-way through the device
#: loops; every device lands on its floor. That keeps this file about *which* rows a filter
#: selects. The order they are curtailed in is
#: `tests/e2e/test_grid_restrictions.py::test_devices_are_curtailed_in_priority_order`'s job,
#: and that needs an exhausted budget to say anything at all.
EXPECTED = {
    ('individual', 'over-consumption'): {'agent_a': {'bat_a': -4200, 'ev_a': -4200,
                                                     'hp_a': -5399},
                                         'agent_b': {'ev_b': -4200}},
    ('individual', 'over-generation'): {},
    ('ems', 'over-consumption'): {'agent_a': {'ems': 13562}, 'agent_b': {'ems': 4438}},
    ('ems', 'over-generation'): {'agent_a': {'ems': -33476}, 'agent_b': {'ems': -13189}},
}

#: Not asserted here, and deliberately: that §14a leaves `grid.load` / `grid.sgen` untouched.
#: Both methods `deepcopy` the net and then `deepcopy` each filtered frame again, but every
#: `.loc[...]` on those frames is a *read* -- neither method ever assigns into one. Dropping any
#: of the copies is therefore behaviour-preserving today, so a test asserting the net is
#: unchanged passes with or without them and pins nothing. Measured, not assumed: eight
#: deepcopy-removal mutations, every one green.

#: `individual` + over-generation commands nothing, and that is not this file's doing: the
#: over-generation budget `total_p_at_bus * (1 - 1/loading)` is negative whenever the bus
#: exports, and every device loop is guarded by `0 < that`. The branch filters its devices and
#: then commands none of them, for any input -- pinned as a characterisation of today's code by
#: `test_over_generation_control_does_nothing_at_all` in the conformance file.
#:
#: **So nothing this file asserts about that case constrains its filters.** An empty result is
#: the same empty result whether the masks select the right rows, the wrong rows, or every row
#: in the grid; `&` swapped for `|` there is invisible to every assertion below. It stays in the
#: sweep for the one thing that still works -- the filters do run, so the reindex test would
#: catch a re-chain -- and for the day the branch is made reachable, when this note goes wrong.
UNREACHABLE_BRANCH = {('individual', 'over-generation')}

CASES = [(method, name, build, trafo, EXPECTED[(method, name)])
         for method in ('individual', 'ems') for name, build, trafo in DIRECTIONS]
IDS = [f'{method} / {name}' for method, name, _, _, _ in CASES]


@pytest.mark.parametrize('method, name, build, trafo_power, expected', CASES, ids=IDS)
def test_exactly_the_selected_devices_are_commanded_and_by_exactly_this_much(
        method, name, build, trafo_power, expected):
    """The filters selected these rows and no others, and gave them these numbers.

    `DECOY` owns only rows the filters exclude -- one device at another bus, one uncontrollable
    load type, one uncontrollable plant type -- so its absence is the selection claim, and the
    watts are what stops a filter naming the wrong device class from passing anyway.
    """
    assert _control(build(), method, trafo_power) == expected


@pytest.mark.parametrize('method, name, build, trafo_power, expected', CASES, ids=IDS)
def test_deleting_the_excluded_rows_changes_nothing(method, name, build, trafo_power, expected):
    """If the filters select what they name, rows they exclude cannot affect the result."""
    net = build()
    assert _control(net, method, trafo_power) == _control(_without_decoys(net), method,
                                                          trafo_power), (
        f'{name}: the command changed when rows the filters exclude were deleted, so at least '
        f'one of them was being selected')


@pytest.mark.parametrize('method, name, build, trafo_power, expected',
                         [case for case in CASES if case[0] == 'ems'],
                         ids=[i for i in IDS if i.startswith('ems')])
def test_the_row_order_of_the_element_tables_does_not_matter(method, name, build, trafo_power,
                                                             expected):
    """Label-aligned selection is order-invariant; positional selection is not.

    `ems` only, and not because `individual` is uninteresting. `__individual_device_control`
    walks its devices in table order and gives the remainder of the budget to whichever device
    it reaches as the budget runs out, so its output is *legitimately* order-dependent as soon
    as the budget binds. An order-invariance assertion over it would hold only while the fixture
    happens to leave the budget unexhausted -- true today, and a trap the moment anyone edits a
    magnitude in `_over_consumption_net`. `__control_via_ems` aggregates per agent before
    distributing, so it is order-invariant for real.

    For `individual` the same claim is carried by the exact commands above: a positional
    de-reindexing selects a different set of rows and produces different watts.
    """
    net = build()
    assert _control(net, method, trafo_power) == _control(_shuffled(net), method, trafo_power), (
        f'{name}: reversing the row order changed the command, so a filter is selecting by '
        f'position rather than by label')


@pytest.mark.parametrize('method, name, build, trafo_power, expected', CASES, ids=IDS)
def test_no_filter_hands_pandas_a_boolean_key_it_has_to_reindex(method, name, build, trafo_power,
                                                                expected):
    """The #210 pin: chaining the masks back together makes this raise.

    pandas warns on every boolean key whose index differs from the frame's, and has said it will
    raise instead in a future release. Promoting that warning to an error here is what makes the
    rewrite hold: without it nothing fails when a filter goes back to `frame[mask_a][mask_b]`,
    because today's reindex still produces the right answer.
    """
    assert _control_without_reindex_warning(build(), method, trafo_power) == expected
