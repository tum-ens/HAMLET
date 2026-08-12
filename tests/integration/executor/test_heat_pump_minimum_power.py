"""The §14a minimum controllable power of a heat pump, written onto its grid element.

`EnWG14a`'s direct power control reduces a device only as far as the regulation lets it, and it
reads that floor for heat pumps from a `hp_min_control` column on the load table. Nothing in
HAMLET wrote that column, so the control raised `KeyError: 'hp_min_control'` the first time a
heat pump took part in a reduction — and nothing noticed, because nothing had ever reached it:
no shipped example enables direct power control at all.

The floor is not the flat threshold that EVs and batteries get. Per BNetzA BK6-22-300, and
equation 2.1 of the thesis this implementation comes from (Chu 2024):

    P_min = 0.4 * P_rated    if the grid connection power exceeds 11 kW
    P_min = threshold        otherwise

Both branches are pinned below, on the rated power rather than the instantaneous draw, because
that is what the regulation means by grid connection power.
"""
import pandapower as pp
import pytest

import hamlet.constants as c
from hamlet.executor.utilities.database.agent_db import AgentDB
from hamlet.executor.utilities.database.grid_db import ElectricityGridDB
from hamlet.executor.utilities.database.region_db import RegionDB

TOPOLOGY_FILE = 'topology.xlsx'
THRESHOLD_W = 4200
ZONE = 'test_region'


def grid_db_for(tmp_path, threshold=THRESHOLD_W):
    """A topology-method GridDB over a two-bus network, with §14a configured."""
    net = pp.create_empty_network()
    pp.create_bus(net, vn_kv=0.4, zone=ZONE, name='slack')
    pp.create_bus(net, vn_kv=0.4, zone=ZONE, name='agent_bus')
    pp.create_ext_grid(net, bus=0)
    pp.create_line(net, from_bus=0, to_bus=1, length_km=0.1, std_type='NAYY 4x50 SE')
    net.bus['agent'] = [None, 'agent_a']
    pp.to_excel(net, str(tmp_path / TOPOLOGY_FILE))

    return ElectricityGridDB(
        grid_type=c.G_ELECTRICITY, grid_path=str(tmp_path),
        grid_config={'generation': {'method': 'topology',
                                    'topology': {'file': TOPOLOGY_FILE},
                                    'file': {'file': 'electricity.xlsx'}},
                     'restrictions': {'apply': ['enwg_14a'],
                                      'enwg_14a': {'direct_power_control':
                                                   {'active': True, 'method': 'ems',
                                                    'threshold': threshold}}}})


def regions_with(plants):
    """One agent owning `plants`, plus a PV.

    The PV is here so that `_create_grid_from_topology` creates an sgen: it finishes by dropping
    on `id_agent`, and that column only exists once something has been created, so an agent with
    no generation at all raises `KeyError: ['id_agent']`. That is a separate defect with its own
    fix and its own test; carrying it here would only couple two merge requests together.
    """
    agent = AgentDB(path='', agent_type='sfh', agent_id='agent_a')
    agent.account = {c.K_GENERAL: {'bus': 1}}
    agent.plants = {**plants, 'p_pv': {'type': c.P_PV, 'sizing': {'power': 5_000}}}
    region = RegionDB(path='')
    region.agents = {'sfh': {'agent_a': agent}}
    return {ZONE: region}


def minimum_of(grid_db, plant_id):
    """The written minimum, back in watts."""
    loads = grid_db.grid.load
    row = loads[loads[c.TC_ID_PLANT] == plant_id].iloc[0]
    return row['hp_min_control'] / c.WH_TO_MWH


@pytest.mark.parametrize('rated_w, expected_w, why', [
    (2_800, THRESHOLD_W, 'well under the 11 kW limit'),
    (11_000, THRESHOLD_W, 'exactly at the limit, which the regulation says is not above it'),
    (11_001, 0.4 * 11_001, 'just over the limit'),
    (20_000, 8_000, 'comfortably over: 40 % of 20 kW'),
])
def test_the_minimum_follows_the_regulation(tmp_path, rated_w, expected_w, why):
    """P_min = 0.4 x P_rated above 11 kW, the flat threshold below it."""
    grid_db = grid_db_for(tmp_path)

    grid_db.register_grid(regions_with(
        {'p_load': {'type': c.P_INFLEXIBLE_LOAD, 'sizing': {}},
         'p_hp': {'type': c.P_HP, 'sizing': {'power': rated_w}}}))

    assert minimum_of(grid_db, 'p_hp') == pytest.approx(expected_w), why


def test_the_flat_minimum_follows_the_configured_threshold(tmp_path):
    """Below the limit the floor is the configured guarantee, not a hardcoded 4.2 kW.

    The rest of `EnWG14a` reduces batteries and EVs to `direct_power_control.threshold`, so a
    heat pump under the limit has to use the same number or the two halves of one regulation
    disagree in the same run.
    """
    grid_db = grid_db_for(tmp_path, threshold=3_000)

    grid_db.register_grid(regions_with(
        {'p_load': {'type': c.P_INFLEXIBLE_LOAD, 'sizing': {}},
         'p_hp': {'type': c.P_HP, 'sizing': {'power': 5_000}}}))

    assert minimum_of(grid_db, 'p_hp') == pytest.approx(3_000)


def test_the_column_exists_for_every_load_so_the_control_cannot_raise(tmp_path):
    """The column is what `enwg_14a` indexes, and it indexes the whole load table.

    This is the guard against the original defect returning: `KeyError: 'hp_min_control'` came
    from the column being absent, not from its value being wrong, and a scenario whose heat pump
    sits beside other loads must still present the column on all of them.
    """
    grid_db = grid_db_for(tmp_path)

    grid_db.register_grid(regions_with(
        {'p_load': {'type': c.P_INFLEXIBLE_LOAD, 'sizing': {}},
         'p_hp': {'type': c.P_HP, 'sizing': {'power': 9_800}},
         'p_ev': {'type': c.P_EV, 'sizing': {'power': 7_200}}}))

    loads = grid_db.grid.load
    assert 'hp_min_control' in loads.columns
    assert not loads['hp_min_control'].isna().any(), (
        'a load with no minimum would make the sum in __control_via_ems NaN, which silently '
        'disables the reduction rather than raising')
    # Only the heat pump carries a real floor; the others are not dimmed by this rule.
    assert minimum_of(grid_db, 'p_ev') == 0.0
    assert minimum_of(grid_db, 'p_load') == 0.0


def test_a_heat_pump_with_no_rated_power_falls_back_to_the_threshold(tmp_path):
    """Missing sizing must not produce NaN, for the reason in the test above."""
    grid_db = grid_db_for(tmp_path)

    grid_db.register_grid(regions_with(
        {'p_load': {'type': c.P_INFLEXIBLE_LOAD, 'sizing': {}},
         'p_hp': {'type': c.P_HP, 'sizing': {}}}))

    assert minimum_of(grid_db, 'p_hp') == pytest.approx(THRESHOLD_W)
