"""The §14a minimum controllable power of a heat pump, written onto its grid element.

`EnWG14a`'s direct power control reduces a device only as far as the regulation lets it, and it
reads that floor for heat pumps from a `hp_min_control` column on the load table. Nothing in
HAMLET wrote that column, so the control raised `KeyError: 'hp_min_control'` the first time a
heat pump took part in a reduction — and nothing noticed, because nothing had ever reached it:
no shipped example enables direct power control at all.

The floor is not the flat threshold that EVs and batteries get. BK6-22-300 Anlage 1 Ziffer 4.5.1:

    P_min = 0.4 * P_connection    if the Netzanschlussleistung exceeds 11 kW
    P_min = threshold             otherwise

**The whole point of this file is which quantity `P_connection` is.** The regulation means the
device's *electrical* grid connection rating. A heat pump's `sizing['power']` in HAMLET is its
*thermal* output -- sized from annual heat demand, and divided by the COP wherever the model needs
an electrical figure -- so using it here compares a thermal number against an electrical limit and
overstates the connection by the COP, which for the shipped specs is 2.15 to 3.52. Every case
below is therefore parametrised on **both** halves of the quotient, and the expectations of
`test_the_minimum_follows_the_regulation` are chosen so that a revert to `sizing['power']` fails
here rather than merely shifting a number nobody pinned -- with the one exception that row names
itself.

`cop_ref` is the device's reference-point COP, carried per plant in the agent's `specs`, keyed by
plant id alongside the specs of every other plant the agent owns.
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


def regions_with(plants, specs=None):
    """One agent owning `plants`, plus a PV.

    The PV is here so that `_create_grid_from_topology` creates an sgen: it finishes by dropping
    on `id_agent`, and that column only exists once something has been created, so an agent with
    no generation at all raises `KeyError: ['id_agent']`. That is a separate defect with its own
    fix and its own test; carrying it here would only couple two merge requests together.

    `specs` is the agent's per-plant spec dict, which is where `cop_ref` lives in a real scenario
    (`agents/*/specs.json`, keyed by plant id).
    """
    agent = AgentDB(path='', agent_type='sfh', agent_id='agent_a')
    agent.account = {c.K_GENERAL: {'bus': 1}}
    agent.plants = {**plants, 'p_pv': {'type': c.P_PV, 'sizing': {'power': 5_000}}}
    agent.specs = dict(specs or {})
    region = RegionDB(path='')
    region.agents = {'sfh': {'agent_a': agent}}
    return {ZONE: region}


def hp_agent(thermal_w, cop_ref, extra_plants=None, decoy_specs=True):
    """`regions_with` for the common case: one heat pump, one inflexible load.

    The agent's `specs` carries a **decoy PV entry with its own `cop_ref`**, because a real
    agent's `specs.json` holds one entry per plant and PV and wind write entries too. Without it
    the dict has a single key, and reading "the only spec" instead of "the spec for this plant id"
    passes every assertion in this file.
    """
    sizing = {} if thermal_w is None else {'power': thermal_w}
    specs = {} if cop_ref is None else {'p_hp': {'type': c.P_HP, 'cop_ref': cop_ref}}
    if decoy_specs:
        specs = {'p_pv': {'type': c.P_PV, 'cop_ref': 1.0}, **specs}
    return regions_with({'p_load': {'type': c.P_INFLEXIBLE_LOAD, 'sizing': {}},
                         'p_hp': {'type': c.P_HP, 'sizing': sizing},
                         **(extra_plants or {})}, specs=specs)


def minimum_of(grid_db, plant_id):
    """The written minimum, back in watts."""
    loads = grid_db.grid.load
    row = loads[loads[c.TC_ID_PLANT] == plant_id].iloc[0]
    return row['hp_min_control'] / c.WH_TO_MWH


@pytest.mark.parametrize('thermal_w, cop_ref, expected_w, why', [
    (8_400, 3.0, THRESHOLD_W,
     'connection 2.8 kW, well under the limit -- and under it thermally too, so this row alone '
     'does not distinguish the two readings; it is here to pin the ordinary small-pump case'),
    (33_000, 3.0, THRESHOLD_W,
     'connection exactly 11 kW, which the regulation says is not *above* the limit'),
    (33_003, 3.0, 0.4 * 11_001, 'connection just over the limit'),
    (60_000, 3.0, 8_000, 'connection 20 kW, comfortably over: 40 % of it'),
    (25_000, 2.0, 5_000, 'a poorer COP puts a smaller thermal rating over the limit'),
])
def test_the_minimum_follows_the_regulation(tmp_path, thermal_w, cop_ref, expected_w, why):
    """P_min = 0.4 x P_connection above 11 kW, the flat threshold below it.

    **Rows 2-5 are the ones that discriminate**, and they do it in both directions: row 2 is over
    the limit thermally and exactly at it electrically, rows 3-5 would otherwise take 40 % of a
    number `cop_ref` times too large. Row 1 is under the limit on either reading and is here for
    the ordinary case, not for the discrimination -- said explicitly because "every row
    discriminates" is the kind of claim that reads as coverage and is not.
    """
    grid_db = grid_db_for(tmp_path)

    grid_db.register_grid(hp_agent(thermal_w, cop_ref))

    assert minimum_of(grid_db, 'p_hp') == pytest.approx(expected_w), why


@pytest.mark.parametrize('thermal_w, cop_ref', [(20_000, 3.0), (60_000, 3.0), (25_000, 2.0)])
def test_the_thermal_rating_is_not_what_the_limit_is_tested_against(tmp_path, thermal_w, cop_ref):
    """The floor never equals what `sizing['power']` alone would have produced.

    The parametrisation above pins the right answers; this pins that the *wrong* answer is absent,
    which is the claim that survives someone rewriting the expectations. Each case is over 11 kW
    thermally and so would take the scaling branch on the thermal figure.
    """
    grid_db = grid_db_for(tmp_path)

    grid_db.register_grid(hp_agent(thermal_w, cop_ref))

    thermal_answer = 0.4 * thermal_w
    assert thermal_answer > THRESHOLD_W, 'the case has to be one the two readings disagree on'
    assert minimum_of(grid_db, 'p_hp') != pytest.approx(thermal_answer)


@pytest.mark.parametrize('cop_ref, why', [
    (None, 'no spec entry at all, so no rating can be derived'),
    (0, 'a zero COP would divide by zero'),
    (0.8, 'a COP below 1 would claim a connection larger than the heat produced'),
])
def test_an_underivable_connection_power_falls_back_to_the_threshold(tmp_path, cop_ref, why):
    """No usable `cop_ref` means the flat threshold, not a guess and not NaN.

    NaN is the failure that matters, and **it maximises the curtailment rather than disabling
    it**: `__individual_device_control`'s `min(budget, p_mw - hp_min_control)` returns the budget
    when the second term is NaN, because every comparison against NaN is false, and
    `__control_via_ems` sums the column with `Series.sum()`, which skips NaN and so reads the
    floor as zero. Both leave the pump curtailed to nothing -- the one outcome §14a forbids.
    """
    grid_db = grid_db_for(tmp_path)

    grid_db.register_grid(hp_agent(60_000, cop_ref))

    assert minimum_of(grid_db, 'p_hp') == pytest.approx(THRESHOLD_W), why


def test_the_flat_minimum_follows_the_configured_threshold(tmp_path):
    """Below the limit the floor is the configured guarantee, not a hardcoded 4.2 kW.

    The rest of `EnWG14a` reduces batteries and EVs to `direct_power_control.threshold`, so a
    heat pump under the limit has to use the same number or the two halves of one regulation
    disagree in the same run.
    """
    grid_db = grid_db_for(tmp_path, threshold=3_000)

    grid_db.register_grid(hp_agent(15_000, 3.0))

    assert minimum_of(grid_db, 'p_hp') == pytest.approx(3_000)


def test_the_column_exists_for_every_load_so_the_control_cannot_raise(tmp_path):
    """The column is what `enwg_14a` indexes, and it indexes the whole load table.

    This is the guard against the original defect returning: `KeyError: 'hp_min_control'` came
    from the column being absent, not from its value being wrong, and a scenario whose heat pump
    sits beside other loads must still present the column on all of them.
    """
    grid_db = grid_db_for(tmp_path)

    grid_db.register_grid(hp_agent(
        9_800, 3.13, extra_plants={'p_ev': {'type': c.P_EV, 'sizing': {'power': 7_200}}}))

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

    grid_db.register_grid(hp_agent(None, 3.0))

    assert minimum_of(grid_db, 'p_hp') == pytest.approx(THRESHOLD_W)
