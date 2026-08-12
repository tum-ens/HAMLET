"""Grid registration — the two grid-file conventions, and the two ways it can go wrong.

`GridDB.register_grid` is where a scenario meets its network, and until #205 nothing exercised it
at all: the grid stage had no test and neither grid-enabled example ran. What it pins:

* **Both conventions are read.** A grid file may carry HAMLET's per-element metadata as real
  columns or packed into `description` as `key:value,key:value`. !216 removed the packed reader
  because a real network's `description` is prose and parsing it crashed — which fixed the paper's
  design 6 network and broke `examples/create_scenario_with_grid`, the one example that uses the
  file method. Both are now supported and both are tested here.
* **A mismatch is named, not swallowed.** An agent with no bus in the topology file, and an agent
  with no matching inflexible load in the grid file, used to surface as a bare `KeyError` on a
  random id and `TypeError: cannot unpack non-iterable NoneType object` respectively (#201, #205).
  Continuing instead would leave the agent out of the network and solve the power flow for a
  feeder that is missing one of its participants.

These build real pandapower networks and drive the real `ElectricityGridDB`, because the defect
was in how the two are read against each other; a fixture standing in for either would have
removed the thing under test.
"""
import pandapower as pp
import pytest

import hamlet.constants as c
from hamlet.executor.utilities.database.agent_db import AgentDB
from hamlet.executor.utilities.database.grid_db import ElectricityGridDB
from hamlet.executor.utilities.database.region_db import RegionDB

GRID_FILE = 'electricity.xlsx'
TOPOLOGY_FILE = 'topology.xlsx'
ZONE = 'test_region'


def grid_config(method):
    """The `electricity:` block of a grids.yaml, as `GridDB` receives it."""
    return {'powerflow': 'dc', 'active': True,
            'generation': {'method': method,
                           'file': {'file': GRID_FILE},
                           'topology': {'file': TOPOLOGY_FILE}},
            'restrictions': {'apply': [], 'max_iteration': 10}}


def make_agent(agent_id, bus, plants):
    """A real AgentDB with the two attributes grid registration reads off it."""
    agent = AgentDB(path='', agent_type='sfh', agent_id=agent_id)
    agent.account = {c.K_GENERAL: {'bus': bus}}
    agent.plants = plants
    return agent


def make_regions(agents):
    """A real RegionDB holding `agents`, shaped the way `register_grid` walks it."""
    region = RegionDB(path='')
    region.agents = {'sfh': {agent.agent_id: agent for agent in agents}}
    return {ZONE: region}


def base_net():
    """Two-bus network: an external grid on bus 0, agents on bus 1."""
    net = pp.create_empty_network()
    pp.create_bus(net, vn_kv=0.4, zone=ZONE, name='slack')
    pp.create_bus(net, vn_kv=0.4, zone=ZONE, name='agent_bus')
    pp.create_ext_grid(net, bus=0)
    pp.create_line(net, from_bus=0, to_bus=1, length_km=0.1, std_type='NAYY 4x50 SE')
    return net


def write(net, tmp_path, filename):
    """Save the network the way a scenario ships it, and return a GridDB reading it back."""
    pp.to_excel(net, str(tmp_path / filename))
    method = 'topology' if filename == TOPOLOGY_FILE else 'file'
    return ElectricityGridDB(grid_type=c.G_ELECTRICITY, grid_path=str(tmp_path),
                             grid_config=grid_config(method))


# --------------------------------------------------------------------------------------------
# The two grid-file conventions
# --------------------------------------------------------------------------------------------

LOAD_PROFILE = 'hh_2455_0.csv'
HP_PROFILE = 'hp_test_air.json'
PV_PROFILE = 'pv_config.json'


def packed_net():
    """A grid file in the packed convention: metadata inside `description`, no metadata columns.

    This is what `examples/create_scenario_with_grid/.../electricity.xlsx` looks like. `owner`
    refers to the row of the inflexible load the element belongs to.
    """
    net = base_net()
    pp.create_load(net, bus=1, p_mw=0.001,
                   description=f'agent_type:sfh,dummy:False,demand:2.455,file:{LOAD_PROFILE},'
                               f'file_add:NaN,cos_phi:1.0,load_type:inflexible-load')
    pp.create_load(net, bus=1, p_mw=0.0004,
                   description=f'agent_type:sfh,owner:0,demand:0.0385,file_add:{HP_PROFILE},'
                               f'file:NaN,cos_phi:0.95,load_type:hp,power:0.077')
    pp.create_sgen(net, bus=1, p_mw=0.005,
                   description=f'plant_type:pv,owner:0,power:0.0051,file:{PV_PROFILE},'
                               f'cos_phi:0.95,orientation:124,angle:60')
    return net


def columns_net():
    """A grid file in the column convention, with a real network's prose in `description`.

    This is what the paper's design 6 network looks like: `load_type` is a real column and
    `description` is the operator's free text — not parseable as `key:value`, and absent
    altogether on some rows.
    """
    net = base_net()
    pp.create_load(net, bus=1, p_mw=0.001, load_type=c.P_INFLEXIBLE_LOAD, agent_type='sfh',
                   owner=-1, file=LOAD_PROFILE, file_add='NaN',
                   description='2022: 17209 kWh')
    pp.create_load(net, bus=1, p_mw=0.0004, load_type=c.P_HP, agent_type='sfh',
                   owner=0, file='NaN', file_add=HP_PROFILE, description=None)
    pp.create_sgen(net, bus=1, p_mw=0.005, plant_type=c.P_PV, owner=0, file=PV_PROFILE,
                   description='Anlagenart: Photovoltaik \n Energieart: Sonne')
    return net


def agent_owning_all_three(agent_id='agent_a', bus=1):
    """An agent whose plant set matches what both minimal networks describe."""
    return make_agent(agent_id, bus=bus, plants={
        'plant_load': {'type': c.P_INFLEXIBLE_LOAD, 'sizing': {'file': LOAD_PROFILE}},
        'plant_hp': {'type': c.P_HP, 'sizing': {'file': HP_PROFILE}},
        'plant_pv': {'type': c.P_PV, 'sizing': {'file': PV_PROFILE}}})


@pytest.mark.parametrize('net_builder, convention', [(packed_net, 'packed into description'),
                                                     (columns_net, 'real columns')])
def test_both_grid_file_conventions_are_read(tmp_path, net_builder, convention):
    """Registration reads the metadata whichever of the two places it lives in.

    The packed arm is #205: it raised `KeyError: 'load_type'` once !216 removed the reader. The
    column arm is #216: it raised inside `add_info_from_col` while the reader was unconditional.
    Neither arm can pass unless the *other* convention is also handled, so one cannot mask the
    other.
    """
    grid_db = write(net_builder(), tmp_path, GRID_FILE)

    grid_db.register_grid(make_regions([agent_owning_all_three()]))

    assert 'load_type' in grid_db.grid.load.columns, (
        f'load metadata was not read from the {convention} convention')
    # Every element of both tables reached the agent: the inflexible load and the heat pump on the
    # load side, the PV on the sgen side. Asserting only on the inflexible load would pass while
    # the owned plants were silently left unassigned.
    assert grid_db.grid.load[c.TC_ID_AGENT].tolist() == ['agent_a', 'agent_a'], (
        f'loads were not all assigned under the {convention} convention')
    assert grid_db.grid.sgen[c.TC_ID_AGENT].tolist() == ['agent_a'], (
        f'the sgen was not assigned under the {convention} convention')
    assert sorted(grid_db.grid.load[c.TC_ID_PLANT]) == ['plant_hp', 'plant_load']
    assert grid_db.grid.sgen[c.TC_ID_PLANT].tolist() == ['plant_pv']


def test_a_grid_file_in_neither_convention_is_rejected_by_name(tmp_path):
    """A file carrying no plant-type information at all fails here, naming both conventions.

    Without this the run dies several frames later on a missing column, which is what made #205
    read as two unrelated defects.
    """
    net = base_net()
    pp.create_load(net, bus=1, p_mw=0.001, description='just some prose')
    grid_db = write(net, tmp_path, GRID_FILE)

    with pytest.raises(ValueError) as excinfo:
        grid_db.register_grid(make_regions([make_agent('agent_a', bus=1, plants={})]))

    message = str(excinfo.value)
    assert 'load_type' in message and 'description' in message and GRID_FILE in message


# --------------------------------------------------------------------------------------------
# The two mismatches
# --------------------------------------------------------------------------------------------

def test_an_agent_with_no_bus_in_the_topology_is_named(tmp_path):
    """#205's first half: `KeyError: '<random id>'` from an unguarded `agents_bus` lookup.

    Reached whenever the scenario's agent ids are not the ones written into the topology file —
    creating the scenario from configs rather than from files redraws them.
    """
    net = base_net()
    net.bus['agent'] = [None, 'agent_in_topology']
    grid_db = write(net, tmp_path, TOPOLOGY_FILE)
    agents = [make_agent('agent_in_topology', bus=1,
                         plants={'p1': {'type': c.P_INFLEXIBLE_LOAD, 'sizing': {}}}),
              make_agent('agent_missing', bus=1,
                         plants={'p2': {'type': c.P_INFLEXIBLE_LOAD, 'sizing': {}}})]

    with pytest.raises(ValueError) as excinfo:
        grid_db.register_grid(make_regions(agents))

    message = str(excinfo.value)
    assert 'agent_missing' in message, 'the error does not name the unassigned agent'
    assert TOPOLOGY_FILE in message, 'the error does not name the file to fix'
    assert 'agent_in_topology' not in message, 'the assigned agent is reported as unassigned'


def test_two_indistinguishable_agents_at_one_bus_each_keep_their_own_load(tmp_path):
    """Two agents the matcher cannot tell apart must not end up sharing one grid element.

    `_create_grid_from_file` iterates agents greedily and mutates `load_df` as it goes, and until
    the candidate filter excluded already-claimed rows, the second agent re-matched the *first*
    agent's inflexible load and overwrote its `id_agent`. The first agent then appeared nowhere in
    the network, `register_grid` returned normally, and `__process_elements` dropped it -- so the
    power flow solved for a feeder missing a participant and reported a loading that was too low.
    Silent, and older than #205: it predates the reader that made this path reachable again.
    """
    net = base_net()
    for _ in range(2):
        pp.create_load(net, bus=1, p_mw=0.001,
                       description=f'agent_type:sfh,owner:NaN,file:{LOAD_PROFILE},'
                                   f'file_add:NaN,load_type:inflexible-load')
    grid_db = write(net, tmp_path, GRID_FILE)
    agents = [make_agent(agent_id, bus=1,
                         plants={f'{agent_id}_load': {'type': c.P_INFLEXIBLE_LOAD,
                                                      'sizing': {'file': LOAD_PROFILE}}})
              for agent_id in ('agent_a', 'agent_b')]

    grid_db.register_grid(make_regions(agents))

    assigned = sorted(grid_db.grid.load[c.TC_ID_AGENT].tolist())
    assert assigned == ['agent_a', 'agent_b'], (
        f'the two agents did not each get their own inflexible load: {assigned}')


def test_an_agent_owning_nothing_electrical_needs_no_bus(tmp_path):
    """A heat-only agent, or the parent of a set of sub-agents, is not required to be in the grid.

    `_create_grid_from_topology` only looks a bus up from inside its plant loop, and only for
    plant types on the electricity side, so such an agent never needed one. The unassigned-agent
    check added for #205 was stricter than the code it guards and rejected scenarios that ran
    before it existed. The empty case is real rather than hypothetical:
    `RegionDB.__register_all_agents` registers a parent of sub-agents via `register_sub_agent`,
    which leaves it with `plants = {}`.
    """
    net = base_net()
    net.bus['agent'] = [None, 'agent_on_the_grid']
    grid_db = write(net, tmp_path, TOPOLOGY_FILE)
    agents = [make_agent('agent_on_the_grid', bus=1,
                         plants={'p1': {'type': c.P_INFLEXIBLE_LOAD, 'sizing': {}}}),
              make_agent('agent_heat_only', bus=1,
                         plants={'p2': {'type': c.P_HEAT_STORAGE, 'sizing': {}}}),
              make_agent('agent_parent_of_sub_agents', bus=1, plants={})]

    grid_db.register_grid(make_regions(agents))

    assert grid_db.grid.load[c.TC_ID_AGENT].tolist() == ['agent_on_the_grid']


def test_an_agent_with_no_matching_inflexible_load_is_named(tmp_path):
    """#201 / #205's second half: `TypeError: cannot unpack non-iterable NoneType object`.

    The agent declares a profile file that no inflexible load in the grid file carries, so the
    match falls through. The old code returned `None` into a tuple unpacking, naming neither the
    agent nor what failed to match.
    """
    grid_db = write(packed_net(), tmp_path, GRID_FILE)
    agent = agent_owning_all_three()
    agent.plants['plant_load']['sizing']['file'] = 'a_profile_the_grid_does_not_have.csv'

    with pytest.raises(ValueError) as excinfo:
        grid_db.register_grid(make_regions([agent]))

    message = str(excinfo.value)
    assert 'agent_a' in message, 'the error does not name the agent'
    assert 'inflexible load' in message.lower()
