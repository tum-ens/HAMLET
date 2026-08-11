"""Unit -- a grid imported from a network operator can be read.

`ElectricityGridDB.__get_grid_element_dataframe` used to unpack pandapower's `description` column
as `key:value,key:value` before filtering. That column is not HAMLET's to read: nothing in HAMLET
writes it -- `_create_grid_from_topology` passes `plant_id`, `agent_id`, `agent_type`, `zone` and
`load_type`/`plant_type` as real pandapower columns -- and in an imported network it holds
free-form text.

Measured on the paper's design 6 grid, which is the first real network anything here has been run
against:

- **96 of 469 loads and 134 of 263 sgens carry no description**, giving
  `AttributeError: 'NoneType' object has no attribute 'split'`.
- Of those that do, most are prose containing colons -- `'Anlagenart: Photovoltaik \\n Energieart:
  Sonne \\n Baujahr: 2022 \\n ...'`, `'2022: 17209 kWh'` -- giving
  `ValueError: too many values to unpack (expected 2)`.

Both fixtures below are taken from that data. The second matters more than the first: a
description that happens to parse would have had its invented columns joined on and written back
into `self.grid.load`, so *succeeding* would have been worse than crashing.

Nothing in the suite caught this because both shipped grid examples fail earlier for unrelated
reasons, so no test has ever reached this code with a network in it.
"""
import pandapower as pp
import pytest

import hamlet.constants as c
from hamlet.executor.utilities.database.grid_db import ElectricityGridDB

#: Verbatim shapes from the design 6 workbook.
PROSE = ('Anlagenart: Photovoltaik \n Energieart: Sonne \n Baujahr: 2022 \n Q-Modellierung: \n '
         'Bis 2011 -> cosphi = 1')
SEMICOLONS = 'hvac: COP_ASHP_radiator_40;dhw: COP_ASHP_water_50'
CONSUMPTION = '2022: 17209 kWh'


def network(descriptions):
    """A two-bus network whose loads carry `descriptions`, one per load."""
    net = pp.create_empty_network()
    first = pp.create_bus(net, vn_kv=0.4, zone='region')
    second = pp.create_bus(net, vn_kv=0.4, zone='region')
    pp.create_ext_grid(net, bus=first)
    pp.create_line(net, from_bus=first, to_bus=second, length_km=0.1, std_type='NAYY 4x50 SE')

    for index, description in enumerate(descriptions):
        pp.create_load(net, bus=second, p_mw=0.001, load_type=c.P_INFLEXIBLE_LOAD,
                       description=description, name=f'load_{index}')
    return net


@pytest.fixture
def grid_db():
    db = ElectricityGridDB.__new__(ElectricityGridDB)
    db.energy_type = c.ET_ELECTRICITY
    db.relevant_plant_type = db.filter_energy_types()
    db.relevant_plant_type[c.OM_STORAGE].remove(c.P_EV)
    db.relevant_plant_type[c.OM_LOAD].append(c.P_EV)
    db.relevant_plant_type['sgen'] = db.relevant_plant_type.pop(c.OM_GENERATION)
    db.relevant_plant_type['sgen'].extend(db.relevant_plant_type[c.OM_STORAGE])
    return db


def elements(db, net):
    """The private method under test, reached the way `_create_grid_from_file` reaches it."""
    db.grid = net
    return db._ElectricityGridDB__get_grid_element_dataframe(
        element_name='load', type_field='load_type', add_columns=[c.TC_ID_PLANT, 'p_mw', 'q_mvar'])


@pytest.mark.parametrize('description, label', [
    (None, 'no description at all'),
    (PROSE, 'prose with colons and newlines'),
    (SEMICOLONS, 'semicolon-separated, not comma'),
    (CONSUMPTION, 'a year and a quantity'),
])
def test_a_real_description_does_not_stop_the_grid_being_read(grid_db, description, label):
    """Each of these raised before the fix -- the first with AttributeError, the rest ValueError."""
    result = elements(grid_db, network([description]))

    assert len(result) == 1, label


def test_a_mixed_network_reads_every_row(grid_db):
    """The real shape: some rows described, some not, none of it HAMLET's format."""
    result = elements(grid_db, network([None, PROSE, SEMICOLONS, CONSUMPTION, None]))

    assert len(result) == 5


def test_no_columns_are_invented_from_the_description(grid_db):
    """The half that would have been silent. `Anlagenart` is not a HAMLET field.

    A parse that succeeded would have joined these on and written them back into `grid.load`.
    """
    result = elements(grid_db, network([PROSE, CONSUMPTION]))

    for invented in ('Anlagenart', 'Energieart', 'Baujahr', '2022', 'hvac', 'dhw'):
        assert invented not in result.columns


def test_the_real_columns_still_arrive(grid_db):
    """What the method is actually for, pinned so the fix cannot be bought at its expense."""
    result = elements(grid_db, network([None, PROSE]))

    assert result['load_type'].tolist() == [c.P_INFLEXIBLE_LOAD] * 2
    assert result['zone'].tolist() == ['region'] * 2
    assert result[c.TC_ID_PLANT].tolist() == [0, 0]


def test_irrelevant_plant_types_are_still_filtered_out(grid_db):
    """The filter runs on `load_type`, a real column, and is unaffected by any of this."""
    net = network([None])
    pp.create_load(net, bus=1, p_mw=0.001, load_type='not-a-hamlet-plant', description=PROSE)

    assert len(elements(grid_db, net)) == 1
