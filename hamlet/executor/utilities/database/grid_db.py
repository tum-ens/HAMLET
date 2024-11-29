__author__ = "jiahechu"
__credits__ = ""
__license__ = ""
__maintainer__ = "jiahechu"
__email__ = "jiahe.chu@tum.de"

import os
import pandas as pd
import polars as pl
import pandapower as pp
from hamlet import constants as c
from hamlet import functions as f


class GridDB:
    """
    Database contains all the information for grids.
    This is only a base class, all types of grids DB should inherit from this class. There should be no other direct
    connection from this class to other object.

    Attributes:
        grid_type: types of the grid, currently only electricity grid is implemented.
        grid_path: path where contains all grid information.
        grid_config: dictionary with grid configurations.
        grid: actual grid object, e.g. a pandapower network.
        results: dictionary with stores all grid simulation results.

    """

    def __init__(self, grid_type: str, grid_path: str, grid_config: dict):

        self.grid_type = grid_type      # types of grid (electricity, heat, h2)

        self.grid_path = grid_path      # path with grid files

        self.grid_config = grid_config  # grid configuration dict

        self.grid = None                # grids object will be created when calling register_grid function.

        self.results = {}               # grid simulation results

        self.restriction_commands = {}  # dict for store and exchange grid restriction commands

    def register_grid(self, regions: dict):
        """Assign class attribute from data in grids folder."""
        raise NotImplementedError('The register_grid function must be implemented for each grids type.')

    def save_grid(self, **kwargs):
        """Save the grid results to the given path."""
        raise NotImplementedError('The save_grid function must be implemented for each grids type.')


class ElectricityGridDB(GridDB):
    """
    Database contains all the information for electricity grids.
    Should only be connected with Database class, no connection with main Executor.

    """

    def __init__(self, grid_type: str, grid_path: str, grid_config: dict):

        super().__init__(grid_type, grid_path, grid_config)

        # TODO: the results dict key is pre-defined here
        # simulation results
        self.results = {
            # grid simulation results
            'res_bus': [],
            'res_line': [],
            'res_trafo': [],
            'res_ext_grid': [],
            'res_load': [],
            'res_sgen': [],
        }

    def register_grid(self, regions: dict):
        """
        Create a pandapower network from either a grid topology and agent data or directly from a complete electricity
        grid file.

        Attributes:
            regions: dictionary contains all RegionDBs.

        """

        match self.grid_config['generation']['method']:

            case 'file':        # create pandapower network from a complete electricity grid file
                self.__create_grid_from_file(regions)

            case 'topology':    # create pandapower network from a grid topology and agent data
                self.__create_grid_from_topology(regions)

        # scaling grid capacity for simulating grid restrictions
        # self.grid.line['max_i_ka'] *= 0.015
        # self.grid.trafo['sn_mva'] *= 0.015

        # this is for direct power control, add a column with minimal possible hp power to prevent infeasibility
        self.grid.load['hp_min_control'] = 0

    def save_grid(self, path):
        """
        Save the grid results to the given path.

        Attributes:
            path: path to save the grid results to.

        """
        # save pandapower grid object
        pp.to_excel(self.grid, os.path.join(path, self.grid_config['generation']['file']['file']))

        # save grid simulation results
        for key, data in self.results.items():
            if data:
                file_name = key + '.csv'
                result_df = pd.concat(data)
                result_df.to_csv(os.path.join(path, file_name))

    def __create_grid_from_file(self, regions: dict):
        """
        Create a pandapower network from the given grid file and assign each load and generation to plant and
        agent.

        Attributes:
            regions: dictionary contains all RegionDBs.

        """

        # create pandapower network object from Excel file
        grid_object = pp.from_excel(os.path.join(self.grid_path, self.grid_config['generation']['file']['file']))

        # unpack all agents to {agent_id: AgentDB}
        all_agents = {}
        for region in regions.values():
            for agents in region.agents.values():
                for agent_id, agent in agents.items():
                    all_agents[agent_id] = agent

        # get relevant load dataframe
        load_df = f.add_info_from_col(df=grid_object.load, col='description', drop=False)   # unpack description column
        load_df = load_df.loc[load_df['load_type'] != c.P_HEAT]  # remove heat from electricity grid
        load_df[c.TC_ID_PLANT] = None
        load_plant_types = load_df['load_type'].unique()

        # get relevant sgen dataframe
        sgen_df = f.add_info_from_col(df=grid_object.sgen, col='description', drop=False)   # unpack description column
        sgen_df = sgen_df.loc[sgen_df['plant_type'] != c.P_HEAT_STORAGE]  # remove heat storage from electricity grid
        sgen_df[c.TC_ID_PLANT] = None
        sgen_df['agent_type'] = None
        sgen_plant_types = sgen_df['plant_type'].unique()

        # iterate rows for each agent in load df and plants config file to assign plant id
        for agent_id, agent in all_agents.items():  # iterate agents
            # get the bus of agent
            agent_bus = agent.account[c.K_GENERAL]['bus']

            # get plants config for agent
            plants_config = agent.plants

            # transfer all plants to a dataframe
            plants_df = {}
            for plant_id, plant in plants_config.items():
                if plant['type'] == c.P_HEAT or plant['type'] == c.P_HEAT_STORAGE:
                    # skip heat since it's not relevant for electricity grid
                    continue

                if 'file' in plant['sizing'].keys():
                    plants_df[plant_id] = {'type': plant['type'], 'file': plant['sizing']['file'], 'bus': agent_bus,
                                           'load_index': None}
                else:
                    plants_df[plant_id] = {'type': plant['type'], 'file': None, 'bus': agent_bus, 'load_index': None}

            plants_df = pd.DataFrame.from_dict(plants_df, orient='index')
            plants_df.index.name = c.TC_ID_PLANT
            plants_df.reset_index(inplace=True)

            # count numbers for each plant for this agent
            agent_load_plants_count = {}  # load plants
            for load_type in load_plant_types:
                agent_load_plants_count[load_type] = len(plants_df[plants_df['type'] == load_type])
            del agent_load_plants_count[c.P_INFLEXIBLE_LOAD]    # ignore inflexible load since its always 1

            agent_sgen_plants_count = {}  # sgen plants
            for sgen_type in sgen_plant_types:
                agent_sgen_plants_count[sgen_type] = len(plants_df[plants_df['type'] == sgen_type])

            # find out which inflexible load is for this agent by comparing the number of each plant
            inflexible_load_for_agent_index = load_df[(load_df['bus'] == agent_bus) &
                                                      (load_df['load_type'] == c.P_INFLEXIBLE_LOAD) &
                                                      (load_df['agent_type'] == agent.agent_type)].index

            # count numbers for each plant for this inflexible load by iterating all inflexible loads at this bus,
            # compare counts with agent to find out which inflexible load is the right one
            for inflexible_load_index in inflexible_load_for_agent_index:
                owned_load_plants_df = load_df[load_df['owner'] == inflexible_load_index]
                owned_sgen_plants_df = sgen_df[sgen_df['owner'] == inflexible_load_index]

                try:
                    for load_type in agent_load_plants_count.keys():  # compare load counts
                        if agent_load_plants_count[load_type] != len(owned_load_plants_df[owned_load_plants_df
                                                                                          ['load_type'] == load_type]):
                            raise StopIteration
                    for sgen_type in agent_sgen_plants_count.keys():  # compare sgen counts
                        if agent_sgen_plants_count[sgen_type] != len(owned_sgen_plants_df[owned_sgen_plants_df
                                                                                          ['plant_type'] == sgen_type]):
                            raise StopIteration

                    for load_index in owned_load_plants_df.index:  # compare if all file names match for load
                        load_type = owned_load_plants_df.loc[load_index, 'load_type']
                        load_plant_df = plants_df[plants_df['type'] == load_type]

                        if not load_plant_df['file'].str.contains(str(owned_load_plants_df.loc[load_index, 'file']))\
                                .any() and not load_plant_df['file'].str.contains(str(owned_load_plants_df
                                                                                      .loc[load_index, 'file_add']))\
                                .any():
                            raise StopIteration

                    for sgen_index in owned_sgen_plants_df.index:  # compare if all file names match for sgen
                        sgen_type = owned_sgen_plants_df.loc[sgen_index, 'plant_type']
                        sgen_plant_df = plants_df[plants_df['type'] == sgen_type]
                        if not sgen_plant_df['file'].str.contains(str(owned_sgen_plants_df.loc[sgen_index, 'file']))\
                                .any() and sgen_type != c.P_BATTERY:
                            raise StopIteration

                except StopIteration:   # in this case this inflexible load is not the match, move to the next
                    continue

                # no StopIteration is raised, this inflexible load fulfill all requirements, assign plant id
                for agent_index in plants_df[plants_df['type'] == c.P_INFLEXIBLE_LOAD].index:  # assign inflexible load
                    if plants_df.loc[agent_index, 'file'] == load_df.loc[inflexible_load_index, 'file']:
                        load_df.loc[inflexible_load_index, c.TC_ID_PLANT] = plants_df.loc[agent_index, c.TC_ID_PLANT]
                        load_df.loc[inflexible_load_index, c.TC_ID_AGENT] = agent_id
                        plants_df.drop(axis='index', index=agent_index, inplace=True)
                        break   # finished assigning inflexible load, further inflexible load won't be considered

                for grid_index in owned_load_plants_df.index:  # assign other load plants
                    load_type = owned_load_plants_df.loc[grid_index, 'load_type']
                    for agent_index in plants_df[plants_df['type'] == load_type].index:
                        # here compare both plant type and plant file name
                        if plants_df.loc[agent_index, 'file'] == owned_load_plants_df.loc[grid_index, 'file'] or \
                                plants_df.loc[agent_index, 'file'] == owned_load_plants_df.loc[grid_index, 'file_add']:
                            load_df.loc[grid_index, c.TC_ID_PLANT] = plants_df.loc[agent_index, c.TC_ID_PLANT]
                            load_df.loc[grid_index, c.TC_ID_AGENT] = agent_id
                            plants_df.drop(axis='index', index=agent_index, inplace=True)
                            break   # finished assigning this plant, further plant won't be considered

                for grid_index in owned_sgen_plants_df.index:  # assign other sgen plants
                    sgen_type = owned_sgen_plants_df.loc[grid_index, 'plant_type']
                    for agent_index in plants_df[plants_df['type'] == sgen_type].index:
                        # here compare both plant type and plant file name
                        if plants_df.loc[agent_index, 'file'] == owned_sgen_plants_df.loc[grid_index, 'file'] or \
                                sgen_type == c.P_BATTERY:
                            sgen_df.loc[grid_index, c.TC_ID_PLANT] = plants_df.loc[agent_index, c.TC_ID_PLANT]
                            sgen_df.loc[grid_index, c.TC_ID_AGENT] = agent_id
                            sgen_df.loc[grid_index, 'agent_type'] = agent.agent_type
                            plants_df.drop(axis='index', index=agent_index, inplace=True)
                            break   # finished assigning this plant, further plant won't be considered

                break   # finished assigning this agent, further inflexible load at the same bus won't be considered

            # is there any other plants un-assigned?
            if len(plants_df) > 0:
                for index in plants_df.index:
                    plant_type = plants_df.loc[index, 'type']
                    plant_file = plants_df.loc[index, 'file']

                    # find remaining unassigned plants in the grid file
                    available_load = load_df.loc[load_df['load_type'] == plant_type].loc[load_df['bus'] == agent_bus] \
                                            .loc[load_df[c.TC_ID_PLANT] == None]
                    available_sgen = sgen_df.loc[sgen_df['plant_type'] == plant_type].loc[sgen_df['bus'] == agent_bus] \
                                            .loc[sgen_df[c.TC_ID_PLANT] == None]

                    try:
                        for grid_index in available_load.index:
                            # since file name is checked before, here also check if file_add name matches
                            if plant_file == available_load.loc[grid_index, 'file'] or \
                                    plant_file == available_load.loc[grid_index, 'file_add']:
                                load_df.loc[grid_index, c.TC_ID_PLANT] = plants_df.loc[index, c.TC_ID_PLANT]
                                load_df.loc[grid_index, c.TC_ID_AGENT] = agent_id

                                # found a matched load, directly jump to the next unassigned plant
                                raise StopIteration

                        for grid_index in available_sgen.index:
                            if plant_file == available_sgen.loc[grid_index, 'file']:
                                sgen_df.loc[grid_index, c.TC_ID_PLANT] = plants_df.loc[index, c.TC_ID_PLANT]
                                sgen_df.loc[grid_index, c.TC_ID_AGENT] = agent_id
                                sgen_df.loc[grid_index, 'agent_type'] = agent.agent_type

                                # found a matched load, directly jump to the next unassigned plant
                                raise StopIteration

                    except StopIteration:
                        continue    # move to the next unassigned plant

                    # no StopIteration is raised, means no match is found, print information
                    print('Grid file not consistent with scenario config, solution needs to be discussed.')

        # add region to load and sgen df
        bus_df = grid_object.bus
        bus_df.index.name = 'bus'
        bus_df.reset_index(inplace=True)
        bus_df = bus_df[['bus', 'zone']]

        load_df = load_df.merge(bus_df, on='bus', how='left')
        sgen_df = sgen_df.merge(bus_df, on='bus', how='left')

        # set init value of p_mw and q_mw to 0
        load_df['p_mw'] = 0
        load_df['q_mvar'] = 0
        sgen_df['p_mw'] = 0
        sgen_df['q_mvar'] = 0

        # write data back to grid
        grid_object.load = load_df
        grid_object.sgen = sgen_df

        # re-index bus
        grid_object.bus.index = grid_object.bus['bus']

        # write grid object to class
        self.grid = grid_object

    def __create_grid_from_topology(self, regions: dict):
        """
        Create a pandapower network from the topology Excel file and add network elements at each bus corresponding
        to agents at that bus.

        Attributes:
            regions: dictionary contains all RegionDBs.

        """
        # TODO: the category of each plant type is pre-defined
        plants_load = [c.P_INFLEXIBLE_LOAD, c.P_FLEXIBLE_LOAD, c.P_EV, c.P_HP]
        plants_sgen = [c.P_PV, c.P_WIND, c.P_BATTERY]
        plants_storage = []

        # create pandapower network from Excel file
        grid_object = pp.from_excel(os.path.join(self.grid_path, self.grid_config['topology']['file']))

        # get agents at each bus
        agents_bus = {}     # {agent_id: bus_id}
        for column in grid_object.bus.columns:
            if 'agent' in column:
                for index in grid_object.bus.index:
                    if grid_object.bus.loc[index, column] is not None:
                        agent_id = grid_object.bus.loc[index, column]
                        agents_bus[agent_id] = index

        # add grid elements according to agent plants
        for region_name, region in regions.items():
            for agent_type, agents in region.agents.items():
                for agent_id, agent in agents.items():
                    for plant_id, plant in agent.plants.items():
                        # generate a kwarg dict for writing data at each column
                        kw_args = {c.TC_ID_PLANT: plant_id, c.TC_ID_AGENT: agent_id, 'agent_type': agent_type,
                                   'zone': region_name, 'cos_phi': 0}

                        # generate grid component
                        if plant['type'] in plants_load:  # TODO: 'load_type', 'plant_type' hardcoded, not clean enough
                            pp.create_load(grid_object, bus=agents_bus[agent_id], p_mw=0, load_type=plant['type'],
                                           **kw_args)
                        elif plant['type'] in plants_sgen:
                            pp.create_sgen(grid_object, bus=agents_bus[agent_id], p_mw=0, plant_type=plant['type'],
                                           **kw_args)
                        elif plant['type'] in plants_storage:
                            capacity = plant['sizing']['capacity'] * c.WH_TO_MWH
                            pp.create_storage(grid_object, bus=agents_bus[agent_id], p_mw=0, max_e_mwh=capacity,
                                              **kw_args)

        # write data back to the grid
        grid_object.load = grid_object.load.dropna(subset=[c.TC_ID_AGENT])
        grid_object.sgen = grid_object.sgen.dropna(subset=[c.TC_ID_AGENT])

        # write grid object to class attributes
        self.grid = grid_object


class HeatGridDB(GridDB):
    def __init__(self, grid_type: str, grid_path: str, grid_config: dict):
        super().__init__(grid_type, grid_path, grid_config)

    def register_grid(self, regions: dict):
        raise NotImplementedError('The heat grids is not implemented yet.')


class HydrogenGridDB(GridDB):
    def __init__(self, grid_type: str, grid_path: str, grid_config: dict):
        super().__init__(grid_type, grid_path, grid_config)

    def register_grid(self, regions: dict):
        raise NotImplementedError('The hydrogen grids is not implemented yet.')
