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

        self.grid_type = grid_type  # type of grid (electricity, heat, h2)

        self.grid_path = grid_path  # path with grid files

        self.grid_config = grid_config  # grid configuration dict

        self.grid = None  # grids object will be created when calling register_grid function.

        self.results = {}  # grid simulation results

        self.restriction_commands = {}  # dict for store and exchange grid restriction commands

        self.energy_type = None  # energy type of the grid

        self.relevant_plant_type = {}  # relevant plant type for the grid

    def register_grid(self, regions: dict):
        """Assign class attribute from data in grids folder."""
        raise NotImplementedError('The register_grid function must be implemented for each grids type.')

    def save_grid(self, **kwargs):
        """Save the grid results to the given path."""
        raise NotImplementedError('The save_grid function must be implemented for each grids type.')

    def filter_energy_types(self) -> dict:
        """
        Filter all relevant plant type for the grid by mapping the energy type. E.g. heat is not relevant for the
        electricity grid, battery is not relevant for the heat grid.

        Returns:
            relevant_plant_type: dict containing all relevant plant types and their operation mode.
        """
        relevant_plant_type = {c.OM_LOAD: [], c.OM_GENERATION: [], c.OM_STORAGE: []}
        for plant_type in c.COMP_MAP.keys():
            if self.energy_type in c.COMP_MAP[plant_type].keys():
                relevant_plant_type[c.COMP_MAP[plant_type][self.energy_type]].append(plant_type)

        return relevant_plant_type


class ElectricityGridDB(GridDB):
    """
    Database contains all the information for electricity grids.
    Should only be connected with Database class, no connection with main Executor.

    """

    def __init__(self, grid_type: str, grid_path: str, grid_config: dict):

        super().__init__(grid_type, grid_path, grid_config)

        self.energy_type = c.ET_ELECTRICITY

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
        # read pandapower network object from Excel file
        self.grid = pp.from_excel(os.path.join(self.grid_path, self.grid_config['generation']
        [self.grid_config['generation']['method']]['file']))

        # assign relevant plant type
        self.relevant_plant_type = self.filter_energy_types()
        self.relevant_plant_type[c.OM_STORAGE].remove(c.P_EV)  # remove ev from storage
        self.relevant_plant_type[c.OM_LOAD].append(c.P_EV)  # add ev to load
        self.relevant_plant_type['sgen'] = self.relevant_plant_type.pop(c.OM_GENERATION)  # rename generation
        self.relevant_plant_type['sgen'].extend(self.relevant_plant_type[c.OM_STORAGE])  # combine sgen and storage

        # create detailed grid components
        match self.grid_config['generation']['method']:

            case 'file':  # create pandapower network from a complete electricity grid file
                self._create_grid_from_file(regions)

            case 'topology':  # create pandapower network from a grid topology and agent data
                self._create_grid_from_topology(regions)

            case _:
                raise ValueError(f"Unknown grid creation method: {self.grid_config['method']}")

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

    def _create_grid_from_file(self, regions: dict):
        """
        Create a pandapower network from the given grid file and assign each load and generation to plant and
        agent.

        Attributes:
            regions: dictionary contains all RegionDBs.

        """
        # unpack all agents to {agent_id: AgentDB}
        all_agents = {}
        for region in regions.values():
            for agents in region.agents.values():
                for agent_id, agent in agents.items():
                    all_agents[agent_id] = agent

        # get relevant load and sgen dataframe
        load_df = self.__get_grid_element_dataframe(element_name='load', type_field='load_type',
                                                    add_columns=[c.TC_ID_PLANT, 'p_mw', 'q_mvar'])
        sgen_df = self.__get_grid_element_dataframe(element_name='sgen', type_field='plant_type',
                                                    add_columns=[c.TC_ID_PLANT, 'agent_type', 'p_mw', 'q_mvar'])

        # iterate rows for each agent in load df and plants config file to assign plant id
        for agent_id, agent in all_agents.items():  # iterate agents
            # get the bus of agent
            agent_bus = agent.account[c.K_GENERAL]['bus']

            # get plants config for agent
            plants_config = agent.plants

            # transfer all plants to a dataframe
            plants_df, agent_plants_count = self.__get_count_plants_for_agent(plants_config, agent_bus)

            # find out which inflexible load is for this agent and assign it in load_df
            load_df, inflexible_load_index = self.__assign_inflexible_load_for_agent(load_df=load_df,
                                                                                     sgen_df=sgen_df, agent=agent,
                                                                                     plants_df=plants_df,
                                                                                     agent_plants_count=
                                                                                     agent_plants_count)

            # assign other plants
            load_df, plants_df = self.__assign_plants_for_agent(element_df=load_df, plants_df=plants_df,
                                                                inflexible_load_index=inflexible_load_index,
                                                                agent=agent, type_field='load_type')
            sgen_df, plants_df = self.__assign_plants_for_agent(element_df=sgen_df, plants_df=plants_df,
                                                                inflexible_load_index=inflexible_load_index,
                                                                agent=agent, type_field='plant_type')

            # is there any other plants un-assigned?
            if len(plants_df) > 0:
                for index in plants_df.index:
                    if self.__find_remaining_unassigned_plants(element_df=load_df, plants_df=plants_df,
                                                               plants_df_index=index, agent=agent,
                                                               type_field='load_type'):
                        load_df = self.__find_remaining_unassigned_plants(element_df=load_df, plants_df=plants_df,
                                                                          plants_df_index=index, agent=agent,
                                                                          type_field='load_type')
                    elif self.__find_remaining_unassigned_plants(element_df=sgen_df, plants_df=plants_df,
                                                                 plants_df_index=index, agent=agent,
                                                                 type_field='plant_type'):
                        sgen_df = self.__find_remaining_unassigned_plants(element_df=sgen_df, plants_df=plants_df,
                                                                          plants_df_index=index, agent=agent,
                                                                          type_field='plant_type')
                    else:
                        # no StopIteration is raised, means no match is found, print information
                        print('Grid file not consistent with scenario config, solution needs to be discussed.')

        # write data back to grid
        self.grid.load = load_df
        self.grid.sgen = sgen_df

    def _create_grid_from_topology(self, regions: dict):
        """
        Create a pandapower network from the topology Excel file and add network elements at each bus corresponding
        to agents at that bus.

        Attributes:
            regions: dictionary contains all RegionDBs.

        """
        # get agents at each bus
        agents_bus = {}  # {agent_id: bus_id}
        for column in self.grid.bus.columns:
            if 'agent' in column:
                for index in self.grid.bus.index:
                    if self.grid.bus.loc[index, column] is not None:
                        agent_id = self.grid.bus.loc[index, column]
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
                        if plant['type'] in self.relevant_plant_type['load']:  # create load
                            pp.create_load(self.grid, bus=agents_bus[agent_id], p_mw=0, load_type=plant['type'],
                                           **kw_args)
                        elif plant['type'] in self.relevant_plant_type['sgen']:
                            pp.create_sgen(self.grid, bus=agents_bus[agent_id], p_mw=0, plant_type=plant['type'],
                                           **kw_args)

        # write data back to the grid
        self.grid.load.dropna(subset=[c.TC_ID_AGENT], inplace=True)
        self.grid.sgen.dropna(subset=[c.TC_ID_AGENT], inplace=True)

    def __get_grid_element_dataframe(self, element_name: str, type_field: str, add_columns: list) -> (
    pd.DataFrame, list):
        """
        Get dataframe and pre-process it for the given grid element.

        Args:
            element_name (string): name of grid element (normally load or sgen).
            type_field (string): name of the column containing plant type (currently 'load_type' for load and '
            plant_type' for sgen).
            add_columns (list): list of additional column names to-be initialized.

        Returns:
            df (pd.DataFrame): grid element dataframe.
            df_plant_types (list): list of plant type names in the dataframe.
        """
        # get dataframe and unpack description column
        df = f.add_info_from_col(df=getattr(self.grid, element_name).copy(), col='description', drop=False)
        df = df.loc[df[type_field].isin(self.relevant_plant_type[element_name])]  # remove unnecessary plants from grid
        df[add_columns] = 0  # add additional columns

        # add region to df
        bus_df = self.grid.bus.copy()
        bus_df.index.name = 'bus'
        bus_df.reset_index(inplace=True)
        bus_df = bus_df[['bus', 'zone']]
        df = df.reset_index().merge(bus_df, how="left").set_index('index')

        return df

    def __get_count_plants_for_agent(self, plants_config: dict, agent_bus: int) -> (pd.DataFrame, dict):
        """
        Get all agent plants as Dataframe and count number of each plant type for the given agent.

        Extract and count the plants associated with a given agent.

        Args:
            plants_config (dict): The configuration of plants for the agent, with plant IDs as keys and their details as
            values.
            agent_bus (int): The bus ID associated with the agent.

        Returns:
            plants_df (pd.DataFrame): A DataFrame containing details of the agent's plants, including plant type, file,
            and bus.
            agent_plants_count (dict): A nested dictionary with counts of plants by type and category (Format:
            {element_type: {plant_type: count}}).

        """
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
        agent_plants_count = {}
        for element_type in self.relevant_plant_type.keys():
            agent_plants_count[element_type] = {}
            for plant_type in self.relevant_plant_type[element_type]:
                agent_plants_count[element_type][plant_type] = len(plants_df[plants_df['type'] == plant_type])
        del agent_plants_count['load'][c.P_INFLEXIBLE_LOAD]  # ignore inflexible load since its always 1

        return plants_df, agent_plants_count

    def __assign_inflexible_load_for_agent(self, load_df: pd.DataFrame, sgen_df: pd.DataFrame, agent,
                                           plants_df: pd.DataFrame, agent_plants_count: dict) -> (pd.DataFrame, int):
        """
        Assign the appropriate inflexible load to an agent based on plant configuration.

        This function identifies the correct inflexible load for the given agent by comparing the
        plant type counts and file details between the agent's plant configuration and the grid's
        load and sgen data. The matched inflexible load is then assigned to the agent.

        Args:
            load_df (pd.DataFrame): DataFrame containing load elements from the grid.
            sgen_df (pd.DataFrame): DataFrame containing generation (sgen) elements from the grid.
            agent: The agent object whose inflexible load is being assigned.
            plants_df (pd.DataFrame): DataFrame of the agent's plants, including plant details.
            agent_plants_count (dict): Dictionary containing counts of plants by type for the agent.

        Returns:
            load_df (pd.DataFrame): Updated `load_df` with the assigned inflexible load's plant ID and agent ID.
            inflexible_load_index (int): Index of the matching inflexible load.

        """
        inflexible_load_for_agent_index = load_df[(load_df['bus'] == agent.account[c.K_GENERAL]['bus']) &
                                                  (load_df['load_type'] == c.P_INFLEXIBLE_LOAD) &
                                                  (load_df['agent_type'] == agent.agent_type)].index

        # count numbers for each plant for this inflexible load by iterating all inflexible loads at this bus,
        # compare counts with agent to find out which inflexible load is the right one
        for inflexible_load_index in inflexible_load_for_agent_index:
            try:
                self.__check_inflexible_load_ownership(element_df=load_df, plants_df=plants_df,
                                                       inflexible_load_index=inflexible_load_index,
                                                       element_name='load', type_field='load_type',
                                                       agent_plants_count=agent_plants_count)

                self.__check_inflexible_load_ownership(element_df=sgen_df, plants_df=plants_df,
                                                       inflexible_load_index=inflexible_load_index,
                                                       element_name='sgen', type_field='plant_type',
                                                       agent_plants_count=agent_plants_count)

            except StopIteration:  # in this case this inflexible load is not the match, move to the next
                continue

            # no StopIteration is raised, this inflexible load fulfill all requirements, assign plant id
            for agent_index in plants_df[plants_df['type'] == c.P_INFLEXIBLE_LOAD].index:  # assign inflexible load
                if plants_df.loc[agent_index, 'file'] == load_df.loc[inflexible_load_index, 'file']:
                    load_df.loc[inflexible_load_index, c.TC_ID_PLANT] = plants_df.loc[agent_index, c.TC_ID_PLANT]
                    load_df.loc[inflexible_load_index, c.TC_ID_AGENT] = agent.agent_id
                    plants_df.drop(axis='index', index=agent_index, inplace=True)

                    # finished assigning inflexible load, further inflexible load won't be considered
                    return load_df, inflexible_load_index

    def __assign_plants_for_agent(self, element_df: pd.DataFrame, plants_df: pd.DataFrame, inflexible_load_index: int,
                                  agent, type_field: str):
        """
        Assign plants from the grid to an agent based on plant type and file details.

        This function iterates through the plants associated with an inflexible load index and assigns
        them to the agent by matching plant types and file names between the agent's configuration
        and the grid's elements.

        Args:
            element_df (pd.DataFrame): DataFrame representing grid elements (e.g., loads or sgen).
            plants_df (pd.DataFrame): DataFrame containing plants for the agent.
            inflexible_load_index (int): Index of the inflexible load used to identify owned plants.
            agent: The agent object to whom the plants are being assigned.
            type_field (str): The column name representing the type of the element (e.g., 'load_type', 'plant_type').

        Returns:
            element_df (pd.DataFrame): Updated `element_df` with assigned plant IDs and agent IDs.
            plants_df (pd.DataFrame): Updated `plants_df` with unassigned plants after allocation.

        """
        owned_plants_df = element_df[element_df['owner'] == inflexible_load_index]

        for grid_index in owned_plants_df.index:  # assign other plants
            plant_type = owned_plants_df.loc[grid_index, type_field]
            for agent_index in plants_df[plants_df['type'] == plant_type].index:
                # here compare both plant type and plant file name
                if (plants_df.loc[agent_index, 'file'] == owned_plants_df.loc[grid_index, 'file'] or (
                        'file_add' in owned_plants_df.columns and plants_df.loc[agent_index, 'file'] ==
                        owned_plants_df.loc[grid_index, 'file_add'])
                        or plant_type in self.relevant_plant_type[c.OM_STORAGE]):
                    element_df.loc[grid_index, c.TC_ID_PLANT] = plants_df.loc[agent_index, c.TC_ID_PLANT]
                    element_df.loc[grid_index, c.TC_ID_AGENT] = agent.agent_id
                    element_df.loc[grid_index, 'agent_type'] = agent.agent_type
                    plants_df.drop(axis='index', index=agent_index, inplace=True)
                    break  # finished assigning this plant, further plant won't be considered

        return element_df, plants_df

    @staticmethod
    def __find_remaining_unassigned_plants(element_df: pd.DataFrame, plants_df: pd.DataFrame,
                                           plants_df_index: int, agent, type_field: str):
        """
        Assign a remaining unassigned plant to an agent by matching type and file.

        This function searches for unassigned plants in the `element_df` that match the type and file
        of the plant at the given `plants_df_index` in `plants_df`. If a match is found, the plant is
        assigned to the agent, and the updated `element_df` is returned.

        Args:
            element_df (pd.DataFrame): DataFrame representing grid elements (e.g., loads or sgen).
            plants_df (pd.DataFrame): DataFrame containing plants for the agent.
            plants_df_index (int): Index of the plant in `plants_df` to find a match for.
            agent: The agent object to whom the plant is being assigned.
            type_field (str): The column name representing the type of the element (e.g., 'load_type', 'plant_type').

        Returns:
            pd.DataFrame: Updated `element_df` with the assigned plant.
        """
        plant_type = plants_df.loc[plants_df_index, 'type']
        plant_file = plants_df.loc[plants_df_index, 'file']

        available_plant = element_df.loc[element_df[type_field] == plant_type] \
            .loc[element_df['bus'] == agent.account[c.K_GENERAL]['bus']] \
            .loc[element_df[c.TC_ID_PLANT] == None]

        for grid_index in available_plant.index:
            # since file name is checked before, here also check if file_add name matches
            if plant_file == available_plant.loc[grid_index, 'file'] or ('file_add' in available_plant.columns and
                                                                         plant_file == available_plant.loc[
                                                                             grid_index, 'file_add']):
                element_df.loc[grid_index, c.TC_ID_PLANT] = plants_df.loc[plants_df_index, c.TC_ID_PLANT]
                element_df.loc[grid_index, c.TC_ID_AGENT] = agent.agent_id
                element_df.loc[grid_index, 'agent_type'] = agent.agent_type

                # found a matched load, directly jump to the next unassigned plant
                return element_df

    def __check_inflexible_load_ownership(self, element_df: pd.DataFrame, plants_df: pd.DataFrame,
                                          inflexible_load_index: int, element_name: str, type_field: str,
                                          agent_plants_count: dict):
        """
        Verify if an inflexible load is owned by an agent based on plant type and file match criteria.

        This function checks if the count of plants of each type and the associated file names
        match the agent's expected configuration for the specified inflexible load index. If the
        criteria are not met, a `StopIteration` exception is raised.

        Args:
            element_df (pd.DataFrame): DataFrame representing grid elements (e.g., loads or sgen).
            plants_df (pd.DataFrame): DataFrame containing plants for the agent.
            inflexible_load_index (int): The index of the inflexible load to verify.
            element_name (str): The name of the element being processed ('load' or 'sgen').
            type_field (str): The column name representing the type of the element (e.g., 'load_type', 'plant_type').
            agent_plants_count (dict): A dictionary with the count of plants by type for the agent.

        Raises:
            StopIteration: If the inflexible load does not match the agent's plant configuration.
        """

        agent_plants_count = agent_plants_count[element_name]
        owned_plants_df = element_df[element_df['owner'] == inflexible_load_index]

        # compare plant count
        for plant_type in agent_plants_count.keys():
            if agent_plants_count[plant_type] != len(owned_plants_df[owned_plants_df[type_field] == plant_type]):
                raise StopIteration

        # compare if all file names match
        for plant_index in owned_plants_df.index:
            plant_type = owned_plants_df.loc[plant_index, type_field]
            current_plant_df = plants_df[plants_df['type'] == plant_type]

            if not current_plant_df['file'].str.contains(str(owned_plants_df.loc[plant_index, 'file'])).any():
                if (('file_add' in owned_plants_df.columns and
                     current_plant_df['file'].str.contains(str(owned_plants_df.loc[plant_index, 'file_add'])).any())
                        or plant_type in self.relevant_plant_type[c.OM_STORAGE]):
                    continue
                else:
                    raise StopIteration


class HeatGridDB(GridDB):
    def __init__(self, grid_type: str, grid_path: str, grid_config: dict):
        super().__init__(grid_type, grid_path, grid_config)

    def register_grid(self, regions: dict):
        raise NotImplementedError('The heat grids are not implemented yet.')


class HydrogenGridDB(GridDB):
    def __init__(self, grid_type: str, grid_path: str, grid_config: dict):
        super().__init__(grid_type, grid_path, grid_config)

    def register_grid(self, regions: dict):
        raise NotImplementedError('The hydrogen grids are not implemented yet.')
