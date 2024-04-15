__author__ = "TUM-Doepfert"
__credits__ = "jiahechu"
__license__ = ""
__maintainer__ = "TUM-Doepfert"
__email__ = "markus.doepfert@tum.de"

from hamlet.creator.agents.agents import Agents
import os
import pandas as pd
import numpy as np
from ruamel.yaml.compat import ordereddict
import random
from pprint import pprint
import math
import hamlet.constants as c


class Aggregator(Agents):
    """
        Sets up aggregator agents. Inherits from Agents class.

        Mainly used for Excel file creation. Afterwards Aggregator class creates the individual agents.
    """

    def __init__(self, input_path: str, config: ordereddict, config_path: str, scenario_path: str, config_root: str):

        # Call the init method of the parent class
        super().__init__(config_path, input_path, scenario_path, config_root)

        # Define agent type
        self.type = c.A_AGGREGATOR


        # Config file
        self.config = config["types"]

        # Grid information (if applicable)
        self.grid = None
        self.bus = None  # bus sheet containing only the bus information of the agent type
        self.load = None  # load sheet containing only the load information of the agent type
        self.agents = None  # load sheet but limited to all agents, i.e. all inflexible_loads
        self.sgen = None  # sgen sheet containing only the sgen information of the agent type

        # Creation method
        self.method = None

        # Number of agents
        self.num = 0

        # Dataframe containing all information
        self.df = None

        # Misc
        self.n_digits = 2  # number of digits values get rounded to in respective value column

    def create_df_from_config(self, dict_agents: dict) -> pd.DataFrame:
        """Function to create the dataframe that makes the Excel sheet
        """

        # set the method
        self.method = 'config'

        for aggregator_type in self.config:

            self.num += self.config[f"{aggregator_type}"]["general"]["number_of_type"]

        if self.num == 0:
            return self.df

        # create a dict for all active agent_devices
        dict_agent_devices = self.create_agents_devices_dict(dict_agents=dict_agents)

        # Create the overall dataframe structure for the worksheet
        self.create_df_structure(dict_agents=dict_agents, dict_agent_devices=dict_agent_devices)

        # If no agents are created, return the empty dataframe
        #if self.config["general"]["active"] == False:
        #    return self.df

        number_of_aggregators = 0
        # loop over all aggregator types
        for aggregator_type in self.config:

            self.num_agents = self.config[f"{aggregator_type}"]["general"]["number_of_type"]

            number_of_types = []
            idx = number_of_aggregators
            for num in range(self.num_agents):
                idx += num
                number_of_types.append(idx)

            number_of_aggregators += self.num_agents

            # loop over all numbers of aggregator types
            for num_agent in number_of_types:

                # Fill the general information in dataframe
                self.fill_general(aggregator_type=aggregator_type, num_agent=num_agent)

                #  Fill all the aggregation information in dataframe
                self.fill_aggregator(dict_agents=dict_agents, dict_agent_devices=dict_agent_devices, aggregator_type=aggregator_type, num_agent=num_agent)

                # Fill the inflexible load information in dataframe
                #self.fill_inflexible_load(dict_agents=dict_agents, aggregator_type=aggregator_type, num_agent=num_agent)

                # Fill the flexible load information in dataframe
                #self.fill_flexible_load(dict_agents=dict_agents, aggregator_type=aggregator_type, num_agent=num_agent)

                # Fill the heat information in dataframe
                #self.fill_heat(dict_agents=dict_agents, aggregator_type=aggregator_type, num_agent=num_agent)

                # Fill the dhw information in dataframe
                #self.fill_dhw(dict_agents=dict_agents, aggregator_type=aggregator_type, num_agent=num_agent)

                # Fill the pv information in dataframe
                #self.fill_pv(dict_agents=dict_agents, aggregator_type=aggregator_type, num_agent=num_agent)

                # Fill the wind information in dataframe
                #self.fill_wind(dict_agents=dict_agents, aggregator_type=aggregator_type, num_agent=num_agent)

                # Fill the fixed generation information in dataframe
                #self.fill_fixed_gen(dict_agents=dict_agents, aggregator_type=aggregator_type, num_agent=num_agent)

                # Fill the heat pump information in dataframe
                #self.fill_hp(dict_agents=dict_agents, aggregator_type=aggregator_type, num_agent=num_agent)

                # Fill the electric vehicle information in dataframe
                #self.fill_ev(dict_agents=dict_agents, aggregator_type=aggregator_type, num_agent=num_agent)

                # Fill the battery information in dataframe
                #self.fill_battery(dict_agents=dict_agents, aggregator_type=aggregator_type, num_agent=num_agent)

                # Fill the heat storage information in dataframe
                #self.fill_heat_storage(dict_agents=dict_agents, aggregator_type=aggregator_type, num_agent=num_agent)

                # Fill the energy management system information in dataframe
                self.fill_ems(aggregator_type=aggregator_type, num_agent=num_agent)

        return self.df

    def create_df_from_grid(self, grid: dict, dict_agents: dict, fill_from_config: bool = False, **kwargs) -> pd.DataFrame:

        # Load the grid information
        self.grid = grid

        # Load the bus sheet
        self.bus = self.grid['bus']

        # Get the rows in the load sheet of the agent type
        self.load = self.grid['load'][self.grid['load']['agent_type'] == self.type]

        # The agents are all the buses that have an inflexible load
        self.agents = self.load[self.load['load_type'] == 'inflexible_load']

        # Get the rows in the sgen sheet that the owners in the owners column match with the index in the load sheet
        self.sgen = self.grid['sgen'][self.grid['sgen']['owner'].isin(self.load.index)]

        # Get the number of agents and set the method
        self.num = self.get_num_from_grid(self.grid['load'], self.type)
        self.method = 'grid'

        # Create the overall dataframe structure for the worksheet
        self.create_df_structure()

        # if no agents are created, return the empty dataframe
        if self.num == 0:
            return self.df

        # Fill the general information in dataframe
        self.fill_general()

        # Fill the inflexible load information in dataframe
        self.fill_inflexible_load(**kwargs)

        # Fill the flexible load information in dataframe (can only be added through config)
        if fill_from_config:
            self.fill_flexible_load()

        # Fill the heat information in dataframe
        self.fill_heat(**kwargs)

        # Fill the dhw information in dataframe
        self.fill_dhw(**kwargs)

        # Fill the pv information in dataframe
        self.fill_pv(**kwargs)

        # Fill the wind information in dataframe
        self.fill_wind(**kwargs)

        # Fill the fixed generation information in dataframe
        self.fill_fixed_gen(**kwargs)

        # Fill the heat pump information in dataframe
        self.fill_hp(**kwargs)

        # Fill the electric vehicle information in dataframe
        self.fill_ev(**kwargs)

        # Fill the battery information in dataframe
        self.fill_battery(**kwargs)

        # Fill the heat storage information in dataframe
        self.fill_heat_storage(**kwargs)

        # Fill the model predictive controller information in dataframe
        self.fill_ems()

        return self.df

    def create_df_structure(self, dict_agents: dict, dict_agent_devices: dict):
        """
            Function to create the dataframe structure with the respective columns
        """
        share_list = []
        for aggregator_type, _ in self.config.items():
            share_list.append(self.config[aggregator_type]["manual_represented_group"]["share"])
        share = max(share_list)


        # Go through file and create the columns for the aggregator worksheet
        columns = ordereddict()
        for aggregator_type, _ in self.config.items():
            config = self.config
            self.config = self.config[aggregator_type]

            for key, _ in self.config.items():

                cols = self.make_list_from_nested_dict(self.config[key], add_string=key)
                # Adjust the columns from "general"
                if key == "general":
                    cols[0] = f"{key}/agent_id"
                    cols[-1] = f"{key}/market_participant"
                    #del cols[1]
                    cols.insert(1, f"{key}/name")
                    cols.insert(2, f"{key}/comment")
                    cols.insert(3, f"{key}/bus")
                    cols.insert(4, f"{key}/aggregated_agents")
                    cols.insert(5, f"{key}/aggregated_by")
                # Adjust the columns from "inflexible_load"
                elif key == "inflexible_load":
                    cols[0] = f"{key}/owner"
                    cols.insert(1, f"{key}/num")
                    cols.insert(2, f"{key}/sizing/demand")
                    cols.insert(3, f"{key}/sizing/file")

                    num = 0
                    for agent in dict_agent_devices[key]:
                        num += len(dict_agent_devices[key][agent])
                    max_num = max(int(math.ceil(share * num)), 1)
                    cols = cols[:2] + self.repeat_columns(columns=cols[2:4], num=max_num) + cols[4:]
                # Adjust the columns from "flexible_load"
                elif key == "flexible_load":
                    cols[0] = f"{key}/owner"
                    cols.insert(1, f"{key}/num")
                    cols.insert(2, f"{key}/sizing/demand")
                    cols.insert(3, f"{key}/sizing/file")
                    cols.insert(4, f"{key}/sizing/time_offset")

                    num = 0
                    for agent in dict_agent_devices[key]:
                        num += len(dict_agent_devices[key][agent])
                    max_num = max(int(math.ceil(share * num)), 1)
                    cols = cols[:2] + self.repeat_columns(columns=cols[2:5], num=max_num) + cols[5:]
                # Adjust the columns from "heat"
                elif key == "heat":
                    cols[0] = f"{key}/owner"
                    cols.insert(1, f"{key}/num")
                    cols.insert(2, f"{key}/sizing/demand")
                    cols.insert(3, f"{key}/sizing/file")
                    cols.insert(4, f"{key}/sizing/temperature")

                    num = 0
                    for agent in dict_agent_devices[key]:
                        num += len(dict_agent_devices[key][agent])
                    max_num = max(int(math.ceil(share * num)), 1)
                    cols = cols[:2] + self.repeat_columns(columns=cols[2:5], num=max_num) + cols[5:]
                # Adjust the columns from "dhw"
                elif key == "dhw":
                    cols[0] = f"{key}/owner"
                    cols.insert(1, f"{key}/num")
                    cols.insert(2, f"{key}/sizing/demand")
                    cols.insert(3, f"{key}/sizing/file")
                    cols.insert(4, f"{key}/sizing/temperature")

                    num = 0
                    for agent in dict_agent_devices[key]:
                        num += len(dict_agent_devices[key][agent])
                    max_num = max(int(math.ceil(share * num)), 1)
                    cols = cols[:2] + self.repeat_columns(columns=cols[2:5], num=max_num) + cols[5:]
                # Adjust the columns from "pv"
                elif key == "pv":
                    cols[0] = f"{key}/owner"
                    cols.insert(1, f"{key}/num")
                    cols.insert(2, f"{key}/sizing/power")
                    cols.insert(3, f"{key}/sizing/file")
                    cols.insert(4, f"{key}/sizing/orientation")
                    cols.insert(5, f"{key}/sizing/angle")
                    cols.insert(6, f"{key}/sizing/controllable")

                    num = 0
                    for agent in dict_agent_devices[key]:
                        num += len(dict_agent_devices[key][agent])
                    max_num = max(int(math.ceil(share * num)), 1)

                    cols = cols[:2] + self.repeat_columns(columns=cols[2:7], num=max_num) + cols[7:]
                # Adjust the columns from "wind"
                elif key == "wind":
                    cols[0] = f"{key}/owner"
                    cols.insert(1, f"{key}/num")
                    cols.insert(2, f"{key}/sizing/power")
                    cols.insert(3, f"{key}/sizing/file")
                    cols.insert(4, f"{key}/sizing/controllable")

                    num = 0
                    for agent in dict_agent_devices[key]:
                        num += len(dict_agent_devices[key][agent])
                    max_num = max(int(math.ceil(share * num)), 1)
                    cols = cols[:2] + self.repeat_columns(columns=cols[2:5], num=max_num) + cols[5:]
                # Adjust the columns from "fixed_gen"
                elif key == "fixed_gen":
                    cols[0] = f"{key}/owner"
                    cols.insert(1, f"{key}/num")
                    cols.insert(2, f"{key}/sizing/power")
                    cols.insert(3, f"{key}/sizing/file")
                    cols.insert(4, f"{key}/sizing/controllable")

                    num = 0
                    for agent in dict_agent_devices[key]:
                        num += len(dict_agent_devices[key][agent])
                    max_num = max(int(math.ceil(share * num)), 1)
                    cols = cols[:2] + self.repeat_columns(columns=cols[2:5], num=max_num) + cols[5:]
                # Adjust the columns from "hp"
                elif key == "hp":
                    cols[0] = f"{key}/owner"
                    cols.insert(1, f"{key}/num")
                    cols.insert(2, f"{key}/sizing/power")
                    cols.insert(3, f"{key}/sizing/file")

                    num = 0
                    for agent in dict_agent_devices[key]:
                        num += len(dict_agent_devices[key][agent])
                    max_num = max(int(math.ceil(share * num)), 1)
                    cols = cols[:2] + self.repeat_columns(columns=cols[2:4], num=max_num) + cols[4:]
                # Adjust the columns from "ev"
                elif key == "ev":
                    cols[0] = f"{key}/owner"
                    cols.insert(1, f"{key}/num")
                    cols.insert(2, f"{key}/sizing/file")
                    cols.insert(3, f"{key}/sizing/capacity")
                    cols.insert(4, f"{key}/sizing/charging_home")
                    cols.insert(5, f"{key}/sizing/charging_AC")
                    cols.insert(6, f"{key}/sizing/charging_DC")
                    cols.insert(7, f"{key}/sizing/charging_efficiency")
                    cols.insert(8, f"{key}/sizing/soc")
                    cols.insert(9, f"{key}/sizing/v2g")
                    cols.insert(10, f"{key}/sizing/v2h")

                    num = 0
                    for agent in dict_agent_devices[key]:
                        num += len(dict_agent_devices[key][agent])
                    max_num = max(int(math.ceil(share * num)), 1)
                    cols = cols[:2] + self.repeat_columns(columns=cols[2:11], num=max_num) + cols[11:]
                # Adjust the columns from "battery"
                elif key == "battery":
                    cols[0] = f"{key}/owner"
                    cols.insert(1, f"{key}/num")
                    cols.insert(2, f"{key}/sizing/power")
                    cols.insert(3, f"{key}/sizing/capacity")
                    cols.insert(4, f"{key}/sizing/efficiency")
                    cols.insert(5, f"{key}/sizing/soc")
                    cols.insert(6, f"{key}/sizing/g2b")
                    cols.insert(7, f"{key}/sizing/b2g")

                    num = 0
                    for agent in dict_agent_devices[key]:
                        num += len(dict_agent_devices[key][agent])
                    max_num = max(int(math.ceil(share * num)), 1)
                    cols = cols[:2] + self.repeat_columns(columns=cols[2:8], num=max_num) + cols[8:]
                # Adjust the columns from "heat_storage"
                elif key == "heat_storage":
                    cols[0] = f"{key}/owner"
                    cols.insert(1, f"{key}/num")
                    cols.insert(2, f"{key}/sizing/power")
                    cols.insert(3, f"{key}/sizing/capacity")
                    cols.insert(4, f"{key}/sizing/efficiency")
                    cols.insert(5, f"{key}/sizing/soc")

                    num = 0
                    for agent in dict_agent_devices[key]:
                        num += len(dict_agent_devices[key][agent])
                    max_num = max(int(math.ceil(share * num)), 1)
                    cols = cols[:2] + self.repeat_columns(columns=cols[2:6], num=max_num) + cols[6:]
                # All columns that do not need to be adjusted
                elif key in ["ems"]:
                    pass
                elif key in ["manual_represented_group"]:
                    pass
                elif key in ["clustering_represented_group"]:
                    pass

                else:
                    raise NotImplementedError(
                        f"The configuration file contains a key word ('{key}') that has not been configured in "
                        "the Aggregator class yet. Aborting scenario creation...")

                # Add the columns to the dictionary
                columns[key] = cols

            # Combine all separate lists into one for the dataframe
            cols_df = []
            for idx, cols in columns.items():
                cols_df += cols

            # set config
            self.config = config

            # Create dataframe with responding columns
            if self.method == 'config':
                # normal indexing
                self.df = pd.DataFrame(index=range(self.num), columns=cols_df)
            elif self.method == 'grid':
                # indexing matches the load sheet (all rows that are empty in owner as those are EVs and HPs)
                self.df = pd.DataFrame(index=self.agents.index, columns=cols_df)
            else:
                raise NotImplementedError(f"The method '{self.method}' has not been implemented yet. "
                                          f"Aborting scenario creation...")

        return self.df

    def fill_general(self, aggregator_type: str, num_agent: int):
        """
            Fills all general columns
        """
        # Key in the config file
        key = "general"


        # Get the config for the key
        config = self.config[aggregator_type][f"{key}"]

        # general
        self.df[f"{key}/agent_id"][num_agent] = self._gen_new_ids(n=1)

        # market participation
        self.df[f"{key}/market_participant"][num_agent] = self._gen_rand_bool_list(n=1,
                                                                        share_ones=config["market_participant_share"])


        # If the method is grid, fill the name, comment and bus columns from grid file
        if self.method == 'config':
            self.df = self._general_config(key=key)
        elif self.method == 'grid':
            self.df = self._general_grid(key=key)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _general_config(self, key: str) -> pd.DataFrame:
        # TODO: In the future this will need to assign a bus from the artificial grid
        return self.df

    def _general_grid(self, key: str) -> pd.DataFrame:
        self.df[f"{key}/name"] = list(self.agents["name"])
        self.df[f"{key}/bus"] = list(self.agents["bus"])

        return self.df

    def fill_inflexible_load(self, dict_agents: dict, aggregator_type: str, num_agent: int, **kwargs) -> pd.DataFrame:
        """
            Fills all inflexible_load columns
        """

        # Key in the config file
        key = "inflexible_load"

        # Get the config for the key
        config = self.config[aggregator_type][f"{key}"]

        if self.method == 'config':
            self.df = self._inflexible_load_config(key=key, config=config, dict_agents=dict_agents, aggregator_type=aggregator_type, num_agent=num_agent)
        elif self.method == 'grid':
            self.df = self._inflexible_load_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _inflexible_load_config(self, key: str, config: dict, dict_agents: dict, aggregator_type: str, num_agent: int,) -> pd.DataFrame:
        """adds the inflexible load from the config file"""

        # loop over all agents aggregated by the aggregator and add the specific plants to the aggregator's dataframe
        idx = 0
        for agent in dict_agents:

            if f"{key}/num" in dict_agents[agent].columns:

                filtered_rows = dict_agents[agent][(dict_agents[agent][f"{key}/num"] == 1) & (dict_agents[agent]["general/agent_id"].apply(lambda x: any(item in x for item in self.df["general/aggregated_agents"][num_agent])))]
                if idx == 0:
                    all_rows = filtered_rows
                    idx += 1
                elif idx == 1:
                    all_rows = pd.concat([all_rows, filtered_rows], ignore_index=True)

        # loop over all rows / agents which are aggregated by the aggregator and add the information
        self.df[f"{key}/num"][num_agent] = 0
        self.df[f"{key}/owner"][num_agent] = []

        for index, row in all_rows.iterrows():

            # num
            self.df[f"{key}/num"][num_agent] += 1

            # owners
            self.df[f"{key}/owner"][num_agent].append(row["general/agent_id"])

            # demand
            self.df[f"{key}/sizing/demand_{index}"][num_agent] = row[f"{key}/sizing/demand_0"]

            # file
            self.df[f"{key}/sizing/file_{index}"][num_agent] = row[f"{key}/sizing/file_0"]



        # forecast
        self.df = self._add_info_simple_aggre(keys=[key, "fcast"], config=config["fcast"], df=self.df, num_agent=num_agent)

        return self.df

    def _inflexible_load_grid(self, key: str, config: dict, **kwargs) -> pd.DataFrame:
        """adds the inflexible load from the grid file"""

        # Get all the kwargs
        from_config_if_empty = kwargs.get('from_config_if_empty', False)

        # Drop all rows that do not contain the load type
        df = self.load[self.load['load_type'] == key]

        # Check if file contains the plant type (load > 0), if not use the config file to generate it
        if from_config_if_empty and df['load_type'].value_counts().get(key, 0) == 0:
            self._inflexible_load_config(key=key, config=config)
            return self.df

        # general
        self.df[f"{key}/owner"] = (df['demand'] > 0).astype(int)
        self.df[f"{key}/num"] = self.df[f"{key}/owner"]  # equals owner as only one inflexible load per agent

        # sizing
        for num in range(max(self.df[f"{key}/num"])):  # currently only one device per agent is supported
            # Get demand from load sheet
            self.df[f"{key}/sizing/demand_{num}"] = (df['demand'] * 1e6).astype('Int64')
            # Check if file column is empty and fill it with the closest file if so
            if df['file'].isnull().all():
                # Pick file that is closest to the demand
                self.df[f"{key}/sizing/file_{num}"] = self._pick_files_by_values(
                    vals=self.df[f"{key}/sizing/demand_{num}"] / 1000, input_path=os.path.join(self.input_path, key))
            else:
                self.df[f"{key}/sizing/file_{num}"] = df['file']

        # forecast
        self.df = self._add_info_simple(keys=[key, "fcast"], config=config["fcast"], df=self.df)

        return self.df

    def fill_flexible_load(self, dict_agents: dict, aggregator_type: str, num_agent: int) -> pd.DataFrame:
        """
            Fills all flexible_load columns
        """
        key = "flexible_load"
        config = self.config[aggregator_type][f"{key}"]
        # loop over all agents aggregated by the aggregator and add the specific plants to the aggregator's dataframe
        idx = 0
        for agent in dict_agents:

            if f"{key}/num" in dict_agents[agent].columns:

                filtered_rows = dict_agents[agent][(dict_agents[agent][f"{key}/num"] == 1) & (dict_agents[agent]["general/agent_id"].apply(lambda x: any(item in x for item in self.df["general/aggregated_agents"][num_agent])))]
                if idx == 0:
                    all_rows = filtered_rows
                    idx += 1
                elif idx == 1:
                    all_rows = pd.concat([all_rows, filtered_rows], ignore_index=True)

        # loop over all rows / agents which are aggregated by the aggregator and add the information
        self.df[f"{key}/num"][num_agent] = 0
        self.df[f"{key}/owner"][num_agent] = []

        for index, row in all_rows.iterrows():

            # num
            self.df[f"{key}/num"][num_agent] += 1

            # owners
            self.df[f"{key}/owner"][num_agent].append(row["general/agent_id"])

            # demand
            self.df[f"{key}/sizing/demand_{index}"][num_agent] = row[f"{key}/sizing/demand_0"]

            # file
            self.df[f"{key}/sizing/file_{index}"][num_agent] = row[f"{key}/sizing/file_0"]



        # forecast
        self.df = self._add_info_simple_aggre(keys=[key, "fcast"], config=config["fcast"], df=self.df, num_agent=num_agent)

        return self.df

    def fill_heat(self, dict_agents: dict, aggregator_type: str, num_agent: int, **kwargs) -> pd.DataFrame:
        """
            Fills all heat columns
        """

        # Key in the config file
        key = "heat"

        # Get the config for the key
        config = self.config[aggregator_type][f"{key}"]

        if self.method == 'config':
            self.df = self._heat_config(key=key, config=config, dict_agents=dict_agents, aggregator_type=aggregator_type, num_agent=num_agent)
        elif self.method == 'grid':
            self.df = self._heat_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _heat_config(self, key: str, config: dict, dict_agents: dict, aggregator_type: str, num_agent: int) -> pd.DataFrame:
        """adds the heat from the config file"""

        # loop over all agents aggregated by the aggregator and add the specific plants to the aggregator's dataframe
        idx = 0
        for agent in dict_agents:

            if f"{key}/num" in dict_agents[agent].columns:

                filtered_rows = dict_agents[agent][(dict_agents[agent][f"{key}/num"] == 1) & (dict_agents[agent]["general/agent_id"].apply(lambda x: any(item in x for item in self.df["general/aggregated_agents"][num_agent])))]
                if idx == 0:
                    all_rows = filtered_rows
                    idx += 1
                elif idx == 1:
                    all_rows = pd.concat([all_rows, filtered_rows], ignore_index=True)

        # loop over all rows / agents which are aggregated by the aggregator and add the information
        self.df[f"{key}/num"][num_agent] = 0
        self.df[f"{key}/owner"][num_agent] = []

        for index, row in all_rows.iterrows():

            # num
            self.df[f"{key}/num"][num_agent] += 1

            # owners
            self.df[f"{key}/owner"][num_agent].append(row["general/agent_id"])

            # demand
            self.df[f"{key}/sizing/demand_{index}"][num_agent] = row[f"{key}/sizing/demand_0"]

            # file
            self.df[f"{key}/sizing/file_{index}"][num_agent] = row[f"{key}/sizing/file_0"]



        # forecast
        self.df = self._add_info_simple_aggre(keys=[key, "fcast"], config=config["fcast"], df=self.df, num_agent=num_agent)

        return self.df

    def _heat_grid(self, key: str, config: dict, **kwargs) -> pd.DataFrame:
        """adds the heat from the grid"""

        # Get all the kwargs
        from_config_if_empty = kwargs.get('from_config_if_empty', False)

        # Drop all rows that do not contain the load type and set the index to the owner
        df = self.load[self.load['load_type'] == key].set_index('owner', drop=False)

        # Check if file contains the plant type (load > 0), if not use the config file to generate it
        if from_config_if_empty and df['load_type'].value_counts().get(key, 0) == 0:
            self._inflexible_load_config(key=key, config=config)
            return self.df

        # Check if there are any ev plants, if not set all owners to 0 and return
        if len(df) == 0:
            self.df[f"{key}/owner"] = 0
            return self.df

        # Check if there is more than one plant per agent
        if df.index.duplicated().any():
            raise NotImplementedError(f"More than one {key} per agent is not implemented yet. "
                                      f"Combine the {key} into one. "
                                      f"Aborting scenario creation...")

        # general
        self.df[f"{key}/num"] = self.df.index.map(df['owner'].value_counts()).fillna(0).astype('Int64')
        self.df[f"{key}/owner"] = (self.df[f"{key}/num"] > 0).fillna(0).astype(
            'Int64')  # all agents that have plant type

        # sizing
        for num in range(max(self.df[f"{key}/num"])):  # currently only one device per agent is supported
            # Get demand from load sheet
            self.df[f"{key}/sizing/demand_{num}"] = self.df.index.map(df['demand'] * 1e6).astype('Int64')
            self.df[f"{key}/sizing/file_{num}"] = self.df.index.map(df['heat_file'])
            self.df[f"{key}/sizing/temperature_{num}"] = self.df.index.map(df['temperature'])

        # forecast
        self.df = self._add_info_simple(keys=[key, "fcast"], config=config["fcast"], df=self.df)

        return self.df

    def fill_dhw(self, dict_agents: dict, aggregator_type: str, num_agent: int, **kwargs) -> pd.DataFrame:
        """
            Fills all dhw columns
        """

        # Key in the config file
        key = "dhw"

        # Get the config for the key
        config = self.config[aggregator_type][f"{key}"]

        if self.method == 'config':
            self.df = self._dhw_config(key=key, config=config, dict_agents=dict_agents, aggregator_type=aggregator_type, num_agent=num_agent)
        elif self.method == 'grid':
            self.df = self._dhw_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _dhw_config(self, key: str, config: dict, dict_agents: dict, aggregator_type: str, num_agent: int) -> pd.DataFrame:
        """adds the dhw from the config file"""

        # loop over all agents aggregated by the aggregator and add the specific plants to the aggregator's dataframe
        idx = 0
        for agent in dict_agents:

            if f"{key}/num" in dict_agents[agent].columns:

                filtered_rows = dict_agents[agent][(dict_agents[agent][f"{key}/num"] == 1) & (dict_agents[agent]["general/agent_id"].apply(lambda x: any(item in x for item in self.df["general/aggregated_agents"][num_agent])))]
                if idx == 0:
                    all_rows = filtered_rows
                    idx += 1
                elif idx == 1:
                    all_rows = pd.concat([all_rows, filtered_rows], ignore_index=True)

        # loop over all rows / agents which are aggregated by the aggregator and add the information
        self.df[f"{key}/num"][num_agent] = 0
        self.df[f"{key}/owner"][num_agent] = []

        for index, row in all_rows.iterrows():

            # num
            self.df[f"{key}/num"][num_agent] += 1

            # owners
            self.df[f"{key}/owner"][num_agent].append(row["general/agent_id"])

            # demand
            self.df[f"{key}/sizing/demand_{index}"][num_agent] = row[f"{key}/sizing/demand_0"]

            # file
            self.df[f"{key}/sizing/file_{index}"][num_agent] = row[f"{key}/sizing/file_0"]



        # forecast
        self.df = self._add_info_simple_aggre(keys=[key, "fcast"], config=config["fcast"], df=self.df, num_agent=num_agent)

        return self.df

    def _dhw_grid(self, key: str, config: dict, **kwargs) -> pd.DataFrame:
        """adds the dhw from the grid"""

        # Get all the kwargs
        from_config_if_empty = kwargs.get('from_config_if_empty', False)

        # Drop all rows that do not contain the load type and set the index to the owner
        df = self.load[self.load['load_type'] == key].set_index('owner', drop=False)

        # Check if file contains the plant type (load > 0), if not use the config file to generate it
        if from_config_if_empty and df['load_type'].value_counts().get(key, 0) == 0:
            self._inflexible_load_config(key=key, config=config)
            return self.df

        # Check if there are any ev plants, if not set all owners to 0 and return
        if len(df) == 0:
            self.df[f"{key}/owner"] = 0
            return self.df

        # Check if there is more than one plant per agent
        if df.index.duplicated().any():
            raise NotImplementedError(f"More than one {key} per agent is not implemented yet. "
                                      f"Combine the {key} into one. "
                                      f"Aborting scenario creation...")

        # general
        self.df[f"{key}/num"] = self.df.index.map(df['owner'].value_counts()).fillna(0).astype('Int64')
        self.df[f"{key}/owner"] = (self.df[f"{key}/num"] > 0).fillna(0).astype(
            'Int64')  # all agents that have plant type

        # sizing
        for num in range(max(self.df[f"{key}/num"])):  # currently only one device per agent is supported
            # Get demand from load sheet
            self.df[f"{key}/sizing/demand_{num}"] = self.df.index.map(df['demand'] * 1e6).astype('Int64')
            self.df[f"{key}/sizing/file_{num}"] = self.df.index.map(df['heat_file'])
            self.df[f"{key}/sizing/temperature_{num}"] = self.df.index.map(df['temperature'])

        # forecast
        self.df = self._add_info_simple(keys=[key, "fcast"], config=config["fcast"], df=self.df)

        return self.df

    def fill_pv(self, dict_agents: dict, aggregator_type: str, num_agent: int, **kwargs) -> pd.DataFrame:
        """
            Fills all pv columns
        """

        # Key in the config file
        key = "pv"

        # Get the config for the key
        config = self.config[aggregator_type][f"{key}"]

        if self.method == 'config':
            self.df = self._pv_config(key=key, config=config, dict_agents=dict_agents, aggregator_type=aggregator_type, num_agent=num_agent)
        elif self.method == 'grid':
            self.df = self._pv_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _pv_config(self, key: str, config: dict, dict_agents: dict, aggregator_type: str, num_agent: int) -> pd.DataFrame:
        """adds the inflexible load from the config file"""

        # loop over all agents aggregated by the aggregator and add the specific plants to the aggregator's dataframe
        idx = 0
        for agent in dict_agents:

            if f"{key}/num" in dict_agents[agent].columns:

                filtered_rows = dict_agents[agent][(dict_agents[agent][f"{key}/num"] == 1) & (
                    dict_agents[agent]["general/agent_id"].apply(
                        lambda x: any(item in x for item in self.df["general/aggregated_agents"][num_agent])))]
                if idx == 0:
                    all_rows = filtered_rows
                    idx += 1
                elif idx == 1:
                    all_rows = pd.concat([all_rows, filtered_rows], ignore_index=True)

        # loop over all rows / agents which are aggregated by the aggregator and add the information
        self.df[f"{key}/num"][num_agent] = 0
        self.df[f"{key}/owner"][num_agent] = []

        for index, row in all_rows.iterrows():
            # num
            self.df[f"{key}/num"][num_agent] += 1

            # owners
            self.df[f"{key}/owner"][num_agent].append(row["general/agent_id"])

            # power
            self.df[f"{key}/sizing/power_{index}"][num_agent] = row[f"{key}/sizing/power_0"]

            # file
            self.df[f"{key}/sizing/file_{index}"][num_agent] = row[f"{key}/sizing/file_0"]

        # forecast
        self.df = self._add_info_simple_aggre(keys=[key, "fcast"], config=config["fcast"], df=self.df,
                                              num_agent=num_agent)

        # quality
        #self.df[f"{key}/quality"] = config["quality"]

        return self.df

    def _pv_grid(self, key: str, config: dict, **kwargs) -> pd.DataFrame:
        """adds the pv plants from the grid file"""

        # Get all the kwargs
        from_config_if_empty = kwargs.get('from_config_if_empty', False)

        # Check if file contains the plant type (count > 0), if not use the config file to generate it
        if from_config_if_empty and self.sgen['sgen_type'].value_counts()[key] == 0:
            self._pv_config(key=key, config=config)
            return self.df

        # Drop all rows that do not contain the plant type and set the index to the owner
        df = self.sgen[self.sgen['plant_type'] == key].set_index('owner', drop=False)

        # Check if there are any pv plants, if not set all owners to 0 and return
        if len(df) == 0:
            self.df[f"{key}/owner"] = 0
            self.df[f"{key}/num"] = 0
            return self.df

        # Check if there is more than one plant per agent
        if df.index.duplicated().any():
            raise NotImplementedError(f"More than one {key} per agent is not implemented yet. "
                                      f"Combine the {key} into one. "
                                      f"Aborting scenario creation...")

        # general
        self.df[f"{key}/num"] = self.df.index.map(df['owner'].value_counts()).fillna(0).astype('Int64')
        self.df[f"{key}/owner"] = (self.df[f"{key}/num"] > 0).fillna(0).astype('Int64')  # all agents that have pv

        # sizing (all parameters that can be indexed)
        for num in range(max(self.df[f"{key}/num"])):  # Currently only one pv per agent is supported
            # Match the power with the power specified in sgen sheet
            self.df[f"{key}/sizing/power_{num}"] = (self.df.index.map(df['power']) * 1e6).astype('Int64')

            # Check if file column exists
            if 'file' in df and not df['file'].isnull().all():
                # Fill rows with values from sgen sheet
                self.df[f"{key}/sizing/file_{num}"] = self.df.index.map(df['file'])
                self.df[f"{key}/sizing/orientation_{num}"] = self.df.index.map(df['orientation']).fillna(0)
                self.df[f"{key}/sizing/angle_{num}"] = self.df.index.map(df['angle']).fillna(0)
            # If file column does not exist, check if orientation and angle columns exist and all rows are filled
            elif 'orientation' in df and 'angle' in df \
                    and not df['orientation'].isnull().any() and not df['angle'].isnull().any():
                # Fill orientation and angle from sgen sheet
                self.df[f"{key}/sizing/orientation_{num}"] = self.df.index.map(df['orientation'])
                self.df[f"{key}/sizing/angle_{num}"] = self.df.index.map(df['angle'])
                # Pick random specs file (> num as num starts at 0)
                self.df[f"{key}/sizing/file_{num}"].loc[self.df[f"{key}/num"] > num] = 'specs'
                self.df[f"{key}/sizing/file_{num}"] = self._pick_files(list_type=self.df[f"{key}/sizing/file_{num}"],
                                                                       device=f"{key}",
                                                                       input_path=os.path.join(self.input_path, key))
            # If all three columns are not filled, pick random timeseries file
            else:
                # Pick random timeseries file (> num as num starts at 0)
                self.df[f"{key}/sizing/file_{num}"].loc[self.df[f"{key}/num"] > num] = 'timeseries'
                self.df[f"{key}/sizing/file_{num}"] = self._pick_files(list_type=self.df[f"{key}/sizing/file_{num}"],
                                                                       device=f"{key}",
                                                                       input_path=os.path.join(self.input_path, key))
                # Assign standard orientation and angle since they do not matter if no file is specified
                self.df[f"{key}/sizing/orientation_{num}"] = 0
                self.df[f"{key}/sizing/angle_{num}"] = 0

            # Make all plants controllable
            self.df[f"{key}/sizing/controllable_{num}"] = True

        # forecast
        self.df = self._add_info_simple(keys=[key, "fcast"], config=config["fcast"], df=self.df)

        # quality
        self.df[f"{key}/quality"] = config["quality"]

        return self.df

    def fill_wind(self, dict_agents: dict, aggregator_type: str, num_agent: int, **kwargs) -> pd.DataFrame:
        """
            Fills all wind columns
        """
        # Key in the config file
        key = "wind"

        # Get the config for the key
        config = self.config[aggregator_type][f"{key}"]

        if self.method == 'config':
            self.df = self._wind_config(key=key, config=config, dict_agents=dict_agents, aggregator_type=aggregator_type, num_agent=num_agent)
        elif self.method == 'grid':
            self.df = self._wind_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _wind_config(self, key: str, config: dict, dict_agents: dict, aggregator_type: str, num_agent: int) -> pd.DataFrame:

        # loop over all agents aggregated by the aggregator and add the specific plants to the aggregator's dataframe
        idx = 0
        for agent in dict_agents:

            if f"{key}/num" in dict_agents[agent].columns:

                filtered_rows = dict_agents[agent][(dict_agents[agent][f"{key}/num"] == 1) & (
                    dict_agents[agent]["general/agent_id"].apply(
                        lambda x: any(item in x for item in self.df["general/aggregated_agents"][num_agent])))]
                if idx == 0:
                    all_rows = filtered_rows
                    idx += 1
                elif idx == 1:
                    all_rows = pd.concat([all_rows, filtered_rows], ignore_index=True)

        # loop over all rows / agents which are aggregated by the aggregator and add the information
        self.df[f"{key}/num"][num_agent] = 0
        self.df[f"{key}/owner"][num_agent] = []

        for index, row in all_rows.iterrows():
            # num
            self.df[f"{key}/num"][num_agent] += 1

            # owners
            self.df[f"{key}/owner"][num_agent].append(row["general/agent_id"])

            # power
            self.df[f"{key}/sizing/power_{index}"][num_agent] = row[f"{key}/sizing/power_0"]

            # file
            self.df[f"{key}/sizing/file_{index}"][num_agent] = row[f"{key}/sizing/file_0"]

        # forecast
        self.df = self._add_info_simple_aggre(keys=[key, "fcast"], config=config["fcast"], df=self.df,
                                              num_agent=num_agent)

        # quality
        #self.df[f"{key}/quality"] = config["quality"]

        return self.df

    def _wind_grid(self, key: str, config: dict, **kwargs) -> pd.DataFrame:

        # Get all the kwargs
        from_config_if_empty = kwargs.get('from_config_if_empty', False)

        # Check if file contains the plant type (count > 0), if not use the config file to generate it
        if from_config_if_empty and self.sgen['sgen_type'].value_counts()[key] == 0:
            self._wind_config(key=key, config=config)
            return self.df

        # Drop all rows that do not contain the plant type and set the index to the owner
        df = self.sgen[self.sgen['plant_type'] == key].set_index('owner', drop=False)

        # Check if there are any pv plants, if not set all owners to 0 and return
        if len(df) == 0:
            self.df[f"{key}/owner"] = 0
            self.df[f"{key}/num"] = 0
            return self.df

        # Check if there is more than one plant per agent
        if df.index.duplicated().any():
            raise NotImplementedError(f"More than one {key} per agent is not implemented yet. "
                                      f"Combine the {key} into one. "
                                      f"Aborting scenario creation...")

        # general
        self.df[f"{key}/num"] = self.df.index.map(df['owner'].value_counts()).fillna(0).astype('Int64')
        self.df[f"{key}/owner"] = (self.df[f"{key}/num"] > 0).fillna(0).astype('Int64')  # all agents that have pv

        # sizing (all parameters that can be indexed)
        for num in range(max(self.df[f"{key}/num"])):  # Currently only one plant per agent is supported
            # Match the power with the power specified in sgen sheet
            self.df[f"{key}/sizing/power_{num}"] = (self.df.index.map(df['power']) * 1e6).astype('Int64')

            # Check if file column exists
            if 'file' in df and not df['file'].isnull().all():
                # Fill rows with values from sgen sheet
                self.df[f"{key}/sizing/file_{num}"] = self.df.index.map(df['file'])
            # If file column does not exist, check if height columns exist and all rows are filled
            elif 'height' in df and not df['height'].isnull().any():
                # TODO: Include height in the wind power calculation (add to config_agents)
                # Fill height from sgen sheet
                # self.df[f"{key}/sizing/height_{num}"] = self.df.index.map(df['height'])
                # Pick random specs file (> num as num starts at 0)
                self.df[f"{key}/sizing/file_{num}"].loc[self.df[f"{key}/num"] > num] = 'specs'
                self.df[f"{key}/sizing/file_{num}"] = self._pick_files(list_type=self.df[f"{key}/sizing/file_{num}"],
                                                                       device=f"{key}",
                                                                       input_path=os.path.join(self.input_path, key))
            # If file column does not exist, pick random timeseries file
            else:
                # Pick random timeseries file (> num as num starts at 0)
                self.df[f"{key}/sizing/file_{num}"].loc[self.df[f"{key}/num"] > num] = 'timeseries'
                self.df[f"{key}/sizing/file_{num}"] = self._pick_files(list_type=self.df[f"{key}/sizing/file_{num}"],
                                                                       device=f"{key}",
                                                                       input_path=os.path.join(self.input_path, key))
                # Assign standard height since they do not matter if no file is specified
                # self.df[f"{key}/sizing/height_{num}"] = 0

            # Make all plants controllable
            self.df[f"{key}/sizing/controllable_{num}"] = True

        # forecast
        self.df = self._add_info_simple(keys=[key, "fcast"], config=config["fcast"], df=self.df)

        # quality
        self.df[f"{key}/quality"] = config["quality"]

        return self.df

    def fill_fixed_gen(self, dict_agents: dict, aggregator_type: str, num_agent: int, **kwargs) -> pd.DataFrame:
        """
            Fills all fixed_gen columns
        """

        # Key in the config file
        key = "fixed_gen"

        # Get the config for the key
        config = self.config[aggregator_type][f"{key}"]

        if self.method == 'config':
            self.df = self._fixed_gen_config(key=key, config=config, dict_agents=dict_agents, aggregator_type=aggregator_type, num_agent=num_agent)
        elif self.method == 'grid':
            self.df = self._fixed_gen_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _fixed_gen_config(self, key: str, config: dict, dict_agents: dict, aggregator_type: str, num_agent: int) -> pd.DataFrame:

        # loop over all agents aggregated by the aggregator and add the specific plants to the aggregator's dataframe
        idx = 0
        for agent in dict_agents:

            if f"{key}/num" in dict_agents[agent].columns:

                filtered_rows = dict_agents[agent][(dict_agents[agent][f"{key}/num"] == 1) & (
                    dict_agents[agent]["general/agent_id"].apply(
                        lambda x: any(item in x for item in self.df["general/aggregated_agents"][num_agent])))]
                if idx == 0:
                    all_rows = filtered_rows
                    idx += 1
                elif idx == 1:
                    all_rows = pd.concat([all_rows, filtered_rows], ignore_index=True)

        # loop over all rows / agents which are aggregated by the aggregator and add the information
        self.df[f"{key}/num"][num_agent] = 0
        self.df[f"{key}/owner"][num_agent] = []

        for index, row in all_rows.iterrows():
            # num
            self.df[f"{key}/num"][num_agent] += 1

            # owners
            self.df[f"{key}/owner"][num_agent].append(row["general/agent_id"])

            # power
            self.df[f"{key}/sizing/demand_{index}"][num_agent] = row[f"{key}/sizing/power_0"]

            # file
            self.df[f"{key}/sizing/file_{index}"][num_agent] = row[f"{key}/sizing/file_0"]

        # forecast
        self.df = self._add_info_simple_aggre(keys=[key, "fcast"], config=config["fcast"], df=self.df,
                                              num_agent=num_agent)

        # quality
        #self.df[f"{key}/quality"] = config["quality"]

        return self.df

    def _fixed_gen_grid(self, key: str, config: dict, **kwargs) -> pd.DataFrame:

        # Get all the kwargs
        from_config_if_empty = kwargs.get('from_config_if_empty', False)

        # Check if file contains the plant type (count > 0), if not use the config file to generate it
        if from_config_if_empty and self.sgen['sgen_type'].value_counts()[key] == 0:
            self._fixed_gen_config(key=key, config=config)
            return self.df

        # Drop all rows that do not contain the plant type and set the index to the owner
        df = self.sgen[self.sgen['plant_type'] == key].set_index('owner', drop=False)

        # Check if there are any pv plants, if not set all owners to 0 and return
        if len(df) == 0:
            self.df[f"{key}/owner"] = 0
            return self.df

        # Check if there is more than one plant per agent
        if df.index.duplicated().any():
            raise NotImplementedError(f"More than one {key} per agent is not implemented yet. "
                                      f"Combine the {key} into one. "
                                      f"Aborting scenario creation...")

        # general
        self.df[f"{key}/num"] = self.df.index.map(df['owner'].value_counts()).fillna(0).astype('Int64')
        self.df[f"{key}/owner"] = (self.df[f"{key}/num"] > 0).fillna(0).astype('Int64')  # all agents that have pv

        # sizing (all parameters that can be indexed)
        for num in range(max(self.df[f"{key}/num"])):  # Currently only one plant per agent is supported
            # Match the power with the power specified in sgen sheet
            self.df[f"{key}/sizing/power_{num}"] = (self.df.index.map(df['power']) * 1e6).astype('Int64')

            # Check if file column exists
            if 'file' in df and not df['file'].isnull().all():
                # Fill rows with values from sgen sheet
                self.df[f"{key}/sizing/file_{num}"] = self.df.index.map(df['file'])
            # If file column does not exist, pick random timeseries file
            else:
                # Pick random timeseries file (> num as num starts at 0)
                self.df[f"{key}/sizing/file_{num}"].loc[self.df[f"{key}/num"] > num] = 'timeseries'
                self.df[f"{key}/sizing/file_{num}"] = self._pick_files(list_type=self.df[f"{key}/sizing/file_{num}"],
                                                                       device=f"{key}",
                                                                       input_path=os.path.join(self.input_path, key))

            # Make all plants controllable
            self.df[f"{key}/sizing/controllable_{num}"] = True

        # forecast
        self.df = self._add_info_simple(keys=[key, "fcast"], config=config["fcast"], df=self.df)

        # quality
        self.df[f"{key}/quality"] = config["quality"]

        return self.df

    def fill_hp(self, dict_agents: dict, aggregator_type: str, num_agent: int, **kwargs) -> pd.DataFrame:
        """
            Fills all hp columns
        """

        # Key in the config file
        key = "hp"

        # Get the config for the key
        config = self.config[aggregator_type][f"{key}"]

        if self.method == 'config':
            self._hp_config(key=key, config=config, dict_agents=dict_agents, aggregator_type=aggregator_type, num_agent=num_agent)
        elif self.method == 'grid':
            self._hp_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _hp_config(self, key: str, config: dict, dict_agents: dict, aggregator_type: str, num_agent: int) -> pd.DataFrame:

        # loop over all agents aggregated by the aggregator and add the specific plants to the aggregator's dataframe
        idx = 0
        for agent in dict_agents:

            if f"{key}/num" in dict_agents[agent].columns:

                filtered_rows = dict_agents[agent][(dict_agents[agent][f"{key}/num"] == 1) & (
                    dict_agents[agent]["general/agent_id"].apply(
                        lambda x: any(item in x for item in self.df["general/aggregated_agents"][num_agent])))]
                if idx == 0:
                    all_rows = filtered_rows
                    idx += 1
                elif idx == 1:
                    all_rows = pd.concat([all_rows, filtered_rows], ignore_index=True)

        # loop over all rows / agents which are aggregated by the aggregator and add the information
        self.df[f"{key}/num"][num_agent] = 0
        self.df[f"{key}/owner"][num_agent] = []

        for index, row in all_rows.iterrows():
            # num
            self.df[f"{key}/num"][num_agent] += 1

            # owners
            self.df[f"{key}/owner"][num_agent].append(row["general/agent_id"])

            # power
            self.df[f"{key}/sizing/demand_{index}"][num_agent] = row[f"{key}/sizing/power_0"]

            # file
            self.df[f"{key}/sizing/file_{index}"][num_agent] = row[f"{key}/sizing/file_0"]


        # quality
        self.df[f"{key}/quality"] = config["quality"]

        return self.df

    def _hp_grid(self, key: str, config: dict, **kwargs) -> pd.DataFrame:

        # Get all the kwargs
        from_config_if_empty = kwargs.get('from_config_if_empty', False)

        # Drop all rows that do not contain the load type and set the index to the owner
        df = self.load[self.load['load_type'] == key].set_index('owner', drop=False)

        # Check if file contains the plant type (load > 0), if not use the config file to generate it
        if from_config_if_empty and df['load_type'].value_counts().get(key, 0) == 0:
            self._hp_config(key=key, config=config)
            return self.df

        # Check if there are any hp plants, if not set all owners to 0 and return
        if len(df) == 0:
            self.df[f"{key}/owner"] = 0
            return self.df

        # Check if there is more than one plant per agent
        if df.index.duplicated().any():
            raise NotImplementedError(f"More than one {key} per agent is not implemented yet. "
                                      f"Combine the {key} into one. "
                                      f"Aborting scenario creation...")

        # general
        self.df[f"{key}/num"] = self.df.index.map(df['owner'].value_counts()).fillna(0).astype('Int64')
        self.df[f"{key}/owner"] = (self.df[f"{key}/num"] > 0).fillna(0).astype(
            'Int64')  # all agents that have plant type

        # sizing
        for num in range(max(self.df[f"{key}/num"])):  # currently only one device per agent is supported
            # Add power of hp
            self.df[f"{key}/sizing/power_{num}"] = self.df.index.map(df['power'] * 1e6).astype('Int64')
            # Check if cop_file column exists
            if 'file' in df and not df['cop_file'].isnull().all():
                # Fill rows with values from sgen sheet
                self.df[f"{key}/sizing/file_{num}"] = self.df.index.map(df['cop_file'])
            else:
                # Add random specs file to all agents that have a plant (> num as num starts at 0)
                self.df[f"{key}/sizing/file_{num}"].loc[self.df[f"{key}/num"] > num] = 'specs'
                self.df[f"{key}/sizing/file_{num}"] = self._pick_files(list_type=self.df[f"{key}/sizing/file_{num}"],
                                                                       device=f"{key}",
                                                                       input_path=os.path.join(self.input_path, key))

        # quality
        self.df[f"{key}/quality"] = config["quality"]

        return self.df

    def fill_ev(self, dict_agents: dict, aggregator_type: str, num_agent: int, **kwargs) -> pd.DataFrame:
        """
            Fills all ev columns
        """

        # Key in the config file
        key = "ev"

        # Get the config for the key
        config = self.config[aggregator_type][f"{key}"]

        if self.method == 'config':
            self._ev_config(key=key, config=config, dict_agents=dict_agents, aggregator_type=aggregator_type, num_agent=num_agent)
        elif self.method == 'grid':
            self._ev_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _ev_config(self, key: str, config: dict, dict_agents: dict, aggregator_type: str, num_agent: int) -> pd.DataFrame:

        # loop over all agents aggregated by the aggregator and add the specific plants to the aggregator's dataframe
        idx = 0
        for agent in dict_agents:

            if f"{key}/num" in dict_agents[agent].columns:

                filtered_rows = dict_agents[agent][(dict_agents[agent][f"{key}/num"] == 1) & (
                    dict_agents[agent]["general/agent_id"].apply(
                        lambda x: any(item in x for item in self.df["general/aggregated_agents"][num_agent])))]
                if idx == 0:
                    all_rows = filtered_rows
                    idx += 1
                elif idx == 1:
                    all_rows = pd.concat([all_rows, filtered_rows], ignore_index=True)

        # loop over all rows / agents which are aggregated by the aggregator and add the information
        self.df[f"{key}/num"][num_agent] = 0
        self.df[f"{key}/owner"][num_agent] = []

        for index, row in all_rows.iterrows():
            # num
            self.df[f"{key}/num"][num_agent] += 1

            # owners
            self.df[f"{key}/owner"][num_agent].append(row["general/agent_id"])

            # power
            self.df[f"{key}/sizing/capacity_{index}"][num_agent] = row[f"{key}/sizing/capacity_0"]

            # file
            self.df[f"{key}/sizing/file_{index}"][num_agent] = row[f"{key}/sizing/file_0"]

        # forecast
        self.df = self._add_info_simple_aggre(keys=[key, "fcast"], config=config["fcast"], df=self.df,
                                              num_agent=num_agent)

        # charging scheme
        #self._add_info_simple(keys=[key, "charging_scheme"], config=config["charging_scheme"], df=self.df)

        # quality
        self.df[f"{key}/quality"] = config["quality"]

        return self.df

    def _ev_grid(self, key: str, config: dict, **kwargs) -> pd.DataFrame:

        # Get all the kwargs
        from_config_if_empty = kwargs.get('from_config_if_empty', False)

        # Drop all rows that do not contain the load type and set the index to the owner
        df = self.load[self.load['load_type'] == key].set_index('owner', drop=False)

        # Check if file contains the plant type (load > 0), if not use the config file to generate it
        if from_config_if_empty and df['load_type'].value_counts().get(key, 0) == 0:
            self._ev_config(key=key, config=config)
            return self.df

        # Check if there are any ev plants, if not set all owners to 0 and return
        if len(df) == 0:
            self.df[f"{key}/owner"] = 0
            return self.df

        # Check if there is more than one plant per agent
        if df.index.duplicated().any():
            raise NotImplementedError(f"More than one {key} per agent is not implemented yet. "
                                      f"Combine the {key} into one. "
                                      f"Aborting scenario creation...")

        # general
        self.df[f"{key}/num"] = self.df.index.map(df['owner'].value_counts()).fillna(0).astype('Int64')
        self.df[f"{key}/owner"] = (self.df[f"{key}/num"] > 0).fillna(0).astype(
            'Int64')  # all agents that have plant type

        # sizing
        for num in range(max(self.df[f"{key}/num"])):  # currently only one device per agent is supported
            self.df[f"{key}/sizing/file_{num}"] = self.df.index.map(df['file_add'])
            self.df[f"{key}/sizing/capacity_{num}"] = (self.df.index.map(df['capacity']) * 1e6).astype('Int64')
            self.df[f"{key}/sizing/charging_home_{num}"] = (self.df.index.map(df['charging_home']) * 1e6).astype(
                'Int64')
            self.df[f"{key}/sizing/charging_AC_{num}"] = (self.df.index.map(df['charging_ac']) * 1e6).astype('Int64')
            self.df[f"{key}/sizing/charging_DC_{num}"] = (self.df.index.map(df['charging_dc']) * 1e6).astype('Int64')
            self.df[f"{key}/sizing/charging_efficiency_{num}"] = self.df.index.map(df['efficiency'])
            self.df[f"{key}/sizing/soc_{num}"] = self.df.index.map(df['soc'])
            self.df[f"{key}/sizing/v2g_{num}"] = self.df.index.map(df['v2g'])
            self.df[f"{key}/sizing/v2h_{num}"] = self.df.index.map(df['v2h'])

        # quality
        self.df[f"{key}/quality"] = config["quality"]

        return self.df

    def fill_battery(self, dict_agents: dict, aggregator_type: str, num_agent: int, **kwargs) -> pd.DataFrame:
        """
            Fills all battery columns
        """

        # Key in the config file
        key = "battery"

        # Get the config for the key
        config = self.config[aggregator_type][f"{key}"]

        if self.method == 'config':
            self.df = self._battery_config(key=key, config=config, dict_agents=dict_agents, aggregator_type=aggregator_type, num_agent=num_agent)
        elif self.method == 'grid':
            self.df = self._battery_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _battery_config(self, key: str, config: dict, dict_agents: dict, aggregator_type: str, num_agent: int) -> pd.DataFrame:

        # loop over all agents aggregated by the aggregator and add the specific plants to the aggregator's dataframe
        idx = 0
        for agent in dict_agents:

            if f"{key}/num" in dict_agents[agent].columns:

                filtered_rows = dict_agents[agent][(dict_agents[agent][f"{key}/num"] == 1) & (
                    dict_agents[agent]["general/agent_id"].apply(
                        lambda x: any(item in x for item in self.df["general/aggregated_agents"][num_agent])))]
                if idx == 0:
                    all_rows = filtered_rows
                    idx += 1
                elif idx == 1:
                    all_rows = pd.concat([all_rows, filtered_rows], ignore_index=True)

        # loop over all rows / agents which are aggregated by the aggregator and add the information
        self.df[f"{key}/num"][num_agent] = 0
        self.df[f"{key}/owner"][num_agent] = []

        for index, row in all_rows.iterrows():
            # num
            self.df[f"{key}/num"][num_agent] += 1

            # owners
            self.df[f"{key}/owner"][num_agent].append(row["general/agent_id"])

            # power
            self.df[f"{key}/sizing/capacity_{index}"][num_agent] = row[f"{key}/sizing/capacity_0"]

            # file
            #self.df[f"{key}/sizing/file_{index}"][num_agent] = row[f"{key}/sizing/file_0"]

        # quality
        self.df[f"{key}/quality"] = config["quality"]

        return self.df

    def _battery_grid(self, key: str, config: dict, **kwargs) -> pd.DataFrame:

        # Get all the kwargs
        from_config_if_empty = kwargs.get('from_config_if_empty', False)

        # Check if file contains the plant type (count > 0), if not use the config file to generate it
        if from_config_if_empty and self.sgen['sgen_type'].value_counts()[key] == 0:
            self._battery_config(key=key, config=config)
            return self.df

        # Drop all rows that do not contain the plant type and set the index to the owner
        df = self.sgen[self.sgen['plant_type'] == key].set_index('owner', drop=False)

        # Check if there are any pv plants, if not set all owners to 0 and return
        if len(df) == 0:
            self.df[f"{key}/owner"] = 0
            return self.df

        # Check if there is more than one plant per agent
        if df.index.duplicated().any():
            raise NotImplementedError(f"More than one {key} per agent is not implemented yet. "
                                      f"Combine the {key} into one. "
                                      f"Aborting scenario creation...")

        # general
        self.df[f"{key}/num"] = self.df.index.map(df['owner'].value_counts()).fillna(0).astype('Int64')
        self.df[f"{key}/owner"] = (self.df[f"{key}/num"] > 0).fillna(0).astype('Int64')  # all agents that have pv

        # sizing (all parameters that can be indexed)
        for num in range(max(self.df[f"{key}/num"])):  # Currently only one plant per agent is supported
            # Match the power with the power specified in sgen sheet
            self.df[f"{key}/sizing/power_{num}"] = (self.df.index.map(df['power']) * 1e6).astype('Int64')
            self.df[f"{key}/sizing/capacity_{num}"] = (self.df.index.map(df['capacity']) * 1e6).astype('Int64')
            self.df[f"{key}/sizing/efficiency_{num}"] = self.df.index.map(df['efficiency'])
            self.df[f"{key}/sizing/soc_{num}"] = self.df.index.map(df['soc'])
            self.df[f"{key}/sizing/g2b_{num}"] = self.df.index.map(df['g2b'])
            self.df[f"{key}/sizing/b2g_{num}"] = self.df.index.map(df['b2g'])

        # quality
        self.df[f"{key}/quality"] = config["quality"]

        return self.df

    def fill_heat_storage(self, dict_agents: dict, aggregator_type: str, num_agent: int, **kwargs) -> pd.DataFrame:
        """
            Fills all heat storage columns
        """

        # Key in the config file
        key = "heat_storage"

        # Get the config for the key
        config = self.config[aggregator_type][f"{key}"]

        if self.method == 'config':
            self.df = self._heat_storage_config(key=key, config=config, dict_agents=dict_agents, aggregator_type=aggregator_type, num_agent=num_agent)
        elif self.method == 'grid':
            self.df = self._heat_storage_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _heat_storage_config(self, key: str, config: dict, dict_agents: dict, aggregator_type: str, num_agent: int) -> pd.DataFrame:

        # loop over all agents aggregated by the aggregator and add the specific plants to the aggregator's dataframe
        idx = 0
        for agent in dict_agents:

            if f"{key}/num" in dict_agents[agent].columns:

                filtered_rows = dict_agents[agent][(dict_agents[agent][f"{key}/num"] == 1) & (
                    dict_agents[agent]["general/agent_id"].apply(
                        lambda x: any(item in x for item in self.df["general/aggregated_agents"][num_agent])))]
                if idx == 0:
                    all_rows = filtered_rows
                    idx += 1
                elif idx == 1:
                    all_rows = pd.concat([all_rows, filtered_rows], ignore_index=True)

        # loop over all rows / agents which are aggregated by the aggregator and add the information
        self.df[f"{key}/num"][num_agent] = 0
        self.df[f"{key}/owner"][num_agent] = []

        for index, row in all_rows.iterrows():
            # num
            self.df[f"{key}/num"][num_agent] += 1

            # owners
            self.df[f"{key}/owner"][num_agent].append(row["general/agent_id"])

            # power
            self.df[f"{key}/sizing/demand_{index}"][num_agent] = row[f"{key}/sizing/power_0"]

            # file
            self.df[f"{key}/sizing/file_{index}"][num_agent] = row[f"{key}/sizing/file_0"]


        # quality
        self.df[f"{key}/quality"] = config["quality"]

        return self.df

    def _heat_storage_grid(self, key: str, config: dict, **kwargs) -> pd.DataFrame:

        # Get all the kwargs
        from_config_if_empty = kwargs.get('from_config_if_empty', False)

        # Check if file contains the plant type (count > 0), if not use the config file to generate it
        if from_config_if_empty and self.sgen['sgen_type'].value_counts()[key] == 0:
            self._battery_config(key=key, config=config)
            return self.df

        # Drop all rows that do not contain the plant type and set the index to the owner
        df = self.sgen[self.sgen['plant_type'] == key].set_index('owner', drop=False)

        # Check if there are any pv plants, if not set all owners to 0 and return
        if len(df) == 0:
            self.df[f"{key}/owner"] = 0
            return self.df

        # Check if there is more than one plant per agent
        if df.index.duplicated().any():
            raise NotImplementedError(f"More than one {key} per agent is not implemented yet. "
                                      f"Combine the {key} into one. "
                                      f"Aborting scenario creation...")

        # general
        self.df[f"{key}/num"] = self.df.index.map(df['owner'].value_counts()).fillna(0).astype('Int64')
        self.df[f"{key}/owner"] = (self.df[f"{key}/num"] > 0).fillna(0).astype('Int64')  # all agents that have pv

        # sizing (all parameters that can be indexed)
        for num in range(max(self.df[f"{key}/num"])):  # Currently only one plant per agent is supported
            # Match the power with the power specified in sgen sheet
            self.df[f"{key}/sizing/power_{num}"] = (self.df.index.map(df['power']) * 1e6).astype('Int64')
            self.df[f"{key}/sizing/capacity_{num}"] = (self.df.index.map(df['capacity']) * 1e6).astype('Int64')
            self.df[f"{key}/sizing/efficiency_{num}"] = self.df.index.map(df['efficiency'])
            self.df[f"{key}/sizing/soc_{num}"] = self.df.index.map(df['soc'])

        # quality
        self.df[f"{key}/quality"] = config["quality"]

        return self.df

    def fill_ems(self, aggregator_type: str, num_agent: int):
        """
            Fills all battery columns
        """

        key = "ems"
        config = self.config[f"{aggregator_type}"][f"{key}"]

        # general
        self.df = self._add_info_simple_aggre(keys=[key], config=config, df=self.df, num_agent=num_agent)

        # Change the values where the value should be randomly picked from a list
        self.df[f'{key}/market/horizon'] = np.random.choice(config['market']['horizon'], size=len(self.df))

    def add_general_info_config(self, key: str) -> pd.DataFrame:

        # fields that exist for all plants
        self.df[f"{key}/owner"] = self._gen_rand_bool_list(n=self.num,
                                                           share_ones=self.config[f"{key}"]["share"])
        self.df[f"{key}/num"] = self._gen_dep_num_list(owner_list=self.df[f"{key}/owner"],
                                                       distr=self.config[f"{key}"]["num"])


        return self.df

    def add_general_info_grid(self, key: str) -> pd.DataFrame:

        # Taken from grid
        self.df[f"{key}/owner"] = (self.load['p_mw'] > 0).astype(int)
        self.df[f"{key}/num"] = self.df[f"{key}/owner"]  # equals owner as only one inflexible load per agent

        # Taken from config

        return self.df

    def _add_general_info_dep(self, key: str, method: str = "inclusive") -> pd.DataFrame:
        """special case battery as it also depends on other plants"""

        # get a list of all the agents that fulfill the dependency requirements
        # note: in both cases the agent needs to have an inflexible load as it is used for sizing
        plants = self.config[f"{key}"]["share_dependent_on"]

        # Check if there are any plants otherwise just treat it as a normal plant
        if len(plants) == 0:
            self.df = self.add_general_info_config(key=key)
            return self.df

        # inclusive: agents needs to have ALL the listed plants to fulfill requirements
        if method == "inclusive":
            list_num = [1] * self.num
            plants += ["inflexible_load"]
            for device in plants:
                list_num = [list_num[idx] * self.df[f"{device}/owner"][idx] for idx, _ in enumerate(list_num)]
        # exclusive: agents needs to have ANY of the listed plants to fulfill requirements
        elif method == "exclusive":
            list_num = [0] * self.num
            for device in plants:
                list_num = [list_num[idx] + self.df[f"{device}/owner"][idx] for idx, _ in enumerate(list_num)]

            # Make sure that there is either zero or one and that owner has inflexible load
            list_num = [min(1, list_num[idx] * self.df[f"inflexible_load/owner"][idx]) for idx, _ in
                        enumerate(list_num)]
        else:
            raise Warning(f"Method '{method}' unknown.")

        # fill the columns
        self.df[f"{key}/owner"] = self._gen_dep_bool_list(list_bool=list_num,
                                                          share_ones=self.config[f"{key}"]["share"])
        self.df[f"{key}/num"] = self._gen_dep_num_list(owner_list=self.df[f"{key}/owner"],
                                                       distr=self.config[f"{key}"]["num"])
        return self.df

    def fill_aggregator(self, dict_agents: dict, dict_agent_devices: dict, aggregator_type: str, num_agent: int,  **kwargs) -> pd.DataFrame:
        """
            Fills all aggregator columns
        """

        # Key in the config file
        #key = "types"

        # Get the config for the key
        config = self.config

        if self.method == 'config':
            self.df = self._aggregator_config(config=config, dict_agents=dict_agents, dict_agent_devices=dict_agent_devices, aggregator_type=aggregator_type, num_agent=num_agent)
        #elif self.method == 'grid':
        #    self.df = self._aggregator_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _aggregator_config(self, config: dict, dict_agents: dict, dict_agent_devices: dict, aggregator_type: str, num_agent: int):

        # ckeck if share of any plant of all aggregators is greater than 1 - Abort scenario
        #number_of_aggregators = {}
        #for key in self.config:

        #    self.num_agents = self.config[f"{key}"]["general"]["number_of_type"]
        #    for agent_devices in dict_agent_devices:
        #        number_of_aggregators[agent_devices] = {}
        #        if agent_devices in self.config[f"{key}"]["manual_represented_group"]["agent_devices"]:
        #            for agent_type in dict_agent_devices[agent_devices]:
        #                number_of_aggregators[agent_devices][agent_type] = []
        #                if agent_type in self.config[f"{key}"]["manual_represented_group"]["agent_types"]:
        #                    for num in self.num_agents:
        #                        number_of_aggregators[agent_devices][agent_type].append(self.config[f"{key}"]["manual_represented_group"]["share"])

        # Get the aggregation method
        method = self.config[f"{aggregator_type}"]["general"]["aggregation_method"]

        # manual method
        if method == ["manual"]:

            # devices_inclusive
            devices_inclusive = self.config[f"{aggregator_type}"]["manual_represented_group"]["devices_inclusive"]

            # make two lists with all agents who have the devices and the owners who are already aggregated
            list_owner, aggregated_owner = self.make_owner_and_aggregator_list(key=aggregator_type,
                                                                               dict_agent_devices=dict_agent_devices,
                                                                               dict_agents=dict_agents,
                                                                               devices_inclusive=devices_inclusive)

            # extract the aggregated agents
            aggregated_agents = self.extract_values(owner_list=list_owner,
                                                          share=self.config[f"{aggregator_type}"]["manual_represented_group"]["share"],
                                                          exclude_values=aggregated_owner)

            # set the aggregated agents to aggregated in the df of the agents
            for aggregated_agent in aggregated_agents:
                for agent_type in dict_agents:
                    idx = 0
                    for agent in dict_agents[agent_type]["general/agent_id"]:
                        if agent == aggregated_agent:
                            dict_agents[agent_type]["general/aggregated_by"][idx] = self.df["general/agent_id"][num_agent]
                        idx += 1

            # add the aggregated agents to the df of the aggregator
            self.df["general/aggregated_agents"][num_agent] = aggregated_agents

            # fill the columns
            self.df = self._add_info_simple_aggre(keys=["manual_represented_group"],
                                                  config=self.config[f"{aggregator_type}"]["manual_represented_group"],
                                                  df=self.df,
                                                  num_agent=num_agent)

        # clustering method
        elif method == ["clustering"]:
            raise NotImplementedError(f"The Method {method} is not implemented yet. "
                                      f"Use another method. "
                                      f"Aborting scenario creation...")

        return self.df

    def extract_values(self, owner_list: list, share: float, exclude_values: list) -> list:

        """extracts values from an owner_list by a share with respect to excluded values

        Args:
            owner_list: list which the values are extracted from
            share: share
            exclude_values: list of excluded values

        Returns:
            extracted_values: list of the extracted values
        """

        # number of values to extract
        num_to_extract = int(round(len(owner_list) * share))

        # Filter values to exclude from extraction
        available_values = [val for val in owner_list if val not in exclude_values]

        # Randomly sample values without the excluded ones
        extracted_values = random.sample(available_values, num_to_extract)

        return extracted_values

    def make_owner_and_aggregator_list(self, key: str, dict_agent_devices: dict, dict_agents: dict, devices_inclusive: str) -> tuple:

        """creates a list of owners of the specific devices and a list of all agents who are already aggregated

        Args:
            key: integer that specifies the number of elements in the list
            dict_agent_devices: dictionary of all available devices
            dict_agents: dictionary of all agents config
            devices_inclusive: exclusive or inclusive

        Returns:
            list_owner: list of owners of the specific devices
            aggregated_owner: list of all agents who are already aggregated

        """

        list_owner = []
        aggregated_owner = []

        # exclusive: agents needs to have ANY of the listed plants to fulfill requirements
        if devices_inclusive == "exclusive":

            # loop over all devices listed in the aggregator's config
            for agent_devices in self.config[f"{key}"]["manual_represented_group"]["agent_devices"]:

                # loop over all agent_types which are specified in the aggregator's config
                for agent_type in self.config[f"{key}"]["manual_represented_group"]["agent_types"]:

                    # add all the agents to the owner list
                    list_owner.extend(dict_agent_devices[agent_devices][agent_type])
                    list_owner = list(set(list_owner))

                    try:
                        # loop over all agents and check if they have the device and create list of owners who are already aggregated
                        idx = 0
                        for agent in dict_agents[agent_type][agent_devices + "/num"]:

                            if agent >= 1:

                                if dict_agents[agent_type]["general/aggregated_by"][idx] is not np.nan:
                                    aggregated_owner.append(dict_agents[agent_type]["general/agent_id"][idx])
                                    aggregated_owner = list(set(aggregated_owner))
                            idx += 1
                    except KeyError:
                        continue

        # inclusive: agents needs to have ALL the listed plants to fulfill requirements
        elif devices_inclusive == "inclusive":

            # loop over all devices listed in the aggregator's config
            for agent_devices in self.config[f"{key}"]["manual_represented_group"]["agent_devices"]:

                # loop over all agent_types which are specified in the aggregator config
                for agent_type in self.config[f"{key}"]["manual_represented_group"]["agent_types"]:

                    try:
                        # loop over all agents
                        idx = 0
                        for agent in dict_agents[agent_type][agent_devices + "/num"]:

                            # check if they have the device
                            if agent >= 1:

                                # create list of owners who are already aggregated
                                if dict_agents[agent_type]["general/aggregated_by"][idx] is not np.nan:
                                    aggregated_owner.append(dict_agents[agent_type]["general/agent_id"][idx])
                                    aggregated_owner = list(set(aggregated_owner))

                                # Make sure they have every other device specified in the aggregator's config
                                all_devices = True
                                for device in self.config[f"{key}"]["manual_represented_group"]["agent_devices"]:
                                    if dict_agents[agent_type][device + "/num"][idx] == 0:
                                        all_devices = False
                                for device in dict_agent_devices:
                                    if device not in self.config[f"{key}"]["manual_represented_group"]["agent_devices"]:
                                        if dict_agents[agent_type][device + "/num"][idx] == 1:
                                            all_devices = False
                                # create list of owners
                                if all_devices:
                                    list_owner.append(dict_agents[agent_type]["general/agent_id"][idx])
                                    list_owner = list(set(list_owner))
                            idx += 1
                    except KeyError:
                        continue

        return list_owner,aggregated_owner

    def _add_info_simple_aggre(cls, keys: list, config: dict, df: pd.DataFrame, num_agent: int, separator: str = "/",
                         preface: str = "", appendix: str = "") -> pd.DataFrame:

        """ This function accepts a  dictionary as argument and fills the according items with the same value
        """

        # Create path from keys
        path_info = cls._create_path(keys=keys, separator=separator)

        # Iterate over all key-value pairs of dict that match the dataframe
        item_list = list(df)
        for item, value in config.items():
            # Check if value is already a value or a dict
            if isinstance(value, dict):
                # If value is a dict call the function again with the new dict
                cls._add_info_simple_aggre(keys=keys + [item], config=value, df=df, separator=separator,
                                     preface=preface, appendix=appendix, num_agent=num_agent)
            else:
                # If value is in item_list, add the value to the dataframe
                if f"{path_info}{preface}{item}{appendix}" in item_list:
                    try:
                        df.loc[num_agent, f"{path_info}{preface}{item}{appendix}"] = value
                    except ValueError:
                        df.loc[num_agent, f"{path_info}{preface}{item}{appendix}"] = str(value)

        return df

    def create_agents_devices_dict(self, dict_agents: dict) -> dict:

        """creates a dictionary of all available devices

        Args:
            dict_agents: dictionary of all agents config


        Returns:
            dict_agent_devices: dictionary of all available devices
        """

        # create a dict for all active agent_devices
        dict_agent_devices = {}
        for agent_devices in self.plants:
            if agent_devices in ["psh", "hydrogen"]:
                continue  # Skips "psh", "hydrogen" agent_devices
            dict_agent_devices[agent_devices] = {}
            for agent_type in self.types:
                if agent_type in ["aggregator"]:
                    continue  # Skips certain agent types

                dict_agent_devices[agent_devices][agent_type] = []
                idx = 0
                try:
                    for agent in dict_agents[agent_type][agent_devices + "/num"]:
                        if agent >= 1:
                            dict_agent_devices[agent_devices][agent_type].append(dict_agents[agent_type]["general/agent_id"][idx])
                        idx += 1
                except KeyError:
                    continue

        return dict_agent_devices

