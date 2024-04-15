__author__ = "TUM-Doepfert"
__credits__ = "jiahechu"
__license__ = ""
__maintainer__ = "TUM-Doepfert"
__email__ = "markus.doepfert@tum.de"

from hamlet.creator.agents.agent_base import AgentBase
import os
import random
import pandas as pd
import numpy as np
from ruamel.yaml.compat import ordereddict
import hamlet.constants as c


class Mfh(AgentBase):
    """
        Sets up mfh agents. Inherits from Agents and Sfhs class.

        Mainly used for excel file creation. Afterwards Mfh class creates the individual agents.
    """

    def __init__(self, input_path: str, config: ordereddict, config_path: str, scenario_path: str, config_root: str):

        # Call the init method of the parent class
        super().__init__(input_path, config, config_path, scenario_path, config_root)

        # Define agent type
        self.type = 'mfh'

        # Path of the input file
        self.input_path = os.path.join(input_path, 'agents', self.type)

        # Number of buildings and apartments
        self.num = self.config["general"]["number_of"]  # number of buildings
        self.num_aps = 0  # number of apartments
        self.num_total = 0  # number of buildings and apartments

        # ID information
        self.ids = {}  # ids of the buildings and their apartments
        self.main_subid = "main"  # sub id of the building

    def create_df_from_config(self) -> pd.DataFrame:
        """
            Function to create the dataframe that makes the Excel sheet
        """

        # Get the number of agents and set the method
        self.num = self.config["general"]["number_of"]
        self.method = 'config'

        # Create the overall dataframe structure for the worksheet
        self.create_df_structure()

        # If no agents are created, return the empty dataframe
        if self.num == 0:
            return self.df

        # Fill the general information in dataframe
        self.fill_general()

        # Fill the inflexible load information in dataframe
        self.fill_inflexible_load()

        # Fill the flexible load information in dataframe
        self.fill_flexible_load()

        # Fill the heat information in dataframe
        self.fill_heat()

        # Fill the dhw information in dataframe
        self.fill_dhw()

        # Fill the pv information in dataframe
        self.fill_pv()

        # Fill the wind information in dataframe
        self.fill_wind()

        # Fill the fixed generation information in dataframe
        self.fill_fixed_gen()

        # Fill the heat pump information in dataframe
        self.fill_hp()

        # Fill the electric vehicle information in dataframe
        self.fill_ev()

        # Fill the battery information in dataframe
        self.fill_battery()

        # Fill the heat storage information in dataframe
        self.fill_heat_storage()

        # Fill the model predictive controller information in dataframe
        self.fill_ems()

        return self.df

    def create_df_from_grid(self, grid: dict, fill_from_config: bool = False, **kwargs) -> pd.DataFrame:

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

        # Get the number of buildings/agents and apartments and set the method
        self.num = self.get_num_from_grid(self.grid['load'], self.type)  # buildings
        self.num_aps = len(self.agents)  # apartments
        self.method = 'grid'

        # Create the overall dataframe structure for the worksheet
        self.create_df_structure()

        # If no agents are created, return the empty dataframe
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

    # def create_df_structure(self):
    #     """
    #         Function to create the dataframe structure with the respective columns
    #     """
    #     # Go through file and create the columns for the mfhs worksheet
    #     columns = ordereddict()
    #     for key, _ in self.config.items():
    #         cols = self.make_list_from_nested_dict(self.config[key], add_string=key)
    #         # Adjust the columns from "general"
    #         if key == c.K_GENERAL:
    #             cols[0] = f"{key}/agent_id"
    #             cols[1] = f"{key}/sub_id"
    #             cols[-1] = f"{key}/market_participant"
    #             cols.insert(1, f"{key}/name")
    #             cols.insert(2, f"{key}/comment")
    #             cols.insert(3, f"{key}/bus")
    #         # Adjust the columns from "inflexible_load"
    #         elif key == c.P_INFLEXIBLE_LOAD:
    #             cols[0] = f"{key}/owner"
    #             cols[5] = f"{key}/sizing/file"
    #             del cols[3]
    #             del cols[1]
    #             max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
    #             cols = cols[:2] + self.repeat_columns(columns=cols[2:4], num=max_num) + cols[4:]
    #         # Adjust the columns from "flexible_load"
    #         elif key == c.P_FLEXIBLE_LOAD:
    #             cols[0] = f"{key}/owner"
    #             cols[5] = f"{key}/sizing/file"
    #             del cols[3]
    #             del cols[1]
    #             max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
    #             cols = cols[:2] + self.repeat_columns(columns=cols[2:5], num=max_num) + cols[5:]
    #         # Adjust the columns from "heat"
    #         elif key == c.P_HEAT:
    #             cols[0] = f"{key}/owner"
    #             cols[3] = f"{key}/sizing/demand"
    #             del cols[1]
    #             max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
    #             cols = cols[:2] + self.repeat_columns(columns=cols[2:5], num=max_num) + cols[5:]
    #         # Adjust the columns from "dhw"
    #         elif key == c.P_DHW:
    #             cols[0] = f"{key}/owner"
    #             del cols[3]
    #             del cols[1]
    #             max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
    #             cols = cols[:2] + self.repeat_columns(columns=cols[2:5], num=max_num) + cols[5:]
    #         # Adjust the columns from "pv"
    #         elif key == c.P_PV:
    #             cols[0] = f"{key}/owner"
    #             del cols[5]
    #             del cols[3]
    #             del cols[1]
    #             max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
    #             cols = cols[:2] + self.repeat_columns(columns=cols[2:7], num=max_num) + cols[7:]
    #         # Adjust the columns from "wind"
    #         elif key == c.P_WIND:
    #             cols[0] = f"{key}/owner"
    #             del cols[5]
    #             del cols[3]
    #             del cols[1]
    #             max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
    #             cols = cols[:2] + self.repeat_columns(columns=cols[2:5], num=max_num) + cols[5:]
    #         # Adjust the columns from "fixed_gen"
    #         elif key == c.P_FIXED_GEN:
    #             cols[0] = f"{key}/owner"
    #             del cols[5]
    #             del cols[3]
    #             del cols[1]
    #             max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
    #             cols = cols[:2] + self.repeat_columns(columns=cols[2:5], num=max_num) + cols[5:]
    #         # Adjust the columns from "hp"
    #         elif key == c.P_HP:
    #             cols[0] = f"{key}/owner"
    #             del cols[3]
    #             del cols[1]
    #             max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
    #             cols = cols[:2] + self.repeat_columns(columns=cols[2:4], num=max_num) + cols[4:]
    #         # Adjust the columns from "ev"
    #         elif key == c.P_EV:
    #             cols[0] = f"{key}/owner"
    #             del cols[3]
    #             del cols[1]
    #             max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
    #             cols = cols[:2] + self.repeat_columns(columns=cols[2:11], num=max_num) + cols[11:]
    #         # Adjust the columns from "battery"
    #         elif key == c.P_BATTERY:
    #             cols[0] = f"{key}/owner"
    #             del cols[4]
    #             del cols[2]
    #             del cols[1]
    #             max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
    #             cols = cols[:2] + self.repeat_columns(columns=cols[2:8], num=max_num) + cols[8:]
    #         # Adjust the columns from "heat_storage"
    #         elif key == c.P_HEAT_STORAGE:
    #             cols[0] = f"{key}/owner"
    #             del cols[4]
    #             del cols[2]
    #             del cols[1]
    #             max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
    #             cols = cols[:2] + self.repeat_columns(columns=cols[2:6], num=max_num) + cols[6:]
    #         # All columns that do not need to be adjusted
    #         elif key in [c.K_EMS]:
    #             pass
    #         else:
    #             raise NotImplementedError(
    #                 f"The configuration file contains a key word ('{key}') that has not been configured in "
    #                 f"the {__class__.__name__} class yet. Aborting scenario creation...")
    #         columns[key] = cols
    #
    #     # Combine all separate lists into one for the dataframe
    #     cols_df = []
    #     for idx, cols in columns.items():
    #         cols_df += cols
    #
    #     # Create dataframe with responding columns
    #     if self.method == 'config':
    #         # normal indexing
    #         self.df = pd.DataFrame(index=range(self.num), columns=cols_df)
    #     elif self.method == 'grid':
    #         # indexing matches the load sheet (all rows that are empty in owner as those are EVs and HPs)
    #         self.df = pd.DataFrame(index=self.agents.index, columns=cols_df)
    #     else:
    #         raise NotImplementedError(f"The method '{self.method}' has not been implemented yet. "
    #                                   f"Aborting scenario creation...")
    #
    #     return self.df

    def _structure_general(self, key, cols):
        cols[0] = f"{key}/agent_id"
        cols[1] = f"{key}/sub_id"
        cols[-1] = f"{key}/market_participant"
        cols.insert(1, f"{key}/name")
        cols.insert(2, f"{key}/comment")
        cols.insert(3, f"{key}/bus")
        cols.insert(4, f"{key}/aggregated_by")
        return cols

    def _structure_inflexible_load(self, key, cols):
        cols[0] = f"{key}/owner"
        cols[5] = f"{key}/sizing/file"
        del cols[3]
        del cols[1]
        max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
        return cols[:2] + self.repeat_columns(columns=cols[2:4], num=max_num) + cols[4:]

    def _structure_flexible_load(self, key, cols):
        cols[0] = f"{key}/owner"
        cols[5] = f"{key}/sizing/file"
        del cols[3]
        del cols[1]
        max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
        return cols[:2] + self.repeat_columns(columns=cols[2:5], num=max_num) + cols[5:]

    def _structure_heat(self, key, cols):
        cols[0] = f"{key}/owner"
        cols[3] = f"{key}/sizing/demand"
        del cols[1]
        max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
        return cols[:2] + self.repeat_columns(columns=cols[2:5], num=max_num) + cols[5:]

    def _structure_dhw(self, key, cols):
        cols[0] = f"{key}/owner"
        del cols[3]
        del cols[1]
        max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
        return cols[:2] + self.repeat_columns(columns=cols[2:5], num=max_num) + cols[5:]

    def _structure_pv(self, key, cols):
        cols[0] = f"{key}/owner"
        del cols[5]
        del cols[3]
        del cols[1]
        max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
        return cols[:2] + self.repeat_columns(columns=cols[2:7], num=max_num) + cols[7:]

    def _structure_wind(self, key, cols):
        cols[0] = f"{key}/owner"
        del cols[5]
        del cols[3]
        del cols[1]
        max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
        return cols[:2] + self.repeat_columns(columns=cols[2:5], num=max_num) + cols[5:]

    def _structure_fixed_gen(self, key, cols):
        cols[0] = f"{key}/owner"
        del cols[5]
        del cols[3]
        del cols[1]
        max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
        return cols[:2] + self.repeat_columns(columns=cols[2:5], num=max_num) + cols[5:]

    def _structure_hp(self, key, cols):
        cols[0] = f"{key}/owner"
        del cols[3]
        del cols[1]
        max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
        return cols[:2] + self.repeat_columns(columns=cols[2:4], num=max_num) + cols[4:]

    def _structure_ev(self, key, cols):
        cols[0] = f"{key}/owner"
        del cols[3]
        del cols[1]
        max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
        return cols[:2] + self.repeat_columns(columns=cols[2:11], num=max_num) + cols[11:]

    def _structure_battery(self, key, cols):
        cols[0] = f"{key}/owner"
        del cols[4]
        del cols[2]
        del cols[1]
        max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
        return cols[:2] + self.repeat_columns(columns=cols[2:8], num=max_num) + cols[8:]

    def _structure_heat_storage(self, key, cols):
        cols[0] = f"{key}/owner"
        del cols[4]
        del cols[2]
        del cols[1]
        max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
        return cols[:2] + self.repeat_columns(columns=cols[2:6], num=max_num) + cols[6:]

    # def fill_general(self) -> pd.DataFrame:
    #     """
    #         Fills all general columns
    #     """
    #     # Key in the config file
    #     key = "general"
    #
    #     # Get the config for the key
    #     config = self.config[f"{key}"]
    #
    #     # Fill the general information based on the method to create the scenario
    #     # In method 'config' first the building agents are created and then the according apartments
    #     if self.method == 'config':
    #         self.df = self._general_config(key=key, config=config)
    #     # In method 'grid' the apartments are first created from the load sheet and afterwards the according buildings
    #     elif self.method == 'grid':
    #         self.df = self._general_grid(key=key, config=config)
    #     else:
    #         raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
    #                                   f"Aborting scenario creation...")
    #
    #     return self.df

    def _general_config(self, key: str, config: dict) -> pd.DataFrame:

        # Fill general information for buildings
        self.df[f"{key}/agent_id"] = self._gen_new_ids(n=self.num)
        self.ids = dict.fromkeys(set(self.df[f"{key}/agent_id"]), [])

        # parameters
        idx_list = self._gen_idx_list_from_distr(n=self.num, distr=config["parameters"]["distribution"])
        self._add_info_indexed(keys=[key, "parameters"], config=config["parameters"], idx_list=idx_list)

        # market participation
        self.df[f"{key}/market_participant"] = self._gen_rand_bool_list(n=self.num,
                                                                        share_ones=config["market_participant_share"])

        # add apartment rows, apartment ids and total area
        self.df = self.df.loc[self.df.index.repeat(self.df[f"{key}/parameters/apartments"] + 1)].reset_index(drop=True)
        for agent_id, _ in self.ids.items():
            # assign main and apartment ids
            aps_num = self.df.loc[self.df[f"{key}/agent_id"] == agent_id, f"{key}/parameters/apartments"].iloc[0]
            self.num_aps += aps_num
            aps_ids = self._gen_new_ids(n=aps_num)
            self.ids[agent_id] = aps_ids
            self.df.loc[self.df[f"{key}/agent_id"] == agent_id, f"{key}/sub_id"] = [self.main_subid] + aps_ids

            # calculate total building area
            self._calc_total(key=key, vals=["parameters/area"], agent_id=agent_id)

        # total number of buildings plus apartments inside
        self.num_total = self.num + self.num_aps

        return self.df

    def _general_grid(self, key: str, config: dict) -> pd.DataFrame:

        # Fill general information for apartments
        self.df[f"{key}/name"] = list(self.agents["name"])
        self.df[f"{key}/bus"] = list(self.agents["bus"])
        self.df[f"{key}/sub_id"] = self._gen_new_ids(n=len(self.df))
        # TODO: adjust as now different approach self.ids = dict.fromkeys(set(self.df[f"{key}/agent_id"]), [])

        # Create a separate df that contains the buildings (i.e. sub ID = self.main_subid)
        df_buildings = self.df.drop_duplicates(subset=f"{key}/bus").copy()

        # Change the indexing to not collide with the apartment indices
        df_buildings.index = range(max(self.df.index) + 1, max(self.df.index) + 1 + len(df_buildings))

        # Assign agent, sub IDs and name
        df_buildings[f"{key}/agent_id"] = self._gen_new_ids(n=self.num)
        df_buildings[f"{key}/sub_id"] = self.main_subid
        df_buildings[f"{key}/name"] = df_buildings[f"{key}/name"].apply(lambda x: x[:-1] + "0")

        # Fill all parameters values
        idx_list = self._gen_idx_list_from_distr(n=self.num, distr=config["parameters"]["distribution"])
        df_buildings = self._add_info_indexed(keys=[key, "parameters"], config=config["parameters"],
                                              idx_list=idx_list, df=df_buildings)

        # Fill market participation
        df_buildings[f"{key}/market_participant"] = self._gen_rand_bool_list(
            n=self.num, share_ones=config["market_participant_share"])

        # Concat both dataframes back to one
        self.df = pd.concat([self.df, df_buildings], axis=0, sort=False)

        # Fill the agent ID based on the same bus (first group by bus, then fill the missing values with the mode)
        self._map_info_on_column(key=f"{key}/agent_id", col=f"{key}/bus")

        # Fill all ./parameters/ values
        self._map_info_on_column(key=f"{key}/parameters/", col=f"{key}/bus", config=config["parameters"])

        # Count the number of apartments based on the number of buses and subtract one to account for main
        self.df[f"{key}/parameters/apartments"] = self.df[f"{key}/bus"].map(self.df[f"{key}/bus"].value_counts()) - 1

        # Fill all market_participant values
        self._map_info_on_column(key=f"{key}/market_participant", col=f"{key}/bus", config=config["parameters"])

        # Sort by index ensuring that 'main' is always first for each building
        self.df['index'] = self.df.index
        self.df = self.df.groupby(f"{key}/bus", sort=False).apply(self.sort_df, col=f"{key}/sub_id",
                                                                  sort_value=self.main_subid)
        self.df.set_index(self.df['index'], inplace=True, drop=True)
        self.df.index.name = None

        # Calculate total building area and number of occupants
        for agent_id in self.df[f"{key}/agent_id"].unique():
            self._calc_total(key=key, vals=["parameters/area"], agent_id=agent_id)
            self._calc_total(key=key, vals=["parameters/occupants"], agent_id=agent_id)

        return self.df

    def _inflexible_load_config(self, key: str, config: dict) -> pd.DataFrame:

        # general
        self._add_general_info_config(key=key)

        # loop through buildings
        max_num = max(config["num"]) if config['share'] else 0
        for agent, aps in self.ids.items():
            # get relevant sub dataframe that either contains building or apartments
            df_sub = self._preprocess_df_sub(agent, aps, key, aps_independent=True)
            # if building does not have device, skip it
            if df_sub is None:
                continue

            # loop through number of plants
            for num in range(max_num):
                # get a list of the agents that own the nth device
                list_owner = np.multiply(np.array(df_sub[f"{key}/num"] - (1 + num) >= 0), 1)

                # sizing
                # file
                df_sub.loc[:, f"{key}/sizing/file_{num}"], idx_list = self._pick_files_from_distr(
                    list_owner=list_owner, distr=config["sizing"]["distribution"], vals=config["sizing"]["demand"],
                    input_path=os.path.join(self.input_path, key),
                    variance=config["sizing"]["demand_deviation"],
                    divisor=1000)
                # demand
                df_sub.loc[:, f"{key}/sizing/demand_{num}"] = self._get_val_from_name(
                    name_list=df_sub.loc[:, f"{key}/sizing/file_{num}"], separator="_", val_idx=1, multiplier=1000)

                # calculate total demand of building
                self._calc_total(key=key, vals=[f"sizing/demand_{num}"], agent_id=agent, df=df_sub)

                # fill main df with sub df
                self.df.loc[df_sub.index, :] = df_sub[:]

        # forecast
        self.df = self._add_info_simple(keys=[key, "fcast"], config=config["fcast"], df=self.df)

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
        self.df[f"{key}/owner"] = (df['demand'] > 0)
        self.df[f"{key}/owner"] = self.df[f"{key}/owner"].fillna(0).astype(int)
        self.df[f"{key}/num"] = self.df[f"{key}/owner"].fillna(0).astype(int)   # equals owner as only one inflexible
                                                                                # load per agent

        # sizing
        for num in range(max(self.df[f"{key}/num"])):  # currently only one device per agent is supported
            # Get demand from load sheet
            self.df[f"{key}/sizing/demand_{num}"] = np.floor(pd.to_numeric(df['demand'] * 1e6, errors='coerce')). \
                astype('Int64')  # the complex formula is needed to convert to integers
            # Check if file column is empty and fill it with the closest file if so
            if df['file'].isnull().all():
                # Pick file that is closest to the demand
                self.df[f"{key}/sizing/file_{num}"] = self._pick_files_by_values(
                    vals=self.df[f"{key}/sizing/demand_{num}"] / 1000, input_path=os.path.join(self.input_path, key))
            else:
                self.df[f"{key}/sizing/file_{num}"] = df['file']

            # Calculate total demand of building
            for agent_id in self.df["general/agent_id"].unique():
                self._calc_total(key=key, vals=[f"sizing/demand_{num}"], agent_id=agent_id)

        # forecast
        self.df = self._add_info_simple(keys=[key, "fcast"], config=config["fcast"], df=self.df)

        return self.df

    def _flexible_load_config(self, key: str, config: dict, **kwargs):
        """
            Fills all flexible_load columns
        """

        # general
        self._add_general_info_config(key=key)

        # loop through buildings
        max_num = max(config["num"]) if config['share'] else 0
        for agent, aps in self.ids.items():
            # get relevant sub dataframe that either contains building or apartments
            df_sub = self._preprocess_df_sub(agent, aps, key, aps_independent=True)
            # if building does not have device, skip it
            if df_sub is None:
                continue

            # loop through number of plants
            for num in range(max_num):
                # get a list of the agents that own the nth device
                list_owner = np.multiply(np.array(df_sub[f"{key}/num"] - (1 + num) >= 0), 1)

                # sizing
                # file
                df_sub.loc[:, f"{key}/sizing/file_{num}"], idx_list = self._pick_files_from_distr(
                    list_owner=list_owner, distr=config["sizing"]["distribution"],
                    vals=config["sizing"]["demand"], input_path=os.path.join(self.input_path, key),
                    variance=config["sizing"]["demand_deviation"], divisor=1000)
                # demand
                df_sub.loc[:, f"{key}/sizing/demand_{num}"] = self._get_val_from_name(
                    name_list=df_sub.loc[:, f"{key}/sizing/file_{num}"], multiplier=1000)
                # time offset
                df_sub.loc[:, f"{key}/sizing/time_offset_{num}"] = self._gen_list_from_idx_list(
                    idx_list=idx_list, distr=config["sizing"]["time_offset"])

                # calculate total demand of building
                self._calc_total(key=key, vals=[f"sizing/demand_{num}"], agent_id=agent, df=df_sub)

                # fill main df with sub df
                self.df.loc[df_sub.index, :] = df_sub[:]

        # forecast
        self.df = self._add_info_simple(keys=[key, "fcast"], config=config["fcast"], df=self.df)

    def _heat_config(self, key: str, config: dict) -> pd.DataFrame:
        """adds the heat from the config file"""
        # general
        self._add_general_info_config(key=key)

        # loop through buildings
        max_num = max(config["num"]) if config['share'] else 0
        for agent, aps in self.ids.items():
            # get relevant sub dataframe that either contains building or apartments
            df_sub = self._preprocess_df_sub(agent, aps, key, aps_independent=True)
            # if building does not have device, skip it
            if df_sub is None:
                continue

            # loop through number of plants
            for num in range(max_num):
                # get a list of the agents that own the nth device
                list_owner = np.multiply(np.array(df_sub[f"{key}/num"] - (1 + num) >= 0), 1)

                # sizing
                idx_list = self._gen_idx_list_from_distr(n=sum(list_owner),
                                                         distr=config["sizing"]["distribution"],
                                                         owner_list=list_owner)

                # add indexed info
                df_sub = self._add_info_indexed(keys=[key, "sizing"], config=config["sizing"],
                                                idx_list=idx_list, df=df_sub, appendix=f"_{num}")

                # postprocessing
                # demand (calculate demand based on efficiency and area of the building)
                df_sub.loc[:, f"{key}/sizing/demand_{num}"] = df_sub[f"general/parameters/efficiency"] * \
                                                              df_sub[f"general/parameters/area"] * 1000

                # file
                # replace all 'linked' with the file of the inflexible load
                df_sub.loc[df_sub[f"{key}/sizing/file_{num}"] == 'linked', f"{key}/sizing/file_{num}"] = df_sub.loc[
                    df_sub[f"{key}/sizing/file_{num}"] == 'linked', f"inflexible_load/sizing/file_0"]
                # shorten the file name to the yearly demand and index
                df_sub.loc[:, f"{key}/sizing/file_{num}"] = [
                    "_".join(item.rsplit(".", 1)[0].split("_")[1:3]) if 'csv' in item else item for item in
                    df_sub[f"{key}/sizing/file_{num}"]]
                df_sub.loc[:, f"{key}/sizing/file_{num}"] = self._pick_files(
                    list_type=df_sub[f"{key}/sizing/file_{num}"], device=f"{key}",
                    input_path=os.path.join(self.input_path, key))

                # calculate total demand of building
                self._calc_total(key=key, vals=[f"sizing/demand_{num}"], agent_id=agent, df=df_sub)

                # fill main df with sub df
                self.df.loc[df_sub.index, :] = df_sub[:]

        # forecast
        self.df = self._add_info_simple(keys=[key, "fcast"], config=config["fcast"], df=self.df)

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

    def _dhw_config(self, key: str, config: dict) -> pd.DataFrame:
        """adds the dhw from the config file"""

        # general
        self._add_general_info_config(key=key)

        # loop through buildings
        max_num = max(config["num"]) if config['share'] else 0
        for agent, aps in self.ids.items():
            # get relevant sub dataframe that either contains building or apartments
            df_sub = self._preprocess_df_sub(agent, aps, key, aps_independent=True)
            # if building does not have device, skip it
            if df_sub is None:
                continue

            # loop through number of plants
            for num in range(max_num):
                # get a list of the agents that own the nth device
                list_owner = np.multiply(np.array(df_sub[f"{key}/num"] - (1 + num) >= 0), 1)

                # sizing
                idx_list = self._gen_idx_list_from_distr(n=sum(list_owner),
                                                         distr=config["sizing"]["distribution"],
                                                         owner_list=list_owner)

                # add indexed info
                df_sub = self._add_info_indexed(keys=[key, "sizing"], config=config["sizing"],
                                                idx_list=idx_list, df=df_sub, appendix=f"_{num}")

                # postprocessing
                # demand (calculate demand based on efficiency and area of the building)
                df_sub.loc[:, f"{key}/sizing/demand_{num}"] *= df_sub[f"general/parameters/occupants"]

                # file
                df_sub.loc[:, f"{key}/sizing/file_{num}"] = self._pick_files(
                    list_type=df_sub[f"{key}/sizing/file_{num}"], device=f"{key}",
                    input_path=os.path.join(self.input_path, key))

                # calculate total demand of building
                self._calc_total(key=key, vals=[f"sizing/demand_{num}"], agent_id=agent, df=df_sub)

                # fill main df with sub df
                self.df.loc[df_sub.index, :] = df_sub[:]

        # forecast
        self.df = self._add_info_simple(keys=[key, "fcast"], config=config["fcast"], df=self.df)

        return self.df

    def _dhw_grid(self, key: str, config: dict, **kwargs) -> pd.DataFrame:
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

    def _pv_config(self, key: str, config: dict) -> pd.DataFrame:

        # general
        self._add_general_info_config(key=key)

        # loop through buildings
        max_num = max(config["num"]) if config['share'] else 0
        for agent, aps in self.ids.items():
            # get relevant sub dataframe that either contains building or apartments
            df_sub = self._preprocess_df_sub(agent, aps, key)
            # if building does not have device, skip it
            if df_sub is None:
                continue

            # loop through number of plants
            for num in range(max_num):
                list_owner = np.multiply(np.array(df_sub[f"{key}/num"] - (1 + num) >= 0), 1)

                # sizing
                idx_list = self._gen_idx_list_from_distr(n=sum(list_owner),
                                                         distr=config["sizing"]["distribution"],
                                                         owner_list=list_owner)
                # add indexed info
                df_sub = self._add_info_indexed(keys=[key, "sizing"], config=config["sizing"],
                                                idx_list=idx_list, df=df_sub, appendix=f"_{num}")
                # postprocessing
                # power
                df_sub.loc[:, f"{key}/sizing/power_{num}"] *= df_sub["inflexible_load/sizing/demand_0"] / 1000
                df_sub.loc[:, f"{key}/sizing/power_{num}"] = self._calc_deviation(
                    idx_list=idx_list, vals=df_sub[f"{key}/sizing/power_{num}"],
                    distr=config["sizing"]["power_deviation"], method="relative")
                df_sub.loc[:, f"{key}/sizing/power_{num}"] = self._round_to_nth_digit(
                    vals=df_sub[f"{key}/sizing/power_{num}"], n=self.n_digits)
                # file
                df_sub.loc[:, f"{key}/sizing/file_{num}"] = self._pick_files(
                    list_type=df_sub[f"{key}/sizing/file_{num}"], device=f"{key}",
                    input_path=os.path.join(self.input_path, key))

            # copy results to main df
            self.df.loc[df_sub.index, :] = df_sub[:]

        # forecast
        self.df = self._add_info_simple(keys=[key, "fcast"], config=config["fcast"], df=self.df)

        # quality
        self.df.loc[:, f"{key}/quality"] = config["quality"]

        # set all other owner values to 0 (only for assurance purposes)
        self.df.loc[self.df[f"{key}/owner"].isnull(), f"{key}/owner"] = 0

        return self.df

    def _pv_grid(self, key: str, config: dict, **kwargs):
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

    def _wind_config(self, key: str, config: dict) -> pd.DataFrame:

        # general
        self._add_general_info_config(key=key)

        # loop through buildings
        max_num = max(config["num"]) if config['share'] else 0
        for agent, aps in self.ids.items():
            # get relevant sub dataframe that either contains building or apartments
            df_sub = self._preprocess_df_sub(agent, aps, key)
            # if building does not have device, skip it
            if df_sub is None:
                continue

            for num in range(max_num):
                list_owner = np.multiply(np.array(df_sub[f"{key}/num"] - (1 + num) >= 0), 1)

                # sizing
                idx_list = self._gen_idx_list_from_distr(n=sum(list_owner),
                                                         distr=config["sizing"]["distribution"],
                                                         owner_list=list_owner)
                # add indexed info
                df_sub = self._add_info_indexed(keys=[key, "sizing"], config=config["sizing"],
                                                idx_list=idx_list, df=df_sub, appendix=f"_{num}")
                # postprocessing
                # power
                df_sub.loc[:, f"{key}/sizing/power_{num}"] *= df_sub["inflexible_load/sizing/demand_0"] / 1000
                df_sub.loc[:, f"{key}/sizing/power_{num}"] = self._calc_deviation(
                    idx_list=idx_list, vals=df_sub[f"{key}/sizing/power_{num}"],
                    distr=config["sizing"]["power_deviation"], method="relative")
                df_sub.loc[:, f"{key}/sizing/power_{num}"] = self._round_to_nth_digit(
                    vals=df_sub[f"{key}/sizing/power_{num}"], n=self.n_digits)
                # file
                df_sub.loc[:, f"{key}/sizing/file_{num}"] = self._pick_files(
                    list_type=df_sub[f"{key}/sizing/file_{num}"], device=f"{key}",
                    input_path=os.path.join(self.input_path, key))

            # copy results to main df
            self.df.loc[df_sub.index, :] = df_sub[:]

        # forecast
        self.df = self._add_info_simple(keys=[key, "fcast"], config=config["fcast"], df=self.df)

        # quality
        self.df.loc[:, f"{key}/quality"] = config["quality"]

        # set all other owner values to 0 (only for assurance purposes)
        self.df.loc[self.df[f"{key}/owner"].isnull(), f"{key}/owner"] = 0

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

    def _fixed_gen_config(self, key: str, config: dict) -> pd.DataFrame:
        """
            Fills all fixed_gen columns based on the config file
        """

        # general
        self._add_general_info_config(key=key)

        # loop through buildings
        max_num = max(config["num"]) if config['share'] else 0
        for agent, aps in self.ids.items():
            # get relevant sub dataframe that either contains building or apartments
            df_sub = self._preprocess_df_sub(agent, aps, key)
            # if building does not have device, skip it
            if df_sub is None:
                continue

            for num in range(max_num):
                list_owner = np.multiply(np.array(df_sub[f"{key}/num"] - (1 + num) >= 0), 1)

                # sizing
                idx_list = self._gen_idx_list_from_distr(n=sum(list_owner),
                                                         distr=config["sizing"]["distribution"],
                                                         owner_list=list_owner)
                # add indexed info
                df_sub = self._add_info_indexed(keys=[key, "sizing"], config=config["sizing"],
                                                idx_list=idx_list, df=df_sub, appendix=f"_{num}")
                # postprocessing
                # power
                df_sub.loc[:, f"{key}/sizing/power_{num}"] *= df_sub["inflexible_load/sizing/demand_0"] / 1000
                df_sub.loc[:, f"{key}/sizing/power_{num}"] = self._calc_deviation(
                    idx_list=idx_list, vals=df_sub[f"{key}/sizing/power_{num}"],
                    distr=config["sizing"]["power_deviation"], method="relative")
                df_sub.loc[:, f"{key}/sizing/power_{num}"] = self._round_to_nth_digit(
                    vals=df_sub[f"{key}/sizing/power_{num}"], n=self.n_digits)
                # file
                df_sub.loc[:, f"{key}/sizing/file_{num}"] = self._pick_files(
                    list_type=df_sub[f"{key}/sizing/file_{num}"], device=f"{key}",
                    input_path=os.path.join(self.input_path, key))

            # copy results to main df
            self.df.loc[df_sub.index, :] = df_sub[:]

        # forecast
        self.df = self._add_info_simple(keys=[key, "fcast"], config=config["fcast"], df=self.df)

        # quality
        self.df.loc[:, f"{key}/quality"] = config["quality"]

        # set all other owner values to 0 (only for assurance purposes)
        self.df.loc[self.df[f"{key}/owner"].isnull(), f"{key}/owner"] = 0

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

    def _hp_config(self, key: str, config: dict) -> pd.DataFrame:

        # general
        self._add_general_info_config(key=key)

        # loop through buildings
        max_num = max(config["num"]) if config['share'] else 0
        for agent, aps in self.ids.items():
            # get relevant sub dataframe that either contains building or apartments
            df_sub = self._preprocess_df_sub(agent, aps, key)
            # if building does not have device, skip it
            if df_sub is None:
                continue

            for num in range(max_num):
                list_owner = np.multiply(np.array(df_sub[f"{key}/num"] - (1 + num) >= 0), 1)

                # sizing
                idx_list = self._gen_idx_list_from_distr(n=sum(list_owner),
                                                         distr=self.config[f"{key}"]["sizing"]["distribution"],
                                                         owner_list=list_owner)
                # add indexed info
                df_sub = self._add_info_indexed(keys=[key, "sizing"], config=config["sizing"],
                                                idx_list=idx_list, df=df_sub, appendix=f"_{num}")
                # postprocessing
                # power
                df_sub.loc[:, f"{key}/sizing/power_{num}"] *= df_sub["heat/sizing/demand_0"] / 2000
                df_sub.loc[:, f"{key}/sizing/power_{num}"] = df_sub.loc[:, f"{key}/sizing/power_{num}"].astype('Int64')
                # TODO: Fix rounding issue. Somehow the only one that is not working (workaround above since int64)
                # df_sub[f"{key}/sizing/power_{num}"] = pd.to_numeric(
                #     df_sub[f"{key}/sizing/power_{num}"], errors='coerce').fillna(
                #     df_sub[f"{key}/sizing/power_{num}"])
                # df_sub.loc[:, f"{key}/sizing/power_{num}"] = self._round_to_nth_digit(
                #     vals=df_sub[f"{key}/sizing/power_{num}"], n=self.n_digits, method='ceil')#.astype('Int64')
                # file
                df_sub.loc[:, f"{key}/sizing/file_{num}"] = self._pick_files(
                    list_type=df_sub[f"{key}/sizing/file_{num}"], device=f"{key}",
                    input_path=os.path.join(self.input_path, key))

            # copy results to main df
            self.df.loc[df_sub.index, :] = df_sub[:]

        # forecast
        self.df = self._add_info_simple(keys=[key, "fcast"], config=config["fcast"], df=self.df)

        # quality
        self.df[f"{key}/quality"] = config["quality"]

        # set all other owner values to 0 (only for assurance purposes)
        self.df.loc[self.df[f"{key}/owner"].isnull(), f"{key}/owner"] = 0

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
            self.df[f"{key}/num"] = 0
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

        # forecast
        self.df = self._add_info_simple(keys=[key, "fcast"], config=config["fcast"], df=self.df)

        # quality
        self.df[f"{key}/quality"] = config["quality"]

        return self.df

    def _ev_config(self, key: str, config: dict) -> pd.DataFrame:

        # general
        self._add_general_info_config(key=key)

        # loop through buildings
        max_num = max(config["num"]) if config['share'] else 0
        for agent, aps in self.ids.items():
            # get relevant sub dataframe that either contains building or apartments
            df_sub = self._preprocess_df_sub(agent, aps, key, aps_independent=True)
            # if building does not have device, skip it
            if df_sub is None:
                continue

            for num in range(max_num):
                # get a list of the agents that own the nth device
                list_owner = np.multiply(np.array(df_sub[f"{key}/num"] - (1 + num) >= 0), 1)

                # sizing
                idx_list = self._gen_idx_list_from_distr(n=sum(list_owner),
                                                         distr=self.config[f"{key}"]["sizing"]["distribution"],
                                                         owner_list=list_owner)
                # add indexed info
                df_sub = self._add_info_indexed(keys=[key, "sizing"], config=config["sizing"],
                                                idx_list=idx_list, df=df_sub, appendix=f"_{num}")
                # postprocessing
                # file
                df_sub.loc[:, f"{key}/sizing/file_{num}"] = self._pick_files(
                    list_type=df_sub[f"{key}/sizing/file_{num}"], device=f"{key}",
                    input_path=os.path.join(self.input_path, key))

            # charging scheme
            df_sub = self._add_info_simple(keys=[key, "charging_scheme"], config=config["charging_scheme"],
                                           df=df_sub)

            # copy results to main df
            self.df.loc[df_sub.index, :] = df_sub[:]

        # forecast
        self.df = self._add_info_simple(keys=[key, "fcast"], config=config["fcast"], df=self.df)

        # quality
        self.df.loc[:, f"{key}/quality"] = config["quality"]

        # set all other owner values to 0 (only for assurance purposes)
        self.df.loc[self.df[f"{key}/owner"].isnull(), f"{key}/owner"] = 0

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
            self.df[f"{key}/num"] = 0
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
            self.df[f"{key}/sizing/consumption_{num}"] = (self.df.index.map(df['consumption']) * 1e6).astype('Int64')
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

    def _battery_config(self, key: str, config: dict) -> pd.DataFrame:

        # general
        self._add_general_info_config(key=key)

        # sizing
        max_num = max(config["num"]) if config['share'] else 0
        for agent, aps in self.ids.items():
            # get relevant sub dataframe that either contains building or apartments
            df_sub = self._preprocess_df_sub(agent, aps, key)
            # if building does not have device, skip it
            if df_sub is None:
                continue

            # check which agents have the required plants to also have a battery
            df_sub = self._sub_dev_dependent_on(key=key, df=df_sub, config=config)

            for num in range(max_num):
                # get a list of the agents that own the nth device
                list_owner = np.multiply(np.array(df_sub[f"{key}/num"] - (1 + num) >= 0), 1)

                # sizing
                idx_list = self._gen_idx_list_from_distr(n=sum(list_owner),
                                                         distr=config["sizing"]["distribution"],
                                                         owner_list=list_owner)
                # add indexed info
                df_sub = self._add_info_indexed(keys=[key, "sizing"], config=config["sizing"],
                                                idx_list=idx_list, df=df_sub, appendix=f"_{num}")
                # postprocessing
                # power
                df_sub.loc[:, f"{key}/sizing/power_{num}"] *= df_sub.loc[:, "inflexible_load/sizing/demand_0"] / 1000
                df_sub[f"{key}/sizing/power_{num}"] = pd.to_numeric(
                    df_sub[f"{key}/sizing/power_{num}"], errors='coerce').fillna(
                    df_sub[f"{key}/sizing/power_{num}"])
                df_sub.loc[:, f"{key}/sizing/power_{num}"] = self._round_to_nth_digit(
                    vals=df_sub[f"{key}/sizing/power_{num}"], n=self.n_digits)
                # capacity
                df_sub.loc[:, f"{key}/sizing/capacity_{num}"] *= df_sub.loc[:, "inflexible_load/sizing/demand_0"] / 1000
                df_sub[f"{key}/sizing/capacity_{num}"] = pd.to_numeric(
                    df_sub[f"{key}/sizing/capacity_{num}"], errors='coerce').fillna(
                    df_sub[f"{key}/sizing/capacity_{num}"])
                df_sub.loc[:, f"{key}/sizing/capacity_{num}"] = self._round_to_nth_digit(
                    vals=df_sub[f"{key}/sizing/capacity_{num}"], n=self.n_digits)

            # copy results to main df
            self.df.loc[df_sub.index, :] = df_sub[:]

        # quality
        self.df.loc[:, f"{key}/quality"] = str(config["quality"])

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
            self.df[f"{key}/num"] = 0
            return self.df

        # Check if there is more than one plant per agent
        if df.index.duplicated().any():
            raise NotImplementedError(f"More than one {key} per agent is not implemented yet. "
                                      f"Combine the {key} into one. "
                                      f"Aborting scenario creation...")

        # general
        self.df[f"{key}/num"] = self.df.index.map(df['owner'].value_counts()).fillna(0).astype('Int64')
        self.df[f"{key}/owner"] = (self.df[f"{key}/num"] > 0).fillna(0).astype('Int64')  # all agents that have battery

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

    def _heat_storage_config(self, key: str, config: dict) -> pd.DataFrame:

        # general
        self._add_general_info_config(key=key)

        # sizing
        max_num = max(config["num"]) if config['share'] else 0
        for agent, aps in self.ids.items():
            # get relevant sub dataframe that either contains building or apartments
            df_sub = self._preprocess_df_sub(agent, aps, key)
            # if building does not have device, skip it
            if df_sub is None:
                continue

            # check which agents have the required plants to also have a battery
            df_sub = self._sub_dev_dependent_on(key=key, df=df_sub, config=config)

            for num in range(max_num):
                # get a list of the agents that own the nth device
                list_owner = np.multiply(np.array(df_sub[f"{key}/num"] - (1 + num) >= 0), 1)

                # sizing
                idx_list = self._gen_idx_list_from_distr(n=sum(list_owner),
                                                         distr=config["sizing"]["distribution"],
                                                         owner_list=list_owner)
                # add indexed info
                df_sub = self._add_info_indexed(keys=[key, "sizing"], config=config["sizing"],
                                                idx_list=idx_list, df=df_sub, appendix=f"_{num}")
                # postprocessing
                # power
                df_sub.loc[:, f"{key}/sizing/power_{num}"] *= df_sub.loc[:, "heat/sizing/demand_0"] / 2000
                df_sub[f"{key}/sizing/power_{num}"] = pd.to_numeric(
                    df_sub[f"{key}/sizing/power_{num}"], errors='coerce').fillna(
                    df_sub[f"{key}/sizing/power_{num}"])
                df_sub.loc[:, f"{key}/sizing/power_{num}"] = self._round_to_nth_digit(
                    vals=df_sub[f"{key}/sizing/power_{num}"], n=self.n_digits)
                # capacity
                df_sub.loc[:, f"{key}/sizing/capacity_{num}"] *= df_sub.loc[:, "heat/sizing/demand_0"] / 2000
                df_sub[f"{key}/sizing/capacity_{num}"] = pd.to_numeric(
                    df_sub[f"{key}/sizing/capacity_{num}"], errors='coerce').fillna(
                    df_sub[f"{key}/sizing/capacity_{num}"])
                df_sub.loc[:, f"{key}/sizing/capacity_{num}"] = self._round_to_nth_digit(
                    vals=df_sub[f"{key}/sizing/capacity_{num}"], n=self.n_digits)

            # copy results to main df
            self.df.loc[df_sub.index, :] = df_sub[:]

        # quality
        self.df.loc[:, f"{key}/quality"] = str(config["quality"])

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

    def _get_df_sub(self, agent: str, aps: list, aps_independent: bool = True) -> pd.DataFrame:
        """ Returns the sub-dataframe with either the building or the apartments of the building"""
        if aps_independent:
            # df_sub are the apartments
            return self.df.loc[self.df[f"general/sub_id"].isin(aps)][:]
        else:
            # df_sub is the building
            return self.df.loc[(self.df[f"general/agent_id"] == agent)
                               & (self.df[f"general/sub_id"] == self.main_subid)][:]

    def _add_general_info_config(self, key: str) -> pd.DataFrame:

        # fields that exist for all plants
        self.df.loc[self.df[f"general/sub_id"] == self.main_subid, f"{key}/owner"] = \
            self._gen_rand_bool_list(n=self.num, share_ones=self.config[f"{key}"]["share"])

        return self.df

    def _preprocess_df_sub(self, agent: str, aps: list, key: str, aps_independent: bool = None):
        """Does all the preprocessing and obtains the relevant df_sub"""

        # get the information about the building
        df_building = self.df.loc[(self.df[f"general/agent_id"] == agent)
                                  & (self.df[f"general/sub_id"] == self.main_subid)][:]
        # check if building has device
        owner = df_building[f"{key}/owner"].item()
        # check if the apartments are independent of building or device is to be calculated for entire building
        # note: can be forced via the optional parameter
        if aps_independent is not None:
            aps_independent = aps_independent
        else:
            aps_independent = df_building["general/parameters/apartments_independent"].item()

        # get sub dataframe if building has device, otherwise set ownership column to zero
        if owner:
            # get the sub dataframe depending on if calculations are for building or apartments
            df_sub = self._get_df_sub(agent=agent, aps=aps, aps_independent=aps_independent)
            if aps_independent:
                # set ownership of building to 0
                self.df.loc[df_building.index, [f"{key}/owner", f"{key}/num"]] = 0
                # calculate number of apartments with device
                df_sub.loc[:, f"{key}/owner"] = self._gen_rand_bool_list(
                    n=len(df_sub), share_ones=random.choice(self.config[f"{key}"]["share_aps"]))
            else:
                # set ownership of apartments to 0
                self.df.loc[self.df[f"general/sub_id"].isin(aps), [f"{key}/owner", f"{key}/num"]] = 0
        else:
            # set ownership of building and all apartments to 0
            self.df.loc[df_building.index, [f"{key}/owner", f"{key}/num"]] = 0
            self.df.loc[self.df[f"general/sub_id"].isin(aps), [f"{key}/owner", f"{key}/num"]] = 0
            df_sub = None

        if df_sub is not None:
            # calculate number of plants for each agent
            df_sub.loc[:, f"{key}/num"] = self._gen_dep_num_list(owner_list=df_sub[f"{key}/owner"],
                                                                 distr=self.config[f"{key}"]["num"])

        return df_sub

    def _calc_total(self, key: str, vals: list, agent_id: str, df: pd.DataFrame = None) -> pd.DataFrame:
        """calculates the total value and inserts it into building row of dataframe"""
        for val in vals:
            if df is not None:
                total = np.nansum(df.loc[:, f"{key}/{val}"])
            else:
                total = np.nansum(self.df.loc[(self.df["general/agent_id"] == agent_id)
                                              & (self.df["general/sub_id"] != self.main_subid), f"{key}/{val}"])
            # insert total into building row
            self.df.loc[(self.df["general/agent_id"] == agent_id) & (self.df["general/sub_id"] == self.main_subid),
            f"{key}/{val}"] = total

        return self.df

    def _sub_dev_dependent_on(self, key: str, df: pd.DataFrame, config: dict) -> pd.DataFrame:
        """Checks which agents are eligible and assigns a number of plants to each depending on share"""
        # TODO: Not properly working (see battery and heat_storage) --> make similar to that of sfh
        # check if agent qualifies for battery and set index appropriately
        list_num = [1] * len(df)
        for device in config["share_dependent_on"]:
            list_num = [list_num[idx] * df[f"{device}/owner"].iloc[idx] for idx, _ in enumerate(list_num)]
        if df["general/sub_id"].iloc[0] != self.main_subid:
            df.loc[:, f"{key}/owner"] = self._gen_dep_bool_list(list_bool=list_num,
                                                                share_ones=random.choice(config["share_aps"]))
        df.loc[:, f"{key}/num"] = self._gen_dep_num_list(owner_list=df[f"{key}/owner"], distr=config["num"])

        return df

    @staticmethod
    def sort_df(df: pd.DataFrame, col, sort_value):
        df_main = df[df[col] == sort_value]
        df_others = df[df[col] != sort_value]
        return pd.concat([df_main, df_others])