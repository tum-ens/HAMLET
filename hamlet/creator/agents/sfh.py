__author__ = "TUM-Doepfert"
__credits__ = "jiahechu"
__license__ = ""
__maintainer__ = "TUM-Doepfert"
__email__ = "markus.doepfert@tum.de"

from hamlet.creator.agents.agent_base import AgentBase
import os
import pandas as pd
import numpy as np
from ruamel.yaml.compat import ordereddict
from pprint import pprint
import hamlet.constants as c
from typing import Callable
import random


class Sfh(AgentBase):
    """
        Sets up sfh agents. Inherits from Agents class.

        Mainly used for Excel file creation. Afterwards Sfh class creates the individual agents.
    """

    def __init__(self, input_path: str, config: ordereddict, config_path: str, scenario_path: str, config_root: str):

        # Call the init method of the parent class
        super().__init__(input_path, config, config_path, scenario_path, config_root)

        # Define agent type
        self.type = c.A_SFH

        # Path of the input file
        self.input_path = os.path.join(input_path, 'agents', self.type)

    def create_df_from_config(self) -> pd.DataFrame:
        """Function to create the dataframe that makes the Excel sheet
        """

        # Get the number of agents and set the method
        self.num = self.config[c.K_GENERAL]["number_of"]
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

        # Fill the energy management system information in dataframe
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
        self.agents = self.load[self.load['load_type'] == c.P_INFLEXIBLE_LOAD]

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

    def create_df_structure(self):
        """
            Function to create the dataframe structure with the respective columns
        """

        # Go through file and create the columns for the sfhs worksheet
        columns = ordereddict()
        for key, _ in self.config.items():
            cols = self.make_list_from_nested_dict(self.config[key], add_string=key)
            # Adjust the columns from "general"
            if key == c.K_GENERAL:
                cols[0] = f"{key}/agent_id"
                cols[-1] = f"{key}/market_participant"
                del cols[1]
                cols.insert(1, f"{key}/name")
                cols.insert(2, f"{key}/comment")
                cols.insert(3, f"{key}/bus")
                cols.insert(4, f"{key}/aggregated_by")
            # Adjust the columns from c.P_INFLEXIBLE_LOAD
            elif key == c.P_INFLEXIBLE_LOAD:
                cols[0] = f"{key}/owner"
                cols[4] = f"{key}/sizing/file"
                del cols[2]
                max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
                cols = cols[:2] + self.repeat_columns(columns=cols[2:4], num=max_num) + cols[4:]
            # Adjust the columns from "flexible_load"
            elif key == c.P_FLEXIBLE_LOAD:
                cols[0] = f"{key}/owner"
                cols[4] = f"{key}/sizing/file"
                del cols[2]
                max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
                cols = cols[:2] + self.repeat_columns(columns=cols[2:5], num=max_num) + cols[5:]
            # Adjust the columns from "heat"
            elif key == c.P_HEAT:
                cols[0] = f"{key}/owner"
                cols[2] = f"{key}/sizing/demand"
                max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
                cols = cols[:2] + self.repeat_columns(columns=cols[2:5], num=max_num) + cols[5:]
            # Adjust the columns from "dhw"
            elif key == c.P_DHW:
                cols[0] = f"{key}/owner"
                del cols[2]
                max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
                cols = cols[:2] + self.repeat_columns(columns=cols[2:5], num=max_num) + cols[5:]
            # Adjust the columns from "pv"
            elif key == c.P_PV:
                cols[0] = f"{key}/owner"
                del cols[4]
                del cols[2]
                max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
                cols = cols[:2] + self.repeat_columns(columns=cols[2:7], num=max_num) + cols[7:]
            # Adjust the columns from "wind"
            elif key == c.P_WIND:
                cols[0] = f"{key}/owner"
                del cols[4]
                del cols[2]
                max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
                cols = cols[:2] + self.repeat_columns(columns=cols[2:5], num=max_num) + cols[5:]
            # Adjust the columns from "fixed_gen"
            elif key == c.P_FIXED_GEN:
                cols[0] = f"{key}/owner"
                del cols[4]
                del cols[2]
                max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
                cols = cols[:2] + self.repeat_columns(columns=cols[2:5], num=max_num) + cols[5:]
            # Adjust the columns from "hp"
            elif key == c.P_HP:
                cols[0] = f"{key}/owner"
                del cols[2]
                max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
                cols = cols[:2] + self.repeat_columns(columns=cols[2:4], num=max_num) + cols[4:]
            # Adjust the columns from "ev"
            elif key == c.P_EV:
                cols[0] = f"{key}/owner"
                del cols[2]
                max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
                cols = cols[:2] + self.repeat_columns(columns=cols[2:11], num=max_num) + cols[11:]
            # Adjust the columns from "battery"
            elif key == c.P_BATTERY:
                cols[0] = f"{key}/owner"
                del cols[3]
                del cols[1]
                max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
                cols = cols[:2] + self.repeat_columns(columns=cols[2:8], num=max_num) + cols[8:]
            # Adjust the columns from "heat_storage"
            elif key == c.P_HEAT_STORAGE:
                cols[0] = f"{key}/owner"
                del cols[3]
                del cols[1]
                max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
                cols = cols[:2] + self.repeat_columns(columns=cols[2:6], num=max_num) + cols[6:]
            # All columns that do not need to be adjusted
            elif key in [c.K_EMS]:
                pass
            else:
                raise NotImplementedError(
                    f"The configuration file contains a key word ('{key}') that has not been configured in "
                    f"the {__class__.__name__} class yet. Aborting scenario creation...")

            # Add the columns to the dictionary
            columns[key] = cols

        # Combine all separate lists into one for the dataframe
        cols_df = []
        for idx, cols in columns.items():
            cols_df += cols

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

    def fill_general(self):
        """
            Fills all general columns
        """
        # TODO: Change structure such as done in mfh
        # Key in the config file
        key = c.K_GENERAL

        # Get the config for the key
        config = self.config[f"{key}"]

        # general
        self.df[f"{key}/agent_id"] = self._gen_new_ids_sfh(n=self.num)

        # parameters
        idx_list = self._gen_idx_list_from_distr(n=self.num, distr=config["parameters"]["distribution"])
        self._add_info_indexed(keys=[key, "parameters"], config=config["parameters"], idx_list=idx_list)

        # market participation
        self.df[f"{key}/market_participant"] = self._gen_rand_bool_list(n=self.num,
                                                                        share_ones=config["market_participant_share"])

        # If the method is grid, fill the name, comment and bus columns from grid file
        if self.method == 'config':
            self.df = self._general_config(key=key, config=config)
        elif self.method == 'grid':
            self.df = self._general_grid(key=key, config=config)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _general_config(self, key: str, config: ordereddict) -> pd.DataFrame:
        # Note: In the future this will need to assign a bus from the artificial grid
        return self.df

    def _general_grid(self, key: str, config: ordereddict, **kwargs) -> pd.DataFrame:
        self.df[f"{key}/name"] = list(self.agents["name"])
        self.df[f"{key}/bus"] = list(self.agents["bus"])

        return self.df

    def fill_columns(self, key: str, config_method: Callable, grid_method: Callable, **kwargs) -> pd.DataFrame:
        """
            Fills all columns based on the provided key
        """

        # Get the config for the key if it exists
        try:
            config = self.config[key]
        except KeyError:
            return self.df

        if self.method == 'config':
            self.df = config_method(key=key, config=config)
        elif self.method == 'grid':
            self.df = grid_method(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _inflexible_load_config(self, key: str, config: dict) -> pd.DataFrame:
        """adds the inflexible load from the config file"""

        # general
        self.df = self.add_general_info_config(key=key)

        # sizing
        max_num = max(config["num"])
        for num in range(max_num):
            # get a list of the agents that own the nth device
            list_owner = np.multiply(np.array(self.df[f"{key}/num"] - (1 + num) >= 0), 1)

            # sizing
            # file
            self.df[f"{key}/sizing/file_{num}"], _ = self._pick_files_from_distr(
                list_owner=list_owner, distr=config["sizing"]["distribution"], vals=config["sizing"]["demand"],
                input_path=os.path.join(self.input_path, key),
                variance=config["sizing"]["demand_deviation"],
                divisor=1000)
            # demand
            self.df[f"{key}/sizing/demand_{num}"] = self._get_val_from_name(
                name_list=self.df[f"{key}/sizing/file_{num}"], separator="_", val_idx=1, multiplier=1000)

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
        self.df[f"{key}/owner"] = (df['demand'] > 0).astype(int)
        self.df[f"{key}/num"] = self.df[f"{key}/owner"]  # equals owner as only one inflexible load per agent

        # sizing
        for num in range(max(self.df[f"{key}/num"])):  # currently only one device per agent is supported
            # Get demand from load sheet
            self.df[f"{key}/sizing/demand_{num}"] = (round(df['demand'] * 1e6)).astype('Int64')
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

    def _flexible_load_config(self, key: str, config: dict) -> pd.DataFrame:
        """
            Fills all flexible_load columns
        """

        # general
        self.df = self.add_general_info_config(key=key)

        # sizing
        max_num = max(config["num"])
        for num in range(max_num):
            # get a list of the agents that own the nth device
            list_owner = np.multiply(np.array(self.df[f"{key}/num"] - (1 + num) >= 0), 1)

            # sizing
            # file
            self.df[f"{key}/sizing/file_{num}"], idx_list = self._pick_files_from_distr(
                list_owner=list_owner, distr=config["sizing"]["distribution"], vals=config["sizing"]["demand"],
                input_path=os.path.join(self.input_path, key),
                variance=config["sizing"]["demand_deviation"],
                divisor=1000)
            # demand
            self.df[f"{key}/sizing/demand_{num}"] = self._get_val_from_name(
                name_list=self.df[f"{key}/sizing/file_{num}"], separator="_", val_idx=1, multiplier=1000)
            # time offset
            self.df[f"{key}/sizing/time_offset_{num}"] = self._gen_list_from_idx_list(
                idx_list=idx_list, distr=self.config[f"{key}"]["sizing"]["time_offset"])

        # forecast
        self.df = self._add_info_simple(keys=[key, "fcast"], config=config["fcast"], df=self.df)

        return self.df

    def _heat_config(self, key: str, config: dict) -> pd.DataFrame:
        """adds the heat from the config file"""

        # general
        self.df = self.add_general_info_config(key=key)

        # sizing
        max_num = max(config["num"]) if config['share'] else 0
        for num in range(max_num):
            # get a list of the agents that own the nth device
            list_owner = np.multiply(np.array(self.df[f"{key}/num"] - (1 + num) >= 0), 1)

            # sizing
            idx_list = self._gen_idx_list_from_distr(n=sum(list_owner),
                                                     distr=config["sizing"]["distribution"],
                                                     owner_list=list_owner)

            # add indexed info
            self._add_info_indexed(keys=[key, "sizing"], config=config["sizing"],
                                   idx_list=idx_list, appendix=f"_{num}")

            # postprocessing
            # demand (calculate demand based on efficiency and area of the building)
            self.df[f"{key}/sizing/demand_{num}"] = self.df[f"general/parameters/efficiency"] * \
                                                    self.df[f"general/parameters/area"] * 1000
            # print(self.df.loc[:, self.df.filter(like=key).columns].to_string())

            # file
            # replace all 'linked' with the file of the inflexible load
            self.df.loc[self.df[f"{key}/sizing/file_{num}"] == 'linked', f"{key}/sizing/file_{num}"] = self.df.loc[
                self.df[f"{key}/sizing/file_{num}"] == 'linked', f"{c.P_INFLEXIBLE_LOAD}/sizing/file_0"]
            # shorten the file name to the yearly demand and index
            self.df[f"{key}/sizing/file_{num}"] = [
                "_".join(item.rsplit(".", 1)[0].split("_")[1:3]) if 'csv' in item else item for item in
                self.df[f"{key}/sizing/file_{num}"]]
            self.df[f"{key}/sizing/file_{num}"] = self._pick_files(list_type=self.df[f"{key}/sizing/file_{num}"],
                                                                   device=f"{key}",
                                                                   input_path=os.path.join(self.input_path, key))

        # forecast
        self.df = self._add_info_simple(keys=[key, "fcast"], config=config["fcast"], df=self.df)

        return self.df

    def _heat_grid(self, key: str, config: dict, **kwargs) -> pd.DataFrame:
        """adds the heat from the grid"""

        # Get all the kwargs
        from_config_if_empty = kwargs.get('from_config_if_empty', False)

        # Drop all rows that do not contain the load type and set the index to the owner
        df = self.load[self.load['load_type'] == key]

        # Check if file contains the plant type (load > 0), if not use the config file to generate it
        if from_config_if_empty and df['load_type'].value_counts().get(key, 0) == 0:
            self._inflexible_load_config(key=key, config=config)
            return self.df

        # Check if there are any ev plants, if not set all owners to 0 and return
        if len(df) == 0:
            self.df[f"{key}/owner"] = 0
            return self.df

        # Set the index to the owner
        df = df.set_index('owner', drop=False)

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
            self.df[f"{key}/sizing/demand_{num}"] = (self.df.index.map(round(df['demand'] * 1e6))).astype('Int64')
            self.df[f"{key}/sizing/file_{num}"] = self.df.index.map(df['file_add'])
            self.df[f"{key}/sizing/temperature_{num}"] = self.df.index.map(df['temperature'])

        # forecast
        self.df = self._add_info_simple(keys=[key, "fcast"], config=config["fcast"], df=self.df)

        return self.df

    def _dhw_config(self, key: str, config: dict) -> pd.DataFrame:
        """adds the dhw from the config file"""

        # general
        self.df = self.add_general_info_config(key=key)

        # sizing
        max_num = max(config["num"]) if config['share'] else 0
        for num in range(max_num):
            # get a list of the agents that own the nth device
            list_owner = np.multiply(np.array(self.df[f"{key}/num"] - (1 + num) >= 0), 1)

            # sizing
            idx_list = self._gen_idx_list_from_distr(n=sum(list_owner),
                                                     distr=config["sizing"]["distribution"],
                                                     owner_list=list_owner)

            # add indexed info
            self._add_info_indexed(keys=[key, "sizing"], config=config["sizing"],
                                   idx_list=idx_list, appendix=f"_{num}")

            # postprocessing
            # demand (calculate demand based on number of occupants)
            self.df[f"{key}/sizing/demand_{num}"] *= self.df[f"general/parameters/occupants"]
            self.df[f"{key}/sizing/demand_{num}"] = (self.df[f"{key}/sizing/demand_{num}"]).astype('Int64')

            # file
            self.df[f"{key}/sizing/file_{num}"] = self._pick_files(list_type=self.df[f"{key}/sizing/file_{num}"],
                                                                   device=f"{key}",
                                                                   input_path=os.path.join(self.input_path, key))

        # forecast
        self.df = self._add_info_simple(keys=[key, "fcast"], config=config["fcast"], df=self.df)

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
            self.df[f"{key}/sizing/demand_{num}"] = (self.df.index.map(round(df['demand'] * 1e6))).astype('Int64')
            self.df[f"{key}/sizing/file_{num}"] = self.df.index.map(df['heat_file'])
            self.df[f"{key}/sizing/temperature_{num}"] = self.df.index.map(df['temperature'])

        # forecast
        self.df = self._add_info_simple(keys=[key, "fcast"], config=config["fcast"], df=self.df)

        return self.df

    def _pv_config(self, key: str, config: dict) -> pd.DataFrame:
        """adds the inflexible load from the config file"""

        # general
        self.df = self.add_general_info_config(key=key)

        # sizing
        max_num = max(config["num"]) if config['share'] else 0  # max number of devices per agent
        for num in range(max_num):
            # get a list of the agents that own the nth device
            list_owner = np.multiply(np.array(self.df[f"{key}/num"] - (1 + num) >= 0), 1)

            # sizing
            idx_list = self._gen_idx_list_from_distr(n=sum(list_owner),
                                                     distr=config["sizing"]["distribution"],
                                                     owner_list=list_owner)
            # add indexed info
            self._add_info_indexed(keys=[key, "sizing"], config=config["sizing"],
                                   idx_list=idx_list, appendix=f"_{num}")
            # postprocessing
            # power
            self.df[f"{key}/sizing/power_{num}"] *= self.df[f"{c.P_INFLEXIBLE_LOAD}/sizing/demand_0"] / 1000
            self.df[f"{key}/sizing/power_{num}"] = self._calc_deviation(idx_list=idx_list,
                                                                        vals=self.df[f"{key}/sizing/power_{num}"],
                                                                        distr=config["sizing"]["power_deviation"],
                                                                        method="relative")
            self.df[f"{key}/sizing/power_{num}"] = self._round_to_nth_digit(vals=self.df[f"{key}/sizing/power_{num}"],
                                                                            n=self.n_digits)
            # file
            self.df[f"{key}/sizing/file_{num}"] = self._pick_files(list_type=self.df[f"{key}/sizing/file_{num}"],
                                                                   device=f"{key}",
                                                                   input_path=os.path.join(self.input_path, key))

        # forecast
        self.df = self._add_info_simple(keys=[key, "fcast"], config=config["fcast"], df=self.df)

        # quality
        self.df[f"{key}/quality"] = config["quality"]

        return self.df

    def _pv_grid(self, key: str, config: dict, **kwargs) -> pd.DataFrame:
        """adds the pv plants from the grid file"""

        # Get all the kwargs
        from_config_if_empty = kwargs.get('from_config_if_empty', False)

        # Drop all rows that do not contain the plant type and set the index to the owner
        try:
            df = self.sgen[self.sgen['plant_type'] == key] #.set_index('owner', drop=False)
        except KeyError:
            return self.df

        # Check if file contains the plant type (count > 0), if not use the config file to generate it
        if from_config_if_empty and df['sgen_type'].value_counts()[key] == 0:
            self._pv_config(key=key, config=config)
            return self.df

        # Check if there are any pv plants, if not set all owners to 0 and return
        if len(df) == 0:
            self.df[f"{key}/owner"] = 0
            self.df[f"{key}/num"] = 0
            return self.df

        # general
        self.df[f"{key}/num"] = self.df.index.map(df['owner'].value_counts()).fillna(0).astype('Int64')
        self.df[f"{key}/owner"] = (self.df[f"{key}/num"] > 0).fillna(0).astype('Int64')  # all agents that have plant

        # Split the dataframe into dataframes with at most one device per owner
        max_devices = df['owner'].value_counts().max()  # Maximum number of devices for a single owner
        plant_dfs = []
        for num in range(max_devices):
            device_df = df.groupby('owner').nth(num)
            plant_dfs.append(device_df)

        # sizing
        # Map each dataframe to the self.df dataframe
        for num, plant_df in enumerate(plant_dfs):
            # Match the power with the power specified in sgen sheet
            self.df[f"{key}/sizing/power_{num}"] = self.df.index.map(
                round(plant_df.set_index('owner')['power'] * 1e6)).astype('Int64')

            # Check if file column exists
            if 'file' in df and not df['file'].isnull().all():
                # Fill rows with values from sgen sheet
                self.df[f"{key}/sizing/file_{num}"] = self.df.index.map(plant_df.set_index('owner')['file'])
                self.df[f"{key}/sizing/orientation_{num}"] = self.df.index.map(
                    plant_df.set_index('owner')['orientation']).fillna(0)
                self.df[f"{key}/sizing/angle_{num}"] = self.df.index.map(
                    plant_df.set_index('owner')['angle']).fillna(0)
            # If file column does not exist, check if orientation and angle columns exist and all rows are filled
            elif 'orientation' in df and 'angle' in df \
                    and not df['orientation'].isnull().any() and not df['angle'].isnull().any():
                # Fill orientation and angle from sgen sheet
                self.df[f"{key}/sizing/orientation_{num}"] = self.df.index.map(
                    plant_df.set_index('owner')['orientation'])
                self.df[f"{key}/sizing/angle_{num}"] = self.df.index.map(
                    plant_df.set_index('owner')['angle'])
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

            # Pick random value from the config file for controllability
            self.df[f"{key}/sizing/controllable_{num}"] = random.choice(config["sizing"]["controllable"])

            # forecast
        self.df = self._add_info_simple(keys=[key, "fcast"], config=config["fcast"], df=self.df)

        # quality
        self.df[f"{key}/quality"] = config["quality"]

        return self.df

    def _wind_config(self, key: str, config: dict) -> pd.DataFrame:

        # general
        self.df = self.add_general_info_config(key=key)

        # sizing
        max_num = max(config["num"]) if config['share'] else 0  # max number of devices per agent
        for num in range(max_num):
            # get a list of the agents that own the nth device
            list_owner = np.multiply(np.array(self.df[f"{key}/num"] - (1 + num) >= 0), 1)

            # sizing
            idx_list = self._gen_idx_list_from_distr(n=sum(list_owner),
                                                     distr=config["sizing"]["distribution"],
                                                     owner_list=list_owner)
            # add indexed info
            self._add_info_indexed(keys=[key, "sizing"], config=config["sizing"],
                                   idx_list=idx_list, appendix=f"_{num}")
            # postprocessing
            # power
            self.df[f"{key}/sizing/power_{num}"] *= self.df[f"{c.P_INFLEXIBLE_LOAD}/sizing/demand_0"] / 1000
            self.df[f"{key}/sizing/power_{num}"] = self._calc_deviation(idx_list=idx_list,
                                                                        vals=self.df[f"{key}/sizing/power_{num}"],
                                                                        distr=config["sizing"]["power_deviation"],
                                                                        method="relative")
            self.df[f"{key}/sizing/power_{num}"] = self._round_to_nth_digit(vals=self.df[f"{key}/sizing/power_{num}"],
                                                                            n=self.n_digits)
            # file
            self.df[f"{key}/sizing/file_{num}"] = self._pick_files(list_type=self.df[f"{key}/sizing/file_{num}"],
                                                                   device=f"{key}",
                                                                   input_path=os.path.join(self.input_path, key))

        # forecast
        self.df = self._add_info_simple(keys=[key, "fcast"], config=config["fcast"], df=self.df)

        # quality
        self.df[f"{key}/quality"] = config["quality"]

        return self.df

    def _wind_grid(self, key: str, config: dict, **kwargs) -> pd.DataFrame:

        # Get all the kwargs
        from_config_if_empty = kwargs.get('from_config_if_empty', False)

        # Check if file contains the plant type (count > 0), if not use the config file to generate it
        if from_config_if_empty and self.sgen['sgen_type'].value_counts()[key] == 0:
            self._wind_config(key=key, config=config)
            return self.df

        # Drop all rows that do not contain the plant type and set the index to the owner
        try:
            df = self.sgen[self.sgen['plant_type'] == key] #.set_index('owner', drop=False)
        except KeyError:
            return self.df

        # Check if there are any pv plants, if not set all owners to 0 and return
        if len(df) == 0:
            self.df[f"{key}/owner"] = 0
            self.df[f"{key}/num"] = 0
            return self.df

        # Set the index to the owner
        df = df.set_index('owner', drop=False)

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
            self.df[f"{key}/sizing/power_{num}"] = (self.df.index.map(round(df['power'] * 1e6))).astype('Int64')

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

            # Pick random value from the config file for controllability
            self.df[f"{key}/sizing/controllable_{num}"] = random.choice(config["sizing"]["controllable"])

            # forecast
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"], df=self.df)

        # quality
        self.df[f"{key}/quality"] = config["quality"]

        return self.df

    def _fixed_gen_config(self, key: str, config: dict) -> pd.DataFrame:

        # general
        self.add_general_info_config(key=key)

        # sizing
        max_num = max(config["num"]) if config['share'] else 0
        for num in range(max_num):
            # get a list of the agents that own the nth device
            list_owner = np.multiply(np.array(self.df[f"{key}/num"] - (1 + num) >= 0), 1)

            # sizing
            idx_list = self._gen_idx_list_from_distr(n=sum(list_owner),
                                                     distr=config["sizing"]["distribution"],
                                                     owner_list=list_owner)
            # add indexed info
            self._add_info_indexed(keys=[key, "sizing"], config=config["sizing"],
                                   idx_list=idx_list, appendix=f"_{num}")
            # postprocessing
            # power
            self.df[f"{key}/sizing/power_{num}"] *= self.df[f"{c.P_INFLEXIBLE_LOAD}/sizing/demand_0"] / 1000
            self.df[f"{key}/sizing/power_{num}"] = self._calc_deviation(idx_list=idx_list,
                                                                        vals=self.df[f"{key}/sizing/power_{num}"],
                                                                        distr=config["sizing"]["power_deviation"],
                                                                        method="relative")
            self.df[f"{key}/sizing/power_{num}"] = self._round_to_nth_digit(vals=self.df[f"{key}/sizing/power_{num}"],
                                                                            n=self.n_digits)
            # file
            self.df[f"{key}/sizing/file_{num}"] = self._pick_files(list_type=self.df[f"{key}/sizing/file_{num}"],
                                                                   device=f"{key}",
                                                                   input_path=os.path.join(self.input_path, key))

        # forecast
        self.df = self._add_info_simple(keys=[key, "fcast"], config=config["fcast"], df=self.df)

        # quality
        self.df[f"{key}/quality"] = config["quality"]

        return self.df

    def _fixed_gen_grid(self, key: str, config: dict, **kwargs) -> pd.DataFrame:

        # Get all the kwargs
        from_config_if_empty = kwargs.get('from_config_if_empty', False)

        # Check if file contains the plant type (count > 0), if not use the config file to generate it
        if from_config_if_empty and self.sgen['sgen_type'].value_counts()[key] == 0:
            self._fixed_gen_config(key=key, config=config)
            return self.df

        # Drop all rows that do not contain the plant type and set the index to the owner
        try:
            df = self.sgen[self.sgen['plant_type'] == key] #.set_index('owner', drop=False)
        except KeyError:
            return self.df

        # Check if there are any pv plants, if not set all owners to 0 and return
        if len(df) == 0:
            self.df[f"{key}/owner"] = 0
            return self.df

        # Set the index to the owner
        df = df.set_index('owner', drop=False)

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
            self.df[f"{key}/sizing/power_{num}"] = (self.df.index.map(round(df['power'] * 1e6))).astype('Int64')

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

            # Pick random value from the config file for controllability
            self.df[f"{key}/sizing/controllable_{num}"] = random.choice(config["sizing"]["controllable"])

            # forecast
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"], df=self.df)

        # quality
        self.df[f"{key}/quality"] = config["quality"]

        return self.df

    def _hp_config(self, key: str, config: dict) -> pd.DataFrame:

        # general
        self.add_general_info_config(key=key)

        # sizing
        max_num = max(config["num"]) if config['share'] else 0
        for num in range(max_num):
            # get a list of the agents that own the nth device
            list_owner = np.multiply(np.array(self.df[f"{key}/num"] - (1 + num) >= 0), 1)

            # sizing
            idx_list = self._gen_idx_list_from_distr(n=sum(list_owner),
                                                     distr=config["sizing"]["distribution"],
                                                     owner_list=list_owner)
            # add indexed info
            self._add_info_indexed(keys=[key, "sizing"], config=config["sizing"],
                                   idx_list=idx_list, appendix=f"_{num}")
            # postprocessing
            # power
            self.df[f"{key}/sizing/power_{num}"] *= self.df["heat/sizing/demand_0"] / 2000
            self.df[f"{key}/sizing/power_{num}"] = self._round_to_nth_digit(vals=self.df[f"{key}/sizing/power_{num}"],
                                                                            n=self.n_digits).astype('Int64')
            # file
            self.df[f"{key}/sizing/file_{num}"] = self._pick_files(list_type=self.df[f"{key}/sizing/file_{num}"],
                                                                   device=f"{key}",
                                                                   input_path=os.path.join(self.input_path, key))

        # forecast
        self.df = self._add_info_simple(keys=[key, "fcast"], config=config["fcast"], df=self.df)

        # quality
        self.df[f"{key}/quality"] = config["quality"]

        return self.df

    def _hp_grid(self, key: str, config: dict, **kwargs) -> pd.DataFrame:

        # Get all the kwargs
        from_config_if_empty = kwargs.get('from_config_if_empty', False)

        # Drop all rows that do not contain the load type and set the index to the owner
        df = self.load[self.load['load_type'] == key]

        # Check if file contains the plant type (load > 0), if not use the config file to generate it
        if from_config_if_empty and df['load_type'].value_counts().get(key, 0) == 0:
            self._hp_config(key=key, config=config)
            return self.df

        # Check if there are any hp plants, if not set all owners to 0 and return
        if len(df) == 0:
            self.df[f"{key}/owner"] = 0
            return self.df

        # Set the index to the owner
        df = df.set_index('owner', drop=False)

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
            self.df[f"{key}/sizing/power_{num}"] = (self.df.index.map(round(df['power'] * 1e6))).astype('Int64')
            # TODO: Needs to be changed to file_add once the correct column name is implemented
            # Check if heat_file column exists
            if 'file' in df and not df['file_add'].isnull().all():
                # Fill rows with values from sgen sheet
                self.df[f"{key}/sizing/file_{num}"] = self.df.index.map(df['file_add'])
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
        self.add_general_info_config(key=key)

        # sizing
        max_num = max(config["num"]) if config['share'] else 0
        for num in range(max_num):
            # get a list of the agents that own the nth device
            list_owner = np.multiply(np.array(self.df[f"{key}/num"] - (1 + num) >= 0), 1)

            # sizing
            idx_list = self._gen_idx_list_from_distr(n=sum(list_owner),
                                                     distr=self.config[f"{key}"]["sizing"]["distribution"],
                                                     owner_list=list_owner)
            # add indexed info
            self._add_info_indexed(keys=[key, "sizing"], config=config["sizing"],
                                   idx_list=idx_list, appendix=f"_{num}")
            # postprocessing
            # file
            self.df[f"{key}/sizing/file_{num}"] = self._pick_files(list_type=self.df[f"{key}/sizing/file_{num}"],
                                                                   device=f"{key}",
                                                                   input_path=os.path.join(self.input_path, key))

        # charging scheme
        self._add_info_simple(keys=[key, "charging_scheme"], config=config["charging_scheme"], df=self.df)

        # forecast
        self.df = self._add_info_simple(keys=[key, "fcast"], config=config["fcast"], df=self.df)

        # quality
        self.df[f"{key}/quality"] = config["quality"]

        return self.df

    def _ev_grid(self, key: str, config: dict, **kwargs) -> pd.DataFrame:

        # Get all the kwargs
        from_config_if_empty = kwargs.get('from_config_if_empty', False)

        # Drop all rows that do not contain the load type and set the index to the owner
        df = self.load[self.load['load_type'] == key]

        # Check if file contains the plant type (load > 0), if not use the config file to generate it
        if from_config_if_empty and df['load_type'].value_counts().get(key, 0) == 0:
            self._ev_config(key=key, config=config)
            return self.df

        # Check if there are any ev plants, if not set all owners to 0 and return
        if len(df) == 0:
            self.df[f"{key}/owner"] = 0
            self.df[f"{key}/num"] = 0
            return self.df

        # general
        self.df[f"{key}/num"] = self.df.index.map(df['owner'].value_counts()).fillna(0).astype('Int64')
        self.df[f"{key}/owner"] = (self.df[f"{key}/num"] > 0).fillna(0).astype('Int64')  # all agents that have plant

        # # Reset the index to remove the 'owner' as an index level TODO: Check if index can simply be set later and does not need to be owner before
        # df_reset = df.reset_index(drop=True)

        # Split the dataframe into dataframes with at most one device per owner
        max_devices = df['owner'].value_counts().max()  # Maximum number of devices for a single owner
        plant_dfs = []
        for num in range(max_devices):
            device_df = df.groupby('owner').nth(num)
            plant_dfs.append(device_df)

        # sizing
        # Map each dataframe to the self.df dataframe
        for num, plant_df in enumerate(plant_dfs):
            self.df[f"{key}/sizing/file_{num}"] = self.df.index.map(plant_df.set_index('owner')['file_add'])
            self.df[f"{key}/sizing/capacity_{num}"] = self.df.index.map(
                round(plant_df.set_index('owner')['capacity'] * 1e6)).astype('Int64')
            self.df[f"{key}/sizing/charging_home_{num}"] = self.df.index.map(
                round(plant_df.set_index('owner')['charging_home'] * 1e6)).astype('Int64')
            self.df[f"{key}/sizing/charging_AC_{num}"] = self.df.index.map(
                round(plant_df.set_index('owner')['charging_ac'] * 1e6)).astype('Int64')
            self.df[f"{key}/sizing/charging_DC_{num}"] = self.df.index.map(
                round(plant_df.set_index('owner')['charging_dc'] * 1e6)).astype('Int64')
            self.df[f"{key}/sizing/charging_efficiency_{num}"] = self.df.index.map(
                plant_df.set_index('owner')['efficiency'])
            self.df[f"{key}/sizing/soc_{num}"] = self.df.index.map(plant_df.set_index('owner')['soc'])
            self.df[f"{key}/sizing/v2g_{num}"] = self.df.index.map(plant_df.set_index('owner')['v2g'])
            self.df[f"{key}/sizing/v2h_{num}"] = self.df.index.map(plant_df.set_index('owner')['v2h'])

        # charging scheme
        self._add_info_simple(keys=[key, "charging_scheme"], config=config["charging_scheme"], df=self.df)

        # forecast
        self.df = self._add_info_simple(keys=[key, "fcast"], config=config["fcast"], df=self.df)

        # quality
        self.df[f"{key}/quality"] = config["quality"]

        return self.df

    def _battery_config(self, key: str, config: dict) -> pd.DataFrame:

        # general
        self._add_general_info_dep(key=key)

        # sizing
        max_num = max(config["num"]) if config['share'] else 0
        for num in range(max_num):
            # get a list of the agents that own the nth device
            list_owner = np.multiply(np.array(self.df[f"{key}/num"] - (1 + num) >= 0), 1)

            # sizing
            idx_list = self._gen_idx_list_from_distr(n=sum(list_owner),
                                                     distr=config["sizing"]["distribution"],
                                                     owner_list=list_owner)
            # add indexed info
            self._add_info_indexed(keys=[key, "sizing"], config=config["sizing"], idx_list=idx_list,
                                   appendix=f"_{num}")
            # postprocessing
            # power
            self.df[f"{key}/sizing/power_{num}"] *= self.df[f"{c.P_INFLEXIBLE_LOAD}/sizing/demand_0"] / 1000
            self.df[f"{key}/sizing/power_{num}"] = self._round_to_nth_digit(
                vals=self.df[f"{key}/sizing/power_{num}"], n=self.n_digits)
            # capacity
            self.df[f"{key}/sizing/capacity_{num}"] *= self.df[f"{c.P_INFLEXIBLE_LOAD}/sizing/demand_0"] / 1000
            self.df[f"{key}/sizing/capacity_{num}"] = self._round_to_nth_digit(
                vals=self.df[f"{key}/sizing/capacity_{num}"], n=self.n_digits)

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
        try:
            df = self.sgen[self.sgen['plant_type'] == key]  # .set_index('owner', drop=False)
        except KeyError:
            return self.df

        # Check if there are any pv plants, if not set all owners to 0 and return
        if len(df) == 0:
            self.df[f"{key}/owner"] = 0
            return self.df

        # Set the index to the owner
        df = df.set_index('owner', drop=False)

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
            self.df[f"{key}/sizing/power_{num}"] = (self.df.index.map(round(df['power'] * 1e6))).astype('Int64')
            self.df[f"{key}/sizing/capacity_{num}"] = (self.df.index.map(round(df['capacity'] * 1e6))).astype('Int64')
            self.df[f"{key}/sizing/efficiency_{num}"] = self.df.index.map(df['efficiency'])
            self.df[f"{key}/sizing/soc_{num}"] = self.df.index.map(df['soc'])
            self.df[f"{key}/sizing/g2b_{num}"] = self.df.index.map(df['g2b'])
            self.df[f"{key}/sizing/b2g_{num}"] = self.df.index.map(df['b2g'])

        # quality
        self.df[f"{key}/quality"] = config["quality"]

        return self.df

    def _heat_storage_config(self, key: str, config: dict) -> pd.DataFrame:

        # general
        self._add_general_info_dep(key=key, method='exclusive')

        # sizing
        max_num = max(config["num"]) if config['share'] else 0
        for num in range(max_num):
            # get a list of the agents that own the nth device
            list_owner = np.multiply(np.array(self.df[f"{key}/num"] - (1 + num) >= 0), 1)

            # sizing
            idx_list = self._gen_idx_list_from_distr(n=sum(list_owner),
                                                     distr=config["sizing"]["distribution"],
                                                     owner_list=list_owner)
            # add indexed info
            self._add_info_indexed(keys=[key, "sizing"], config=config["sizing"], idx_list=idx_list,
                                   appendix=f"_{num}")
            # postprocessing
            # power
            self.df[f"{key}/sizing/power_{num}"] *= self.df["heat/sizing/demand_0"] / 2000
            self.df[f"{key}/sizing/power_{num}"] = self._round_to_nth_digit(
                vals=self.df[f"{key}/sizing/power_{num}"], n=self.n_digits).astype('Int64')
            # capacity
            self.df[f"{key}/sizing/capacity_{num}"] *= self.df["heat/sizing/demand_0"] / 2000
            self.df[f"{key}/sizing/capacity_{num}"] = self._round_to_nth_digit(
                vals=self.df[f"{key}/sizing/capacity_{num}"], n=self.n_digits).astype('Int64')

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

        # Drop all rows that do not contain the plant type
        try:
            df = self.sgen[self.sgen['plant_type'] == key]
        except KeyError:
            return self.df

        # Check if there are any plants, if not set all owners to 0 and return
        if len(df) == 0:
            self.df[f"{key}/owner"] = 0
            return self.df

        # Set the index to the owner
        df = df.set_index('owner', drop=False)

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
            self.df[f"{key}/sizing/power_{num}"] = (self.df.index.map(round(df['power'] * 1e6))).astype('Int64')
            self.df[f"{key}/sizing/capacity_{num}"] = (self.df.index.map(round(df['capacity'] * 1e6))).astype('Int64')
            self.df[f"{key}/sizing/efficiency_{num}"] = self.df.index.map(df['efficiency'])
            self.df[f"{key}/sizing/soc_{num}"] = self.df.index.map(df['soc'])

        # quality
        self.df[f"{key}/quality"] = config["quality"]

        return self.df

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
            plants += [c.P_INFLEXIBLE_LOAD]
            for device in plants:
                list_num = [list_num[idx] * self.df[f"{device}/owner"][idx] for idx, _ in enumerate(list_num)]
        # exclusive: agents needs to have ANY of the listed plants to fulfill requirements
        elif method == "exclusive":
            list_num = [0] * self.num
            for device in plants:
                list_num = [list_num[idx] + self.df[f"{device}/owner"][idx] for idx, _ in enumerate(list_num)]

            # Make sure that there is either zero or one and that owner has inflexible load
            list_num = [min(1, list_num[idx] * self.df[f"{c.P_INFLEXIBLE_LOAD}/owner"][idx]) for idx, _ in
                        enumerate(list_num)]
        else:
            raise Warning(f"Method '{method}' unknown.")

        # fill the columns
        self.df[f"{key}/owner"] = self._gen_dep_bool_list(list_bool=list_num,
                                                          share_ones=self.config[f"{key}"]["share"])
        self.df[f"{key}/num"] = self._gen_dep_num_list(owner_list=self.df[f"{key}/owner"],
                                                       distr=self.config[f"{key}"]["num"])
        return self.df