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
import hamlet.constants as c


class Industry(AgentBase):
    """
        Sets up industry agents. Inherits from Agents class.

        Mainly used for excel file creation. Afterwards Sfh class creates the individual agents.
    """

    def __init__(self, input_path: str, config: ordereddict, config_path: str, scenario_path: str, config_root: str):

        # Call the init method of the parent class
        super().__init__(input_path, config, config_path, scenario_path, config_root)

        # Define agent type
        self.type = c.A_INDUSTRY

        # Path of the input file
        self.input_path = os.path.join(input_path, 'agents', self.type)

        # Index list that is adhered to throughout the creation process to ensure correct order
        self.idx_list = None  # gets created in create_general()

        # Misc
        self.n_digits = 4  # number of digits values get rounded to in respective value column

    def create_df_from_config(self) -> pd.DataFrame:
        """
            Function to create the dataframe that makes the Excel sheet
        """

        # Get the number of agents and set the method
        self.num = self.config["general"]["number_of"]
        self.method = 'config'

        # Create the overall dataframe structure for the worksheet
        self.create_df_structure()

        # Check if there are any agents to create
        if self.num == 0:
            return self.df

        # Fill the general information in dataframe
        self.fill_general()

        # Fill the inflexible load information in dataframe
        self.fill_inflexible_load()

        # Fill the flexible load information in dataframe
        self.fill_flexible_load()

        # Fill the pv information in dataframe
        self.fill_pv()

        # Fill the wind information in dataframe
        self.fill_wind()

        # Fill the fixed generation information in dataframe
        self.fill_fixed_gen()

        # Fill the electric vehicle information in dataframe
        self.fill_ev()

        # Fill the battery information in dataframe
        self.fill_battery()

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
        self.agents = self.load[self.load['load_type'] == c.P_INFLEXIBLE_LOAD]

        # Get the rows in the sgen sheet that the owners in the owners column match with the index in the load sheet
        self.sgen = self.grid['sgen'][self.grid['sgen']['owner'].isin(self.load.index)]

        # Get the number of agents and set the method
        self.num = self.get_num_from_grid(self.grid['load'], self.type)
        self.method = 'grid'

        # Create the overall dataframe structure for the worksheet
        self.create_df_structure()

        # Check if there are any agents to create
        if self.num == 0:
            return self.df

        # Fill the general information in dataframe
        self.fill_general()

        # Fill the inflexible load information in dataframe
        self.fill_inflexible_load(**kwargs)

        # Fill the flexible load information in dataframe (can only be added through config)
        if fill_from_config:
            self.fill_flexible_load()

        # Fill the pv information in dataframe
        self.fill_pv(**kwargs)

        # Fill the wind information in dataframe
        self.fill_wind(**kwargs)

        # Fill the fixed generation information in dataframe
        self.fill_fixed_gen(**kwargs)

        # Fill the electric vehicle information in dataframe
        self.fill_ev(**kwargs)

        # Fill the battery information in dataframe
        self.fill_battery(**kwargs)

        # Fill the model predictive controller information in dataframe
        self.fill_ems()

        return self.df

    # def create_df_structure(self):
    #     """
    #         Function to create the dataframe structure with the respective columns
    #     """
    #     # Go through file and create the columns for the ctss worksheet
    #     columns = ordereddict()
    #     for key, _ in self.config.items():
    #         cols = self.make_list_from_nested_dict(self.config[key], add_string=key)
    #         # Adjust the columns from "general"
    #         if key == "general":
    #             cols[0] = f"{key}/agent_id"
    #             cols[-1] = f"{key}/market_participant"
    #             del cols[4]
    #             del cols[1]
    #             cols.insert(1, f"{key}/name")
    #             cols.insert(2, f"{key}/comment")
    #             cols.insert(3, f"{key}/bus")
    #         # Adjust the columns from "inflexible_load"
    #         elif key == "inflexible_load":
    #             cols[0] = f"{key}/owner"
    #             cols[3] = f"{key}/sizing/file"
    #             max_num = max(self.config[key]["num"])
    #             cols = cols[:2] + self.repeat_columns(columns=cols[2:4], num=max_num) + cols[4:]
    #         # Adjust the columns from "flexible_load"
    #         elif key == "flexible_load":
    #             cols[0] = f"{key}/owner"
    #             cols[3] = f"{key}/sizing/file"
    #             max_num = max(self.config[key]["num"])
    #             cols = cols[:2] + self.repeat_columns(columns=cols[2:5], num=max_num) + cols[5:]
    #         # Adjust the columns from "pv"
    #         elif key == "pv":
    #             cols[0] = f"{key}/owner"
    #             del cols[3]
    #             max_num = max(self.config[key]["num"])
    #             cols = cols[:2] + self.repeat_columns(columns=cols[2:7], num=max_num) + cols[7:]
    #         # Adjust the columns from "wind"
    #         elif key == "wind":
    #             cols[0] = f"{key}/owner"
    #             del cols[3]
    #             max_num = max(self.config[key]["num"])
    #             cols = cols[:2] + self.repeat_columns(columns=cols[2:5], num=max_num) + cols[5:]
    #         # Adjust the columns from "fixed_gen"
    #         elif key == "fixed_gen":
    #             cols[0] = f"{key}/owner"
    #             del cols[3]
    #             max_num = max(self.config[key]["num"])
    #             cols = cols[:2] + self.repeat_columns(columns=cols[2:5], num=max_num) + cols[5:]
    #         # Adjust the columns from "ev"
    #         elif key == "ev":
    #             cols[0] = f"{key}/owner"
    #             max_num = max(self.config[key]["num"])
    #             cols = cols[:2] + self.repeat_columns(columns=cols[2:11], num=max_num) + cols[11:]
    #         # Adjust the columns from "battery"
    #         elif key == "battery":
    #             cols[0] = f"{key}/owner"
    #             del cols[1]
    #             max_num = max(self.config[key]["num"])
    #             cols = cols[:2] + self.repeat_columns(columns=cols[2:-1], num=max_num) + [cols[-1]]
    #         # All columns that do not need to be adjusted
    #         elif key in ["ems"]:
    #             pass
    #         else:
    #             raise NotImplementedError(
    #                 f"The configuration file contains a key word ('{key}') that has not been configured in "
    #                 f"the {__class__.__name__} class yet. Aborting scenario creation...")
    #
    #         # Add the columns to the dictionary
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
        cols[-1] = f"{key}/market_participant"
        del cols[4]
        del cols[1]
        cols.insert(1, f"{key}/name")
        cols.insert(2, f"{key}/comment")
        cols.insert(3, f"{key}/bus")
        return cols

    def _structure_inflexible_load(self, key, cols):
        cols[0] = f"{key}/owner"
        cols[3] = f"{key}/sizing/file"
        max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
        return cols[:2] + self.repeat_columns(columns=cols[2:4], num=max_num) + cols[4:]

    def _structure_flexible_load(self, key, cols):
        cols[0] = f"{key}/owner"
        cols[3] = f"{key}/sizing/file"
        max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
        return cols[:2] + self.repeat_columns(columns=cols[2:5], num=max_num) + cols[5:]

    def _structure_pv(self, key, cols):
        cols[0] = f"{key}/owner"
        del cols[3]
        max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
        return cols[:2] + self.repeat_columns(columns=cols[2:7], num=max_num) + cols[7:]

    def _structure_wind(self, key, cols):
        cols[0] = f"{key}/owner"
        del cols[3]
        max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
        return cols[:2] + self.repeat_columns(columns=cols[2:5], num=max_num) + cols[5:]

    def _structure_fixed_gen(self, key, cols):
        cols[0] = f"{key}/owner"
        del cols[3]
        max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
        return cols[:2] + self.repeat_columns(columns=cols[2:5], num=max_num) + cols[5:]

    def _structure_ev(self, key, cols):
        cols[0] = f"{key}/owner"
        max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
        return cols[:2] + self.repeat_columns(columns=cols[2:11], num=max_num) + cols[11:]

    def _structure_battery(self, key, cols):
        cols[0] = f"{key}/owner"
        del cols[1]
        max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
        return cols[:2] + self.repeat_columns(columns=cols[2:-1], num=max_num) + [cols[-1]]

    def fill_general(self):
        """
            Fills all general columns
        """
        # TODO: Make thise similar to mfh
        key = c.K_GENERAL
        config = self.config[f"{key}"]

        # general
        self.df[f"{key}/agent_id"] = self._gen_new_ids(n=self.num)
        self.idx_list = self._gen_idx_list_from_distr(n=self.num,
                                                      distr=config["parameters"]["distribution"])

        # parameters
        # add indexed info
        self._add_info_indexed(keys=[key, "parameters"], config=config["parameters"], idx_list=self.idx_list)
        # postprocessing
        # area
        self.df[f"{key}/parameters/area"] = self._calc_deviation(idx_list=self.idx_list,
                                                                 vals=self.df[f"{key}/parameters/area"],
                                                                 distr=config["parameters"]["area_deviation"],
                                                                 method="relative")
        self.df[f"{key}/parameters/area"] = self._round_to_nth_digit(
            vals=self.df[f"{key}/parameters/area"], n=self.n_digits)

        # market participation
        self.df[f"{key}/market_participant"] = self._gen_rand_bool_list(n=self.num,
                                                                        share_ones=config["market_participant_share"])

        # If the method is grid, fill the name, comment, bus and type columns from grid file
        if self.method == 'config':
            self.df = self._general_config(key=key, config=config)
        elif self.method == 'grid':
            self.df = self._general_grid(key=key, config=config)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _general_config(self, key: str, config: ordereddict) -> pd.DataFrame:
        # TODO: In the future this will need to assign a bus from the artificial grid
        return self.df

    def _general_grid(self, key: str, config: ordereddict, **kwargs) -> pd.DataFrame:
        self.df[f"{key}/name"] = self.agents["name"]
        self.df[f"{key}/bus"] = self.agents["bus"]
        self.df[f"{key}/parameters/type"] = self.agents['sub_type']

        return self.df

    def _inflexible_load_config(self, key: str, config: dict) -> pd.DataFrame:
        """
            Fills all inflexible_load columns
        """

        # general
        self._add_general_info(key=key)

        # sizing
        max_num = max(config["num"]) if config['share'] else 0
        for num in range(max_num):
            # index list indicating ownership of device
            idx_list = self._get_idx_list(key=key, num=num)

            # add indexed info
            self._add_info_indexed(keys=[key, "sizing"], config=config["sizing"],
                                   idx_list=idx_list, appendix=f"_{num}")
            # postprocessing
            # power
            self.df[f"{key}/sizing/demand_{num}"] *= self.df["general/parameters/area"]
            self.df[f"{key}/sizing/demand_{num}"] = self._calc_deviation(idx_list=idx_list,
                                                                         vals=self.df[f"{key}/sizing/demand_{num}"],
                                                                         distr=config["sizing"]["demand_deviation"],
                                                                         method="relative")
            self.df[f"{key}/sizing/demand_{num}"] = self._round_to_nth_digit(
                vals=self.df[f"{key}/sizing/demand_{num}"], n=self.n_digits)
            self.df[f"{key}/sizing/demand_{num}"] = self.df[f"{key}/sizing/demand_{num}"].astype('Int64')
            # file
            self.df[f"{key}/sizing/file_{num}"] = self._pick_files(list_type=self.df["general/parameters/type"],
                                                                   device=key,
                                                                   input_path=os.path.join(self.input_path, key),
                                                                   idx_type=0)

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
            self.df[f"{key}/sizing/demand_{num}"] = np.floor(pd.to_numeric(df['demand'] * 1e6, errors='coerce')).astype(
                'Int64')
            # (df['demand'] * 1e6).astype('Int64')
            # Check if file column is empty and fill it with the closest file if so
            if df['file'].isnull().all():
                # file
                self.df[f"{key}/sizing/file_{num}"] = self._pick_files(list_type=self.df["general/parameters/type"],
                                                                       device=key,
                                                                       input_path=os.path.join(self.input_path, key),
                                                                       idx_type=0)
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
        self._add_general_info(key=key)

        # sizing
        max_num = max(config["num"]) if config['share'] else 0
        for num in range(max_num):
            # index list indicating ownership of device
            idx_list = self._get_idx_list(key=key, num=num)

            # add indexed info
            self._add_info_indexed(keys=[key, "sizing"], config=config["sizing"],
                                   idx_list=idx_list, appendix=f"_{num}")
            # postprocessing
            # power
            self.df[f"{key}/sizing/demand_{num}"] *= self.df["general/parameters/area"]
            self.df[f"{key}/sizing/demand_{num}"] = self._calc_deviation(idx_list=idx_list,
                                                                         vals=self.df[f"{key}/sizing/demand_{num}"],
                                                                         distr=config["sizing"]["demand_deviation"],
                                                                         method="relative")
            self.df[f"{key}/sizing/demand_{num}"] = self._round_to_nth_digit(
                vals=self.df[f"{key}/sizing/demand_{num}"], n=self.n_digits)
            self.df[f"{key}/sizing/demand_{num}"] = self.df[f"{key}/sizing/demand_{num}"].astype('Int64')
            # file
            self.df[f"{key}/sizing/file_{num}"] = self._pick_files(list_type=self.df["general/parameters/type"],
                                                                   device=key,
                                                                   input_path=os.path.join(self.input_path, key),
                                                                   idx_type=0)

        # forecast
        self.df = self._add_info_simple(keys=[key, "fcast"], config=config["fcast"], df=self.df)

        return self.df

    def _pv_config(self, key: str, config: dict) -> pd.DataFrame:
        """
            Fills all pv columns
        """

        # general
        self._add_general_info(key=key)

        # sizing
        max_num = max(config["num"]) if config['share'] else 0
        for num in range(max_num):
            # index list indicating ownership of device
            idx_list = self._get_idx_list(key=key, num=num)

            # add indexed info
            self._add_info_indexed(keys=[key, "sizing"], config=config["sizing"],
                                   idx_list=idx_list, appendix=f"_{num}")
            # postprocessing
            # power
            self.df[f"{key}/sizing/power_{num}"] *= self.df[f"{c.P_INFLEXIBLE_LOAD}/sizing/demand_0"]
            self.df[f"{key}/sizing/power_{num}"] = self._calc_deviation(idx_list=idx_list,
                                                                        vals=self.df[f"{key}/sizing/power_{num}"],
                                                                        distr=config["sizing"]["power_deviation"],
                                                                        method="relative")
            self.df[f"{key}/sizing/power_{num}"] = self._round_to_nth_digit(vals=self.df[f"{key}/sizing/power_{num}"],
                                                                            n=self.n_digits)
            self.df[f"{key}/sizing/power_{num}"] = self.df[f"{key}/sizing/power_{num}"].astype('Int64')
            # file
            self.df[f"{key}/sizing/file_{num}"] = self._pick_files(list_type=self.df[f"{key}/sizing/file_{num}"],
                                                                   device=f"{key}",
                                                                   input_path=os.path.join(self.input_path, key))
            # orientation
            self.df[f"{key}/sizing/orientation_{num}"] = self.df[f"{key}/sizing/orientation_{num}"].astype('Int16')
            # angle
            self.df[f"{key}/sizing/angle_{num}"] = self.df[f"{key}/sizing/angle_{num}"].astype('Int16')

        # forecast
        self.df = self._add_info_simple(keys=[key, "fcast"], config=config["fcast"], df=self.df)

        # quality
        self.df[f"{key}/quality"] = config["quality"]

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
        """
            Fills all wind columns
        """

        # general
        self._add_general_info(key=key)

        # sizing
        max_num = max(config["num"]) if config['share'] else 0
        for num in range(max_num):
            # index list indicating ownership of device
            idx_list = self._get_idx_list(key=key, num=num)

            # add indexed info
            self._add_info_indexed(keys=[key, "sizing"], config=config["sizing"],
                                   idx_list=idx_list, appendix=f"_{num}")
            # postprocessing
            # power
            self.df[f"{key}/sizing/power_{num}"] *= self.df[f"{c.P_INFLEXIBLE_LOAD}/sizing/demand_0"]
            self.df[f"{key}/sizing/power_{num}"] = self._calc_deviation(idx_list=idx_list,
                                                                        vals=self.df[f"{key}/sizing/power_{num}"],
                                                                        distr=config["sizing"]["power_deviation"],
                                                                        method="relative")
            self.df[f"{key}/sizing/power_{num}"] = self._round_to_nth_digit(vals=self.df[f"{key}/sizing/power_{num}"],
                                                                            n=self.n_digits)
            self.df[f"{key}/sizing/power_{num}"] = self.df[f"{key}/sizing/power_{num}"].astype('Int64')
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
            Fills all fixed_gen columns
        """

        # general
        self._add_general_info(key=key)

        # sizing
        max_num = max(config["num"]) if config['share'] else 0
        for num in range(max_num):
            # index list indicating ownership of device
            idx_list = self._get_idx_list(key=key, num=num)

            # add indexed info
            self._add_info_indexed(keys=[key, "sizing"], config=config["sizing"],
                                   idx_list=idx_list, appendix=f"_{num}")
            # postprocessing
            # power
            self.df[f"{key}/sizing/power_{num}"] *= self.df[f"{c.P_INFLEXIBLE_LOAD}/sizing/demand_0"]
            self.df[f"{key}/sizing/power_{num}"] = self._calc_deviation(idx_list=idx_list,
                                                                        vals=self.df[f"{key}/sizing/power_{num}"],
                                                                        distr=config["sizing"]["power_deviation"],
                                                                        method="relative")
            self.df[f"{key}/sizing/power_{num}"] = self._round_to_nth_digit(vals=self.df[f"{key}/sizing/power_{num}"],
                                                                            n=self.n_digits)
            self.df[f"{key}/sizing/power_{num}"] = self.df[f"{key}/sizing/power_{num}"].astype('Int64')
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
            self._wind_config(key=key, config=config)
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

    def _ev_config(self, key: str, config: dict) -> pd.DataFrame:
        """
            Fills all ev columns
        """

        # general
        self._add_general_info(key=key)

        # sizing
        max_num = max(config["num"]) if config['share'] else 0
        for num in range(max_num):
            # index list indicating ownership of device
            idx_list = self._get_idx_list(key=key, num=num)

            # sizing
            self._add_info_indexed(keys=[key, "sizing"], config=config["sizing"],
                                   idx_list=idx_list, appendix=f"_{num}")
            # postprocessing
            # file
            self.df[f"{key}/sizing/file_{num}"] = self._pick_files(list_type=self.df[f"{key}/sizing/file_{num}"],
                                                                   device=f"{key}",
                                                                   input_path=os.path.join(self.input_path, key))
            # capacity
            self.df[f"{key}/sizing/capacity_{num}"] = self.df[f"{key}/sizing/capacity_{num}"].astype('Int64')
            # charging home
            self.df[f"{key}/sizing/charging_home_{num}"] = self.df[f"{key}/sizing/charging_home_{num}"].astype('Int32')
            # charging AC
            self.df[f"{key}/sizing/charging_AC_{num}"] = self.df[f"{key}/sizing/charging_AC_{num}"].astype('Int32')
            # charging DC
            self.df[f"{key}/sizing/charging_DC_{num}"] = self.df[f"{key}/sizing/charging_DC_{num}"].astype('Int32')

            # things only necessary to add once
            if num == 0:
                # charging scheme
                self._add_info_indexed(keys=[key, "charging_scheme"], config=config["charging_scheme"],
                                       idx_list=idx_list)

        # forecast
        self.df = self._add_info_simple(keys=[key, "fcast"], config=config["fcast"], df=self.df)

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
        """
            Fills all battery columns
        """

        # general
        self._add_general_info_bat(key=key)

        # sizing
        max_num = max(config["num"]) if config['share'] else 0
        for num in range(max_num):
            # index list indicating ownership of device
            idx_list = self._get_idx_list(key=key, num=num)

            # add indexed info
            self._add_info_indexed(keys=[key, "sizing"], config=config["sizing"], idx_list=idx_list,
                                   appendix=f"_{num}")
            # postprocessing
            # power
            self.df[f"{key}/sizing/power_{num}"] *= self.df[f"{c.P_INFLEXIBLE_LOAD}/sizing/demand_0"] / 1e3
            self.df[f"{key}/sizing/power_{num}"] = self._round_to_nth_digit(
                vals=self.df[f"{key}/sizing/power_{num}"], n=2)
            self.df[f"{key}/sizing/power_{num}"] = self.df[f"{key}/sizing/power_{num}"].astype('Int64')
            # capacity
            self.df[f"{key}/sizing/capacity_{num}"] *= self.df[f"{c.P_INFLEXIBLE_LOAD}/sizing/demand_0"] / 1e3
            self.df[f"{key}/sizing/capacity_{num}"] = self._round_to_nth_digit(
                vals=self.df[f"{key}/sizing/capacity_{num}"], n=2)
            self.df[f"{key}/sizing/capacity_{num}"] = self.df[f"{key}/sizing/capacity_{num}"].astype('Int64')

        # quality
        self.df[f"{key}/quality"] = str(config["quality"])

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

    def _get_idx_list(self, key: str, num: int) -> list:
        """creates the index list for the given run"""

        # Check who owns the device
        list_owner = np.multiply(np.array(self.df[f"{key}/num"] - (1 + num) >= 0), 1)
        list_owner = [np.nan if elem == 0 else elem for elem in list_owner]

        # Return index list based on ownership
        idx_list = np.multiply(list_owner, self.idx_list)

        return [int(elem) if not np.isnan(elem) else np.nan for elem in idx_list]

    def _add_general_info(self, key: str) -> None:

        # fields that exist for all plants
        self.df[f"{key}/owner"] = self._gen_list_from_idx_list(idx_list=self.idx_list,
                                                               distr=self.config[f"{key}"]["share"])
        self.df[f"{key}/owner"] = self._gen_idx_bool_list(weight_list=self.df[f"{key}/owner"])
        self.df[f"{key}/num"] = self._gen_list_from_idx_list(idx_list=self.idx_list,
                                                             distr=self.config[f"{key}"]["num"])
        self.df[f"{key}/num"] *= self.df[f"{key}/owner"]

    def _add_general_info_bat(self, key: str):

        # find all potential owners of a battery system dependent on type
        # note: this setup considers the different dependencies for each type and loops through each separately
        agent_types = self.config["general"]["parameters"]["type"]
        list_owner = [0] * self.num
        list_num = [0] * self.num
        for idx, agent_type in enumerate(agent_types):
            # get all agents of given type
            list_type = list(self.df["general/parameters/type"] == agent_type)
            plants = self.config[f"{key}"]["share_dependent_on"][idx] + [c.P_INFLEXIBLE_LOAD]
            # check which agents of that type have the dependent plants
            for device in plants:
                list_type = [ltype * lowner for ltype, lowner in zip(list_type, self.df[f"{device}/owner"])]
            # create list of owners and their number of plants and add them to the lists
            temp_owner = self._gen_dep_bool_list(list_bool=list_type,
                                                 share_ones=self.config[f"{key}"]["share"][idx])
            temp_num = self._gen_dep_num_list(owner_list=temp_owner,
                                              distr=[self.config[f"{key}"]["num"][idx]])
            list_owner = [lowner + towner for lowner, towner in zip(list_owner, temp_owner)]
            list_num = [lnum + tnum for lnum, tnum in zip(list_num, temp_num)]

            self.df[f"{key}/owner"] = list_owner
            self.df[f"{key}/num"] = list_num