__author__ = "TUM-Doepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "TUM-Doepfert"
__email__ = "markus.doepfert@tum.de"

from hamlet.creator.agents.agents import Agents
import os
import pandas as pd
import numpy as np
from ruamel.yaml.compat import ordereddict
from pprint import pprint
import hamlet.constants as c
from typing import Callable


class AgentBase(Agents):
    def __init__(self, input_path: str, config: ordereddict, config_path: str, scenario_path: str, config_root: str):

        # Call the init method of the parent class
        super().__init__(config_path, input_path, scenario_path, config_root)

        # Define agent type
        self.type = None

        # Path of the input file
        self.input_path = None

        # Config file
        self.config = config

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

    def create_df_structure(self):
        """
            Function to create the dataframe structure with the respective columns
        """

        # Define other adjust functions...

        adjust_functions = {
            c.K_GENERAL: self._structure_general,
            c.P_INFLEXIBLE_LOAD: self._structure_inflexible_load,
            c.P_FLEXIBLE_LOAD: self._structure_flexible_load,
            c.P_HEAT: self._structure_heat,
            c.P_DHW: self._structure_dhw,
            c.P_PV: self._structure_pv,
            c.P_WIND: self._structure_wind,
            c.P_FIXED_GEN: self._structure_fixed_gen,
            c.P_HP: self._structure_hp,
            c.P_EV: self._structure_ev,
            c.P_BATTERY: self._structure_battery,
            c.P_HEAT_STORAGE: self._structure_heat_storage
        }

        # Go through file and create the columns for the sfhs worksheet
        columns = ordereddict()
        for key, _ in self.config.items():
            cols = self.make_list_from_nested_dict(self.config[key], add_string=key)

            if key in adjust_functions:
                cols = adjust_functions[key](key, cols)
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

    def _structure_general(self, key, cols):
        cols[0] = f"{key}/agent_id"
        cols[-1] = f"{key}/market_participant"
        del cols[1]
        cols.insert(1, f"{key}/name")
        cols.insert(2, f"{key}/comment")
        cols.insert(3, f"{key}/bus")
        return cols

    def _structure_inflexible_load(self, key, cols):
        cols[0] = f"{key}/owner"
        cols[4] = f"{key}/sizing/file"
        del cols[2]
        max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
        return cols[:2] + self.repeat_columns(columns=cols[2:4], num=max_num) + cols[4:]

    def _structure_flexible_load(self, key, cols):
        cols[0] = f"{key}/owner"
        cols[4] = f"{key}/sizing/file"
        del cols[2]
        max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
        return cols[:2] + self.repeat_columns(columns=cols[2:5], num=max_num) + cols[5:]

    def _structure_heat(self, key, cols):
        cols[0] = f"{key}/owner"
        cols[2] = f"{key}/sizing/demand"
        max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
        return cols[:2] + self.repeat_columns(columns=cols[2:5], num=max_num) + cols[5:]

    def _structure_dhw(self, key, cols):
        cols[0] = f"{key}/owner"
        del cols[2]
        max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
        return cols[:2] + self.repeat_columns(columns=cols[2:5], num=max_num) + cols[5:]

    def _structure_pv(self, key, cols):
        cols[0] = f"{key}/owner"
        del cols[4]
        del cols[2]
        max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
        return cols[:2] + self.repeat_columns(columns=cols[2:7], num=max_num) + cols[7:]

    def _structure_wind(self, key, cols):
        cols[0] = f"{key}/owner"
        del cols[4]
        del cols[2]
        max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
        return cols[:2] + self.repeat_columns(columns=cols[2:5], num=max_num) + cols[5:]

    def _structure_fixed_gen(self, key, cols):
        cols[0] = f"{key}/owner"
        del cols[4]
        del cols[2]
        max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
        return cols[:2] + self.repeat_columns(columns=cols[2:5], num=max_num) + cols[5:]

    def _structure_hp(self, key, cols):
        cols[0] = f"{key}/owner"
        del cols[2]
        max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
        return cols[:2] + self.repeat_columns(columns=cols[2:4], num=max_num) + cols[4:]

    def _structure_ev(self, key, cols):
        cols[0] = f"{key}/owner"
        del cols[2]
        max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
        return cols[:2] + self.repeat_columns(columns=cols[2:11], num=max_num) + cols[11:]

    def _structure_battery(self, key, cols):
        cols[0] = f"{key}/owner"
        del cols[3]
        del cols[1]
        max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
        return cols[:2] + self.repeat_columns(columns=cols[2:8], num=max_num) + cols[8:]

    def _structure_heat_storage(self, key, cols):
        cols[0] = f"{key}/owner"
        del cols[3]
        del cols[1]
        max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
        return cols[:2] + self.repeat_columns(columns=cols[2:6], num=max_num) + cols[6:]

    def fill_columns(self, key: str, config_method: Callable, grid_method: Callable, **kwargs) -> pd.DataFrame:
        """
            Fills all columns based on the provided key
        """

        # Get the config for the key
        config = self.config[key]

        if self.method == 'config':
            self.df = config_method(key=key, config=config)
        elif self.method == 'grid':
            self.df = grid_method(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def fill_general(self, **kwargs) -> pd.DataFrame:
        """
            Fills all general columns
        """
        return self.fill_columns(c.K_GENERAL, self._general_config, self._general_grid, **kwargs)

    def _general_config(self, key: str, config: ordereddict) -> pd.DataFrame:
        """
            Fills all general columns based on the config file
        """
        raise NotImplementedError(f"The method '_general_config' is not implemented yet for agent {self.type}.")

    def _general_grid(self, key: str, config: ordereddict, **kwargs) -> pd.DataFrame:
        """
            Fills all general columns based on the grid file
        """
        raise NotImplementedError(f"The method '_general_grid' is not implemented yet for agent {self.type}.")

    def fill_inflexible_load(self, **kwargs) -> pd.DataFrame:
        """
            Fills all inflexible_load columns
        """
        return self.fill_columns(c.P_INFLEXIBLE_LOAD, self._inflexible_load_config, self._inflexible_load_grid,
                                 **kwargs)

    def _inflexible_load_config(self, key: str, config: ordereddict) -> pd.DataFrame:
        """
            Fills all inflexible_load columns based on the config file
        """
        raise NotImplementedError(f"The method '_inflexible_load_config' is not implemented yet for agent {self.type}.")

    def _inflexible_load_grid(self, key: str, config: ordereddict, **kwargs) -> pd.DataFrame:
        """
            Fills all inflexible_load columns based on the grid file
        """
        raise NotImplementedError(f"The method '_inflexible_load_grid' is not implemented yet for agent {self.type}.")

    def fill_flexible_load(self, **kwargs) -> pd.DataFrame:
        """
            Fills all flexible_load columns
        """
        return self.fill_columns(c.P_FLEXIBLE_LOAD, self._flexible_load_config, self._flexible_load_grid, **kwargs)

    def _flexible_load_config(self, key: str, config: ordereddict) -> pd.DataFrame:
        """
            Fills all flexible_load columns based on the config file
        """
        raise NotImplementedError(f"The method '_flexible_load_config' is not implemented yet for agent {self.type}.")

    def _flexible_load_grid(self, key: str, config: ordereddict, **kwargs) -> pd.DataFrame:
        """
            Fills all flexible_load columns based on the grid file
        """
        raise NotImplementedError(f"The method '_flexible_load_grid' is not implemented yet for agent {self.type}.")

    def fill_heat(self, **kwargs) -> pd.DataFrame:
        """
            Fills all heat columns
        """
        return self.fill_columns(c.P_HEAT, self._heat_config, self._heat_grid, **kwargs)

    def _heat_config(self, key: str, config: ordereddict) -> pd.DataFrame:
        """
            Fills all heat columns based on the config file
        """
        raise NotImplementedError(f"The method '_heat_config' is not implemented yet for agent {self.type}.")

    def _heat_grid(self, key: str, config: ordereddict, **kwargs) -> pd.DataFrame:
        """
            Fills all heat columns based on the grid file
        """
        raise NotImplementedError(f"The method '_heat_grid' is not implemented yet for agent {self.type}.")

    def fill_dhw(self, **kwargs) -> pd.DataFrame:
        """
            Fills all dhw columns
        """
        return self.fill_columns(c.P_DHW, self._dhw_config, self._dhw_grid, **kwargs)

    def _dhw_config(self, key: str, config: ordereddict) -> pd.DataFrame:
        """
            Fills all dhw columns based on the config file
        """
        raise NotImplementedError(f"The method '_dhw_config' is not implemented yet for agent {self.type}.")

    def _dhw_grid(self, key: str, config: ordereddict, **kwargs) -> pd.DataFrame:
        """
            Fills all dhw columns based on the grid file
        """
        raise NotImplementedError(f"The method '_dhw_grid' is not implemented yet for agent {self.type}.")

    def fill_pv(self, **kwargs) -> pd.DataFrame:
        """
            Fills all pv columns
        """
        return self.fill_columns(c.P_PV, self._pv_config, self._pv_grid, **kwargs)

    def _pv_config(self, key: str, config: ordereddict) -> pd.DataFrame:
        """
            Fills all pv columns based on the config file
        """
        raise NotImplementedError(f"The method '_pv_config' is not implemented yet for agent {self.type}.")

    def _pv_grid(self, key: str, config: ordereddict, **kwargs) -> pd.DataFrame:
        """
            Fills all pv columns based on the grid file
        """
        raise NotImplementedError(f"The method '_pv_grid' is not implemented yet for agent {self.type}.")

    def fill_wind(self, **kwargs) -> pd.DataFrame:
        """
            Fills all wind columns
        """
        return self.fill_columns(c.P_WIND, self._wind_config, self._wind_grid, **kwargs)

    def _wind_config(self, key: str, config: ordereddict) -> pd.DataFrame:
        """
            Fills all wind columns based on the config file
        """
        raise NotImplementedError(f"The method '_wind_config' is not implemented yet for agent {self.type}.")

    def _wind_grid(self, key: str, config: ordereddict, **kwargs) -> pd.DataFrame:
        """
            Fills all wind columns based on the grid file
        """
        raise NotImplementedError(f"The method '_wind_grid' is not implemented yet for agent {self.type}.")

    def fill_fixed_gen(self, **kwargs) -> pd.DataFrame:
        """
            Fills all fixed_gen columns
        """
        return self.fill_columns(c.P_FIXED_GEN, self._fixed_gen_config, self._fixed_gen_grid, **kwargs)

    def _fixed_gen_config(self, key: str, config: ordereddict) -> pd.DataFrame:
        """
            Fills all fixed_gen columns based on the config file
        """
        raise NotImplementedError(f"The method '_fixed_gen_config' is not implemented yet for agent {self.type}.")

    def _fixed_gen_grid(self, key: str, config: ordereddict, **kwargs) -> pd.DataFrame:
        """
            Fills all fixed_gen columns based on the grid file
        """
        raise NotImplementedError(f"The method '_fixed_gen_grid' is not implemented yet for agent {self.type}.")

    def fill_hp(self, **kwargs) -> pd.DataFrame:
        """
            Fills all hp columns
        """
        return self.fill_columns(c.P_HP, self._hp_config, self._hp_grid, **kwargs)

    def _hp_config(self, key: str, config: ordereddict) -> pd.DataFrame:
        """
            Fills all hp columns based on the config file
        """
        raise NotImplementedError(f"The method '_hp_config' is not implemented yet for agent {self.type}.")

    def _hp_grid(self, key: str, config: ordereddict, **kwargs) -> pd.DataFrame:
        """
            Fills all hp columns based on the grid file
        """
        raise NotImplementedError(f"The method '_hp_grid' is not implemented yet for agent {self.type}.")

    def fill_ev(self, **kwargs) -> pd.DataFrame:
        """
            Fills all ev columns
        """
        return self.fill_columns(c.P_EV, self._ev_config, self._ev_grid, **kwargs)

    def _ev_config(self, key: str, config: ordereddict) -> pd.DataFrame:
        """
            Fills all ev columns based on the config file
        """
        raise NotImplementedError(f"The method '_ev_config' is not implemented yet for agent {self.type}.")

    def _ev_grid(self, key: str, config: ordereddict, **kwargs) -> pd.DataFrame:
        """
            Fills all ev columns based on the grid file
        """
        raise NotImplementedError(f"The method '_ev_grid' is not implemented yet for agent {self.type}.")

    def fill_battery(self, **kwargs) -> pd.DataFrame:
        """
            Fills all battery columns
        """
        return self.fill_columns(c.P_BATTERY, self._battery_config, self._battery_grid, **kwargs)

    def _battery_config(self, key: str, config: ordereddict) -> pd.DataFrame:
        """
            Fills all battery columns based on the config file
        """
        raise NotImplementedError(f"The method '_battery_config' is not implemented yet for agent {self.type}.")

    def _battery_grid(self, key: str, config: ordereddict, **kwargs) -> pd.DataFrame:
        """
            Fills all battery columns based on the grid file
        """
        raise NotImplementedError(f"The method '_battery_grid' is not implemented yet for agent {self.type}.")

    def fill_heat_storage(self, **kwargs) -> pd.DataFrame:
        """
            Fills all heat_storage columns
        """
        return self.fill_columns(c.P_HEAT_STORAGE, self._heat_storage_config, self._heat_storage_grid, **kwargs)

    def _heat_storage_config(self, key: str, config: ordereddict) -> pd.DataFrame:
        """
            Fills all heat_storage columns based on the config file
        """
        raise NotImplementedError(f"The method '_heat_storage_config' is not implemented yet for agent {self.type}.")

    def _heat_storage_grid(self, key: str, config: ordereddict, **kwargs) -> pd.DataFrame:
        """
            Fills all heat_storage columns based on the grid file
        """
        raise NotImplementedError(f"The method '_heat_storage_grid' is not implemented yet for agent {self.type}.")

    def fill_ems(self):
        """
            Fills all battery columns
        """
        key = c.K_EMS
        config = self.config[f"{key}"]

        # general
        self.df = self._add_info_simple(keys=[key], config=config, df=self.df)

        # Change the values where the value should be randomly picked from a list
        self.df[f'{key}/market/horizon'] = np.random.choice(config['market']['horizon'], size=len(self.df))


