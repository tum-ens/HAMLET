__author__ = "TUM-Doepfert"
__license__ = ""
__maintainer__ = "TUM-Doepfert"
__email__ = "markus.doepfert@tum.de"

import time
import datetime
import shutil
import os
import string
import json
import math
import random
from ruamel.yaml import YAML
from typing import Tuple, Union
import pandas as pd
import numpy as np
from pprint import pprint
from ruamel.yaml.compat import ordereddict
from collections import OrderedDict, Counter
from bisect import bisect_left
import pvlib
from pvlib.pvsystem import PVSystem
from pvlib.location import Location
from windpowerlib import ModelChain, WindTurbine
import ast
import pytz
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'


class Agents:
    """
        A class used to generate all agents. All sub-agents inherit from this class.
        ...

        Attributes
        ----------
        agent_id : str
            ID of the agent

        Public Methods
        -------
        __init__() -> None
        """

    def __init__(self, config_path: str, input_path: str, scenario_path: str, config_root: str = None):
        # Set paths and names
        self.config_path = config_path
        self.config_root = config_root if config_root is not None else config_path
        self.input_path = input_path
        self.scenario_path = scenario_path

        # Load setup plus configuration and/or agent file
        self.setup = self._load_file(path=os.path.join(self.config_root, 'config_general.yaml'))
        self.grid = None    # grid file only required if agents are created from grid file
        self.config = None
        self.excel = None
        try:
            self.config = self._load_file(path=os.path.join(self.config_path, 'config_agents.yaml'))
        except FileNotFoundError:
            try:
                self.excel = self._load_file(path=os.path.join(self.config_path, 'agents.xlsx'))
            except FileNotFoundError as exc:
                raise FileNotFoundError("Configuration requires either a yaml or an xlsx file to create agents.") \
                    from exc

        # Information about the agents
        self.agents = None      # information of all agents
        self.id = None          # information of the current agent id
        self.account = None     # information of the current account

        # Available types of agents
        self.types = {
            'sfh': Sfh,
            'mfh': Mfh,
            'ctsp': Ctsp,
            'industry': Industry,
            'producer': Producer,
            'storage': Storage,
        }
        # Available types of plants
        self.plants = {
            'inflexible_load': {
                'type': 'inflexible_load',
                'info': self.__info_inflexible_load,
            },
            'flexible_load': {
                'type': 'flexible_load',
                'info': self.__info_flexible_load,
            },
            'heat': {
                'type': 'heat',
                'info': self.__info_heat,
                'file_func': self.__make_timeseries_heat,
            },
            'dhw': {
                'type': 'dhw',
                'info': self.__info_dhw,
            },
            'pv': {
                'type': 'pv',
                'info': self.__info_pv,
                'specs': self.__timeseries_from_specs_pv,
            },
            'wind': {
                'type': 'wind',
                'info': self.__info_wind,
                'specs': self.__timeseries_from_specs_wind,
            },
            'fixed_gen': {
                'type': 'fixed_gen',
                'info': self.__info_fixed_gen,
            },
            'hp': {
                'type': 'hp',
                'info': self.__info_hp,
                'specs': self.__timeseries_from_specs_hp,
            },
            'ev': {
                'type': 'ev',
                'info': self.__info_ev,
            },
            'battery': {
                'type': 'battery',
                'info': self.__info_battery,
            },
            'psh': {
                'type': 'psh',
                'info': self.__info_psh,
            },
            'hydrogen': {
                'type': 'hydrogen',
                'info': self.__info_hydrogen,
            },
            'heat_storage': {
                'type': 'heat_storage',
                'info': self.__info_heat_storage,
            },
        }
        # Available types of ctsp (obtained from the input data folder)
        self.ctsp = self.__get_types(
            path=os.path.join(self.input_path, 'agents', 'ctsp', 'inflexible_load'), idx=0, sep='_')
        # Available types of industry (obtained from the input data folder)
        self.industry = self.__get_types(
            path=os.path.join(self.input_path, 'agents', 'industry', 'inflexible_load'), idx=0, sep='_')

    def create_agents_file_from_config(self, overwrite: bool = False):
        """Creates the Excel file from the config file
        """

        # Dictionary to store the dataframes
        dict_agents = {}

        # Create dataframes for each type of agent using their specific class
        for key, config in self.config.items():
            if key in self.types:
                # Create the agents of the specific type
                agents = self.types[key](input_path=self.input_path,
                                         config=config,
                                         config_path=self.config_path,
                                         scenario_path=self.scenario_path)

                # Create the dataframe
                dict_agents[key] = agents.create_df_from_config()
            else:
                raise KeyError(f"Agent type '{key}' is not available.")


        # Save each dataframe as worksheet in Excel
        if os.path.exists(f"{self.config_path}/agents.xlsx") and not overwrite:
            raise Warning("File already exists and overwrite is turned off.")
        else:
            try:
                writer = pd.ExcelWriter(f"{self.config_path}/agents.xlsx", engine="xlsxwriter")
                for key, df in dict_agents.items():
                    df.to_excel(writer, sheet_name=key)
                writer.save()
            except PermissionError:
                raise PermissionError("The file 'agents.xlsx' needs to be closed before running this function.")

    def create_agents_file_from_grid(self, fill_from_config: bool = False, overwrite: bool = False):
        """Creates the Excel file from the grid file
        """

        # Dictionaries to store the dataframes
        dict_grids = {}
        dict_agents = {}

        # Load grid file if not already loaded
        self.grid = self._load_file(path=os.path.join(self.config_path, 'grid.xlsx')) if self.grid is None \
            else FileNotFoundError("Grid file not found.")

        # Load sheets from grid file
        for sheet in self.grid.sheet_names:
            # Load the dataframe from the sheet
            dict_grids[sheet] = self.grid.parse(sheet, index_col=0)

            # Add the info in the description column as new columns
            try:
                dict_grids[sheet] = self.__add_info_from_col(df=dict_grids[sheet], col='description', drop=True)
            except (AttributeError, KeyError):
                pass

        # Create dataframes for each type of agent using their specific class
        for key, _ in self.config.items():
            if key in self.types:
                dict_agents[key] = self.types[key](config=self.config[key],
                                                   input_path=self.input_path,
                                                   config_path=self.config_path,
                                                   scenario_path=self.scenario_path).\
                    create_df_from_grid(grid=dict_grids, fill_from_config=fill_from_config).reset_index(drop=True)
            else:
                print(f"Not there yet at {key}")

        # Save each dataframe as worksheet in Excel
        if os.path.exists(f"{self.config_path}/agents.xlsx") and not overwrite:
            raise Warning("File already exists and overwrite is turned off.")
        else:
            try:
                writer = pd.ExcelWriter(f"{self.config_path}/agents.xlsx", engine="xlsxwriter")
                for key, df in dict_agents.items():
                    df.to_excel(writer, sheet_name=key)
                writer.save()
            except PermissionError:
                raise PermissionError("The file 'agents.xlsx' needs to be closed before running this function.")

    def create_agents_from_file(self, id_check=False) -> None:
        """
            Creates the agents from Excel file. If file does not exist, it is created first
        """

        # Load Excel file if not done yet and throw error if it does not exist
        self.excel = self._load_file(f"{self.config_path}/agents.xlsx") if self.excel is None \
            else FileNotFoundError("Agent xlsx file does not exist. Call 'create_agents_file_from_config()' or "
                                   "'create_agents_file_from_grid()' first to create the file from the config or "
                                   "grid file.")

        # TODO: Check if there are non-unique IDs and replace them in worksheet and in the dataframe
        if id_check:
            agent_ids = list()
            for sheet in self.excel.sheet_names:
                df = self.excel.parse(sheet, index_col=0)
                agent_ids += df["general/agent_id"].to_list()
            print(agent_ids)
            # non-unique items
            print([item for item, count in Counter(agent_ids).items() if count > 1])
            # Find solution for mfhs as they will always give more than one time same ID
            raise NotImplementedError('Not yet implemented...')

        # Create agents from Excel file by looping over the sheets
        for sheet in self.excel.sheet_names:
            # Load the dataframe from the sheet
            df = self.excel.parse(sheet, index_col=0)

            # Create agents dictionary from dataframe
            self.agents = self.create_agents_dict_from_file(df=df)

            # Create agents folders and files from
            self.create_agents_from_dict(agents=self.agents, agent_type=sheet)

    def create_agents_dict_from_file(self, df: pd.DataFrame) -> dict:
        """Creates dictionary from dataframe"""

        # Get the names of the categories in the dict and the names of their according parameters
        groups, param_names = self.__get_groups_and_param_names(df)

        # Get the account information of each agent of the given type
        accounts = self._get_accounts_info(df, groups, param_names)

        return accounts

    def _get_accounts_info(self, df: pd.DataFrame, groups: list, param_names: dict) -> dict:
        # Note: This function can be made even more flexible by splitting it into a recursive function in case more sub IDs are added (but good enough for now)

        # Contains the names of the parameters that are used to identify the agents in descending order
        ids = ['general/agent_id', 'general/sub_id']

        # Get all main accounts sorted by the ID of the agent
        main_accounts = dict.fromkeys(df.loc[:, ids[0]])
        accounts = dict()   # stores the information of all the accounts

        # Get all categories of the accounts, i.e. all that are not defined in self.plants
        acc_categories = [item for item in groups if item not in self.plants.keys()]

        # Loop through all the accounts and get the information of the non-device categories
        for main_account in main_accounts:
            # Find the index/indeces of the account by matching its agent ID
            # Note: Several indices are found if there are multiple agents with the same ID, e.g. for mfh
            acc_idxs = df[df[ids[0]] == main_account].index

            # Loop through all the indices of the account and add the information for each subaccount
            for acc_idx in acc_idxs:

                # Create the account name from the main ID and, if applicable, the sub IDs by separating them with a '/'
                account = main_account
                level = 0
                while True:
                    try:
                        level += 1
                        account += '/' + df[ids[level]].iat[acc_idx]
                    except (KeyError, IndexError):
                        break

                # Create a dict with all categories and add one for the plants of the account
                accounts[account] = OrderedDict.fromkeys(acc_categories + ["plants"])

                # Add all parameters of the non-device categories for the account
                for group in acc_categories:
                    # Gets the value from the df(match names that are not followed by underscore), then uses squeeze to
                    #   only obtain the value, then renames the columns to the names defined in param_names and lastly
                    #   converts it all into a dict
                    info = df.loc[acc_idx, df.columns.str.match(f"{group}(?!_)")].squeeze().\
                        rename(param_names[group]).to_dict(into=OrderedDict)
                    accounts[account][group] = info

                # Get all plant categories
                acc_plants = OrderedDict.fromkeys(self.plants.keys())
                for device in acc_plants:
                    # print(group_params[device])
                    try:
                        # Gets the value from the df (match names that are not followed by underscore), then uses
                        #   squeeze to only obtain the value, then renames the columns to the names defined in
                        #   param_names and lastly converts it all into a dict
                        info = df.loc[acc_idx, df.columns.str.match(f"{device}(?!_)")].squeeze().\
                            rename(param_names[device]).to_dict(into=OrderedDict)
                        acc_plants[device] = info
                    except KeyError:
                        pass

                accounts[account]["plants"] = acc_plants

        return accounts

    def create_agents_from_dict(self, agents: dict, agent_type: str):
        """Creates agents with folders and files"""

        # Create every agent individually
        # Note: the agent ID can be subdivided using "/", e.g. when creating an mfh with multiple households
        for agent, account in agents.items():
            # Store id and account information of the current agent
            self.id = agent
            self.account = account

            # Path of the folder in which all the agent's files are to be stored
            path = os.path.join(self.scenario_path, 'agents', agent_type, agent)

            # Create folder for the agent
            self.__create_folder(path)
            
            # Create all the data that is needed for the agent
            account["plants"], plants, meters, timeseries, socs = self._create_plants_for_agent(
                plants=account["plants"], agent_type=agent_type)

            # Create agent from the data
            data = {
                "account.json": account,
                "plants.json": plants,
                "meters.csv": meters,
                "timeseries.csv": timeseries,
                "socs.csv": socs}
            self._create_agent(path, data)

    def _create_plants_for_agent(self, plants: dict, agent_type: str) -> Tuple:
        """Creates the plants for the agent"""

        # Setup
        plants_dict = OrderedDict()     # Contains all plant information
        plant_dict = OrderedDict()      # Contains single plant information
        plants_ids = []                 # Contains the plant IDs
        start = self.setup['simulation']['sim']['start'].replace(tzinfo=datetime.timezone.utc)
        end = start + datetime.timedelta(days=self.setup['simulation']['sim']['duration'])
        start, end = int(start.timestamp()), int(end.timestamp())
        timeseries = pd.DataFrame(index=range(start, end, self.setup['simulation']['timestep']))  # Contains the time series data of each power plant (e.g. power output)
        timeseries.index.name = 'timestamp'         # Name of the index (must be equal to name in input files)
        meters = pd.DataFrame(index=['in', 'out'])  # Contains the meter values of each plant
        socs = pd.DataFrame(index=['soc'])          # Contains the SOCs of each plant

        # Loop through plants and check if agent has the plant
        for plant, info in plants.items():
            # Check if plant information is actually available, otherwise skip this type of plant
            if info is None:
                continue

            # Check how many plants the agent has of respective plant type (try/except for cells that are empty)
            try:
                num_plants = int(info["owner"] * info["num"])
            except ValueError:
                num_plants = 0

            # Add general information to plant_dict
            plant_dict["type"] = plant
            plant_dict["has_submeter"] = info["has_submeter"]

            # Loop through every device individually as agent can have more than one of each type
            for num_plant in range(num_plants):
                # Generate plant ID and add to plant info dict and to ID list of all plants
                plant_id = self._gen_new_ids()
                plant_dict["id"] = plant_id
                plants_ids += [plant_id]

                # Add meter to meters dataframe
                meters = meters.join(self.__make_meter(plant_id=plant_id))

                # Add specific plant information
                if plant in self.plants:
                    plant_dict = self.plants[plant]['info'](info=info, plant_dict=plant_dict, idx=num_plant)

                # Add time series to timeseries dataframe (if applicable)
                try:
                    timeseries = timeseries.join(self.__make_timeseries(
                        file_path=os.path.join(self.input_path, 'agents', agent_type, plant, plant_dict['file']),
                        plant_id=plant_id, plant_dict=plant_dict,
                        delta=int(self.setup['simulation']['timestep'])))
                except KeyError:
                    pass

                # Add SoC to SoCs dataframe (if applicable)
                try:
                    socs = socs.join(self.__make_soc(plant_id=plant_id,
                                                     soc=info[f"soc_{num_plant}"] * info[f"capacity_{num_plant}"]))
                except KeyError:
                    pass

                # Add plant information to plant dict
                plants_dict[plant_id] = plant_dict

            # Reset plant dictionary to add next entry
            plant_dict = OrderedDict()

        return plants_ids, plants_dict, meters, timeseries, socs

    def _create_agent(self, path: str, data: dict) -> None:
        """Creates the agent files in the specified path

        Args:
            path (str): Path to the folder in which the agent files are to be stored
            data (dict): Dictionary containing the data for the agent files and the names of the files

        Returns:
            None

        """

        # Create the agent files
        for key, value in data.items():
            self._save_file(path=os.path.join(path, key), data=value)

    @classmethod
    def _load_file(cls, path: str, index: int = 0):
        file_type = path.rsplit('.', 1)[-1]
        if file_type == 'yaml' or file_type == 'yml':
            with open(path) as file:
                file = YAML().load(file)
        elif file_type == 'json':
            with open(path) as file:
                file = json.load(file)
        elif file_type == 'csv':
            file = pd.read_csv(path, index_col=index)
        elif file_type == 'xlsx':
            file = pd.ExcelFile(path)
        elif file_type == 'ft':
            file = pd.read_feather(path)
        else:
            raise ValueError(f'File type "{file_type}" not supported')

        return file

    @classmethod
    def _save_file(cls, path: str, data, index: bool = True) -> None:
        file_type = path.rsplit('.', 1)[-1]

        if file_type == 'yaml' or file_type == 'yml':
            with open(path, 'w') as file:
                YAML().dump(data, file)
        elif file_type == 'json':
            with open(path, 'w') as file:
                json.dump(data, file, indent=4)
        elif file_type == 'csv':
            data.to_csv(path, index=index)
        elif file_type == 'xlsx':
            data.to_excel(path, index=index)
        elif file_type == 'ft':
            data.reset_index(inplace=True)
            data.to_feather(path)
        else:
            raise ValueError(f'File type "{file_type}" not supported')

    @staticmethod
    def __make_meter(plant_id: str, vals: list = (0, 0)) -> pd.Series:
        """initialize a meter file with initial positive and negative meter readings

        Args:
            meters: dictionary that contains all meter readings
            id_meter: ID of the meter that is to be initialized
            init_in: initial positive meter reading (energy flowing into plant)
            init_out: initial negative meter reading (energy flowing out of plant)

        Returns:
            None

        """
        return pd.Series(vals, index=["in", "out"], name=plant_id)

    def __make_timeseries(self, file_path: str, plant_id: str, plant_dict: dict, delta: int) -> pd.DataFrame:
        """Creates a time series for the given plant.

        Args:
            file_path (str): The file path to the time series data.
            plant_id (str): The ID of the plant for which to create a time series.
            plant_dict (dict): A dictionary of information about the plant, including the file format and power rating.

        Returns:
            pd.Series: A Pandas Series object representing the time series data for the plant.

        """

        # Load required file as series
        file = self._load_file(file_path)

        # Multiply with power or demand if time series is per unit
        if "pu" in plant_dict["file"].split("_")[-1]:
            try:
                file.iloc[:, 0] *= plant_dict["power"]
            except KeyError:
                try:
                    file.iloc[:, 0] *= plant_dict["demand"]
                except KeyError:
                    raise KeyError("Plant type neither has power nor demand information to multiply with the pu file.")

        # If file is a spec file, create time series accordingly
        if type(file) is dict:
            try:
                # Use the plant-specific specs function to create a time series from the spec data
                file = self.plants[plant_dict['type']]['specs'](specs=file, plant=plant_dict)
            except:
                # If the specs function is not available for this plant type, raise a warning
                raise Warning('Time series creation from spec file not available for this plant type.')

        # Check if a special function is to be applied to the time series
        # Note: they are stored in self.plants under the key 'file_func'
        if self.plants[plant_dict['type']].get('file_func', None):
            # If so, apply the function to the time series
            file = self.plants[plant_dict['type']]['file_func'](file, plant_dict)

        # Name the series after the plant ID
        if len(file.columns) == 1:
            # If there's only one column, set the name to "<plant_id>_<column name>"
            file = file.squeeze()
            file.name = f'{plant_id}_{file.name}'
        else:
            # If there are multiple columns, add the plant ID as a prefix to each column name
            file.columns = [f'{plant_id}_{col}' for col in file.columns]

        # Resample time series data to ensure all rows are filled
        file = self.resample_timeseries(timeseries=file, delta=delta)

        return file

    def __make_timeseries_heat(self, df: pd.DataFrame, plant_dict: dict) -> pd.DataFrame:

        # Get the goal values that are to be searched for: efficiency, occupants, temperature
        goal = [self.account['general']['efficiency'], self.account['general']['occupants'], plant_dict['temperature']]

        # Check if there are multiple columns
        if len(df.columns) == 1:
            df['heat'] = df.iloc[:, 0]
        else:
            # Check columns one by one, starting with the first value
            # Note: For each value, we calculate the distance between each column's numeric value for that position and the
            #       corresponding goal value. We then filter the list of columns to only include those with the smallest
            #       distance for that value. We repeat this process for each value, so that we end up with a list of columns
            #       that have the smallest distance for each value.
            closest_cols = df.columns
            for i, goal_value in enumerate(goal):
                # Calculate distance for each column
                distances = [abs(int(col.split('_')[i + 1]) - goal_value) for col in closest_cols]
                # Get column(s) with the smallest distance
                closest_cols = [closest_cols[j] for j in range(len(closest_cols)) if distances[j] == min(distances)]

            # First value is chosen as column and added to the file
            df['heat'] = df[closest_cols[0]]

        # Calculate the duration of the time series
        # Note: We add the delta to the duration as the last index is not included in the duration
        delta = df.index[1] - df.index[0]
        duration = (df.index[-1] - df.index[0] + delta) / 3600 / 8760  # in years
        
        # Scale the heat to the demand and take into account the duration of the time series
        df['heat'] *= plant_dict['demand'] * duration / sum(df['heat'])

        # Drop all columns except the heat column
        df = df['heat'].astype(int).to_frame()

        return df

    @staticmethod
    def __make_soc(plant_id: str, soc: float = 0.8) -> pd.Series:
        """initialize a meter file with initial positive and negative meter readings

        Args:
            meters: dictionary that contains all meter readings
            id_meter: ID of the meter that is to be initialized
            init_in: initial positive meter reading (energy flowing into plant)
            init_out: initial negative meter reading (energy flowing out of plant)

        Returns:
            None

        """
        return pd.Series(soc, index=['soc'], name=plant_id)

    @classmethod
    def __create_folder(cls, path: str, delete: bool = True) -> None:
        """Creates a folder at the given path

        Args:
            path: path to the folder
            delete: if True, the folder will be deleted if it already exists

        Returns:
            None
        """

        # Create main folder if it does not exist
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            if delete:
                shutil.rmtree(path)
                os.mkdirs(path)
        time.sleep(0.0001)

    @staticmethod
    def __del_file_contents(path: str) -> None:
        """deletes all files and folders within the specified path

        Args:
            path: string that contains the path of the directory that needs to be deleted

        Returns:
            None

        TODO: Move to higher level file

        """

        for root, directories, files in os.walk(path):
            for file in files:
                os.unlink(os.path.join(root, file))
            for directory in directories:
                shutil.rmtree(os.path.join(root, directory))

    @staticmethod
    def _gen_new_ids(n: int = 1, length: int = 15) -> Union[str, list[str]]:
        """creates random ID"""
        ids = []
        for _ in range(n):
            ids.append("".join(random.choices(string.ascii_letters + string.digits, k=length)))

        if len(ids) == 1:
            return ids[0]
        else:
            return ids

    @staticmethod
    def _gen_rand_bool_list(n: int, share_ones: float) -> list:
        """generates a randomly ordered boolean list of specified length and share of ones

        Args:
            n: integer that specifies the number of elements in the list
            share_ones: float that specifies the share of ones in the list with a value between 0 and 1

        Returns:
            list_bool: list with boolean values

        """

        list_bool = [0] * n
        list_bool[:round(n * share_ones)] = [1] * round(n * share_ones)
        random.shuffle(list_bool)

        return list_bool

    @staticmethod
    def _gen_idx_bool_list(weight_list: list) -> list:
        """generates a boolean list based on the weights

        Args:
            weight_list: list of probability for value to be 1

        Returns:
            list with boolean values

        """

        return [random.choices([0, 1], [1 - weight, weight])[0] for weight in weight_list]

    @staticmethod
    def _gen_dep_bool_list(list_bool: list, share_ones: float) -> list:
        """generates an ordered boolean list with a specified share of ones that depends on another boolean list

        Comment:
            The new list depends on the provided list in that ones can only be created in the positions where the
            provided list has ones. The absolute number of ones therefore depends on the number of ones in the original
            list. Example: list_bool has length 10 and 8 ones. share_ones = 0.5. The dependent list will have 4 ones.
            If list_bool has length 10 and 4 ones, the dependent list will have 2 ones if share_1s = 0.5.

        Args:
            list_bool: list of boolean values
            share_ones: float that specifies the share of ones in the list in relation to the share of ones in list_bool
                      with a value between 0 and 1.

        Returns:
            list_dep: list of boolean values

        """

        list_dep = list_bool.copy()
        n_ones = round(sum(list_bool) * share_ones)

        # Alternative approach
        # idxs_1s = np.nonzero(list_bool)
        # idxs_n = np.random.choice(idxs_1s, n_1s, replace=False)
        # list_dep = [0] * len(list_bool)
        # list_dep[idxs_n] = 1

        # Reduce the number of ones until n_1s is reached. A while-loop was used as it is the fastest method for n<1000
        while sum(list_dep) > n_ones:
            idx = random.randint(0, len(list_dep) - 1)
            list_dep[idx] = 0

        return list_dep

    @staticmethod
    def __get_groups_and_param_names(df: pd.DataFrame) -> tuple:
        # Get all unique first values of headers in original order and create the names of the parameters
        groups = list(OrderedDict.fromkeys([column.split("/", 1)[0] for column in df.columns.values.tolist()]))
        group_params = OrderedDict.fromkeys(groups)
        for group in group_params:
            group_params[group] = dict.fromkeys([column for column in df.columns.values.tolist()
                                                 if column.split("/", 1)[0] == group])
            for param in group_params[group]:
                group_params[group][param] = param.rsplit("/", 1)[-1]

        return groups, group_params

    @classmethod
    def _gen_idx_list_from_distr(cls, n: int, distr: list, owner_list: list = None, equal_one: bool = True) -> list:
        """
            Generates an index list according to distribution
        """

        # If n = 1 simply give back a random index
        # if n == 1:
        # print("I am used")
        # return random.choices(range(len(distr)), distr)

        # Ensure that some of distribution equals one
        owner_list = [] if owner_list is None else owner_list

        distr = [x / sum(distr) for x in distr] if equal_one else distr
        idx_list = []
        for idx in range(len(distr)):
            idx_list += round(n * distr[idx]) * [idx]
        idx_list = idx_list[:n]
        random.shuffle(idx_list)

        # Adjust idx_list if sum does not match num
        if len(idx_list) != n:
            diff_num = n - len(idx_list)
            idxs = random.choices(range(len(distr)), weights=distr, k=abs(diff_num))
            if diff_num > 0:
                idx_list += idxs
            elif diff_num < 0:
                idx_list -= idxs

        # Fill list with nans if length of owners does not match length of idx list
        if len(owner_list) > n:
            idx_list = iter(idx_list)
            idx_list = [next(idx_list) if item else np.nan for item in owner_list]

        return idx_list

    @classmethod
    def _gen_list_from_idx_list(cls, idx_list: list, distr: list) -> list:
        """
            Picks elements from the list according to
        """

        try:
            idx_max = max(list(filter(None, idx_list)))
        except ValueError:
            return [distr[0]] * len(idx_list)
        if idx_max < len(distr) - 1:
            # print(f"Highest index value ({max(idx_list)}) lower than length of distribution list ({len(distr)}). "
            #       f"List was generated with deprecated distribution list.")
            # print(f"Old distribution list: {distr}")
            distr = distr[:idx_max + 1]
            # print(f"New distribution list: {distr}")
        elif idx_max > len(distr) - 1:
            # print(f"Highest index value ({max(idx_list)}) higher than length of distribution list ({len(distr)}). "
            #       f"List was generated with changed index list which will distort the distribution.")
            # print(f"Old index list: {idx_list}")
            idx_list = [min(idx, len(distr) - 1) for idx in idx_list]
            # print(f"New index list: {idx_list}")
        elem_list = [distr[idx] if not np.isnan(idx) else np.nan for idx in idx_list]
        return elem_list

    @classmethod
    def _pick_files_from_distr(cls, list_owner: list, distr: list, vals: list, input_path: str,
                               variance: list = None, divisor: int = None, input_idx: int = 1) -> tuple:
        """Generates a list from files in input folder
        """
        if all(val == 0 for val in list_owner):
            return [None] * len(list_owner), [None] * len(list_owner)

        # Variance
        variance = variance if variance else [0] * len(vals)
        divisor = divisor if divisor else 1

        # Assign values to each owner
        list_idxs = cls._gen_idx_list_from_distr(n=sum(list_owner), distr=distr)

        # Pick values based on variance
        input_files = [file for file in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, file))]
        input_vals = [int(val.split("_")[input_idx]) for val in input_files]
        input_idxs = []
        # Get indices of values that lie within range
        for idx, val in enumerate(vals):
            input_idxs.append([iidx for iidx, ival in enumerate(input_vals)
                               if (val - variance[idx]) / divisor <= ival <= (val + variance[idx]) / divisor])

        # Replace values in idx_list with random indices of the matching input_idx
        list_vals = [0] * len(list_idxs)
        for idx, val in enumerate(list_idxs):
            list_vals[idx] = random.choice(input_idxs[val])

        # Create list of the corresponding files of list_idxs
        list_files = iter([input_files[val] for val in list_vals])
        list_files = [next(list_files) if item else np.nan for item in list_owner]
        list_idxs = iter(list_idxs)
        list_idxs = [next(list_idxs) if item else np.nan for item in list_owner]

        return list_files, list_idxs

    @classmethod
    def _pick_files_by_values(cls, vals: list, input_path: str, input_idx: int = 1) -> list:
        # Load input files and their values
        input_files = [file for file in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, file))]
        input_vals = [int(val.split("_")[input_idx]) for val in input_files]

        # Get indices (first value of tuple [0]) of values that are closest to the desired values
        input_idxs = [cls._get_closest(input_vals, val)[0]
                      if (not pd.isna(val)) and val > 0 else np.nan for val in vals]

        return [input_files[idx] if not np.isnan(idx) else np.nan for idx in input_idxs]

    @classmethod
    def _pick_files_at_random(cls, list_owner: list, input_path: str, file_type: str = "csv",
                              key: str = None, key_idx: int = -1) -> list:
        """
            Generates a random list from files in input folder
        """
        # Pick values based on variance
        input_files = [file for file in os.listdir(input_path)
                       if os.path.isfile(os.path.join(input_path, file)) and file.split(".")[-1] == file_type]

        # Get only files with the correct key if provided
        if key:
            # key_idx = key_idx if key_idx is not None else -1
            # input_files = [file for file in input_files if file.split(".")[0].split("_")[key_idx] == key]
            input_files = [file for file in input_files if key in file]

        return [random.choice(input_files) if owner else np.nan for owner in list_owner]

    def _pick_files(self, list_type: list, device: str, input_path: str = None, idx_type: int = None) -> list:
        """
            Configures the load from part where it either picks a file at random or sets it to weather to subsequently
            calculate the values.
        """
        unique_files = {x for x in set(list_type) if x == x}
        # print(unique_files)
        list_files = [np.nan] * len(list_type)
        # print(list_files)
        # exit()
        # TODO: This function needs to become more flexible. Instead of looking for the item key, it should simply take
        #  the key and look where it is in the name. Otherwise it collects all files that have the same type. So input
        #  should be either file_type or a key. The rest should be deleted.
        for item in unique_files:
            list_owner = np.multiply(np.array(list_type == item), 1).tolist()
            if item == "specs":
                list_temp = self._pick_files_at_random(list_owner=list_owner, input_path=input_path,
                                                       file_type="json")
            elif item == "timeseries":
                list_temp = self._pick_files_at_random(list_owner=list_owner, input_path=input_path,
                                                       file_type="csv")
            elif item in ["ground", "water", "air"]:
                list_temp = self._pick_files_at_random(list_owner=list_owner, input_path=input_path,
                                                       file_type="json", key=item, key_idx=idx_type)
            elif item in self.industry or item in self.ctsp:
                list_temp = self._pick_files_at_random(list_owner=list_owner, input_path=input_path,
                                                       file_type="csv", key=item, key_idx=idx_type)
            elif item == "pu" or "_" in item:
                list_temp = self._pick_files_at_random(list_owner=list_owner, input_path=input_path,
                                                       file_type="csv", key=item, key_idx=idx_type)
            elif os.path.exists(f"{input_path}/{item}"):
                list_temp = [item if owner else np.nan for owner in list_owner]
            else:
                raise ValueError(f"Input value '{item}' unknown.")
            list_files = [new if type(new) == str else current for current, new in zip(list_files, list_temp)]

        return list_files

    @classmethod
    def _get_val_from_name(cls, name_list: list, separator: str = "_", val_idx: int = 1, multiplier: int = 1):
        """
            Obtains the values from the columns based on the separator and val_position
        """

        return [int(item.split(separator)[val_idx]) * multiplier if type(item) is str else item for item in name_list]

    @classmethod
    def _gen_dep_num_list(cls, owner_list: list, distr: list):
        """
            Generates dependent list with random values from distr
        """
        return [random.choice(distr) if item else 0 for item in owner_list]

    @staticmethod
    def __gen_distr_list(num: int, distr: list, equal_one=True) -> list:
        """NOT IN USE --> demand is not given with intervals anymore but with absolute value and variance

        generates a sorted list according to the provided distribution with its total equal to num

        Comment:
            If the sum of the list does not match up with num, the list will be adjusted according to the provided
            probability to ensure that the sum of the list and num match up.

        Args:
            num: integer specifying the sum of the new list (represents the number of participants)
            distr: list that contains the distribution pattern for the new list
            equal_one: makes sure that sum of values in distr equals 1

        Returns:
            list_distr: list according to the provided distribution pattern and the sum equal to num

        """
        # NOT IN USE -->
        # Generate list according to distribution pattern
        list_distr = [round(x / sum(distr) * num) for x in distr] if equal_one \
            else [round(x * num) for x in distr]

        # Adjust list_distr if sum does not match num
        while sum(list_distr) != num:
            idx = random.choices(range(len(distr)), distr)[0]
            if sum(list_distr) > num and list_distr[idx] > 0:  # subtract one element by one
                list_distr[idx] -= 1
            elif sum(list_distr) < num:  # add one element by one
                list_distr[idx] += 1

        return list_distr

    @staticmethod
    def repeat_columns(columns: list, num: int) -> list:
        """
            Repeats the specified columns and adds number
        """
        start = 0
        end = len(columns)
        columns = num * columns
        for iters in range(num):
            for idx, item in enumerate(columns[start + (end - start) * iters:end + (end - start) * iters]):
                columns[idx + start + (end - start) * iters] = item + "_" + str(iters)
        return columns

    @classmethod
    def make_list_from_nested_dict(cls, dict_obj: dict, sign: str = "/", add_string: str = None) -> list:
        ''' This function accepts a nested dictionary as argument
            and iterate over all values of nested dictionaries
        '''
        # Iterate over all key-value pairs of dict argument
        lst = []
        for key, value in dict_obj.items():
            # Check if value is of dict type
            if isinstance(value, dict):
                # If value is dict then iterate over all its values
                for pair in cls.make_list_from_nested_dict(value):
                    lst.append(sign.join([key, pair]))
            else:
                # If value is not dict type then yield the value
                lst.append(key)
        if add_string:
            for idx, item in enumerate(lst):
                lst[idx] = sign.join([add_string, item])
        return lst

    @staticmethod
    def _round_to_nth_digit(vals: list, n: int = 2, method: str = 'round') -> list:
        """ Rounds to the nth digit. Differentiates between values greater and smaller one.
        Based on smallest value in list."""

        # Find smallest value if it exists
        try:
            min_val = min(vals[~np.isnan(vals)])
        except ValueError:
            return vals

        # Calculate length of the smallest value
        min_len = len(str(min_val).split(".")[0])

        if method == "round":
            return round(vals, -min_len + n)
        elif method == "floor":
            return np.floor(vals, -min_len + n)
        elif method == "ceil":
            return np.ceil(vals, -min_len + n)
        else:
            raise ValueError(f"Method '{method}' unknown.")

    @staticmethod
    def _create_path(keys, separator: str = "/"):
        """ Creates a path from the provided keys"""

        c_path = ""
        for key in keys:
            c_path += f"{key}{separator}"

        return c_path

    @classmethod
    def _calc_deviation(cls, idx_list: list, vals: list, distr: list, method: str) -> list:
        """Calculates the deviation"""

        # Calculate the deviation
        deviation = cls._gen_list_from_idx_list(idx_list=idx_list, distr=distr)
        deviation = [x * y for x, y in zip(deviation, np.random.uniform(-1, 1, size=len(idx_list)))]

        # Return the updated values based on the deviation and method
        if method == "relative":
            return [round(max(0, x * (1 + y))) if not np.isnan(y) else np.nan for x, y in zip(vals, deviation)]
        elif method == "absolute":
            return [round(max(0, x + y)) if not np.isnan(y) else np.nan for x, y in zip(vals, deviation)]
        else:
            raise Warning(f"Unknown method ({method}) used in _calc_deviation.")

    def _add_info_simple(self, keys: list, config: dict, df: pd.DataFrame = None, separator: str = "/",
                         preface: str = "", appendix: str = "") -> pd.DataFrame:
        """ This function accepts a  dictionary as argument and fills the according items with the same value
        """

        # Create path from keys
        path_info = self._create_path(keys=keys, separator=separator)

        # Iterate over all key-value pairs of dict that match the dataframe
        item_list = list(df) if df is not None else list(self.df)
        for item, value in config.items():
            if f"{path_info}{preface}{item}{appendix}" in item_list:
                try:
                    if df is not None:
                        df.loc[:, f"{path_info}{preface}{item}{appendix}"] = value
                    else:
                        self.df[f"{path_info}{preface}{item}{appendix}"] = value
                except ValueError:
                    if df is not None:
                        df.loc[:, f"{path_info}{preface}{item}{appendix}"] = str(value)
                    else:
                        self.df[f"{path_info}{preface}{item}{appendix}"] = str(value)

        if df is not None:
            return df
        else:
            return self.df

    def _add_info_random(self, keys: list, config: dict, df: pd.DataFrame = None, separator: str = "/",
                         preface: str = "", appendix: str = "") -> pd.DataFrame:
        """ This function accepts a  dictionary as argument and fills the according items with a randomly chosen value
        """

        # Create path from keys
        path_info = self._create_path(keys=keys, separator=separator)

        # Iterate over all key-value pairs of dict that match the dataframe
        item_list = list(df) if df is not None else list(self.df)
        for item, value in config.items():
            if f"{path_info}{preface}{item}{appendix}" in item_list:
                try:
                    if df is not None:
                        df.loc[:, f"{path_info}{preface}{item}{appendix}"] = random.choices(value, k=len(df))
                    else:
                        self.df[f"{path_info}{preface}{item}{appendix}"] = random.choices(value, k=len(self.df))
                except ValueError:
                    if df is not None:
                        df.loc[: f"{path_info}{preface}{item}{appendix}"] = str(random.choices(value, k=len(df)))
                    else:
                        self.df[f"{path_info}{preface}{item}{appendix}"] = str(random.choices(value, k=len(self.df)))

        if df is not None:
            return df
        else:
            return self.df

    def _add_info_indexed(self, keys: list, config: dict, idx_list: list, df: pd.DataFrame = None,
                          separator: str = "/", preface: str = "", appendix: str = "") -> pd.DataFrame:
        """ This function accepts a  dictionary as argument and fills the according items with the indexed value
        """

        # Create path from keys
        path_info = self._create_path(keys=keys, separator=separator)

        # Iterate over all key-value pairs of dict that match the dataframe
        item_list = list(df) if df is not None else list(self.df)
        for item, value in config.items():
            if f"{path_info}{preface}{item}{appendix}" in item_list:
                if df is not None:
                    df.loc[:, f"{path_info}{preface}{item}{appendix}"] = self._gen_list_from_idx_list(
                        idx_list=idx_list, distr=value)
                else:
                    self.df[f"{path_info}{preface}{item}{appendix}"] = self._gen_list_from_idx_list(
                        idx_list=idx_list, distr=value)

        if df is not None:
            return df
        else:
            return self.df

    def _map_info_on_column(self, key: str, col: str, df: pd.DataFrame = None, config: dict = None,
                            separator: str = "/", preface: str = "", appendix: str = "") -> pd.DataFrame:
        """ This function accepts a  dictionary as argument and fills the according items with the indexed value
        """

        # Iterate over all key-value pairs of dict that match the dataframe
        item_list = list(df) if df is not None else list(self.df)
        for item in item_list:
            if key in item:
                if df is not None:
                    df.loc[:, item] = self.df.groupby(col)[item].apply(lambda x: x.fillna(x.mode().iloc[0]))
                else:
                    self.df[item] = self.df.groupby(col)[item].apply(lambda x: x.fillna(x.mode().iloc[0]))

        if df is not None:
            return df
        else:
            return self.df

    def __info_inflexible_load(self, info: dict, plant_dict: dict, idx: int) -> dict:

        # Add specific plant information (all entries with an index value at the end)
        entries = ["demand", "file"]
        plant_dict = self.__add_entries(info=info, plant_dict=plant_dict, idx=idx, entries=entries)

        # Add forecast information
        # TODO: Forecast does not work yet if there is more than one (see EV for example)
        plant_dict["fcast"] = dict(list(info.items())[5:])
        
        return plant_dict

    def __info_flexible_load(self, info: dict, plant_dict: dict, idx: int) -> dict:

        # Add specific plant information (all entries with an index value at the end)
        entries = ["demand", "file"]
        plant_dict = self.__add_entries(info=info, plant_dict=plant_dict, idx=idx, entries=entries)

        # Add forecast information
        plant_dict["fcast"] = dict(list(info.items())[5:])

        return plant_dict

    def __info_heat(self, info: dict, plant_dict: dict, idx: int) -> dict:

        # Add specific plant information (all entries with an index value at the end)
        entries = ["demand", "file", "temperature"]
        plant_dict = self.__add_entries(info=info, plant_dict=plant_dict, idx=idx, entries=entries)

        # Add forecast information
        plant_dict["fcast"] = dict(list(info.items())[5:])

        return plant_dict

    def __info_dhw(self, info: dict, plant_dict: dict, idx: int) -> dict:

        # Add specific plant information (all entries with an index value at the end)
        entries = ["demand", "file", "temperature"]
        plant_dict = self.__add_entries(info=info, plant_dict=plant_dict, idx=idx, entries=entries)

        # Add forecast information
        plant_dict["fcast"] = dict(list(info.items())[5:])

        return plant_dict

    def __info_pv(self, info: dict, plant_dict: dict, idx: int) -> dict:

        # Add specific plant information (all entries with an index value at the end)
        entries = ["power", "file", "orientation", "angle", "controllable"]
        plant_dict = self.__add_entries(info=info, plant_dict=plant_dict, idx=idx, entries=entries)

        # Add forecast information
        plant_dict["fcast"] = dict(list(info.items())[8:-1])

        # Add quality of energy
        plant_dict["quality"] = info["quality"]

        return plant_dict

    def __info_wind(self, info: dict, plant_dict: dict, idx: int) -> dict:

        # Add specific plant information (all entries with an index value at the end)
        entries = ["power", "file", "controllable"]
        plant_dict = self.__add_entries(info=info, plant_dict=plant_dict, idx=idx, entries=entries)

        # Add forecast information
        plant_dict["fcast"] = dict(list(info.items())[6:-1])

        # Add quality of energy
        plant_dict["quality"] = info["quality"]

        return plant_dict

    def __info_fixed_gen(self, info: dict, plant_dict: dict, idx: int) -> dict:

        # Add specific plant information (all entries with an index value at the end)
        entries = ["power", "file", "controllable"]
        plant_dict = self.__add_entries(info=info, plant_dict=plant_dict, idx=idx, entries=entries)

        # Add forecast information
        plant_dict["fcast"] = dict(list(info.items())[6:-1])

        # Add quality of energy
        plant_dict["quality"] = info["quality"]

        return plant_dict

    def __info_hp(self, info: dict, plant_dict: dict, idx: int) -> dict:

        # Add specific plant information (all entries with an index value at the end)
        entries = ["power", "file"]
        plant_dict = self.__add_entries(info=info, plant_dict=plant_dict, idx=idx, entries=entries)

        # Add quality of energy
        plant_dict["quality"] = info["quality"]

        return plant_dict

    def __info_ev(self, info: dict, plant_dict: dict, idx: int) -> dict:

        # Add specific plant information (all entries with an index value at the end)
        entries = ["file", "capacity", "consumption", "charging_home", "charging_AC", "charging_DC",
                   "charging_efficiency", "soc", "v2g", "v2h"]
        plant_dict = self.__add_entries(info=info, plant_dict=plant_dict, idx=idx, entries=entries)

        # Add forecast information
        plant_dict["fcast"] = dict(list(info.items())[13:])

        return plant_dict

    def __info_battery(self, info: dict, plant_dict: dict, idx: int) -> dict:

        # Add specific plant information (all entries with an index value at the end)
        entries = ["power", "capacity", "efficiency", "soc", "g2b", "b2g"]
        plant_dict = self.__add_entries(info=info, plant_dict=plant_dict, idx=idx, entries=entries)

        # Add quality of energy
        plant_dict["quality"] = info["quality"]

        return plant_dict

    def __info_psh(self, info: dict, plant_dict: dict, idx: int) -> dict:

        # Add specific plant information (all entries with an index value at the end)
        entries = ["power", "capacity", "efficiency", "soc", "g2psh", "psh2g"]
        plant_dict = self.__add_entries(info=info, plant_dict=plant_dict, idx=idx, entries=entries)

        # Add quality of energy
        plant_dict["quality"] = info["quality"]

        return plant_dict

    def __info_hydrogen(self, info: dict, plant_dict: dict, idx: int) -> dict:
        # Add specific plant information (all entries with an index value at the end)
        entries = ["power", "capacity", "efficiency", "soc", "g2h2", "h22g"]
        plant_dict = self.__add_entries(info=info, plant_dict=plant_dict, idx=idx, entries=entries)

        # Add quality of energy
        plant_dict["quality"] = info["quality"]

        return plant_dict

    def __info_heat_storage(self, info: dict, plant_dict: dict, idx: int) -> dict:

        # Add specific plant information (all entries with an index value at the end)
        entries = ["capacity", "efficiency", "soc"]
        plant_dict = self.__add_entries(info=info, plant_dict=plant_dict, idx=idx, entries=entries)

        # Add quality of energy
        plant_dict["quality"] = info["quality"]

        return plant_dict

    @staticmethod
    def __create_pv_system_from_config(config: dict, orientation: tuple) -> PVSystem:
        """create PV system from hardware config file.

        Args:
            config: pv hardware configuration from json file (dic)
            orientation: pv orientation (surface tilt, surface azimuth) (tuple)

        Returns:
            system: a PV system abject (PVSystem)

        """
        # The class supports basic system topologies consisting of:
        # N total modules arranged in series (modules_per_string=N, strings_per_inverter=1).
        # M total modules arranged in parallel (modules_per_string=1, strings_per_inverter=M).
        # NxM total modules arranged in M strings of N modules each (modules_per_string=N, strings_per_inverter=M).

        # get pv orientation
        surface_tilt, surface_azimuth = orientation

        # get hardware data
        module = pd.Series(config['module'])
        inverter = pd.Series(config['inverter'])

        # set temperature model
        temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

        # generate pv system
        system = PVSystem(
            surface_tilt=surface_tilt,
            surface_azimuth=surface_azimuth,
            module_parameters=module,
            inverter_parameters=inverter,
            temperature_model_parameters=temperature_model_parameters
        )

        return system

    @staticmethod
    def __adjust_weather_data_for_pv(weather_path: str, location: tuple) -> pd.DataFrame:
        """adjust weather data to the right format that pvlib needed.

        Args:
            weather_path: path of the original weather data (string)
            location: location of the weather data (latitude, longitude, name, altitude, timezone) (tuple)

        Returns:
            weather: adjusted weather data (dataframe)

        """
        # get location data
        latitude, longitude, name, altitude = location

        # get weather data from csv
        weather = pd.read_csv(weather_path)
        weather = weather[weather['ts_delivery_current'] == weather['ts_delivery_fcast']]  # remove forcasting data

        # convert time data to datetime (use utc time overall in pvlib)
        time = pd.DatetimeIndex(pd.to_datetime(weather['ts_delivery_current'], unit='s', utc=True))
        weather.index = time
        weather.index.name = 'utc_time'

        # adjust temperature data
        weather.rename(columns={'temp': 'temp_air'}, inplace=True)  # rename to pvlib format
        weather['temp_air'] -= 273.15  # convert unit to celsius

        # get solar position
        # test data find in https://www.suncalc.org/#/48.1364,11.5786,15/2022.02.15/16:21/1/3
        solpos = pvlib.solarposition.get_solarposition(
            time=time,
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            temperature=weather['temp_air'],
            pressure=weather['pressure'],
        )

        # calculate dni with solar position
        weather.loc[:, 'dni'] = (weather['ghi'] - weather['dhi']) / np.cos(solpos['zenith'])

        return weather

    def __timeseries_from_specs_pv(self, specs: dict, plant: dict) -> pd.DataFrame:
        """Creates a time series for pv power from config spec file using pv power model.

        Args:
            specs (dict): A dictionary of pv hardware config.
            plant (dict): A dictionary of pv planet information.

        Returns:
            power: A Pandas Dataframe object representing the time series data for the pv plant.

        """
        # get location information from config
        location = self.setup['simulation']['location']
        location = (location['latitude'], location['longitude'], location['name'], location['altitude'])

        # get plant orientation
        orientation = (plant['orientation'], plant['angle'])

        # get weather path
        weather_path = os.path.join(self.input_path, 'general', 'weather', self.setup['simulation']['location']['weather'])

        # create PVSystem and adjust weather data
        system = self.__create_pv_system_from_config(config=specs, orientation=orientation)
        weather = self.__adjust_weather_data_for_pv(weather_path=weather_path, location=location)

        # get location data and create corresponding pvlib Location object
        latitude, longitude, name, altitude = location
        location = Location(
            latitude,
            longitude,
            name=name,
            altitude=altitude
        )

        # create calculation model for the given pv system and location
        mc = pvlib.modelchain.ModelChain(system, location)

        # calculate model under given weather data and get output ac power from it
        mc.run_model(weather)
        power = mc.results.ac

        # calculate nominal power
        nominal_power = specs['module']['Impo']*specs['module']['Vmpo']

        # set time index to origin timestamp
        power.index = weather['ts_delivery_current']
        power.index.name = 'timestamp'

        # rename and round data column
        power.rename('power', inplace=True)
        power = power.to_frame()

        # calculate and round power
        power = power / nominal_power * plant['power']
        power = power.round().astype(int)

        # replace all negative values
        power[power < 0] = 0

        return power

    @staticmethod
    def __adjust_weather_data_for_wind(weather_path: str) -> pd.DataFrame:
        """adjust weather data to the right format that windpowerlib needed.

        Args:
            weather_path: path of the original weather data (string)

        Returns:
            weather: adjusted weather data (dataframe)

        """
        # get weather data from csv
        weather = pd.read_csv(weather_path)
        weather = weather[weather['ts_delivery_current'] == weather['ts_delivery_fcast']]  # remove forcasting data

        # convert time data to datetime
        time = pd.DatetimeIndex(pd.to_datetime(weather['ts_delivery_current'], unit='s', utc=True))
        weather.index = time
        weather.index.name = None

        # delete unnecessary columns and rename
        weather.drop(['ts_delivery_fcast', 'cloud_cover', 'sunrise', 'sunset', 'ghi', 'dhi',
                      'visibility', 'pop'], axis=1, inplace=True)
        weather.rename(columns={'temp': 'temperature'}, inplace=True)

        if 'roughness_length' not in weather.columns:
            weather['roughness_length'] = 0.15

        # generate height level hard-coded
        weather.columns = pd.MultiIndex.from_tuples(tuple(zip(weather.columns, [2, 2, 2, 2, 2, 2, 2, 10, 10, 2])),
                                                    names=('', 'height'))

        return weather

    def __timeseries_from_specs_wind(self, specs: dict, plant: dict) -> pd.DataFrame:
        """Creates a time series for wind power from config spec file using wind power model.

        Args:
            specs (dict): A dictionary of wind hardware config.
            plant (dict): A dictionary of wind planet information.

        Returns:
            power (dataframe): A Pandas Dataframe object representing the time series data for the wind plant.

        """
        # get weather path
        weather_path = os.path.join(self.input_path, 'general', 'weather', self.setup['simulation']['location']['weather'])

        # get weather data
        weather = self.__adjust_weather_data_for_wind(weather_path=weather_path)

        # get nominal power
        nominal_power = specs['nominal_power']

        # convert power curve to dataframe
        specs['power_curve'] = pd.DataFrame(data={
            "value": specs['power_curve'],
            "wind_speed": specs['wind_speed']})

        # convert power coefficient curve to dataframe
        specs['power_coefficient_curve'] = pd.DataFrame(data={
            "value": specs['power_coefficient_curve'],
            "wind_speed": specs['wind_speed']})

        # generate a WindTurbine object from data
        turbine = WindTurbine(**specs)

        # calculate turbine model
        mc_turbine = ModelChain(turbine).run_model(weather)

        # get output power
        power = mc_turbine.power_output

        # set time index to origin timestamp
        power.index = weather['ts_delivery_current'].unstack(level=0).values
        power.index.name = 'timestamp'

        # rename data column
        power.rename('power', inplace=True)
        power = power.to_frame()

        # calculate and round power
        power = power / nominal_power * plant['power']
        power = power.round().astype(int)

        return power

    def __timeseries_from_specs_hp(self, specs: dict, plant: dict):
        # TODO: @Zhengjie
        # input is the specs of the plant and the plant dict
        print(specs)
        print(plant)
        # output is the timeseries of the plant. This should be a pandas dataframe with the index being the unix timestamp and the columns named "power" and "cop" (if applicable)
        pass

    @staticmethod
    def __add_entries(info: dict, plant_dict: dict, idx: int, entries: list) -> dict:
        """Adds specified entries from info in plant_dict"""

        for entry in entries:
            plant_dict[entry] = info[f"{entry}_{idx}"]

        return plant_dict

    @staticmethod
    def resample_timeseries(timeseries: pd.DataFrame, delta: int) -> pd.DataFrame:
        """
        Resamples a timeseries with the index being the unix timestamp.
        The resampling method (interpolate or mean) is decided based on the comparison between
        the input delta and the time delta of the input timeseries.

        Parameters:
        df (pd.DataFrame): The input timeseries
        delta (int): The desired time delta in seconds between two time steps

        Returns:
        pd.DataFrame: The resampled timeseries
        """

        # Calculate the delta of the input timeseries
        original_delta = int(timeseries.index[1] - timeseries.index[0])

        # Return the original timeseries if the delta is the same
        if delta == original_delta:
            return timeseries

        # Convert the index to datetime
        timeseries.index = pd.to_datetime(timeseries.index, unit='s')

        # Resample the timeseries (mean when delta > original_delta, interpolate otherwise)
        if delta > original_delta:
            # Copy the original timeseries
            resampled = timeseries.copy()

            # Calculate the number of times the original delta can be divided by the desired delta
            multiple = int(delta / original_delta)

            # Add the shifted timeseries to the original timeseries (fillna(0) to ensure there is no NaN)
            for i in range(1, multiple):
                resampled += timeseries.shift(-i).fillna(0)

            # Calculate the mean of the timeseries by dividing multiple and selecting every nth value
            resampled = (resampled / multiple)[::multiple]
        else:
            # Interpolate the timeseries
            resampled = timeseries.resample(f'{delta}s').interpolate()

        # Convert the index back to unix timestamp
        resampled.index = resampled.index.astype(int) // 10 ** 9

        # Convert the data types back to the original ones
        if type(timeseries) == pd.DataFrame:
            for col, dtype in timeseries.dtypes.items():
                resampled[col] = resampled[col].astype(dtype)
        elif type(timeseries) == pd.Series:
            resampled = resampled.astype(timeseries.dtype)
        else:
            raise TypeError(f"Type {type(timeseries)} is not supported")

        return resampled

    @classmethod
    def get_num_from_grid(cls, df: pd.DataFrame, type: str, col: str = None, unique: str = 'bus') -> int:
        """Returns the number of plants of a specific type in a grid"""

        # Assign the column to the column containing '_type' if no column is specified
        col = [col for col in df.columns if '_type' in str(col)][0] if col is None else col

        # Get the number of agents of the specified type
        num = df[df[col] == type]

        if unique:
            num = len(num[unique].unique())

        return num

    @staticmethod
    def __add_info_from_col(df: pd.DataFrame, col: str, drop: bool = False, sep: str = ',', key_val_sep: str = ':') \
            -> pd.DataFrame:
        """Adds information from a column to the dataframe

        The column should contain a string with the following format:
        'key1:value1,key2:value2,...,keyN:valueN'

        Parameters:
            df (pd.DataFrame): The input dataframe
            col (str): The column containing the information
            drop (bool): Whether to drop the column containing the information
            sep (str): The separator between key-value pairs
            key_val_sep (str): The separator between keys and values

        Returns:
            pd.DataFrame: The dataframe with the information added as separate columns

        Alternative method:
            # Split the strings into separate key-value pairs
            df['parsed'] = df[col].apply(lambda x: dict(tuple(i.split(key_val_sep)) for i in x.split(sep)))

            # Get all the keys from the parsed strings
            keys = set().union(*df['parsed'].apply(lambda x: x.keys()))

            # Create separate columns for each key-value pair and fill with np.nan if not present
            for key in keys:
                df[key] = df['parsed'].apply(
                    lambda x: x.get(key, np.nan) if x.get(key, None) in ['NaN', 'nan'] else x.get(key, np.nan))

            # Drop the original column and the intermediate parsed column
            df.drop(columns=['col', 'parsed'], inplace=True)
        """

        # Turn the column into a dictionary
        info = df[col].to_dict()

        # Loop through the dictionary and create entries for the key-value pairs
        for idx, val in info.items():
            # Split the key-value pairs

            key_value_pairs = val.split(sep)
            # Create a dictionary for the key-value pairs
            info[idx] = dict()

            for key_value_pair in key_value_pairs:
                # Split the key and value
                key, value = key_value_pair.split(key_val_sep)

                # Add the key-value pair to the dictionary and convert them to the desired data type
                try:
                    info[idx][key] = int(value)
                except ValueError:
                    try:
                        info[idx][key] = float(value)
                    except ValueError:
                        info[idx][key] = str(value)

        # Create a dataframe from the dictionary
        df = df.join(pd.DataFrame(info).T)

        # Fill empty values and cells with string NaN and nan with NaN
        df = df.replace(r'^\s*$', np.nan, regex=True)
        df = df.replace(["NaN", "nan"], np.nan, regex=True)

        # Drop the original column
        if drop:
            df.drop(columns=col, inplace=True)

        return df

    @staticmethod
    def _get_closest_sorted(search_list, val):
        """
        Assumes search_list is sorted. Returns the closest value to find_val.

        If two numbers are equally close, return the smallest number.
        """
        pos = bisect_left(search_list, val)
        if pos == 0:
            return search_list[0], pos
        if pos == len(search_list):
            return search_list[-1], pos
        before = search_list[pos - 1]
        after = search_list[pos]
        if after - val < val - before:
            print('after')
            return after, pos
        else:
            print('before')
            return before, pos

    @staticmethod
    def _get_closest(search_list: list, val: float, sorted_list: bool = False):
        """ Returns the closest value to val in search_list."""

        if sorted_list:
            return Agents._get_closest_sorted(search_list, val)

        return min(enumerate(search_list), key=lambda x: abs(x[1] - val))

    @staticmethod
    def __get_types(path: str, idx: int = 0, sep: str = '_') -> list:
        """Returns a list of all types of a specific agent type"""

        # Return the unique types from the files in the directory
        return list(set([file.split(sep)[idx] for file in os.listdir(path)]))


class Sfh(Agents):
    """
        Sets up sfh agents. Inherits from Agents class.

        Mainly used for Excel file creation. Afterwards Sfh class creates the individual agents.
    """

    def __init__(self, input_path: str, config: ordereddict, config_path: str, scenario_path: str):

        # Call the init method of the parent class
        super().__init__(config_path, input_path, scenario_path)

        # Define agent type
        self.type = 'sfh'

        # Path of the input file
        self.input_path = os.path.join(input_path, 'agents', self.type)

        # Config file
        self.config = config

        # Grid information (if applicable)
        self.grid = None
        self.bus = None  # bus sheet containing only the bus information of the agent type
        self.load = None  # load sheet containing only the load information of the agent type
        self.agents = None # load sheet but limited to all agents, i.e. all inflexible_loads
        self.sgen = None  # sgen sheet containing only the sgen information of the agent type

        # Creation method
        self.method = None

        # Number of agents
        self.num = 0

        # Dataframe containing all information
        self.df = None

        # Misc
        self.n_digits = 2           # number of digits values get rounded to in respective value column

    def create_df_from_config(self) -> pd.DataFrame:
        """Function to create the dataframe that makes the Excel sheet
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
        self.fill_mpc()

        # Fill the market agent information in dataframe
        self.fill_market_agent()

        # Fill the metering information in dataframe
        self.fill_meter()

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
        self.fill_mpc()

        # Fill the market agent information in dataframe
        self.fill_market_agent()

        # Fill the metering information in dataframe
        self.fill_meter()

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
            if key == "general":
                cols[0] = f"{key}/agent_id"
                cols[-1] = f"{key}/market_participant"
                del cols[1]
                cols.insert(1, f"{key}/name")
                cols.insert(2, f"{key}/comment")
                cols.insert(3, f"{key}/bus")
            # Adjust the columns from "inflexible_load"
            elif key == "inflexible_load":
                cols[0] = f"{key}/owner"
                cols[5] = f"{key}/sizing/file"
                del cols[3]
                max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
                cols = cols[:3] + self.repeat_columns(columns=cols[3:5], num=max_num) + cols[5:]
            # Adjust the columns from "flexible_load"
            elif key == "flexible_load":
                cols[0] = f"{key}/owner"
                cols[5] = f"{key}/sizing/file"
                del cols[3]
                max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
                cols = cols[:3] + self.repeat_columns(columns=cols[3:6], num=max_num) + cols[6:]
            # Adjust the columns from "heat"
            elif key == "heat":
                cols[0] = f"{key}/owner"
                cols[3] = f"{key}/sizing/demand"
                max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
                cols = cols[:3] + self.repeat_columns(columns=cols[3:6], num=max_num) + cols[6:]
            # Adjust the columns from "dhw"
            elif key == "dhw":
                cols[0] = f"{key}/owner"
                del cols[3]
                max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
                cols = cols[:3] + self.repeat_columns(columns=cols[3:6], num=max_num) + cols[6:]
            # Adjust the columns from "pv"
            elif key == "pv":
                cols[0] = f"{key}/owner"
                del cols[5]
                del cols[3]
                max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
                cols = cols[:3] + self.repeat_columns(columns=cols[3:8], num=max_num) + cols[8:]
            # Adjust the columns from "wind"
            elif key == "wind":
                cols[0] = f"{key}/owner"
                del cols[5]
                del cols[3]
                max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
                cols = cols[:3] + self.repeat_columns(columns=cols[3:6], num=max_num) + cols[6:]
            # Adjust the columns from "fixed_gen"
            elif key == "fixed_gen":
                cols[0] = f"{key}/owner"
                del cols[5]
                del cols[3]
                max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
                cols = cols[:3] + self.repeat_columns(columns=cols[3:6], num=max_num) + cols[6:]
            # Adjust the columns from "hp"
            elif key == "hp":
                cols[0] = f"{key}/owner"
                del cols[3]
                max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
                cols = cols[:3] + self.repeat_columns(columns=cols[3:5], num=max_num) + cols[5:]
            # Adjust the columns from "ev"
            elif key == "ev":
                cols[0] = f"{key}/owner"
                del cols[3]
                max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
                cols = cols[:3] + self.repeat_columns(columns=cols[3:13], num=max_num) + cols[13:]
            # Adjust the columns from "battery"
            elif key == "battery":
                cols[0] = f"{key}/owner"
                del cols[4]
                del cols[1]
                max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
                cols = cols[:3] + self.repeat_columns(columns=cols[3:9], num=max_num) + cols[9:]
            # Adjust the columns from "heat_storage"
            elif key == "heat_storage":
                cols[0] = f"{key}/owner"
                del cols[4]
                del cols[1]
                max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
                cols = cols[:3] + self.repeat_columns(columns=cols[3:7], num=max_num) + cols[7:]
            # Adjust the columns from "mpc"
            elif key == "mpc":
                pass
            # Adjust the columns from "market_agent"
            elif key == "market_agent":
                pass
            # Adjust the columns from "meter"
            elif key == "meter":
                pass
            else:
                raise NotImplementedError(
                    f"The configuration file contains a key word ('{key}') that has not been configured in "
                    "the Sfhs class yet. Aborting scenario creation...")

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
        # Key in the config file
        key = "general"

        # Get the config for the key
        config = self.config[f"{key}"]

        # general
        self.df[f"{key}/agent_id"] = self._gen_new_ids(n=self.num)

        # parameters
        idx_list = self._gen_idx_list_from_distr(n=self.num, distr=config["parameters"]["distribution"])
        self._add_info_indexed(keys=[key, "parameters"], config=config["parameters"], idx_list=idx_list)

        # market participation
        self.df[f"{key}/market_participant"] = self._gen_rand_bool_list(n=self.num,
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

    def fill_inflexible_load(self, **kwargs) -> pd.DataFrame:
        """
            Fills all inflexible_load columns
        """

        # Key in the config file
        key = "inflexible_load"

        # Get the config for the key
        config = self.config[f"{key}"]

        if self.method == 'config':
            self.df = self._inflexible_load_config(key=key, config=config)
        elif self.method == 'grid':
            self.df = self._inflexible_load_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _inflexible_load_config(self, key: str, config: dict) -> pd.DataFrame:
        """adds the inflexible load from the config file"""

        # general
        self.add_general_info_config(key=key)

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
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

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
        # note: always taken from config
        self.df[f"{key}/has_submeter"] = self.config[f"{key}"]["has_submeter"]

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
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

        return self.df

    def fill_flexible_load(self) -> pd.DataFrame:
        """
            Fills all flexible_load columns
        """
        key = "flexible_load"
        config = self.config[f"{key}"]
        max_num = max(config["num"])

        # general
        self.add_general_info_config(key=key)

        # sizing
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
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

        return self.df

    def fill_heat(self, **kwargs) -> pd.DataFrame:
        """
            Fills all heat columns
        """

        # Key in the config file
        key = "heat"

        # Get the config for the key
        config = self.config[f"{key}"]

        if self.method == 'config':
            self.df = self._heat_config(key=key, config=config)
        elif self.method == 'grid':
            self.df = self._heat_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _heat_config(self, key: str, config: dict) -> pd.DataFrame:
        """adds the heat from the config file"""

        # general
        self.add_general_info_config(key=key)

        # sizing
        max_num = max(config["num"])
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
            self.df.loc[self.df[f"{key}/sizing/file_{num}"] == 'linked', f"{key}/sizing/file_{num}"] = self.df.loc[self.df[f"{key}/sizing/file_{num}"] == 'linked', f"inflexible_load/sizing/file_0"]
            # shorten the file name to the yearly demand and index
            self.df[f"{key}/sizing/file_{num}"] = ["_".join(item.rsplit(".", 1)[0].split("_")[1:3]) if 'csv' in item else item for item in self.df[f"{key}/sizing/file_{num}"]]
            self.df[f"{key}/sizing/file_{num}"] = self._pick_files(list_type=self.df[f"{key}/sizing/file_{num}"],
                                                                   device=f"{key}",
                                                                   input_path=os.path.join(self.input_path, key))

        # forecast
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

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
        self.df[f"{key}/owner"] = (self.df[f"{key}/num"] > 0).fillna(0).astype('Int64')  # all agents that have plant type
        # note: always taken from config
        self.df[f"{key}/has_submeter"] = self.config[f"{key}"]["has_submeter"]

        # sizing
        for num in range(max(self.df[f"{key}/num"])):  # currently only one device per agent is supported
            # Get demand from load sheet
            self.df[f"{key}/sizing/demand_{num}"] = self.df.index.map(df['demand'] * 1e6).astype('Int64')
            self.df[f"{key}/sizing/file_{num}"] = self.df.index.map(df['heat_file'])
            self.df[f"{key}/sizing/temperature_{num}"] = self.df.index.map(df['temperature'])

        # forecast
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

        return self.df

    def fill_dhw(self, **kwargs) -> pd.DataFrame:
        """
            Fills all dhw columns
        """

        # Key in the config file
        key = "dhw"

        # Get the config for the key
        config = self.config[f"{key}"]

        if self.method == 'config':
            self.df = self._dhw_config(key=key, config=config)
        elif self.method == 'grid':
            self.df = self._dhw_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _dhw_config(self, key: str, config: dict) -> pd.DataFrame:
        """adds the dhw from the config file"""

        # general
        self.add_general_info_config(key=key)

        # sizing
        max_num = max(config["num"])
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
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

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
        self.df[f"{key}/owner"] = (self.df[f"{key}/num"] > 0).fillna(0).astype('Int64')  # all agents that have plant type
        # note: always taken from config
        self.df[f"{key}/has_submeter"] = self.config[f"{key}"]["has_submeter"]

        # sizing
        for num in range(max(self.df[f"{key}/num"])):  # currently only one device per agent is supported
            # Get demand from load sheet
            self.df[f"{key}/sizing/demand_{num}"] = self.df.index.map(df['demand'] * 1e6).astype('Int64')
            self.df[f"{key}/sizing/file_{num}"] = self.df.index.map(df['heat_file'])
            self.df[f"{key}/sizing/temperature_{num}"] = self.df.index.map(df['temperature'])

        # forecast
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

        return self.df

    def fill_pv(self, **kwargs) -> pd.DataFrame:
        """
            Fills all pv columns
        """

        # Key in the config file
        key = "pv"

        # Get the config for the key
        config = self.config[f"{key}"]

        if self.method == 'config':
            self.df = self._pv_config(key=key, config=config)
        elif self.method == 'grid':
            self.df = self._pv_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _pv_config(self, key: str, config: dict) -> pd.DataFrame:
        """adds the inflexible load from the config file"""

        # general
        self.add_general_info_config(key=key)

        # sizing
        max_num = max(config["num"])  # max number of devices per agent
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
            self.df[f"{key}/sizing/power_{num}"] *= self.df["inflexible_load/sizing/demand_0"] / 1000
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
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

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
        # note: always taken from config
        self.df[f"{key}/has_submeter"] = self.config[f"{key}"]["has_submeter"]

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
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

        # quality
        self.df[f"{key}/quality"] = config["quality"]

        return self.df

    def fill_wind(self, **kwargs) -> pd.DataFrame:
        """
            Fills all wind columns
        """
        # Key in the config file
        key = "wind"

        # Get the config for the key
        config = self.config[f"{key}"]

        if self.method == 'config':
            self.df = self._wind_config(key=key, config=config)
        elif self.method == 'grid':
            self.df = self._wind_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _wind_config(self, key: str, config: dict) -> pd.DataFrame:

        # general
        self.add_general_info_config(key=key)

        # sizing
        max_num = max(config["num"])  # max number of devices per agent
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
            self.df[f"{key}/sizing/power_{num}"] *= self.df["inflexible_load/sizing/demand_0"] / 1000
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
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

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
        # note: always taken from config
        self.df[f"{key}/has_submeter"] = self.config[f"{key}"]["has_submeter"]

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
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

        # quality
        self.df[f"{key}/quality"] = config["quality"]

        return self.df

    def fill_fixed_gen(self, **kwargs) -> pd.DataFrame:
        """
            Fills all fixed_gen columns
        """

        # Key in the config file
        key = "fixed_gen"

        # Get the config for the key
        config = self.config[f"{key}"]

        if self.method == 'config':
            self.df = self._fixed_gen_config(key=key, config=config)
        elif self.method == 'grid':
            self.df = self._fixed_gen_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _fixed_gen_config(self, key: str, config: dict) -> pd.DataFrame:

        # general
        self.add_general_info_config(key=key)

        # sizing
        max_num = max(config["num"])
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
            self.df[f"{key}/sizing/power_{num}"] *= self.df["inflexible_load/sizing/demand_0"] / 1000
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
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

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
        # note: always taken from config
        self.df[f"{key}/has_submeter"] = self.config[f"{key}"]["has_submeter"]

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
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

        # quality
        self.df[f"{key}/quality"] = config["quality"]

        return self.df

    def fill_hp(self, **kwargs) -> pd.DataFrame:
        """
            Fills all hp columns
        """

        # Key in the config file
        key = "hp"

        # Get the config for the key
        config = self.config[f"{key}"]

        if self.method == 'config':
            self._hp_config(key=key, config=config)
        elif self.method == 'grid':
            self._hp_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _hp_config(self, key: str, config: dict) -> pd.DataFrame:

        # general
        self.add_general_info_config(key=key)

        # sizing
        max_num = max(config["num"])
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
        self.df[f"{key}/owner"] = (self.df[f"{key}/num"] > 0).fillna(0).astype('Int64')  # all agents that have plant type
        # note: always taken from config
        self.df[f"{key}/has_submeter"] = self.config[f"{key}"]["has_submeter"]

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

    def fill_ev(self, **kwargs) -> pd.DataFrame:
        """
            Fills all ev columns
        """

        # Key in the config file
        key = "ev"

        # Get the config for the key
        config = self.config[f"{key}"]

        if self.method == 'config':
            self._ev_config(key=key, config=config)
        elif self.method == 'grid':
            self._ev_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _ev_config(self, key: str, config: dict) -> pd.DataFrame:

        # general
        self.add_general_info_config(key=key)

        # sizing
        max_num = max(config["num"])
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
        self._add_info_simple(keys=[key, "charging_scheme"], config=config["charging_scheme"])

        # forecast
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

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
        self.df[f"{key}/owner"] = (self.df[f"{key}/num"] > 0).fillna(0).astype('Int64')  # all agents that have plant type
        # note: always taken from config
        self.df[f"{key}/has_submeter"] = self.config[f"{key}"]["has_submeter"]

        # sizing
        for num in range(max(self.df[f"{key}/num"])):  # currently only one device per agent is supported
            self.df[f"{key}/sizing/file_{num}"] = self.df.index.map(df['file_add'])
            self.df[f"{key}/sizing/capacity_{num}"] = (self.df.index.map(df['capacity']) * 1e6).astype('Int64')
            self.df[f"{key}/sizing/consumption_{num}"] = (self.df.index.map(df['consumption']) * 1e6).astype('Int64')
            self.df[f"{key}/sizing/charging_home_{num}"] = (self.df.index.map(df['charging_home']) * 1e6).astype('Int64')
            self.df[f"{key}/sizing/charging_AC_{num}"] = (self.df.index.map(df['charging_ac']) * 1e6).astype('Int64')
            self.df[f"{key}/sizing/charging_DC_{num}"] = (self.df.index.map(df['charging_dc']) * 1e6).astype('Int64')
            self.df[f"{key}/sizing/charging_efficiency_{num}"] = self.df.index.map(df['efficiency'])
            self.df[f"{key}/sizing/soc_{num}"] = self.df.index.map(df['soc'])
            self.df[f"{key}/sizing/v2g_{num}"] = self.df.index.map(df['v2g'])
            self.df[f"{key}/sizing/v2h_{num}"] = self.df.index.map(df['v2h'])

        # quality
        self.df[f"{key}/quality"] = config["quality"]

        return self.df

    def fill_battery(self, **kwargs) -> pd.DataFrame:
        """
            Fills all battery columns
        """

        # Key in the config file
        key = "battery"

        # Get the config for the key
        config = self.config[f"{key}"]

        if self.method == 'config':
            self.df = self._battery_config(key=key, config=config)
        elif self.method == 'grid':
            self.df = self._battery_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _battery_config(self, key: str, config: dict) -> pd.DataFrame:

        # general
        self._add_general_info_dep(key=key)

        # sizing
        max_num = max(config["num"])
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
            self.df[f"{key}/sizing/power_{num}"] *= self.df["inflexible_load/sizing/demand_0"] / 1000
            self.df[f"{key}/sizing/power_{num}"] = self._round_to_nth_digit(
                vals=self.df[f"{key}/sizing/power_{num}"], n=self.n_digits)
            # capacity
            self.df[f"{key}/sizing/capacity_{num}"] *= self.df["inflexible_load/sizing/demand_0"] / 1000
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
        # note: always taken from config
        self.df[f"{key}/has_submeter"] = self.config[f"{key}"]["has_submeter"]

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

    def fill_heat_storage(self, **kwargs) -> pd.DataFrame:
        """
            Fills all heat storage columns
        """

        # Key in the config file
        key = "heat_storage"

        # Get the config for the key
        config = self.config[f"{key}"]

        if self.method == 'config':
            self.df = self._heat_storage_config(key=key, config=config)
        elif self.method == 'grid':
            self.df = self._heat_storage_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _heat_storage_config(self, key: str, config: dict) -> pd.DataFrame:

        # general
        self._add_general_info_dep(key=key, method='exclusive')

        # sizing
        max_num = max(config["num"])
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
        # note: always taken from config
        self.df[f"{key}/has_submeter"] = self.config[f"{key}"]["has_submeter"]

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

    def fill_mpc(self):
        """
            Fills all battery columns
        """
        key = "mpc"
        config = self.config[f"{key}"]

        # general
        self._add_info_simple(keys=[key], config=config)

    def fill_market_agent(self):
        """
            Fills all battery columns
        """
        key = "market_agent"
        config = self.config[f"{key}"]

        # general
        self._add_info_random(keys=[key], config=config)

    def fill_meter(self):
        """
            Fills all battery columns
        """
        key = "meter"
        config = self.config[f"{key}"]

        # general
        self._add_info_simple(keys=[key], config=config)

    def add_general_info_config(self, key: str) -> None:

        # fields that exist for all plants
        self.df[f"{key}/owner"] = self._gen_rand_bool_list(n=self.num,
                                                           share_ones=self.config[f"{key}"]["share"])
        self.df[f"{key}/num"] = self._gen_dep_num_list(owner_list=self.df[f"{key}/owner"],
                                                       distr=self.config[f"{key}"]["num"])
        self.df[f"{key}/has_submeter"] = self.config[f"{key}"]["has_submeter"]

    def add_general_info_grid(self, key: str) -> None:

        # Taken from grid
        self.df[f"{key}/owner"] = (self.load['p_mw'] > 0).astype(int)
        self.df[f"{key}/num"] = self.df[f"{key}/owner"]  # equals owner as only one inflexible load per agent

        # Taken from config
        self.df[f"{key}/has_submeter"] = self.config[f"{key}"]["has_submeter"]

    def _add_general_info_dep(self, key: str, method: str = "inclusive"):
        """special case battery as it also depends on other plants"""

        # get a list of all the agents that fulfill the dependency requirements
        # note: in both cases the agent needs to have an inflexible load as it is used for sizing
        plants = self.config[f"{key}"]["share_dependent_on"]

        # Check if there are any plants otherwise just treat it as a normal plant
        if len(plants) == 0:
            self.add_general_info_config(key=key)
            return

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
            list_num = [min(1, list_num[idx] * self.df[f"inflexible_load/owner"][idx]) for idx, _ in enumerate(list_num)]
        else:
            raise Warning(f"Method '{method}' unknown.")

        # fill the columns
        self.df[f"{key}/owner"] = self._gen_dep_bool_list(list_bool=list_num,
                                                          share_ones=self.config[f"{key}"]["share"])
        self.df[f"{key}/num"] = self._gen_dep_num_list(owner_list=self.df[f"{key}/owner"],
                                                       distr=self.config[f"{key}"]["num"])
        self.df[f"{key}/has_submeter"] = str(self.config[f"{key}"]["has_submeter"])


class Mfh(Agents):
    """
        Sets up mfh agents. Inherits from Agents and Sfhs class.

        Mainly used for excel file creation. Afterwards Mfh class creates the individual agents.
    """

    def __init__(self, input_path: str, config: ordereddict, config_path: str, scenario_path: str):

        # Call the init method of the parent class
        super().__init__(config_path, input_path, scenario_path)

        # Define agent type
        self.type = 'mfh'

        # Path of the input file
        self.input_path = os.path.join(input_path, 'agents', self.type)

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

        # Number of buildings and apartments
        self.num = self.config["general"]["number_of"]  # number of buildings
        self.num_aps = 0  # number of apartments
        self.num_total = 0  # number of buildings and apartments

        # ID information
        self.ids = {}  # ids of the buildings and their apartments
        self.main_subid = "main"  # sub id of the building

        # Dataframe containing all information
        self.df = None

        # Misc
        self.n_digits = 2           # number of digits values get rounded to in respective value column

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
        self.fill_mpc()

        # Fill the market agent information in dataframe
        self.fill_market_agent()

        # Fill the metering information in dataframe
        self.fill_meter()

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
        self.fill_mpc()

        # Fill the market agent information in dataframe
        self.fill_market_agent()

        # Fill the metering information in dataframe
        self.fill_meter()

        return self.df

    def create_df_structure(self):
        """
            Function to create the dataframe structure with the respective columns
        """
        # Go through file and create the columns for the mfhs worksheet
        columns = ordereddict()
        for key, _ in self.config.items():
            cols = self.make_list_from_nested_dict(self.config[key], add_string=key)
            # Adjust the columns from "general"
            if key == "general":
                cols[0] = f"{key}/agent_id"
                cols[1] = f"{key}/sub_id"
                cols[-1] = f"{key}/market_participant"
                cols.insert(1, f"{key}/name")
                cols.insert(2, f"{key}/comment")
                cols.insert(3, f"{key}/bus")
            # Adjust the columns from "inflexible_load"
            elif key == "inflexible_load":
                cols[0] = f"{key}/owner"
                cols[6] = f"{key}/sizing/file"
                del cols[4]
                del cols[1]
                max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
                cols = cols[:3] + self.repeat_columns(columns=cols[3:5], num=max_num) + cols[5:]
            # Adjust the columns from "flexible_load"
            elif key == "flexible_load":
                cols[0] = f"{key}/owner"
                cols[6] = f"{key}/sizing/file"
                del cols[4]
                del cols[1]
                max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
                cols = cols[:3] + self.repeat_columns(columns=cols[3:6], num=max_num) + cols[6:]
            # Adjust the columns from "heat"
            elif key == "heat":
                cols[0] = f"{key}/owner"
                cols[4] = f"{key}/sizing/demand"
                del cols[1]
                max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
                cols = cols[:3] + self.repeat_columns(columns=cols[3:6], num=max_num) + cols[6:]
            # Adjust the columns from "dhw"
            elif key == "dhw":
                cols[0] = f"{key}/owner"
                del cols[4]
                del cols[1]
                max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
                cols = cols[:3] + self.repeat_columns(columns=cols[3:6], num=max_num) + cols[6:]
            # Adjust the columns from "pv"
            elif key == "pv":
                cols[0] = f"{key}/owner"
                del cols[6]
                del cols[4]
                del cols[1]
                max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
                cols = cols[:3] + self.repeat_columns(columns=cols[3:8], num=max_num) + cols[8:]
            # Adjust the columns from "wind"
            elif key == "wind":
                cols[0] = f"{key}/owner"
                del cols[6]
                del cols[4]
                del cols[1]
                max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
                cols = cols[:3] + self.repeat_columns(columns=cols[3:6], num=max_num) + cols[6:]
            # Adjust the columns from "fixed_gen"
            elif key == "fixed_gen":
                cols[0] = f"{key}/owner"
                del cols[6]
                del cols[4]
                del cols[1]
                max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
                cols = cols[:3] + self.repeat_columns(columns=cols[3:6], num=max_num) + cols[6:]
            # Adjust the columns from "hp"
            elif key == "hp":
                cols[0] = f"{key}/owner"
                del cols[4]
                del cols[1]
                max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
                cols = cols[:3] + self.repeat_columns(columns=cols[3:5], num=max_num) + cols[5:]
            # Adjust the columns from "ev"
            elif key == "ev":
                cols[0] = f"{key}/owner"
                del cols[4]
                del cols[1]
                max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
                cols = cols[:3] + self.repeat_columns(columns=cols[3:13], num=max_num) + cols[13:]
            # Adjust the columns from "battery"
            elif key == "battery":
                cols[0] = f"{key}/owner"
                del cols[5]
                del cols[2]
                del cols[1]
                max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
                cols = cols[:3] + self.repeat_columns(columns=cols[3:9], num=max_num) + cols[9:]
            # Adjust the columns from "heat_storage"
            elif key == "heat_storage":
                cols[0] = f"{key}/owner"
                del cols[5]
                del cols[2]
                del cols[1]
                max_num = max(max(self.config[key]["num"]), 1)  # ensure at least 1 entrance
                cols = cols[:3] + self.repeat_columns(columns=cols[3:7], num=max_num) + cols[7:]
            # Adjust the columns from "mpc"
            elif key == "mpc":
                pass
            # Adjust the columns from "market_agent"
            elif key == "market_agent":
                pass
            # Adjust the columns from "meter"
            elif key == "meter":
                pass
            else:
                raise NotImplementedError(
                    f"The configuration file contains a key word ('{key}') that has not been configured in "
                    "the Sfhs class yet. Aborting scenario creation...")
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

    def fill_general(self) -> pd.DataFrame:
        """
            Fills all general columns
        """
        # Key in the config file
        key = "general"

        # Get the config for the key
        config = self.config[f"{key}"]

        # Fill the general information based on the method to create the scenario
        # In method 'config' first the building agents are created and then the according apartments
        if self.method == 'config':
            self.df = self._general_config(key=key, config=config)
        # In method 'grid' the apartments are first created from the load sheet and afterwards the according buildings
        elif self.method == 'grid':
            self.df = self._general_grid(key=key, config=config)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

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
        self.df = self.df.groupby(f"{key}/bus", sort=False).apply(self.sort_df, col=f"{key}/sub_id", sort_value=self.main_subid)
        self.df.set_index(self.df['index'], inplace=True, drop=True)
        self.df.index.name = None

        # Calculate total building area and number of occupants
        for agent_id in self.df[f"{key}/agent_id"].unique():
            self._calc_total(key=key, vals=["parameters/area"], agent_id=agent_id)
            self._calc_total(key=key, vals=["parameters/occupants"], agent_id=agent_id)

        return self.df

    def fill_inflexible_load(self, **kwargs):
        """
            Fills all inflexible_load columns
        """

        # Key in the config file
        key = "inflexible_load"

        # Get the config for the key
        config = self.config[f"{key}"]

        if self.method == 'config':
            self.df = self._inflexible_load_config(key=key, config=config)
        elif self.method == 'grid':
            self.df = self._inflexible_load_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _inflexible_load_config(self, key: str, config: dict) -> pd.DataFrame:

        # general
        self._add_general_info_config(key=key)

        # loop through buildings
        max_num = max(config["num"])
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
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

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
        self.df[f"{key}/num"] = self.df[f"{key}/owner"].fillna(0).astype(int)  # equals owner as only one inflexible load per agent
        # note: always taken from config
        self.df[f"{key}/has_submeter"] = self.config[f"{key}"]["has_submeter"]

        # sizing
        for num in range(max(self.df[f"{key}/num"])):  # currently only one device per agent is supported
            # Get demand from load sheet
            self.df[f"{key}/sizing/demand_{num}"] = np.floor(pd.to_numeric(df['demand'] * 1e6, errors='coerce')).\
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
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

        return self.df

    def fill_flexible_load(self):
        """
            Fills all flexible_load columns
        """

        key = "flexible_load"
        config = self.config[f"{key}"]
        max_num = max(config["num"])

        # general
        self._add_general_info_config(key=key)

        # loop through buildings
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
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

    def fill_heat(self, **kwargs) -> pd.DataFrame:
        """
            Fills all heat columns
        """

        # Key in the config file
        key = "heat"

        # Get the config for the key
        config = self.config[f"{key}"]

        if self.method == 'config':
            self.df = self._heat_config(key=key, config=config)
        elif self.method == 'grid':
            self.df = self._heat_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _heat_config(self, key: str, config: dict) -> pd.DataFrame:
        """adds the heat from the config file"""

        # general
        self._add_general_info_config(key=key)

        # loop through buildings
        max_num = max(config["num"])
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
                df_sub.loc[df_sub[f"{key}/sizing/file_{num}"] == 'linked', f"{key}/sizing/file_{num}"] = df_sub.loc[df_sub[f"{key}/sizing/file_{num}"] == 'linked', f"inflexible_load/sizing/file_0"]
                # shorten the file name to the yearly demand and index
                df_sub.loc[:, f"{key}/sizing/file_{num}"] = ["_".join(item.rsplit(".", 1)[0].split("_")[1:3]) if 'csv' in item else item for item in df_sub[f"{key}/sizing/file_{num}"]]
                df_sub.loc[:, f"{key}/sizing/file_{num}"] = self._pick_files(
                    list_type=df_sub[f"{key}/sizing/file_{num}"], device=f"{key}",
                    input_path=os.path.join(self.input_path, key))

                # calculate total demand of building
                self._calc_total(key=key, vals=[f"sizing/demand_{num}"], agent_id=agent, df=df_sub)

                # fill main df with sub df
                self.df.loc[df_sub.index, :] = df_sub[:]

        # forecast
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

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
        self.df[f"{key}/owner"] = (self.df[f"{key}/num"] > 0).fillna(0).astype('Int64')  # all agents that have plant type
        # note: always taken from config
        self.df[f"{key}/has_submeter"] = self.config[f"{key}"]["has_submeter"]

        # sizing
        for num in range(max(self.df[f"{key}/num"])):  # currently only one device per agent is supported
            # Get demand from load sheet
            self.df[f"{key}/sizing/demand_{num}"] = self.df.index.map(df['demand'] * 1e6).astype('Int64')
            self.df[f"{key}/sizing/file_{num}"] = self.df.index.map(df['heat_file'])
            self.df[f"{key}/sizing/temperature_{num}"] = self.df.index.map(df['temperature'])

        # forecast
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

        return self.df

    def fill_dhw(self, **kwargs) -> pd.DataFrame:
        """
            Fills all dhw columns
        """

        # Key in the config file
        key = "dhw"

        # Get the config for the key
        config = self.config[f"{key}"]

        if self.method == 'config':
            self.df = self._dhw_config(key=key, config=config)
        elif self.method == 'grid':
            self.df = self._dhw_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _dhw_config(self, key: str, config: dict) -> pd.DataFrame:
        """adds the dhw from the config file"""

        # general
        self._add_general_info_config(key=key)

        # loop through buildings
        max_num = max(config["num"])
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
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

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
        self.df[f"{key}/owner"] = (self.df[f"{key}/num"] > 0).fillna(0).astype('Int64')  # all agents that have plant type
        # note: always taken from config
        self.df[f"{key}/has_submeter"] = self.config[f"{key}"]["has_submeter"]

        # sizing
        for num in range(max(self.df[f"{key}/num"])):  # currently only one device per agent is supported
            # Get demand from load sheet
            self.df[f"{key}/sizing/demand_{num}"] = self.df.index.map(df['demand'] * 1e6).astype('Int64')
            self.df[f"{key}/sizing/file_{num}"] = self.df.index.map(df['heat_file'])
            self.df[f"{key}/sizing/temperature_{num}"] = self.df.index.map(df['temperature'])

        # forecast
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

        return self.df

    def fill_pv(self, **kwargs):
        """
            Fills all pv columns
        """

        # Key in the config file
        key = "pv"

        # Get the config for the key
        config = self.config[f"{key}"]

        if self.method == 'config':
            self.df = self._pv_config(key=key, config=config)
        elif self.method == 'grid':
            self.df = self._pv_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _pv_config(self, key: str, config: dict) -> pd.DataFrame:

        # general
        self._add_general_info_config(key=key)

        # loop through buildings
        max_num = max(config["num"])
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
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

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
        # note: always taken from config
        self.df[f"{key}/has_submeter"] = self.config[f"{key}"]["has_submeter"]

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
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

        # quality
        self.df[f"{key}/quality"] = config["quality"]

        return self.df

    def fill_wind(self, **kwargs) -> pd.DataFrame:
        """
            Fills all wind columns
        """
        # Key in the config file
        key = "wind"

        # Get the config for the key
        config = self.config[f"{key}"]

        if self.method == 'config':
            self.df = self._wind_config(key=key, config=config)
        elif self.method == 'grid':
            self.df = self._wind_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _wind_config(self, key: str, config: dict) -> pd.DataFrame:

        # general
        self._add_general_info_config(key=key)

        # loop through buildings
        max_num = max(config["num"])
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
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

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
        # note: always taken from config
        self.df[f"{key}/has_submeter"] = self.config[f"{key}"]["has_submeter"]

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
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

        # quality
        self.df[f"{key}/quality"] = config["quality"]

        return self.df

    def fill_fixed_gen(self, **kwargs) -> pd.DataFrame:
        """
            Fills all fixed_gen columns
        """

        # Key in the config file
        key = "fixed_gen"

        # Get the config for the key
        config = self.config[f"{key}"]

        if self.method == 'config':
            self.df = self._fixed_gen_config(key=key, config=config)
        elif self.method == 'grid':
            self.df = self._fixed_gen_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _fixed_gen_config(self, key: str, config: dict) -> pd.DataFrame:
        """
            Fills all fixed_gen columns based on the config file
        """

        # general
        self._add_general_info_config(key=key)

        # loop through buildings
        max_num = max(config["num"])
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
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

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
        # note: always taken from config
        self.df[f"{key}/has_submeter"] = self.config[f"{key}"]["has_submeter"]

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
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

        # quality
        self.df[f"{key}/quality"] = config["quality"]

        return self.df

    def fill_hp(self, **kwargs) -> pd.DataFrame:
        """
            Fills all hp columns
        """
        # Key in the config file
        key = "hp"

        # Get the config for the key
        config = self.config[f"{key}"]

        if self.method == 'config':
            self._hp_config(key=key, config=config)
        elif self.method == 'grid':
            self._hp_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _hp_config(self, key: str, config: dict) -> pd.DataFrame:

        # general
        self._add_general_info_config(key=key)

        # loop through buildings
        max_num = max(config["num"])
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
                # TODO: Fix rounding issue. Somehow the only one that is not working
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
        self.df[f"{key}/owner"] = (self.df[f"{key}/num"] > 0).fillna(0).astype('Int64')  # all agents that have plant type
        # note: always taken from config
        self.df[f"{key}/has_submeter"] = self.config[f"{key}"]["has_submeter"]

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

    def fill_ev(self, **kwargs) -> pd.DataFrame:
        """
            Fills all ev columns
        """

        # Key in the config file
        key = "ev"

        # Get the config for the key
        config = self.config[f"{key}"]

        if self.method == 'config':
            self._ev_config(key=key, config=config)
        elif self.method == 'grid':
            self._ev_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _ev_config(self, key: str, config: dict) -> pd.DataFrame:

        # general
        self._add_general_info_config(key=key)

        # loop through buildings
        max_num = max(config["num"])
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
            # postprocessing
            # method
            df_sub.loc[:, f"{key}/charging_scheme/method"] = random.choices(config["charging_scheme"]["method"],
                                                                            k=len(df_sub))

            # copy results to main df
            self.df.loc[df_sub.index, :] = df_sub[:]

        # forecast
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

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
        self.df[f"{key}/owner"] = (self.df[f"{key}/num"] > 0).fillna(0).astype('Int64')  # all agents that have plant type
        # note: always taken from config
        self.df[f"{key}/has_submeter"] = self.config[f"{key}"]["has_submeter"]

        # sizing
        for num in range(max(self.df[f"{key}/num"])):  # currently only one device per agent is supported
            self.df[f"{key}/sizing/file_{num}"] = self.df.index.map(df['file_add'])
            self.df[f"{key}/sizing/capacity_{num}"] = (self.df.index.map(df['capacity']) * 1e6).astype('Int64')
            self.df[f"{key}/sizing/consumption_{num}"] = (self.df.index.map(df['consumption']) * 1e6).astype('Int64')
            self.df[f"{key}/sizing/charging_home_{num}"] = (self.df.index.map(df['charging_home']) * 1e6).astype('Int64')
            self.df[f"{key}/sizing/charging_AC_{num}"] = (self.df.index.map(df['charging_ac']) * 1e6).astype('Int64')
            self.df[f"{key}/sizing/charging_DC_{num}"] = (self.df.index.map(df['charging_dc']) * 1e6).astype('Int64')
            self.df[f"{key}/sizing/charging_efficiency_{num}"] = self.df.index.map(df['efficiency'])
            self.df[f"{key}/sizing/soc_{num}"] = self.df.index.map(df['soc'])
            self.df[f"{key}/sizing/v2g_{num}"] = self.df.index.map(df['v2g'])
            self.df[f"{key}/sizing/v2h_{num}"] = self.df.index.map(df['v2h'])

        # quality
        self.df[f"{key}/quality"] = config["quality"]

        return self.df

    def fill_battery(self, **kwargs) -> pd.DataFrame:
        """
            Fills all battery columns
        """

        # Key in the config file
        key = "battery"

        # Get the config for the key
        config = self.config[f"{key}"]

        if self.method == 'config':
            self.df = self._battery_config(key=key, config=config)
        elif self.method == 'grid':
            self.df = self._battery_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _battery_config(self, key: str, config: dict) -> pd.DataFrame:

        # general
        self._add_general_info_config(key=key)

        # sizing
        max_num = max(config["num"])
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
        # note: always taken from config
        self.df[f"{key}/has_submeter"] = self.config[f"{key}"]["has_submeter"]

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

    def fill_heat_storage(self, **kwargs) -> pd.DataFrame:
        """
            Fills all heat storage columns
        """

        # Key in the config file
        key = "heat_storage"

        # Get the config for the key
        config = self.config[f"{key}"]

        if self.method == 'config':
            self.df = self._heat_storage_config(key=key, config=config)
        elif self.method == 'grid':
            self.df = self._heat_storage_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _heat_storage_config(self, key: str, config: dict) -> pd.DataFrame:

        # general
        self._add_general_info_config(key=key)

        # sizing
        max_num = max(config["num"])
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
        # note: always taken from config
        self.df[f"{key}/has_submeter"] = self.config[f"{key}"]["has_submeter"]

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

    def fill_mpc(self):
        """
            Fills all battery columns
        """
        key = "mpc"
        config = self.config[f"{key}"]

        # general
        self._add_info_simple(keys=[key], config=config)

    def fill_market_agent(self):
        """
            Fills all battery columns
        """
        key = "market_agent"
        config = self.config[f"{key}"]

        # general
        self._add_info_random(keys=[key], config=config)

    def fill_meter(self):
        """
            Fills all battery columns
        """
        key = "meter"

        # general
        self.df.loc[:, f"{key}/prob_late"] = self.config[f"{key}"]["prob_late"]
        self.df.loc[:, f"{key}/prob_late_95"] = self.config[f"{key}"]["prob_late_95"]
        self.df.loc[:, f"{key}/prob_missing"] = self.config[f"{key}"]["prob_missing"]

    def _get_df_sub(self, agent: str, aps: list, aps_independent: bool = True) -> pd.DataFrame:
        """ Returns the sub-dataframe with either the building or the apartments of the building"""
        if aps_independent:
            # df_sub are the apartments
            return self.df.loc[self.df[f"general/sub_id"].isin(aps)][:]
        else:
            # df_sub is the building
            return self.df.loc[(self.df[f"general/agent_id"] == agent)
                               & (self.df[f"general/sub_id"] == self.main_subid)][:]

    def _add_general_info_config(self, key: str) -> None:

        # fields that exist for all plants
        self.df.loc[self.df[f"general/sub_id"] == self.main_subid, f"{key}/owner"] = \
            self._gen_rand_bool_list(n=self.num, share_ones=self.config[f"{key}"]["share"])
        self.df.loc[:, f"{key}/has_submeter"] = self.config[f"{key}"]["has_submeter"]

    def _preprocess_df_sub(self, agent: str, aps: list, key:str, aps_independent: bool = None):
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


class Ctsp(Agents):
    """
        Sets up cts agents. Inherits from Agents class.

        Mainly used for excel file creation. Afterwards Sfh class creates the individual agents.
    """

    def __init__(self, input_path: str, config: ordereddict, config_path: str, scenario_path: str):

        # Call the init method of the parent class
        super().__init__(config_path, input_path, scenario_path)

        # Define agent type
        self.type = 'ctsp'

        # Path of the input file
        self.input_path = os.path.join(input_path, 'agents', self.type)

        # Config file
        self.config = config

        # Grid information (if applicable)
        self.grid = None
        self.bus = None  # bus sheet containing only the bus information of the agent type
        self.load = None  # load sheet containing only the load information of the agent type
        self.agents = None # load sheet but limited to all agents, i.e. all inflexible_loads
        self.sgen = None  # sgen sheet containing only the sgen information of the agent type

        # Creation method
        self.method = None

        # Number of agents
        self.num = 0

        # Dataframe containing all information
        self.df = None

        # Index list that is adhered to throughout the creation process to ensure correct order
        self.idx_list = None        # gets created in create_general()

        # Misc
        self.n_digits = 3           # number of digits values get rounded to in respective value column

    def create_df_from_config(self) -> pd.DataFrame:
        """
            Function to create the dataframe that makes the Excel sheet
        """

        # Get the number of agents and set the method
        self.num = self.config["general"]["number_of"]
        self.method = 'config'

        # Create the overall dataframe structure for the worksheet
        self.create_df_structure()

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
        self.fill_mpc()

        # Fill the market agent information in dataframe
        self.fill_market_agent()

        # Fill the metering information in dataframe
        self.fill_meter()

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

        # Get the number of agents and set the method
        self.num = self.get_num_from_grid(self.grid['load'], self.type)
        self.method = 'grid'

        # Create the overall dataframe structure for the worksheet
        self.create_df_structure()

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
        self.fill_mpc()

        # Fill the market agent information in dataframe
        self.fill_market_agent()

        # Fill the metering information in dataframe
        self.fill_meter()

        return self.df

    def create_df_structure(self):
        """
            Function to create the dataframe structure with the respective columns
        """
        # Go through file and create the columns for the ctss worksheet
        columns = ordereddict()
        for key, _ in self.config.items():
            cols = self.make_list_from_nested_dict(self.config[key], add_string=key)
            # Adjust the columns from "general"
            if key == "general":
                cols[0] = f"{key}/agent_id"
                cols[-1] = f"{key}/market_participant"
                del cols[4]
                del cols[1]
                cols.insert(1, f"{key}/name")
                cols.insert(2, f"{key}/comment")
                cols.insert(3, f"{key}/bus")
            # Adjust the columns from "inflexible_load"
            elif key == "inflexible_load":
                cols[0] = f"{key}/owner"
                cols[4] = f"{key}/sizing/file"
                max_num = max(self.config[key]["num"])
                cols = cols[:3] + self.repeat_columns(columns=cols[3:5], num=max_num) + cols[5:]
            # Adjust the columns from "flexible_load"
            elif key == "flexible_load":
                cols[0] = f"{key}/owner"
                cols[4] = f"{key}/sizing/file"
                max_num = max(self.config[key]["num"])
                cols = cols[:3] + self.repeat_columns(columns=cols[3:6], num=max_num) + cols[6:]
            # Adjust the columns from "pv"
            elif key == "pv":
                cols[0] = f"{key}/owner"
                del cols[4]
                max_num = max(self.config[key]["num"])
                cols = cols[:3] + self.repeat_columns(columns=cols[3:8], num=max_num) + cols[8:]
            # Adjust the columns from "wind"
            elif key == "wind":
                cols[0] = f"{key}/owner"
                del cols[4]
                max_num = max(self.config[key]["num"])
                cols = cols[:3] + self.repeat_columns(columns=cols[3:6], num=max_num) + cols[6:]
            # Adjust the columns from "fixed_gen"
            elif key == "fixed_gen":
                cols[0] = f"{key}/owner"
                del cols[4]
                max_num = max(self.config[key]["num"])
                cols = cols[:3] + self.repeat_columns(columns=cols[3:6], num=max_num) + cols[6:]
            # Adjust the columns from "ev"
            elif key == "ev":
                cols[0] = f"{key}/owner"
                max_num = max(self.config[key]["num"])
                cols = cols[:3] + self.repeat_columns(columns=cols[3:13], num=max_num) + cols[13:]
            # Adjust the columns from "battery"
            elif key == "battery":
                cols[0] = f"{key}/owner"
                del cols[1]
                max_num = max(self.config[key]["num"])
                cols = cols[:3] + self.repeat_columns(columns=cols[3:-1], num=max_num) + [cols[-1]]
            # All columns that do not need to be adjusted
            elif key in ["mpc", "market_agent", "meter"]:
                pass
            else:
                raise NotImplementedError(
                    f"The configuration file contains a key word ('{key}') that has not been configured in "
                    "the Sfhs class yet. Aborting scenario creation...")

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
        key = "general"
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

        # forecast
        self.df[f"{key}/fcast_retraining_frequency"] = config["fcast_retraining_frequency"]

        # market participation
        self.df[f"{key}/market_participant"] = self._gen_rand_bool_list(n=self.num,
                                                                        share_ones=config["market_participant_share"])

        # If the method is grid, fill the name, comment, bus and type columns from grid file
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
        self.df[f"{key}/name"] = self.agents["name"]
        self.df[f"{key}/bus"] = self.agents["bus"]
        self.df[f"{key}/parameters/type"] = self.agents['sub_type']

        return self.df

    def fill_inflexible_load(self, **kwargs) -> pd.DataFrame:
        """
            Fills all inflexible_load columns
        """

        # Key in the config file
        key = "inflexible_load"

        # Get the config for the key
        config = self.config[f"{key}"]

        if self.method == 'config':
            self.df = self._inflexible_load_config(key=key, config=config)
        elif self.method == 'grid':
            self.df = self._inflexible_load_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _inflexible_load_config(self, key: str, config: dict) -> pd.DataFrame:
        """
            Fills all inflexible_load columns
        """

        # general
        self._add_general_info(key=key)

        # sizing
        max_num = max(config["num"])
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
            # file
            self.df[f"{key}/sizing/file_{num}"] = self._pick_files(list_type=self.df["general/parameters/type"],
                                                                   device=key,
                                                                   input_path=os.path.join(self.input_path, key),
                                                                   idx_type=0)

        # forecast
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

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
        # note: always taken from config
        self.df[f"{key}/has_submeter"] = self.config[f"{key}"]["has_submeter"]

        # sizing
        for num in range(max(self.df[f"{key}/num"])): # currently only one device per agent is supported
            # Get demand from load sheet
            self.df[f"{key}/sizing/demand_{num}"] = (df['demand'] * 1e6).astype('Int64')
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
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

        return self.df

    def fill_flexible_load(self):
        """
            Fills all flexible_load columns
        """
        key = "flexible_load"
        config = self.config[f"{key}"]

        # general
        self._add_general_info(key=key)

        # sizing
        max_num = max(config["num"])
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
            # file
            self.df[f"{key}/sizing/file_{num}"] = self._pick_files(list_type=self.df["general/parameters/type"],
                                                                   device=key,
                                                                   input_path=os.path.join(self.input_path, key),
                                                                   idx_type=0)

        # forecast
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

    def fill_pv(self, **kwargs) -> pd.DataFrame:
        """
            Fills all pv columns
        """

        # Key in the config file
        key = "pv"

        # Get the config for the key
        config = self.config[f"{key}"]

        if self.method == 'config':
            self.df = self._pv_config(key=key, config=config)
        elif self.method == 'grid':
            self.df = self._pv_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _pv_config(self, key: str, config: dict) -> pd.DataFrame:
        """
            Fills all pv columns
        """

        # general
        self._add_general_info(key=key)

        # sizing
        max_num = max(config["num"])
        for num in range(max_num):
            # index list indicating ownership of device
            idx_list = self._get_idx_list(key=key, num=num)

            # add indexed info
            self._add_info_indexed(keys=[key, "sizing"], config=config["sizing"],
                                   idx_list=idx_list, appendix=f"_{num}")
            # postprocessing
            # power
            self.df[f"{key}/sizing/power_{num}"] *= self.df["inflexible_load/sizing/demand_0"]
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
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

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
        # note: always taken from config
        self.df[f"{key}/has_submeter"] = self.config[f"{key}"]["has_submeter"]

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
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

        # quality
        self.df[f"{key}/quality"] = config["quality"]

        return self.df

    def fill_wind(self, **kwargs) -> pd.DataFrame:
        """
            Fills all wind columns
        """
        # Key in the config file
        key = "wind"

        # Get the config for the key
        config = self.config[f"{key}"]

        if self.method == 'config':
            self.df = self._wind_config(key=key, config=config)
        elif self.method == 'grid':
            self.df = self._wind_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _wind_config(self, key: str, config: dict) -> pd.DataFrame:
        """
            Fills all wind columns
        """

        # general
        self._add_general_info(key=key)

        # sizing
        max_num = max(config["num"])
        for num in range(max_num):
            # index list indicating ownership of device
            idx_list = self._get_idx_list(key=key, num=num)

            # add indexed info
            self._add_info_indexed(keys=[key, "sizing"], config=config["sizing"],
                                   idx_list=idx_list, appendix=f"_{num}")
            # postprocessing
            # power
            self.df[f"{key}/sizing/power_{num}"] *= self.df["inflexible_load/sizing/demand_0"]
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
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

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
        # note: always taken from config
        self.df[f"{key}/has_submeter"] = self.config[f"{key}"]["has_submeter"]

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
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

        # quality
        self.df[f"{key}/quality"] = config["quality"]

        return self.df

    def fill_fixed_gen(self, **kwargs) -> pd.DataFrame:
        """
            Fills all fixed_gen columns
        """

        # Key in the config file
        key = "fixed_gen"

        # Get the config for the key
        config = self.config[f"{key}"]

        if self.method == 'config':
            self.df = self._fixed_gen_config(key=key, config=config)
        elif self.method == 'grid':
            self.df = self._fixed_gen_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _fixed_gen_config(self, key: str, config: dict) -> pd.DataFrame:
        """
            Fills all fixed_gen columns
        """

        # general
        self._add_general_info(key=key)

        # sizing
        max_num = max(config["num"])
        for num in range(max_num):
            # index list indicating ownership of device
            idx_list = self._get_idx_list(key=key, num=num)

            # add indexed info
            self._add_info_indexed(keys=[key, "sizing"], config=config["sizing"],
                                   idx_list=idx_list, appendix=f"_{num}")
            # postprocessing
            # power
            self.df[f"{key}/sizing/power_{num}"] *= self.df["inflexible_load/sizing/demand_0"]
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
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

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
        # note: always taken from config
        self.df[f"{key}/has_submeter"] = self.config[f"{key}"]["has_submeter"]

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
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

        # quality
        self.df[f"{key}/quality"] = config["quality"]

        return self.df

    def fill_ev(self, **kwargs) -> pd.DataFrame:
        """
            Fills all ev columns
        """

        # Key in the config file
        key = "ev"

        # Get the config for the key
        config = self.config[f"{key}"]

        if self.method == 'config':
            self._ev_config(key=key, config=config)
        elif self.method == 'grid':
            self._ev_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _ev_config(self, key: str, config: dict) -> pd.DataFrame:
        """
            Fills all ev columns
        """

        # general
        self._add_general_info(key=key)

        # sizing
        max_num = max(config["num"])
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

            # things only necessary to add once
            if num == 0:
                # charging scheme
                self._add_info_indexed(keys=[key, "charging_scheme"], config=config["charging_scheme"],
                                       idx_list=idx_list)

        # forecast
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

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
        self.df[f"{key}/owner"] = (self.df[f"{key}/num"] > 0).fillna(0).astype('Int64')  # all agents that have plant type
        # note: always taken from config
        self.df[f"{key}/has_submeter"] = self.config[f"{key}"]["has_submeter"]

        # sizing
        for num in range(max(self.df[f"{key}/num"])):  # currently only one device per agent is supported
            self.df[f"{key}/sizing/file_{num}"] = self.df.index.map(df['file_add'])
            self.df[f"{key}/sizing/capacity_{num}"] = (self.df.index.map(df['capacity']) * 1e6).astype('Int64')
            self.df[f"{key}/sizing/consumption_{num}"] = (self.df.index.map(df['consumption']) * 1e6).astype('Int64')
            self.df[f"{key}/sizing/charging_home_{num}"] = (self.df.index.map(df['charging_home']) * 1e6).astype('Int64')
            self.df[f"{key}/sizing/charging_AC_{num}"] = (self.df.index.map(df['charging_ac']) * 1e6).astype('Int64')
            self.df[f"{key}/sizing/charging_DC_{num}"] = (self.df.index.map(df['charging_dc']) * 1e6).astype('Int64')
            self.df[f"{key}/sizing/charging_efficiency_{num}"] = self.df.index.map(df['efficiency'])
            self.df[f"{key}/sizing/soc_{num}"] = self.df.index.map(df['soc'])
            self.df[f"{key}/sizing/v2g_{num}"] = self.df.index.map(df['v2g'])
            self.df[f"{key}/sizing/v2h_{num}"] = self.df.index.map(df['v2h'])

        # quality
        self.df[f"{key}/quality"] = config["quality"]

        return self.df

    def fill_battery(self, **kwargs) -> pd.DataFrame:
        """
            Fills all battery columns
        """

        # Key in the config file
        key = "battery"

        # Get the config for the key
        config = self.config[f"{key}"]

        if self.method == 'config':
            self.df = self._battery_config(key=key, config=config)
        elif self.method == 'grid':
            self.df = self._battery_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _battery_config(self, key: str, config: dict) -> pd.DataFrame:
        """
            Fills all battery columns
        """

        # general
        self._add_general_info_bat(key=key)

        # sizing
        max_num = max(config["num"])
        for num in range(max_num):
            # index list indicating ownership of device
            idx_list = self._get_idx_list(key=key, num=num)

            # add indexed info
            self._add_info_indexed(keys=[key, "sizing"], config=config["sizing"], idx_list=idx_list,
                                   appendix=f"_{num}")
            # postprocessing
            # power
            self.df[f"{key}/sizing/power_{num}"] *= self.df["inflexible_load/sizing/demand_0"] / 1e3
            self.df[f"{key}/sizing/power_{num}"] = self._round_to_nth_digit(
                vals=self.df[f"{key}/sizing/power_{num}"], n=2)
            # capacity
            self.df[f"{key}/sizing/capacity_{num}"] *= self.df["inflexible_load/sizing/demand_0"] / 1e3
            self.df[f"{key}/sizing/capacity_{num}"] = self._round_to_nth_digit(
                vals=self.df[f"{key}/sizing/capacity_{num}"], n=2)

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
        # note: always taken from config
        self.df[f"{key}/has_submeter"] = self.config[f"{key}"]["has_submeter"]

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

    def fill_mpc(self):
        """
            Fills all battery columns
        """
        key = "mpc"
        config = self.config[f"{key}"]

        # general
        self._add_info_simple(keys=[key], config=config)

    def fill_market_agent(self):
        """
            Fills all battery columns
        """
        key = "market_agent"
        config = self.config[f"{key}"]

        # general
        self._add_info_random(keys=[key], config=config)

    def fill_meter(self):
        """
            Fills all battery columns
        """
        key = "meter"
        config = self.config[f"{key}"]

        # general
        self._add_info_simple(keys=[key], config=config)

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
        self.df[f"{key}/has_submeter"] = self.config[f"{key}"]["has_submeter"]

    def _add_general_info_bat(self, key: str):

        # find all potential owners of a battery system dependent on type
        # note: this setup considers the different dependencies for each type and loops through each separately
        agent_types = self.config["general"]["parameters"]["type"]
        list_owner = [0] * self.num
        list_num = [0] * self.num
        for idx, agent_type in enumerate(agent_types):
            # get all agents of given type
            list_type = list(self.df["general/parameters/type"] == agent_type)
            plants = self.config[f"{key}"]["share_dependent_on"][idx] + ["inflexible_load"]
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
            self.df[f"{key}/has_submeter"] = str(self.config[f"{key}"]["has_submeter"])


class Industry(Agents):
    """
        Sets up industry agents. Inherits from Agents class.

        Mainly used for excel file creation. Afterwards Sfh class creates the individual agents.
    """

    def __init__(self, input_path: str, config: ordereddict, config_path: str, scenario_path: str):

        # Call the init method of the parent class
        super().__init__(config_path, input_path, scenario_path)

        # Define agent type
        self.type = 'industry'

        # Path of the input file
        self.input_path = os.path.join(input_path, 'agents', self.type)

        # Config file
        self.config = config

        # Grid information (if applicable)
        self.grid = None
        self.bus = None  # bus sheet containing only the bus information of the agent type
        self.load = None  # load sheet containing only the load information of the agent type
        self.agents = None # load sheet but limited to all agents, i.e. all inflexible_loads
        self.sgen = None  # sgen sheet containing only the sgen information of the agent type

        # Creation method
        self.method = None

        # Number of agents
        self.num = 0

        # Dataframe containing all information
        self.df = None

        # Index list that is adhered to throughout the creation process to ensure correct order
        self.idx_list = None        # gets created in create_general()

        # Misc
        self.n_digits = 4           # number of digits values get rounded to in respective value column

    def create_df_from_config(self) -> pd.DataFrame:
        """
            Function to create the dataframe that makes the Excel sheet
        """

        # Get the number of agents and set the method
        self.num = self.config["general"]["number_of"]
        self.method = 'config'

        # Create the overall dataframe structure for the worksheet
        self.create_df_structure()

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
        self.fill_mpc()

        # Fill the market agent information in dataframe
        self.fill_market_agent()

        # Fill the metering information in dataframe
        self.fill_meter()

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

        # Get the number of agents and set the method
        self.num = self.get_num_from_grid(self.grid['load'], self.type)
        self.method = 'grid'

        # Create the overall dataframe structure for the worksheet
        self.create_df_structure()

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
        self.fill_mpc()

        # Fill the market agent information in dataframe
        self.fill_market_agent()

        # Fill the metering information in dataframe
        self.fill_meter()

        return self.df

    def create_df_structure(self):
        """
            Function to create the dataframe structure with the respective columns
        """
        # Go through file and create the columns for the ctss worksheet
        columns = ordereddict()
        for key, _ in self.config.items():
            cols = self.make_list_from_nested_dict(self.config[key], add_string=key)
            # Adjust the columns from "general"
            if key == "general":
                cols[0] = f"{key}/agent_id"
                cols[-1] = f"{key}/market_participant"
                del cols[4]
                del cols[1]
                cols.insert(1, f"{key}/name")
                cols.insert(2, f"{key}/comment")
                cols.insert(3, f"{key}/bus")
            # Adjust the columns from "inflexible_load"
            elif key == "inflexible_load":
                cols[0] = f"{key}/owner"
                cols[4] = f"{key}/sizing/file"
                max_num = max(self.config[key]["num"])
                cols = cols[:3] + self.repeat_columns(columns=cols[3:5], num=max_num) + cols[5:]
            # Adjust the columns from "flexible_load"
            elif key == "flexible_load":
                cols[0] = f"{key}/owner"
                cols[4] = f"{key}/sizing/file"
                max_num = max(self.config[key]["num"])
                cols = cols[:3] + self.repeat_columns(columns=cols[3:6], num=max_num) + cols[6:]
            # Adjust the columns from "pv"
            elif key == "pv":
                cols[0] = f"{key}/owner"
                del cols[4]
                max_num = max(self.config[key]["num"])
                cols = cols[:3] + self.repeat_columns(columns=cols[3:8], num=max_num) + cols[8:]
            # Adjust the columns from "wind"
            elif key == "wind":
                cols[0] = f"{key}/owner"
                del cols[4]
                max_num = max(self.config[key]["num"])
                cols = cols[:3] + self.repeat_columns(columns=cols[3:6], num=max_num) + cols[6:]
            # Adjust the columns from "fixed_gen"
            elif key == "fixed_gen":
                cols[0] = f"{key}/owner"
                del cols[4]
                max_num = max(self.config[key]["num"])
                cols = cols[:3] + self.repeat_columns(columns=cols[3:6], num=max_num) + cols[6:]
            # Adjust the columns from "ev"
            elif key == "ev":
                cols[0] = f"{key}/owner"
                max_num = max(self.config[key]["num"])
                cols = cols[:3] + self.repeat_columns(columns=cols[3:13], num=max_num) + cols[13:]
            # Adjust the columns from "battery"
            elif key == "battery":
                cols[0] = f"{key}/owner"
                del cols[1]
                max_num = max(self.config[key]["num"])
                cols = cols[:3] + self.repeat_columns(columns=cols[3:-1], num=max_num) + [cols[-1]]
            # All columns that do not need to be adjusted
            elif key in ["mpc", "market_agent", "meter"]:
                pass
            else:
                raise NotImplementedError(
                    f"The configuration file contains a key word ('{key}') that has not been configured in "
                    "the Sfhs class yet. Aborting scenario creation...")

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
        key = "general"
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

        # forecast
        self.df[f"{key}/fcast_retraining_frequency"] = config["fcast_retraining_frequency"]

        # market participation
        self.df[f"{key}/market_participant"] = self._gen_rand_bool_list(n=self.num,
                                                                        share_ones=config["market_participant_share"])

        # If the method is grid, fill the name, comment, bus and type columns from grid file
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
        self.df[f"{key}/name"] = self.agents["name"]
        self.df[f"{key}/bus"] = self.agents["bus"]
        self.df[f"{key}/parameters/type"] = self.agents['sub_type']

        return self.df

    def fill_inflexible_load(self, **kwargs) -> pd.DataFrame:
        """
            Fills all inflexible_load columns
        """

        # Key in the config file
        key = "inflexible_load"

        # Get the config for the key
        config = self.config[f"{key}"]

        if self.method == 'config':
            self.df = self._inflexible_load_config(key=key, config=config)
        elif self.method == 'grid':
            self.df = self._inflexible_load_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _inflexible_load_config(self, key: str, config: dict) -> pd.DataFrame:
        """
            Fills all inflexible_load columns
        """

        # general
        self._add_general_info(key=key)

        # sizing
        max_num = max(config["num"])
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
            # file
            self.df[f"{key}/sizing/file_{num}"] = self._pick_files(list_type=self.df["general/parameters/type"],
                                                                   device=key,
                                                                   input_path=os.path.join(self.input_path, key),
                                                                   idx_type=0)

        # forecast
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

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
        # note: always taken from config
        self.df[f"{key}/has_submeter"] = self.config[f"{key}"]["has_submeter"]

        # sizing
        for num in range(max(self.df[f"{key}/num"])): # currently only one device per agent is supported
            # Get demand from load sheet
            self.df[f"{key}/sizing/demand_{num}"] = np.floor(pd.to_numeric(df['demand'] * 1e6, errors='coerce')).astype('Int64')
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
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

        return self.df

    def fill_flexible_load(self):
        """
            Fills all flexible_load columns
        """
        key = "flexible_load"
        config = self.config[f"{key}"]

        # general
        self._add_general_info(key=key)

        # sizing
        max_num = max(config["num"])
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
            # file
            self.df[f"{key}/sizing/file_{num}"] = self._pick_files(list_type=self.df["general/parameters/type"],
                                                                   device=key,
                                                                   input_path=os.path.join(self.input_path, key),
                                                                   idx_type=0)

        # forecast
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

    def fill_pv(self, **kwargs) -> pd.DataFrame:
        """
            Fills all pv columns
        """

        # Key in the config file
        key = "pv"

        # Get the config for the key
        config = self.config[f"{key}"]

        if self.method == 'config':
            self.df = self._pv_config(key=key, config=config)
        elif self.method == 'grid':
            self.df = self._pv_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _pv_config(self, key: str, config: dict) -> pd.DataFrame:
        """
            Fills all pv columns
        """

        # general
        self._add_general_info(key=key)

        # sizing
        max_num = max(config["num"])
        for num in range(max_num):
            # index list indicating ownership of device
            idx_list = self._get_idx_list(key=key, num=num)

            # add indexed info
            self._add_info_indexed(keys=[key, "sizing"], config=config["sizing"],
                                   idx_list=idx_list, appendix=f"_{num}")
            # postprocessing
            # power
            self.df[f"{key}/sizing/power_{num}"] *= self.df["inflexible_load/sizing/demand_0"]
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
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

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
        # note: always taken from config
        self.df[f"{key}/has_submeter"] = self.config[f"{key}"]["has_submeter"]

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
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

        # quality
        self.df[f"{key}/quality"] = config["quality"]

        return self.df

    def fill_wind(self, **kwargs) -> pd.DataFrame:
        """
            Fills all wind columns
        """
        # Key in the config file
        key = "wind"

        # Get the config for the key
        config = self.config[f"{key}"]

        if self.method == 'config':
            self.df = self._wind_config(key=key, config=config)
        elif self.method == 'grid':
            self.df = self._wind_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _wind_config(self, key: str, config: dict) -> pd.DataFrame:
        """
            Fills all wind columns
        """

        # general
        self._add_general_info(key=key)

        # sizing
        max_num = max(config["num"])
        for num in range(max_num):
            # index list indicating ownership of device
            idx_list = self._get_idx_list(key=key, num=num)

            # add indexed info
            self._add_info_indexed(keys=[key, "sizing"], config=config["sizing"],
                                   idx_list=idx_list, appendix=f"_{num}")
            # postprocessing
            # power
            self.df[f"{key}/sizing/power_{num}"] *= self.df["inflexible_load/sizing/demand_0"]
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
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

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
        # note: always taken from config
        self.df[f"{key}/has_submeter"] = self.config[f"{key}"]["has_submeter"]

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
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

        # quality
        self.df[f"{key}/quality"] = config["quality"]

        return self.df

    def fill_fixed_gen(self, **kwargs) -> pd.DataFrame:
        """
            Fills all fixed_gen columns
        """

        # Key in the config file
        key = "fixed_gen"

        # Get the config for the key
        config = self.config[f"{key}"]

        if self.method == 'config':
            self.df = self._fixed_gen_config(key=key, config=config)
        elif self.method == 'grid':
            self.df = self._fixed_gen_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _fixed_gen_config(self, key: str, config: dict) -> pd.DataFrame:
        """
            Fills all fixed_gen columns
        """

        # general
        self._add_general_info(key=key)

        # sizing
        max_num = max(config["num"])
        for num in range(max_num):
            # index list indicating ownership of device
            idx_list = self._get_idx_list(key=key, num=num)

            # add indexed info
            self._add_info_indexed(keys=[key, "sizing"], config=config["sizing"],
                                   idx_list=idx_list, appendix=f"_{num}")
            # postprocessing
            # power
            self.df[f"{key}/sizing/power_{num}"] *= self.df["inflexible_load/sizing/demand_0"]
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
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

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
        # note: always taken from config
        self.df[f"{key}/has_submeter"] = self.config[f"{key}"]["has_submeter"]

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
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

        # quality
        self.df[f"{key}/quality"] = config["quality"]

        return self.df

    def fill_ev(self, **kwargs) -> pd.DataFrame:
        """
            Fills all ev columns
        """

        # Key in the config file
        key = "ev"

        # Get the config for the key
        config = self.config[f"{key}"]

        if self.method == 'config':
            self._ev_config(key=key, config=config)
        elif self.method == 'grid':
            self._ev_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _ev_config(self, key: str, config: dict) -> pd.DataFrame:
        """
            Fills all ev columns
        """

        # general
        self._add_general_info(key=key)

        # sizing
        max_num = max(config["num"])
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

            # things only necessary to add once
            if num == 0:
                # charging scheme
                self._add_info_indexed(keys=[key, "charging_scheme"], config=config["charging_scheme"],
                                       idx_list=idx_list)

        # forecast
        self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])

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
        self.df[f"{key}/owner"] = (self.df[f"{key}/num"] > 0).fillna(0).astype('Int64')  # all agents that have plant type
        # note: always taken from config
        self.df[f"{key}/has_submeter"] = self.config[f"{key}"]["has_submeter"]

        # sizing
        for num in range(max(self.df[f"{key}/num"])):  # currently only one device per agent is supported
            self.df[f"{key}/sizing/file_{num}"] = self.df.index.map(df['file_add'])
            self.df[f"{key}/sizing/capacity_{num}"] = (self.df.index.map(df['capacity']) * 1e6).astype('Int64')
            self.df[f"{key}/sizing/consumption_{num}"] = (self.df.index.map(df['consumption']) * 1e6).astype('Int64')
            self.df[f"{key}/sizing/charging_home_{num}"] = (self.df.index.map(df['charging_home']) * 1e6).astype('Int64')
            self.df[f"{key}/sizing/charging_AC_{num}"] = (self.df.index.map(df['charging_ac']) * 1e6).astype('Int64')
            self.df[f"{key}/sizing/charging_DC_{num}"] = (self.df.index.map(df['charging_dc']) * 1e6).astype('Int64')
            self.df[f"{key}/sizing/charging_efficiency_{num}"] = self.df.index.map(df['efficiency'])
            self.df[f"{key}/sizing/soc_{num}"] = self.df.index.map(df['soc'])
            self.df[f"{key}/sizing/v2g_{num}"] = self.df.index.map(df['v2g'])
            self.df[f"{key}/sizing/v2h_{num}"] = self.df.index.map(df['v2h'])

        # quality
        self.df[f"{key}/quality"] = config["quality"]

        return self.df

    def fill_battery(self, **kwargs) -> pd.DataFrame:
        """
            Fills all battery columns
        """

        # Key in the config file
        key = "battery"

        # Get the config for the key
        config = self.config[f"{key}"]

        if self.method == 'config':
            self.df = self._battery_config(key=key, config=config)
        elif self.method == 'grid':
            self.df = self._battery_grid(key=key, config=config, **kwargs)
        else:
            raise NotImplementedError(f"The method '{self.method}' is not implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def _battery_config(self, key: str, config: dict) -> pd.DataFrame:
        """
            Fills all battery columns
        """

        # general
        self._add_general_info_bat(key=key)

        # sizing
        max_num = max(config["num"])
        for num in range(max_num):
            # index list indicating ownership of device
            idx_list = self._get_idx_list(key=key, num=num)

            # add indexed info
            self._add_info_indexed(keys=[key, "sizing"], config=config["sizing"], idx_list=idx_list,
                                   appendix=f"_{num}")
            # postprocessing
            # power
            self.df[f"{key}/sizing/power_{num}"] *= self.df["inflexible_load/sizing/demand_0"] / 1e3
            self.df[f"{key}/sizing/power_{num}"] = self._round_to_nth_digit(
                vals=self.df[f"{key}/sizing/power_{num}"], n=2)
            # capacity
            self.df[f"{key}/sizing/capacity_{num}"] *= self.df["inflexible_load/sizing/demand_0"] / 1e3
            self.df[f"{key}/sizing/capacity_{num}"] = self._round_to_nth_digit(
                vals=self.df[f"{key}/sizing/capacity_{num}"], n=2)

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
        # note: always taken from config
        self.df[f"{key}/has_submeter"] = self.config[f"{key}"]["has_submeter"]

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
    #             cols[4] = f"{key}/sizing/file"
    #             max_num = max(self.config[key]["num"])
    #             cols = cols[:3] + self.repeat_columns(columns=cols[3:5], num=max_num) + cols[5:]
    #         # Adjust the columns from "flexible_load"
    #         elif key == "flexible_load":
    #             cols[0] = f"{key}/owner"
    #             cols[4] = f"{key}/sizing/file"
    #             max_num = max(self.config[key]["num"])
    #             cols = cols[:3] + self.repeat_columns(columns=cols[3:6], num=max_num) + cols[6:]
    #         # Adjust the columns from "pv"
    #         elif key == "pv":
    #             cols[0] = f"{key}/owner"
    #             del cols[8]
    #             del cols[4]
    #             max_num = max(self.config[key]["num"])
    #             cols = cols[:3] + self.repeat_columns(columns=cols[3:8], num=max_num) + cols[8:]
    #         # Adjust the columns from "wind"
    #         elif key == "wind":
    #             cols[0] = f"{key}/owner"
    #             del cols[4]
    #             max_num = max(self.config[key]["num"])
    #             cols = cols[:3] + self.repeat_columns(columns=cols[3:6], num=max_num) + cols[6:]
    #         # Adjust the columns from "fixed_gen"
    #         elif key == "fixed_gen":
    #             cols[0] = f"{key}/owner"
    #             del cols[4]
    #             max_num = max(self.config[key]["num"])
    #             cols = cols[:3] + self.repeat_columns(columns=cols[3:6], num=max_num) + cols[6:]
    #         # Adjust the columns from "ev"
    #         elif key == "ev":
    #             cols[0] = f"{key}/owner"
    #             max_num = max(self.config[key]["num"])
    #             cols = cols[:3] + self.repeat_columns(columns=cols[3:13], num=max_num) + cols[13:]
    #         # Adjust the columns from "battery"
    #         elif key == "battery":
    #             cols[0] = f"{key}/owner"
    #             del cols[1]
    #             max_num = max(self.config[key]["num"])
    #             cols = cols[:3] + self.repeat_columns(columns=cols[3:-1], num=max_num) + [cols[-1]]
    #         # Adjust the columns from "mpc"
    #         elif key == "mpc":
    #             pass
    #         # Adjust the columns from "market_agent"
    #         elif key == "market_agent":
    #             pass
    #         # Adjust the columns from "meter"
    #         elif key == "meter":
    #             pass
    #         else:
    #             raise NotImplementedError(
    #                 f"The configuration file contains a key word ('{key}') that has not been configured in "
    #                 "the Sfhs class yet. Aborting scenario creation...")
    #         columns[key] = cols
    #
    #     # Combine all separate lists into one for the dataframe
    #     cols_df = []
    #     for idx, cols in columns.items():
    #         cols_df += cols
    #
    #     # Create dataframe with responding columns
    #     self.df = pd.DataFrame(index=range(self.config["general"]["number_of"]), columns=cols_df)
    #
    #     return self.df
    #
    # def fill_general(self):
    #     """
    #         Fills all general columns
    #     """
    #     key = "general"
    #     config = self.config[f"{key}"]
    #
    #     # general
    #     self.df[f"{key}/agent_id"] = self._gen_new_ids(n=self.num)
    #     self.idx_list = self._gen_idx_list_from_distr(n=self.num, distr=config["parameters"]["distribution"])
    #
    #     # parameters
    #     # add indexed info
    #     self._add_info_indexed(keys=[key, "parameters"], config=config["parameters"], idx_list=self.idx_list)
    #     # postprocessing
    #     # area
    #     self.df[f"{key}/parameters/area"] = self._calc_deviation(idx_list=self.idx_list,
    #                                                              vals=self.df[f"{key}/parameters/area"],
    #                                                              distr=config["parameters"]["area_deviation"],
    #                                                              method="relative")
    #     self.df[f"{key}/parameters/area"] = self._round_to_nth_digit(
    #         vals=self.df[f"{key}/parameters/area"], n=self.n_digits)
    #
    #     # forecast
    #     self.df[f"{key}/fcast_retraining_frequency"] = config["fcast_retraining_frequency"]
    #
    #     # market participation
    #     self.df[f"{key}/market_participant"] = self._gen_rand_bool_list(n=self.num,
    #                                                                     share_ones=config["market_participant_share"])
    #
    #
    #     print(self.df.loc[:, self.df.filter(like=key).columns].to_string())
    #     exit()
    #
    #     return self.df
    #
    # def fill_inflexible_load(self):
    #     """
    #         Fills all inflexible_load columns
    #     """
    #     key = "inflexible_load"
    #     config = self.config[f"{key}"]
    #
    #     # general
    #     self._add_general_info(key=key)
    #
    #     # sizing
    #     max_num = max(config["num"])
    #     for num in range(max_num):
    #         # index list indicating ownership of device
    #         idx_list = self._get_idx_list(key=key, num=num)
    #
    #         # add indexed info
    #         self._add_info_indexed(keys=[key, "sizing"], config=config["sizing"],
    #                                idx_list=idx_list, appendix=f"_{num}")
    #         # postprocessing
    #         # power
    #         self.df[f"{key}/sizing/demand_{num}"] *= self.df["general/parameters/area"]
    #         self.df[f"{key}/sizing/demand_{num}"] = self._calc_deviation(idx_list=idx_list,
    #                                                                      vals=self.df[f"{key}/sizing/demand_{num}"],
    #                                                                      distr=config["sizing"]["demand_deviation"],
    #                                                                      method="relative")
    #         self.df[f"{key}/sizing/demand_{num}"] = self._round_to_nth_digit(
    #             vals=self.df[f"{key}/sizing/demand_{num}"], n=self.n_digits)
    #         # file
    #         self.df[f"{key}/sizing/file_{num}"] = self._pick_files(list_type=self.df["general/parameters/type"],
    #                                                                device=key,
    #                                                                input_path=os.path.join(self.input_path, key),
    #                                                                idx_type=0)
    #
    #     # forecast
    #     self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])
    #
    # def fill_flexible_load(self):
    #     """
    #         Fills all flexible_load columns
    #     """
    #     key = "flexible_load"
    #     config = self.config[f"{key}"]
    #
    #     # general
    #     self._add_general_info(key=key)
    #
    #     # sizing
    #     max_num = max(config["num"])
    #     for num in range(max_num):
    #         # index list indicating ownership of device
    #         idx_list = self._get_idx_list(key=key, num=num)
    #
    #         # add indexed info
    #         self._add_info_indexed(keys=[key, "sizing"], config=config["sizing"],
    #                                idx_list=idx_list, appendix=f"_{num}")
    #         # postprocessing
    #         # power
    #         self.df[f"{key}/sizing/demand_{num}"] *= self.df["general/parameters/area"]
    #         self.df[f"{key}/sizing/demand_{num}"] = self._calc_deviation(idx_list=idx_list,
    #                                                                      vals=self.df[f"{key}/sizing/demand_{num}"],
    #                                                                      distr=config["sizing"]["demand_deviation"],
    #                                                                      method="relative")
    #         self.df[f"{key}/sizing/demand_{num}"] = self._round_to_nth_digit(
    #             vals=self.df[f"{key}/sizing/demand_{num}"], n=self.n_digits)
    #         # file
    #         self.df[f"{key}/sizing/file_{num}"] = self._pick_files(list_type=self.df["general/parameters/type"],
    #                                                                device=key,
    #                                                                input_path=os.path.join(self.input_path, key),
    #                                                                idx_type=0)
    #
    #     # forecast
    #     self._add_info_simple(keys=[key, "fcast"], config=self.config[key]["fcast"])
    #
    # def fill_pv(self):
    #     """
    #         Fills all pv columns
    #     """
    #     key = "pv"
    #     config = self.config[f"{key}"]
    #
    #     # general
    #     self._add_general_info(key=key)
    #
    #     # sizing
    #     max_num = max(config["num"])
    #     for num in range(max_num):
    #         # index list indicating ownership of device
    #         idx_list = self._get_idx_list(key=key, num=num)
    #
    #         # add indexed info
    #         self._add_info_indexed(keys=[key, "sizing"], config=config["sizing"],
    #                                idx_list=idx_list, appendix=f"_{num}")
    #         # postprocessing
    #         # power
    #         self.df[f"{key}/sizing/power_{num}"] *= self.df["inflexible_load/sizing/demand_0"]
    #         self.df[f"{key}/sizing/power_{num}"] = self._calc_deviation(idx_list=idx_list,
    #                                                                     vals=self.df[f"{key}/sizing/power_{num}"],
    #                                                                     distr=config["sizing"]["power_deviation"],
    #                                                                     method="relative")
    #         self.df[f"{key}/sizing/power_{num}"] = self._round_to_nth_digit(vals=self.df[f"{key}/sizing/power_{num}"],
    #                                                                         n=self.n_digits)
    #         # file
    #         self.df[f"{key}/sizing/file_{num}"] = self._pick_files(list_type=self.df[f"{key}/sizing/file_{num}"],
    #                                                                device=f"{key}",
    #                                                                input_path=os.path.join(self.input_path, key))
    #
    #     # forecast
    #     self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])
    #
    #     # quality
    #     self.df[f"{key}/quality"] = config["quality"]
    #
    # def fill_wind(self):
    #     """
    #         Fills all wind columns
    #     """
    #     key = "wind"
    #     config = self.config[f"{key}"]
    #
    #     # general
    #     self._add_general_info(key=key)
    #
    #     # sizing
    #     max_num = max(config["num"])
    #     for num in range(max_num):
    #         # index list indicating ownership of device
    #         idx_list = self._get_idx_list(key=key, num=num)
    #
    #         # add indexed info
    #         self._add_info_indexed(keys=[key, "sizing"], config=config["sizing"],
    #                                idx_list=idx_list, appendix=f"_{num}")
    #         # postprocessing
    #         # power
    #         self.df[f"{key}/sizing/power_{num}"] *= self.df["inflexible_load/sizing/demand_0"]
    #         self.df[f"{key}/sizing/power_{num}"] = self._calc_deviation(idx_list=idx_list,
    #                                                                     vals=self.df[f"{key}/sizing/power_{num}"],
    #                                                                     distr=config["sizing"]["power_deviation"],
    #                                                                     method="relative")
    #         self.df[f"{key}/sizing/power_{num}"] = self._round_to_nth_digit(vals=self.df[f"{key}/sizing/power_{num}"],
    #                                                                         n=self.n_digits)
    #         # file
    #         self.df[f"{key}/sizing/file_{num}"] = self._pick_files(list_type=self.df[f"{key}/sizing/file_{num}"],
    #                                                                device=f"{key}",
    #                                                                input_path=os.path.join(self.input_path, key))
    #
    #     # forecast
    #     self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])
    #
    #     # quality
    #     self.df[f"{key}/quality"] = config["quality"]
    #
    # def fill_fixed_gen(self):
    #     """
    #         Fills all fixed_gen columns
    #     """
    #
    #     key = "fixed_gen"
    #     config = self.config[f"{key}"]
    #
    #     # general
    #     self._add_general_info(key=key)
    #
    #     # sizing
    #     max_num = max(config["num"])
    #     for num in range(max_num):
    #         # index list indicating ownership of device
    #         idx_list = self._get_idx_list(key=key, num=num)
    #
    #         # add indexed info
    #         self._add_info_indexed(keys=[key, "sizing"], config=config["sizing"],
    #                                idx_list=idx_list, appendix=f"_{num}")
    #         # postprocessing
    #         # power
    #         self.df[f"{key}/sizing/power_{num}"] *= self.df["inflexible_load/sizing/demand_0"]
    #         self.df[f"{key}/sizing/power_{num}"] = self._calc_deviation(idx_list=idx_list,
    #                                                                     vals=self.df[f"{key}/sizing/power_{num}"],
    #                                                                     distr=config["sizing"]["power_deviation"],
    #                                                                     method="relative")
    #         self.df[f"{key}/sizing/power_{num}"] = self._round_to_nth_digit(vals=self.df[f"{key}/sizing/power_{num}"],
    #                                                                         n=self.n_digits)
    #         # file
    #         self.df[f"{key}/sizing/file_{num}"] = self._pick_files(list_type=self.df[f"{key}/sizing/file_{num}"],
    #                                                                device=f"{key}",
    #                                                                input_path=os.path.join(self.input_path, key))
    #
    #     # forecast
    #     self._add_info_simple(keys=[key, "fcast"], config=self.config[key]["fcast"])
    #
    #     # quality
    #     self.df[f"{key}/quality"] = self.config[f"{key}"]["quality"]
    #
    # def fill_ev(self):
    #     """
    #         Fills all ev columns
    #     """
    #     key = "ev"
    #     config = self.config[f"{key}"]
    #
    #     # general
    #     self._add_general_info(key=key)
    #
    #     # sizing
    #     max_num = max(config["num"])
    #     for num in range(max_num):
    #         # index list indicating ownership of device
    #         idx_list = self._get_idx_list(key=key, num=num)
    #
    #         # sizing
    #         self._add_info_indexed(keys=[key, "sizing"], config=config["sizing"],
    #                                idx_list=idx_list, appendix=f"_{num}")
    #         # postprocessing
    #         # file
    #         self.df[f"{key}/sizing/file_{num}"] = self._pick_files(list_type=self.df[f"{key}/sizing/file_{num}"],
    #                                                                device=f"{key}",
    #                                                                input_path=os.path.join(self.input_path, key))
    #
    #         # things only necessary to add once
    #         if num == 0:
    #             # charging scheme
    #             self._add_info_indexed(keys=[key, "charging_scheme"], config=config["charging_scheme"],
    #                                    idx_list=idx_list)
    #
    #     # forecast
    #     self._add_info_simple(keys=[key, "fcast"], config=config["fcast"])
    #
    #     # quality
    #     self.df[f"{key}/quality"] = config["quality"]
    #
    # def fill_battery(self):
    #     """
    #         Fills all battery columns
    #     """
    #     key = "battery"
    #     config = self.config[f"{key}"]
    #
    #     # general
    #     self._add_general_info_dep(key=key)
    #
    #     # sizing
    #     max_num = max(config["num"])
    #     for num in range(max_num):
    #         # index list indicating ownership of device
    #         idx_list = self._get_idx_list(key=key, num=num)
    #
    #         # add indexed info
    #         self._add_info_indexed(keys=[key, "sizing"], config=config["sizing"], idx_list=idx_list,
    #                                appendix=f"_{num}")
    #         # postprocessing
    #         # power
    #         self.df[f"{key}/sizing/power_{num}"] *= self.df["inflexible_load/sizing/demand_0"] / 1000
    #         self.df[f"{key}/sizing/power_{num}"] = self._round_to_nth_digit(
    #             vals=self.df[f"{key}/sizing/power_{num}"], n=self.n_digits)
    #         # capacity
    #         self.df[f"{key}/sizing/capacity_{num}"] *= self.df["inflexible_load/sizing/demand_0"] / 1000
    #         self.df[f"{key}/sizing/capacity_{num}"] = self._round_to_nth_digit(
    #             vals=self.df[f"{key}/sizing/capacity_{num}"], n=self.n_digits)
    #
    #     # quality
    #     self.df[f"{key}/quality"] = str(config["quality"])

    def fill_mpc(self):
        """
            Fills all battery columns
        """
        key = "mpc"
        config = self.config[f"{key}"]

        # general
        self._add_info_simple(keys=[key], config=config)

    def fill_market_agent(self):
        """
            Fills all market agent columns
        """
        key = "market_agent"
        config = self.config[f"{key}"]

        # general
        self._add_info_random(keys=[key], config=config)

    def fill_meter(self):
        """
            Fills all battery columns
        """
        key = "meter"
        config = self.config[f"{key}"]

        # general
        self._add_info_simple(keys=[key], config=config)

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
        self.df[f"{key}/has_submeter"] = self.config[f"{key}"]["has_submeter"]

    def _add_general_info_bat(self, key: str):

        # find all potential owners of a battery system dependent on type
        # note: this setup considers the different dependencies for each type and loops through each separately
        agent_types = self.config["general"]["parameters"]["type"]
        list_owner = [0] * self.num
        list_num = [0] * self.num
        for idx, agent_type in enumerate(agent_types):
            # get all agents of given type
            list_type = list(self.df["general/parameters/type"] == agent_type)
            plants = self.config[f"{key}"]["share_dependent_on"][idx] + ["inflexible_load"]
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
            self.df[f"{key}/has_submeter"] = str(self.config[f"{key}"]["has_submeter"])


class Producer(Agents):
    """
        Sets up producer agents. Inherits from Agents class.

        Mainly used for excel file creation. Afterwards Sfh class creates the individual agents.
    """

    def __init__(self, input_path: str, config: ordereddict, config_path: str, scenario_path: str):

        # Call the init method of the parent class
        super().__init__(config_path, input_path, scenario_path)

        # Define agent type
        self.type = 'producer'

        # Path of the input file
        self.input_path = os.path.join(input_path, 'agents', self.type)

        # Config file
        self.config = config

        # Grid information (if applicable)
        self.grid = None
        self.bus = None  # bus sheet containing only the bus information of the agent type
        self.load = None  # load sheet containing only the load information of the agent type
        self.agents = None # load sheet but limited to all agents, i.e. all inflexible_loads
        self.sgen = None  # sgen sheet containing only the sgen information of the agent type

        # Creation method
        self.method = None

        # Number of agents
        self.num = 0
        self.num_agents = 0           # number of agents (changes depending on which "add_xxx()" function is called)

        # Dataframe containing all information
        self.df = None

        # Index list that is adhered to throughout the creation process to ensure correct order
        self.idx_list = None        # gets created in create_general()
        self.idx_start = 0                # start index where to insert values (changes based on dataframe length)
        self.idx_end = 0

        # Misc
        self.n_digits = 3           # number of digits values get rounded to in respective value column

    def create_df_from_config(self) -> pd.DataFrame:
        """
            Function to create the dataframe that makes the Excel sheet
        """

        # Set the method
        self.method = 'config'

        # Create the overall dataframe structure for the worksheet
        self.create_df_structure()

        # Fill the pv information in dataframe
        self.add_pv()

        # Fill the wind information in dataframe
        self.add_wind()

        # Fill the fixed generation information in dataframe
        self.add_fixed_gen()

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

        # Get the number of agents and set the method
        self.num = self.get_num_from_grid(self.grid['load'], self.type)
        self.method = 'grid'

        # Create the overall dataframe structure for the worksheet
        self.create_df_structure()

        # Note: As the grid file does not include producer type agents this cannot be done yet
        return self.df

        # Fill the general information in dataframe
        self.fill_general()

        # Fill the pv information in dataframe
        self.fill_pv(**kwargs)

        # Fill the wind information in dataframe
        self.fill_wind(**kwargs)

        # Fill the fixed generation information in dataframe
        self.fill_fixed_gen(**kwargs)

        # Fill the battery information in dataframe
        self.fill_battery(**kwargs)

        # Fill the model predictive controller information in dataframe
        self.fill_mpc()

        # Fill the market agent information in dataframe
        self.fill_market_agent()

        # Fill the metering information in dataframe
        self.fill_meter()

        return self.df

    def create_df_structure(self):
        """
            Function to create the dataframe structure with the respective columns
        """
        # Go through file and create the columns for the ctss worksheet
        columns = ordereddict()
        before = True   # variable to insert entries before the plants
        max_bat = 1     # variable to keep track of how many batteries need to be added
        for key, _ in self.config.items():
            cols = self.make_list_from_nested_dict(self.config[key])

            # Define and insert all columns that come before the plants
            if before:
                befkeys = ["general"]
                for befkey in befkeys:
                    befcols = [col for col in cols if befkey in col]
                    # Adjust the columns from "general"
                    if befkey == "general":
                        befcols[0] = f"{befkey}/agent_id"
                        befcols[-1] = f"{befkey}/market_participant"
                        befcols.insert(1, f"{befkey}/name")
                        befcols.insert(2, f"{befkey}/comment")
                        befcols.insert(3, f"{befkey}/bus")
                    columns[befkey] = befcols

                # Set before to False as it only runs once
                before = False

            # Adjust the columns from "pv"
            if key == "pv":
                self.num += self.config[key]["general"]["number_of"]
                del cols[16:]
                del cols[7]
                del cols[5]
                del cols[:3]
                cols.insert(0, f"{key}/owner")
                max_num = max(self.config[key][key]["num"])
                cols = cols[:3] + self.repeat_columns(columns=cols[3:8], num=max_num) + cols[8:]
            # Adjust the columns from "wind"
            elif key == "wind":
                self.num += self.config[key]["general"]["number_of"]
                del cols[14:]
                del cols[7]
                del cols[5]
                del cols[:3]
                cols.insert(0, f"{key}/owner")
                max_num = max(self.config[key][key]["num"])
                cols = cols[:3] + self.repeat_columns(columns=cols[3:6], num=max_num) + cols[6:]
            # Adjust the columns from "fixed_gen"
            elif key == "fixed_gen":
                self.num += self.config[key]["general"]["number_of"]
                del cols[14:]
                del cols[7]
                del cols[5]
                del cols[:3]
                cols.insert(0, f"{key}/owner")
                max_num = max(self.config[key][key]["num"])
                cols = cols[:3] + self.repeat_columns(columns=cols[3:6], num=max_num) + cols[6:]
            else:
                raise NotImplementedError(
                    f"The configuration file contains a key word ('{key}') that has not been configured in "
                    "the Sfhs class yet. Aborting scenario creation...")

            # Update maximum number of batteries
            max_bat = max(max(self.config[key]["battery"]["num"]), max_bat)

            # Save columns
            columns[key] = cols

        # Define and insert all columns that come before the plants
        aftcols = self.make_list_from_nested_dict(self.config[key])
        aftkeys = ["battery", "mpc", "market_agent", "meter"]
        for aftkey in aftkeys:
            # Get all columns that match the key
            cols = [col for col in aftcols if aftkey == col.split("/", 1)[0]]
            # Adjust the columns from "battery"
            if aftkey == "battery":
                del cols[2]
                cols.insert(0, f"{aftkey}/owner")
                cols = cols[:3] + self.repeat_columns(columns=cols[3:-1], num=max_bat) + [cols[-1]]
            # All columns that do not need to be adjusted
            elif aftkey in ["mpc", "market_agent", "meter"]:
                pass
            else:
                raise NotImplementedError(
                    f"The configuration file contains a key word ('{aftkey}') that has not been configured in "
                    "the Sfhs class yet. Aborting scenario creation...")
            columns[aftkey] = cols

        # Combine all separate lists into one for the dataframe
        cols_df = []
        for _, cols in columns.items():
            cols_df += cols

        # Create dataframe with responding columns
        if self.method == 'config':
            # normal indexing
            self.df = pd.DataFrame(columns=cols_df)
        elif self.method == 'grid':
            # indexing matches the load sheet (all rows that are empty in owner as those are EVs and HPs)
            self.df = pd.DataFrame(index=self.agents.index, columns=cols_df)
        else:
            raise NotImplementedError(f"The method '{self.method}' has not been implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def add_pv(self):
        """
            Adds all PV producers
        """
        key = "pv"
        self.num_agents = self.config[f"{key}"]["general"]["number_of"]
        self.idx_start = len(self.df)

        # general
        self.fill_general(device=key)

        # pv
        self.fill_pv(device=key)

        # battery
        self.fill_battery(device=key)

        # mpc
        self.fill_mpc(device=key)

        # market_agent
        self.fill_market_agent(device=key)

        # meter
        self.fill_meter(device=key)

    def add_wind(self):
        """
            Adds all wind producers
        """
        key = "wind"
        self.num_agents = self.config[f"{key}"]["general"]["number_of"]

        # general
        self.fill_general(device=key)

        # wind
        self.fill_wind(device=key)

        # battery
        self.fill_battery(device=key)

        # mpc
        self.fill_mpc(device=key)

        # market_agent
        self.fill_market_agent(device=key)

        # meter
        self.fill_meter(device=key)

    def add_fixed_gen(self):
        """
            Adds all wind producers
        """
        key = "fixed_gen"
        self.num_agents = self.config[f"{key}"]["general"]["number_of"]

        # general
        self.fill_general(device=key)

        # wind
        self.fill_fixed_gen(device=key)

        # battery
        self.fill_battery(device=key)

        # mpc
        self.fill_mpc(device=key)

        # market_agent
        self.fill_market_agent(device=key)

        # meter
        self.fill_meter(device=key)

    def fill_general(self, device: str):
        """
            Fills all general columns
        """
        key = "general"
        config = self.config[f"{device}"][f"{key}"]
        self.idx_start = len(self.df)
        self.idx_end = self.idx_start + self.num_agents

        # add the required rows
        for i in range(self.idx_start, self.idx_end):
            # TODO: Throws a FutureWarning. That needs to be checked.
            self.df.loc[i] = np.nan

        # general
        self.df.loc[self.idx_start:self.idx_end, f"{key}/agent_id"] = self._gen_new_ids(n=self.num_agents)

        # forecast
        self.df.loc[self.idx_start:self.idx_end, f"{key}/fcast_retraining_frequency"] = config["fcast_retraining_frequency"]

        # market participation
        self.df.loc[self.idx_start:self.idx_end, f"{key}/market_participant"] = self._gen_rand_bool_list(
            n=self.num_agents, share_ones=config["market_participant_share"])

    def fill_pv(self, device: str):
        """
            Fills all pv columns
        """
        key = "pv"
        config = self.config[f"{device}"][f"{key}"]

        # general
        self._add_general_info(key=key, config=config)

        # sizing
        max_num = max(config["num"])
        for num in range(max_num):
            # index list indicating ownership of device
            idx_list = self._get_idx_list(key=key, num=num, config=config)

            # sizing
            self.df.loc[self.idx_start:self.idx_end] = self._add_info_indexed(
                keys=[key, "sizing"], config=config["sizing"], df=self.df[self.idx_start:self.idx_end][:],
                idx_list=idx_list, appendix=f"_{num}")
            # postprocessing
            # power
            self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/power_{num}"] = self._calc_deviation(
                idx_list=idx_list, vals=self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/power_{num}"],
                distr=config["sizing"]["power_deviation"], method="absolute")
            self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/power_{num}"] = self._round_to_nth_digit(
                vals=self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/power_{num}"], n=self.n_digits)
            # file
            self.df[f"{key}/sizing/file_{num}"] = self._pick_files(
                list_type=self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/file_{num}"],
                device=f"{key}",
                input_path=os.path.join(self.input_path, key))

        # forecast
        self.df.loc[self.idx_start:self.idx_end] = self._add_info_simple(keys=[key, "fcast"], config=config["fcast"],
                                                                         df=self.df[self.idx_start:self.idx_end][:])

        # quality
        self.df.loc[self.idx_start:self.idx_end, f"{key}/quality"] = config["quality"]

    def fill_wind(self, device: str):
        """
            Fills all wind columns
        """
        key = "wind"
        config = self.config[f"{device}"][f"{key}"]

        # general
        self._add_general_info(key=key, config=config)

        # sizing
        max_num = max(config["num"])
        for num in range(max_num):
            # index list indicating ownership of device
            idx_list = self._get_idx_list(key=key, num=num, config=config)

            # sizing
            self.df.loc[self.idx_start:self.idx_end] = self._add_info_indexed(
                keys=[key, "sizing"], config=config["sizing"], df=self.df[self.idx_start:self.idx_end][:],
                idx_list=idx_list, appendix=f"_{num}")
            # postprocessing
            # power
            self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/power_{num}"] = self._calc_deviation(
                idx_list=idx_list, vals=self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/power_{num}"],
                distr=config["sizing"]["power_deviation"], method="absolute")
            self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/power_{num}"] = self._round_to_nth_digit(
                vals=self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/power_{num}"], n=self.n_digits)
            # file
            self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/file_{num}"] = self._pick_files(
                list_type=self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/file_{num}"],
                device=f"{key}",
                input_path=os.path.join(self.input_path, key))

        # forecast
        self.df.loc[self.idx_start:self.idx_end] = self._add_info_simple(keys=[key, "fcast"], config=config["fcast"],
                                                                         df=self.df[self.idx_start:self.idx_end][:])

        # quality
        self.df.loc[self.idx_start:self.idx_end, f"{key}/quality"] = config["quality"]

    def fill_fixed_gen(self, device: str):
        """
            Fills all fixed_gen columns
        """

        key = "fixed_gen"
        config = self.config[f"{device}"][f"{key}"]

        # general
        self._add_general_info(key=key, config=config)

        # sizing
        max_num = max(config["num"])
        for num in range(max_num):
            # index list indicating ownership of device
            idx_list = self._get_idx_list(key=key, num=num, config=config)

            # sizing
            self.df.loc[self.idx_start:self.idx_end] = self._add_info_indexed(
                keys=[key, "sizing"], config=config["sizing"], df=self.df[self.idx_start:self.idx_end][:],
                idx_list=idx_list, appendix=f"_{num}")
            # postprocessing
            # power
            self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/power_{num}"] = self._calc_deviation(
                idx_list=idx_list, vals=self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/power_{num}"],
                distr=config["sizing"]["power_deviation"], method="absolute")
            self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/power_{num}"] = self._round_to_nth_digit(
                vals=self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/power_{num}"], n=self.n_digits)
            # file
            self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/file_{num}"] = self._pick_files(
                list_type=self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/file_{num}"],
                device=f"{key}",
                input_path=os.path.join(self.input_path, key))

        # forecast
        self.df.loc[self.idx_start:self.idx_end] = self._add_info_simple(keys=[key, "fcast"], config=config["fcast"],
                                                                         df=self.df[self.idx_start:self.idx_end][:])

        # quality
        self.df.loc[self.idx_start:self.idx_end, f"{key}/quality"] = config["quality"]

    def fill_battery(self, device: str):
        """
            Fills all battery columns
        """
        key = "battery"
        config = self.config[f"{device}"][f"{key}"]

        # general
        self._add_general_info_dependent(key=key, config=config)

        # sizing
        max_num = max(config["num"])
        for num in range(max_num):
            # index list indicating ownership of device
            idx_list = self._get_idx_list(key=key, num=num, config=config)

            # sizing
            self.df.loc[self.idx_start:self.idx_end] = self._add_info_indexed(
                keys=[key, "sizing"], config=config["sizing"], df=self.df[self.idx_start:self.idx_end][:],
                idx_list=idx_list, appendix=f"_{num}")
            # postprocessing
            # power
            self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/power_{num}"] *= self.df[f"{device}/sizing/power_0"]
            # capacity
            self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/capacity_{num}"] *= self.df[f"{device}/sizing/power_0"]

        # quality
        self.df.loc[self.idx_start:self.idx_end, f"{key}/quality"] = config["quality"]

    def fill_mpc(self, device: str):
        """
            Fills all battery columns
        """
        key = "mpc"
        config = self.config[f"{device}"][f"{key}"]

        # general
        self.df.loc[self.idx_start:self.idx_end] = self._add_info_simple(keys=[key], config=config,
                                                                         df=self.df[self.idx_start:self.idx_end][:])

    def fill_market_agent(self, device: str):
        """
            Fills all market agent columns
        """
        key = "market_agent"
        config = self.config[f"{device}"][f"{key}"]

        # general
        self.df.loc[self.idx_start:self.idx_end] = self._add_info_random(keys=[key], config=config,
                                                                         df=self.df[self.idx_start:self.idx_end][:])

    def fill_meter(self, device: str):
        """
            Fills all battery columns
        """
        key = "meter"
        config = self.config[f"{device}"][f"{key}"]

        # general
        self.df.loc[self.idx_start:self.idx_end] = self._add_info_simple(keys=[key], config=config,
                                                                         df=self.df[self.idx_start:self.idx_end][:])

    def _get_idx_list(self, key: str, num: int, config: dict) -> list:
        """creates the index list for the given run"""

        # Check who owns the device
        list_owner = np.multiply(np.array(self.df.loc[self.idx_start:self.idx_end, f"{key}/num"] - (1 + num) >= 0), 1)
        list_owner = [np.nan if elem == 0 else elem for elem in list_owner]

        # Return according index list based on ownership to choose plants
        idx_list = self._gen_idx_list_from_distr(n=self.num_agents, distr=config["sizing"]["distribution"])
        idx_list = np.multiply(list_owner, idx_list)

        return [int(elem) if not np.isnan(elem) else np.nan for elem in idx_list]

    def _add_general_info(self, key: str, config: dict) -> None:

        # fields that exist for all plants
        self.df.loc[self.idx_start:self.idx_end, f"{key}/owner"] = 1
        self.df.loc[self.idx_start:self.idx_end, f"{key}/num"] = self._gen_dep_num_list(
            owner_list=self.df.loc[self.idx_start:self.idx_end, f"{key}/owner"], distr=config["num"])
        self.df.loc[self.idx_start:self.idx_end, f"{key}/num"] *= self.df[f"{key}/owner"]
        self.df.loc[self.idx_start:self.idx_end, f"{key}/has_submeter"] = config["has_submeter"]

    def _add_general_info_dependent(self, key: str, config: dict) -> None:

        # fields that exist for all plants
        self.df.loc[self.idx_start:self.idx_end, f"{key}/owner"] = 1
        self.df.loc[self.idx_start:self.idx_end, f"{key}/num"] = self._gen_dep_num_list(
            owner_list=self.df.loc[self.idx_start:self.idx_end, f"{key}/owner"], distr=config["num"])
        self.df.loc[self.idx_start:self.idx_end, f"{key}/owner"] = \
            (self.df.loc[self.idx_start:self.idx_end, f"{key}/num"] > 0) * 1
        self.df.loc[self.idx_start:self.idx_end, f"{key}/has_submeter"] = config["has_submeter"]

    def _add_general_info_bat(self, key: str):

        # find all potential owners of a battery system dependent on type
        # note: this setup considers the different dependencies for each type and loops through each separately
        agent_types = self.config["general"]["parameters"]["type"]
        list_owner = [0] * self.num
        list_num = [0] * self.num
        for idx, agent_type in enumerate(agent_types):
            # get all agents of given type
            list_type = list(self.df["general/parameters/type"] == agent_type)
            plants = self.config[f"{key}"]["share_dependent_on"][idx] + ["inflexible_load"]
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

            self.df.loc[self.idx_start:self.idx_end, f"{key}/owner"] = list_owner
            self.df.loc[self.idx_start:self.idx_end, f"{key}/num"] = list_num
            self.df.loc[self.idx_start:self.idx_end, f"{key}/has_submeter"] = str(self.config[f"{key}"]["has_submeter"])


class Storage(Agents):
    """
        Sets up storage agents. Inherits from Agents class.

        Mainly used for excel file creation. Afterwards Sfh class creates the individual agents.
    """

    def __init__(self, input_path: str, config: ordereddict, config_path: str, scenario_path: str):

        # Call the init method of the parent class
        super().__init__(config_path, input_path, scenario_path)

        # Define agent type
        self.type = 'storage'

        # Path of the input file
        self.input_path = os.path.join(input_path, 'agents', self.type)

        # Config file
        self.config = config

        # Grid information (if applicable)
        self.grid = None
        self.bus = None  # bus sheet containing only the bus information of the agent type
        self.load = None  # load sheet containing only the load information of the agent type
        self.agents = None # load sheet but limited to all agents, i.e. all inflexible_loads
        self.sgen = None  # sgen sheet containing only the sgen information of the agent type

        # Creation method
        self.method = None

        # Number of agents
        self.num = 0
        self.num_agents = 0           # number of agents (changes depending on which "add_xxx()" function is called)

        # Dataframe containing all information
        self.df = None

        # Index list that is adhered to throughout the creation process to ensure correct order
        self.idx_list = None        # gets created in create_general()
        self.idx_start = 0                # start index where to insert values (changes based on dataframe length)
        self.idx_end = 0

        # Misc
        self.n_digits = 3           # number of digits values get rounded to in respective value column

    def create_df_from_config(self) -> pd.DataFrame:
        """
            Function to create the dataframe that makes the Excel sheet
        """

        # Set the method
        self.method = 'config'

        # Create the overall dataframe structure for the worksheet
        self.create_df_structure()

        # Fill the battery information in dataframe
        self.add_battery()

        # Fill the psh information in dataframe
        self.add_psh()

        # Fill the hydrogen information in dataframe
        self.add_hydrogen()

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

        # Get the number of agents and set the method
        self.num = self.get_num_from_grid(self.grid['load'], self.type)
        self.method = 'grid'

        # Create the overall dataframe structure for the worksheet
        self.create_df_structure()

        # Note: As the grid file does not include storage type agents this cannot be done yet
        return self.df

        # Fill the general information in dataframe
        self.fill_general()

        # Fill the battery information in dataframe
        self.fill_battery(**kwargs)

        # Fill the psh information in dataframe
        self.add_psh()

        # Fill the hydrogen information in dataframe
        self.add_hydrogen()

        # Fill the model predictive controller information in dataframe
        self.fill_mpc()

        # Fill the market agent information in dataframe
        self.fill_market_agent()

        # Fill the metering information in dataframe
        self.fill_meter()

        return self.df

    def create_df_structure(self):
        """
            Function to create the dataframe structure with the respective columns
        """
        # Go through file and create the columns for the ctss worksheet
        columns = ordereddict()
        before = True   # variable to insert entries before the plants
        for key, _ in self.config.items():
            cols = self.make_list_from_nested_dict(self.config[key])

            # Define and insert all columns that come before the plants
            if before:
                befkeys = ["general"]
                for befkey in befkeys:
                    befcols = [col for col in cols if befkey in col]
                    # Adjust the columns from "general"
                    if befkey == "general":
                        befcols[0] = f"{befkey}/agent_id"
                        befcols[-1] = f"{befkey}/market_participant"
                        befcols.insert(1, f"{befkey}/name")
                        befcols.insert(2, f"{befkey}/comment")
                        befcols.insert(3, f"{befkey}/bus")
                    columns[befkey] = befcols

                # Set before to False as it only runs once
                before = False

            # Adjust the columns from "battery"
            if key == "battery":
                self.num += self.config[key]["general"]["number_of"]
                del cols[15:]
                del cols[9]
                del cols[7]
                del cols[5]
                del cols[:3]
                cols.insert(0, f"{key}/owner")
                max_num = max(self.config[key][key]["num"])
                cols = cols[:3] + self.repeat_columns(columns=cols[3:9], num=max_num) + cols[9:]
            # Adjust the columns from "psh"
            elif key == "psh":
                self.num += self.config[key]["general"]["number_of"]
                del cols[15:]
                del cols[9]
                del cols[7]
                del cols[5]
                del cols[:3]
                cols.insert(0, f"{key}/owner")
                max_num = max(self.config[key][key]["num"])
                cols = cols[:3] + self.repeat_columns(columns=cols[3:9], num=max_num) + cols[9:]
            # Adjust the columns from "hydrogen"
            elif key == "hydrogen":
                self.num += self.config[key]["general"]["number_of"]
                del cols[15:]
                del cols[9]
                del cols[7]
                del cols[5]
                del cols[:3]
                cols.insert(0, f"{key}/owner")
                max_num = max(self.config[key][key]["num"])
                cols = cols[:3] + self.repeat_columns(columns=cols[3:9], num=max_num) + cols[9:]
            else:
                raise NotImplementedError(
                    f"The configuration file contains a key word ('{key}') that has not been configured in "
                    "the class yet. Aborting scenario creation...")
            # Save columns
            columns[key] = cols

        # Define and insert all columns that come before the plants
        aftcols = self.make_list_from_nested_dict(self.config[key])
        aftkeys = ["mpc", "market_agent", "meter"]
        for aftkey in aftkeys:
            # Get all columns that match the key
            cols = [col for col in aftcols if aftkey == col.split("/", 1)[0]]
            # All columns that do not need to be adjusted
            if aftkey in ["mpc", "market_agent", "meter"]:
                pass
            else:
                raise NotImplementedError(
                    f"The configuration file contains a key word ('{aftkey}') that has not been configured in "
                    "the Sfhs class yet. Aborting scenario creation...")
            columns[aftkey] = cols

        # Combine all separate lists into one for the dataframe
        cols_df = []
        for _, cols in columns.items():
            cols_df += cols

        # Create dataframe with responding columns
        if self.method == 'config':
            # normal indexing
            self.df = pd.DataFrame(columns=cols_df)
        elif self.method == 'grid':
            # indexing matches the load sheet (all rows that are empty in owner as those are EVs and HPs)
            self.df = pd.DataFrame(index=self.agents.index, columns=cols_df)
        else:
            raise NotImplementedError(f"The method '{self.method}' has not been implemented yet. "
                                      f"Aborting scenario creation...")

        return self.df

    def add_battery(self):
        """
            Adds all battery storages
        """
        key = "battery"
        self.num_agents = self.config[f"{key}"]["general"]["number_of"]
        self.idx_start = len(self.df)

        # general
        self.fill_general(device=key)

        # pv
        self.fill_battery(device=key)

        # mpc
        self.fill_mpc(device=key)

        # market_agent
        self.fill_market_agent(device=key)

        # meter
        self.fill_meter(device=key)

    def add_psh(self):
        """
            Adds all psh storages
        """
        key = "psh"
        self.num_agents = self.config[f"{key}"]["general"]["number_of"]

        # general
        self.fill_general(device=key)

        # wind
        self.fill_psh(device=key)

        # mpc
        self.fill_mpc(device=key)

        # market_agent
        self.fill_market_agent(device=key)

        # meter
        self.fill_meter(device=key)

    def add_hydrogen(self):
        """
            Adds all hydrogen storages
        """
        key = "hydrogen"
        self.num_agents = self.config[f"{key}"]["general"]["number_of"]

        # general
        self.fill_general(device=key)

        # wind
        self.fill_hydrogen(device=key)

        # mpc
        self.fill_mpc(device=key)

        # market_agent
        self.fill_market_agent(device=key)

        # meter
        self.fill_meter(device=key)

    def fill_general(self, device: str):
        """
            Fills all general columns
        """
        key = "general"
        config = self.config[f"{device}"][f"{key}"]
        self.idx_start = len(self.df)
        self.idx_end = self.idx_start + self.num_agents

        # add the required rows
        for i in range(self.idx_start, self.idx_end):
            self.df.loc[i] = np.nan

        # general
        self.df.loc[self.idx_start:self.idx_end, f"{key}/agent_id"] = self._gen_new_ids(n=self.num_agents)

        # forecast
        self.df.loc[self.idx_start:self.idx_end, f"{key}/fcast_retraining_frequency"] = config["fcast_retraining_frequency"]

        # market participation
        self.df.loc[self.idx_start:self.idx_end, f"{key}/market_participant"] = self._gen_rand_bool_list(
            n=self.num_agents, share_ones=config["market_participant_share"])

    def fill_battery(self, device: str):
        """
            Fills all pv columns
        """
        key = "battery"
        config = self.config[f"{device}"][f"{key}"]

        # general
        self._add_general_info(key=key, config=config)

        # sizing
        max_num = max(config["num"])
        for num in range(max_num):
            # index list indicating ownership of device
            idx_list = self._get_idx_list(key=key, num=num, config=config)

            # sizing
            self.df.loc[self.idx_start:self.idx_end] = self._add_info_indexed(
                keys=[key, "sizing"], config=config["sizing"], df=self.df[self.idx_start:self.idx_end][:],
                idx_list=idx_list, appendix=f"_{num}")
            # postprocessing
            # power
            self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/power_{num}"] = self._calc_deviation(
                idx_list=idx_list, vals=self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/power_{num}"],
                distr=config["sizing"]["power_deviation"], method="absolute")
            self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/power_{num}"] = self._round_to_nth_digit(
                vals=self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/power_{num}"], n=self.n_digits)
            # capacity
            self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/capacity_{num}"] = self._calc_deviation(
                idx_list=idx_list, vals=self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/capacity_{num}"],
                distr=config["sizing"]["capacity_deviation"], method="absolute")
            self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/capacity_{num}"] = self._round_to_nth_digit(
                vals=self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/capacity_{num}"], n=self.n_digits)

        # quality
        self.df.loc[self.idx_start:self.idx_end, f"{key}/quality"] = config["quality"]

    def fill_psh(self, device: str):
        """
            Fills all psh columns
        """
        key = "psh"
        config = self.config[f"{device}"][f"{key}"]

        # general
        self._add_general_info(key=key, config=config)

        # sizing
        max_num = max(config["num"])
        for num in range(max_num):
            # index list indicating ownership of device
            idx_list = self._get_idx_list(key=key, num=num, config=config)

            # sizing
            self.df.loc[self.idx_start:self.idx_end] = self._add_info_indexed(
                keys=[key, "sizing"], config=config["sizing"], df=self.df[self.idx_start:self.idx_end][:],
                idx_list=idx_list, appendix=f"_{num}")
            # postprocessing
            # power
            self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/power_{num}"] = self._calc_deviation(
                idx_list=idx_list, vals=self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/power_{num}"],
                distr=config["sizing"]["power_deviation"], method="absolute")
            self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/power_{num}"] = self._round_to_nth_digit(
                vals=self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/power_{num}"], n=self.n_digits)
            # capacity
            self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/capacity_{num}"] = self._calc_deviation(
                idx_list=idx_list, vals=self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/capacity_{num}"],
                distr=config["sizing"]["capacity_deviation"], method="absolute")
            self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/capacity_{num}"] = self._round_to_nth_digit(
                vals=self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/capacity_{num}"], n=self.n_digits)

        # quality
        self.df.loc[self.idx_start:self.idx_end, f"{key}/quality"] = config["quality"]

    def fill_hydrogen(self, device: str):
        """
            Fills all hydrogen columns
        """

        key = "hydrogen"
        config = self.config[f"{device}"][f"{key}"]

        # general
        self._add_general_info(key=key, config=config)

        # sizing
        max_num = max(config["num"])
        for num in range(max_num):
            # index list indicating ownership of device
            idx_list = self._get_idx_list(key=key, num=num, config=config)

            # sizing
            self.df.loc[self.idx_start:self.idx_end] = self._add_info_indexed(
                keys=[key, "sizing"], config=config["sizing"], df=self.df[self.idx_start:self.idx_end][:],
                idx_list=idx_list, appendix=f"_{num}")
            # postprocessing
            # power
            self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/power_{num}"] = self._calc_deviation(
                idx_list=idx_list, vals=self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/power_{num}"],
                distr=config["sizing"]["power_deviation"], method="absolute")
            self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/power_{num}"] = self._round_to_nth_digit(
                vals=self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/power_{num}"], n=self.n_digits)
            # capacity
            self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/capacity_{num}"] = self._calc_deviation(
                idx_list=idx_list, vals=self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/capacity_{num}"],
                distr=config["sizing"]["capacity_deviation"], method="absolute")
            self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/capacity_{num}"] = self._round_to_nth_digit(
                vals=self.df.loc[self.idx_start:self.idx_end, f"{key}/sizing/capacity_{num}"], n=self.n_digits)

        # quality
        self.df.loc[self.idx_start:self.idx_end, f"{key}/quality"] = config["quality"]

    def fill_mpc(self, device: str):
        """
            Fills all battery columns
        """
        key = "mpc"
        config = self.config[f"{device}"][f"{key}"]

        # general
        self.df.loc[self.idx_start:self.idx_end] = self._add_info_simple(keys=[key], config=config,
                                                                         df=self.df[self.idx_start:self.idx_end][:])

    def fill_market_agent(self, device: str):
        """
            Fills all market agent columns
        """
        key = "market_agent"
        config = self.config[f"{device}"][f"{key}"]

        # general
        self.df.loc[self.idx_start:self.idx_end] = self._add_info_random(keys=[key], config=config,
                                                                         df=self.df[self.idx_start:self.idx_end][:])

    def fill_meter(self, device: str):
        """
            Fills all battery columns
        """
        key = "meter"
        config = self.config[f"{device}"][f"{key}"]

        # general
        self.df.loc[self.idx_start:self.idx_end] = self._add_info_simple(keys=[key], config=config,
                                                                         df=self.df[self.idx_start:self.idx_end][:])

    def _get_idx_list(self, key: str, num: int, config: dict) -> list:
        """creates the index list for the given run"""

        # Check who owns the device
        list_owner = np.multiply(np.array(self.df.loc[self.idx_start:self.idx_end, f"{key}/num"] - (1 + num) >= 0), 1)
        list_owner = [np.nan if elem == 0 else elem for elem in list_owner]

        # Return according index list based on ownership to choose plants
        idx_list = self._gen_idx_list_from_distr(n=self.num_agents, distr=config["sizing"]["distribution"])
        idx_list = np.multiply(list_owner, idx_list)

        return [int(elem) if not np.isnan(elem) else np.nan for elem in idx_list]

    def _add_general_info(self, key: str, config: dict) -> None:

        # fields that exist for all plants
        self.df.loc[self.idx_start:self.idx_end, f"{key}/owner"] = 1
        self.df.loc[self.idx_start:self.idx_end, f"{key}/num"] = self._gen_dep_num_list(
            owner_list=self.df.loc[self.idx_start:self.idx_end, f"{key}/owner"], distr=config["num"])
        self.df.loc[self.idx_start:self.idx_end, f"{key}/num"] *= self.df[f"{key}/owner"]
        self.df.loc[self.idx_start:self.idx_end, f"{key}/has_submeter"] = config["has_submeter"]

    def _add_general_info_dependent(self, key: str, config: dict) -> None:

        # fields that exist for all plants
        self.df.loc[self.idx_start:self.idx_end, f"{key}/owner"] = 1
        self.df.loc[self.idx_start:self.idx_end, f"{key}/num"] = self._gen_dep_num_list(
            owner_list=self.df.loc[self.idx_start:self.idx_end, f"{key}/owner"], distr=config["num"])
        self.df.loc[self.idx_start:self.idx_end, f"{key}/owner"] = \
            (self.df.loc[self.idx_start:self.idx_end, f"{key}/num"] > 0) * 1
        self.df.loc[self.idx_start:self.idx_end, f"{key}/has_submeter"] = config["has_submeter"]

    def _add_general_info_bat(self, key: str):

        # find all potential owners of a battery system dependent on type
        # note: this setup considers the different dependencies for each type and loops through each separately
        agent_types = self.config["general"]["parameters"]["type"]
        list_owner = [0] * self.num
        list_num = [0] * self.num
        for idx, agent_type in enumerate(agent_types):
            # get all agents of given type
            list_type = list(self.df["general/parameters/type"] == agent_type)
            plants = self.config[f"{key}"]["share_dependent_on"][idx] + ["inflexible_load"]
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

            self.df.loc[self.idx_start:self.idx_end, f"{key}/owner"] = list_owner
            self.df.loc[self.idx_start:self.idx_end, f"{key}/num"] = list_num
            self.df.loc[self.idx_start:self.idx_end, f"{key}/has_submeter"] = str(self.config[f"{key}"]["has_submeter"])


# Playground
if __name__ == "__main__":
    print('Not working anymore ;) Call the example function to create a scenario instead.')
