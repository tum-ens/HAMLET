__author__ = "TUM-Doepfert"
__credits__ = "jiahechu"
__license__ = ""
__maintainer__ = "TUM-Doepfert"
__email__ = "markus.doepfert@tum.de"

import copy
import time
import datetime
import shutil
import os
import string
import json
import math
import random

from pandas import DataFrame
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
from hplib import hplib
import warnings
import re
from hamlet import functions as f
from hamlet import constants as c
from pathlib import Path
import ast

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'


class Agents:
    """
    A class used to generate all agents, including different types such as single-family homes (sfh), multi-family homes (mfh),
    commercial, industrial, and storage. It provides methods to create agents from different sources such as configuration
    files, grid files, or Excel files, and to organize the agents' information into directories and files.

    Agents are key elements in a simulation model, representing various entities that can produce, consume, or store energy.
    Different subtypes of agents can be customized to model specific behaviors and characteristics.

    Attributes
    ----------
    agent_id : str
        ID of the agent.
    config_path : str
        Path to the configuration file.
    input_path : str
        Path to the input data.
    scenario_path : str
        Path to the scenario data.
    config_root : str
        Root path for the configuration, defaults to config_path if not provided.
    setup : dict
        Setup configuration loaded from a YAML file.
    grid : dict
        Grid file representing electricity grid, loaded when needed.
    config : dict
        Configuration for agents loaded from a YAML file.
    excel : pd.ExcelFile
        Excel file containing agents' information.
    agents : dict
        Information of all agents.
    id : str
        Information of the current agent id.
    account : dict
        Information of the current account.
    types : dict
        Dictionary containing the available types of agents.
    plants : dict
        Dictionary containing the available types of plants.
    ctsp : dict
        Available types of ctsp, obtained from the input data folder.
    industry : dict
        Available types of industry, obtained from the input data folder.

    Public Methods
    -------
    __init__(config_path, input_path, scenario_path, config_root=None) -> None:
        Initializes the Agents class with given paths and loads required files.

    create_agents_file_from_config(overwrite=False):
        Creates an Excel file with agents' data from the config file.

    create_agents_file_from_grid(grid='electricity.xlsx', fill_from_config=False, overwrite=False):
        Creates an Excel file with agents' data from the electricity grid file.

    create_agents_from_file(id_check=False) -> None:
        Creates the agents from an Excel file, checking for non-unique IDs if requested.

    create_agents_dict_from_file(df: pd.DataFrame) -> dict:
        Creates a dictionary of agents from a given DataFrame.

    create_agents_from_dict(agents: dict, agent_type: str):
        Creates agent folders and files from a given dictionary for a specific agent type.
    """

    def __init__(self, config_path: str, input_path: str, scenario_path: str, config_root: str = None):

        # Set paths and names
        self.config_path = config_path
        self.config_root = config_root if config_root is not None else config_path
        self.input_path = input_path
        self.scenario_path = scenario_path

        # Load setup plus configuration and/or agent file
        self.setup = f.load_file(path=os.path.join(self.config_root, 'config_setup.yaml'))
        self.grid = None  # grid file only required if agents are created from grid file
        self.config = None
        self.excel = None
        try:
            self.config = f.load_file(path=os.path.join(self.config_path, 'config_agents.yaml'))
        except FileNotFoundError:
            try:
                self.excel = f.load_file(path=os.path.join(self.config_path, 'agents.xlsx'))
            except FileNotFoundError as exc:
                raise FileNotFoundError("Configuration requires either a yaml or an xlsx file to create agents.") \
                    from exc

        # Information about the agents
        self.agents = None  # information of all agents
        self.id = None  # information of the current agent id
        self.account = None  # information of the current account

        # Available types of agents
        from hamlet.creator.agents.sfh import Sfh
        from hamlet.creator.agents.mfh import Mfh
        from hamlet.creator.agents.ctsp import Ctsp
        from hamlet.creator.agents.industry import Industry
        from hamlet.creator.agents.producer import Producer
        from hamlet.creator.agents.storage import Storage
        from hamlet.creator.agents.aggregator import Aggregator
        self.types = {
            c.A_SFH: Sfh,
            c.A_MFH: Mfh,
            c.A_CTSP: Ctsp,
            c.A_INDUSTRY: Industry,
            c.A_PRODUCER: Producer,
            c.A_STORAGE: Storage,
            c.A_AGGREGATOR: Aggregator,
        }
        # Available types of plants
        self.plants = {
            c.P_INFLEXIBLE_LOAD: {
            },
            c.P_FLEXIBLE_LOAD: {
            },
            c.P_HEAT: {
                'func_ts': self.__make_timeseries_heat,
            },
            c.P_DHW: {
            },
            c.P_PV: {
                'specs': self.__timeseries_from_specs_pv,
            },
            c.P_WIND: {
                'specs': self.__timeseries_from_specs_wind,
            },
            c.P_FIXED_GEN: {
            },
            c.P_HP: {
                'specs': self.__timeseries_from_specs_hp,
            },
            c.P_EV: {
            },
            c.P_BATTERY: {
            },
            c.P_PSH: {
            },
            c.P_HYDROGEN: {
            },
            c.P_HEAT_STORAGE: {
            },
        }
        # Available types of ctsp (obtained from the input data folder)
        try:
            self.ctsp = self.__get_types(
                path=os.path.join(self.input_path, 'agents', c.A_CTSP, c.P_INFLEXIBLE_LOAD), idx=0, sep='_')
        except FileNotFoundError:
            self.ctsp = []
        # Available types of industry (obtained from the input data folder)
        try:
            self.industry = self.__get_types(
                path=os.path.join(self.input_path, 'agents', c.A_INDUSTRY, c.P_INFLEXIBLE_LOAD), idx=0, sep='_')
        except FileNotFoundError:
            self.industry = []

    def create_agents_file_from_config(self, overwrite: bool = False):
        """Creates the Excel file from the config file

        Args:
            overwrite (bool, optional): Flag to determine if the existing file should be overwritten. Default is False.
        """

        # Dictionary to store the dataframes corresponding to different agent types
        dict_agents = {}

        # Loop through the configuration items to create dataframes for each agent type
        for key, config in self.config.items():
            if key in self.types:  # Check if the agent type is valid
                # Instantiate the agent class for the specific type
                agents = self.types[key](input_path=self.input_path,
                                         config=config,
                                         config_path=self.config_path,
                                         scenario_path=self.scenario_path,
                                         config_root=self.config_root, )

                # Create a dataframe from the config and store it in the dictionary
                dict_agents[key] = agents.create_df_from_config(dict_agents=dict_agents)
            else:
                # Raise an error if an unsupported agent type is encountered
                raise KeyError(f"Agent type '{key}' is not available.")

        # Check if the file exists and the overwrite flag is turned off
        if os.path.exists(f"{self.config_path}/agents.xlsx") and not overwrite:
            raise Warning("File already exists and overwrite is turned off.")
        else:
            # Attempt to write the dataframes to Excel
            try:
                with pd.ExcelWriter(f"{self.config_path}/agents.xlsx", engine="xlsxwriter") as writer:
                    # Loop through the dictionary and write each dataframe to a separate worksheet
                    for key, df in dict_agents.items():
                        if df is not None:
                            df.to_excel(writer, sheet_name=key)
            except PermissionError:
                # Raise an error if the file is currently open and cannot be accessed
                raise PermissionError("The file 'agents.xlsx' needs to be closed before running this function.")

    def create_agents_file_from_grid(self, grid: str = 'electricity.xlsx', fill_from_config: bool = False,
                                     overwrite: bool = False):
        """Creates the Excel file from the electricity grid file

        Args:
            grid (str, optional): The path to the grid file. Default is 'electricity.xlsx'.
            fill_from_config (bool, optional): Flag to determine if missing information should be filled from the config file. Default is False.
            overwrite (bool, optional): Flag to determine if the existing file should be overwritten. Default is False.
        """

        # Dictionaries to store the dataframes for grids and agents
        dict_grids = {}
        dict_agents = {}

        # Load the grid file if it's not already loaded
        self.grid = f.load_file(path=os.path.join(self.config_path, grid)) if self.grid is None \
            else FileNotFoundError("Grid file not found.")

        # Iterate through sheets in the grid file and load them as dataframes
        for sheet in self.grid.sheet_names:
            # Load the dataframe from the sheet
            dict_grids[sheet] = self.grid.parse(sheet, index_col=0)

            # Extract additional information from the 'description' column and add as new columns
            try:
                dict_grids[sheet] = self.__add_info_from_col(df=dict_grids[sheet], col='description', drop=True)
            except (AttributeError, KeyError):
                pass

        # Create dataframes for each type of agent using their specific class
        for key, _ in self.config.items():
            if key in self.types:

                if key == c.A_AGGREGATOR:

                    agents = self.types[c.A_AGGREGATOR](input_path=self.input_path,
                                                        config=self.config[key],
                                                        config_path=self.config_path,
                                                        scenario_path=self.scenario_path,
                                                        config_root=self.config_root, )

                    # Create a dataframe from the config and store it in the dictionary
                    dict_agents[c.A_AGGREGATOR] = agents.create_df_from_config(dict_agents=dict_agents)

                else:

                    dict_agents[key] = self.types[key](config=self.config[key],
                                                       input_path=self.input_path,
                                                       config_path=self.config_path,
                                                       scenario_path=self.scenario_path,
                                                       config_root=self.config_root). \
                        create_df_from_grid(grid=dict_grids, fill_from_config=fill_from_config).reset_index(drop=True)
            else:
                raise Warning(f'Agent type {key} not available. Check self.types for the available types.')

        # Check if the file exists and the overwrite flag is turned off
        if os.path.exists(f"{self.config_path}/agents.xlsx") and not overwrite:
            raise Warning("File already exists and overwrite is turned off.")
        else:
            # Attempt to write the dataframes to Excel
            try:
                with pd.ExcelWriter(f"{self.config_path}/agents.xlsx", engine="xlsxwriter") as writer:
                    # Loop through the dictionary and write each dataframe to a separate worksheet
                    for key, df in dict_agents.items():
                        if df is not None:
                            df.to_excel(writer, sheet_name=key)
            except PermissionError:
                # Raise an error if the file is currently open and cannot be accessed
                raise PermissionError("The file 'agents.xlsx' needs to be closed before running this function.")

    def create_agents_from_file(self, id_check=False) -> None:
        """
        Creates the agents from an Excel file. If the file does not exist, an error is raised.

        Args:
            id_check (bool, optional): If set to True, will check for non-unique IDs and raise an error.
        """

        # Load the Excel file if it's not already loaded, or raise an error if it does not exist
        self.excel = f.load_file(f"{self.config_path}/agents.xlsx") if self.excel is None \
            else FileNotFoundError("Agent xlsx file does not exist. Call 'create_agents_file_from_config()' or "
                                   "'create_agents_file_from_grid()' first to create the file from the config or "
                                   "grid file.")

        # Optional step to check if there are non-unique IDs, and if so, raise an error (currently not implemented)
        if id_check:
            agent_ids = list()
            for sheet in self.excel.sheet_names:
                df = self.excel.parse(sheet, index_col=0)
                agent_ids += df[f"{c.K_GENERAL}/agent_id"].to_list()
            # Print the list of agent IDs
            print(agent_ids)
            # Print the non-unique items (IDs)
            print([item for item, count in Counter(agent_ids).items() if count > 1])
            # Raise an error as this functionality is not yet implemented
            raise NotImplementedError('Not yet implemented...')

        agents_account_data = {}

        # Loop through the sheets in the Excel file to create agents
        for sheet in self.excel.sheet_names:
            # Load the dataframe from the sheet
            df = self.excel.parse(sheet, index_col=0)

            # Convert the dataframe to a dictionary of agents
            self.agents = self.create_agents_dict_from_file(df=df)

            # save the dataframes for every agent_type
            agents_account_data[sheet] = copy.deepcopy(self.agents)

            # Create folders and files for each agent based on the agent dictionary
            self.create_agents_from_dict(agents=self.agents, agent_type=sheet, agents_account_data=agents_account_data)

    def create_agents_dict_from_file(self, df: pd.DataFrame) -> dict:
        """
        Creates a dictionary from a dataframe and extracts the account information for each agent of the given type.

        Args:
            df (pd.DataFrame): A DataFrame containing the agent information.

        Returns:
            dict: A dictionary containing the agent information organized by IDs, with plants grouped in a nested subdictionary.
        """

        # Identify the columns in the DataFrame that end with '_id', these are used as keys to identify the agents
        ids = [col for col in df.columns if col.endswith('_id')]

        # Convert the DataFrame into a nested dictionary using the identified IDs as keys
        accounts = self.__dataframe_to_dict(df=df, key_column=ids)

        # Iterate through the dictionary and group plant-related keys under a nested 'plants' subdictionary
        for key, val in accounts.items():
            accounts[key] = self.__group_keys_under_nested_dict(input_dict=val, output_name='plants',
                                                                keys=tuple(self.plants.keys()))

        return accounts

    def create_agents_from_dict(self, agents: dict, agent_type: str, agents_account_data: dict):
        """
        Creates agent folders and files from a given dictionary for a specific agent type.

        Args:
            agents (dict): A dictionary containing the agent information.
            agent_type (str): The type of agent to create (e.g. sfh, mfh, etc.).
        """

        # Iterate through each agent in the given dictionary
        for agent, account in agents.items():
            # Store the current agent's ID and account information
            self.id = agent
            self.account = account

            # Define the path where the agent's files will be stored
            path = os.path.join(self.scenario_path, 'agents', agent_type, agent)

            # Create a folder for the current agent
            f.create_folder(path)

            if agent_type == c.A_AGGREGATOR:
                # Call the internal method to create the necessary data for the aggregator (e.g. plants, meters, etc.)
                account["plants"], plants, meters, timeseries, socs, specs, setpoints, fcasts = (
                    self._create_plants_for_aggregator(account=account, agent_type=agent_type, agents_account_data=agents_account_data))

            else:

                # Call the internal method to create the necessary data for the agent (e.g. plants, meters, etc.)
                account[c.K_PLANTS], plants, meters, timeseries, socs, specs, setpoints, fcasts = (
                    self._create_plants_for_agent(account=account, agent_type=agent_type))

            # Organize the created data into a dictionary
            data = {
                "account.json": account,
                "plants.json": plants,
                "meters.ft": meters,
                "timeseries.ft": timeseries,
                "socs.ft": socs,
                "specs.json": specs,
                "setpoints.ft": setpoints,
                "forecasts.ft": fcasts,
            }

            # Call the internal method to create the agent using the organized data
            self._create_agent(path, data)

    def _create_plants_for_agent(self, account: dict, agent_type: str) -> Tuple:
        """Creates the plants for the agent, including plant IDs, meters, time series, state of charges (SOCs), and specs.

        Args:
            plants (dict): Dictionary containing information about the plants.
            agent_type (str): Type of agent for which the plants are being created.

        Returns:
            tuple: Contains the following information:
                - Plant IDs
                - Dictionary with plant information
                - DataFrame with meter values
                - DataFrame with time series for each plant
                - DataFrame with SOCs
                - Dictionary with plant specs.
        """
        # Retrieve info from account
        plants = account[c.K_PLANTS]

        # Set time ranges
        # Length of the training period in days
        train_period = self.__find_max_train_period(plants=plants)
        # Length of the forecasting period in seconds
        fcast_period = account[c.K_EMS]['fcasts']['horizon']
        # Start of the simulation in UTC
        start = self.setup['time']['start'].replace(tzinfo=datetime.timezone.utc)
        # Start of the forecasting period in UTC
        start_fcast_train = start - train_period
        # End of the simulation in UTC (one day added to ensure no foreward forecasting issues)
        # Make sure that fractions are properly read
        if isinstance(self.setup['time']['duration'], str):
            numerator, denominator = map(int, self.setup['time']['duration'].split('/'))
            self.setup['time']['duration'] = numerator / denominator
        # TODO: Make the end of the simulation the end of the timetable instead of guessing to add one day
        end = start + datetime.timedelta(days=self.setup['time']['duration'] + 1)
        # End of the first forecasting period in UTC
        end_fcast_period = start + datetime.timedelta(seconds=fcast_period)
        # Time range for the simulation
        timerange = pd.date_range(start=start, end=end, freq=f"{int(self.setup['time']['timestep'])}S")[:-1]
        # Time range for the forecasting training period
        timerange_fcast_train = pd.date_range(start=start_fcast_train, end=end,
                                              freq=f"{int(self.setup['time']['timestep'])}S")[:-1]
        # Time range for the first forecasting period (first simulation period)
        timerange_fcast_period = pd.date_range(start=start, end=end_fcast_period,
                                              freq=f"{int(self.setup['time']['timestep'])}S")[:-1]

        # Initialize data structures
        # Time series of each plant
        timeseries = pd.DataFrame(index=timerange_fcast_train)
        timeseries.index.name = c.TC_TIMESTAMP
        # Meter values
        meters = pd.DataFrame(index=timerange)
        meters.index.name = c.TC_TIMESTAMP
        # SOCs
        socs = pd.DataFrame(index=timerange)
        socs.index.name = c.TC_TIMESTAMP
        # Setpoints
        setpoints = pd.DataFrame(index=timerange_fcast_period)
        setpoints.index.name = c.TC_TIMESTAMP
        # Forecasts
        forecasts = pd.DataFrame(index=timerange_fcast_period)
        forecasts.index.name = c.TC_TIMESTAMP
        # All plant information
        plants_dict = {}
        # Single plant information
        plant_dict = {}
        # Plant IDs
        plants_ids = []
        # Plant specs
        specs_plants = {}

        # Loop through the provided plants
        for plant, info in plants.items():
            # Skip if plant information is unavailable
            if info is None:
                continue

            # Determine the number of plants for this agent and type, skip if invalid
            try:
                num_plants = int(info["owner"] * info["num"])
            except ValueError:
                continue

            # Add general information to the plant dictionary
            plant_dict["type"] = plant

            # Process each device individually (agent can have more than one of each type)
            for num_plant in range(num_plants):
                # Generate and store plant ID
                plant_id = self._gen_new_ids()
                plant_dict["id"] = plant_id
                plants_ids += [plant_id]

                # Add meter data
                energy_types = c.COMP_MAP[plant].keys()
                for key in energy_types:
                    col_id = f'{plant_id}_{plant}_{key}'
                    meters[col_id] = self.__init_vals(df=meters)

                # Add setpoints
                setpoints[plant_id] = self.__init_vals(df=setpoints)  # TODO: Ponder if rows need to be added here

                # Add and process additional plant information
                plant_dict.update(info)
                plant_dict = self.__clean_indexed_info(data=plant_dict, key='sizing', index=num_plant)

                # Add time series if applicable
                try:
                    ts, specs = self.__make_timeseries(
                        file_path=os.path.join(self.input_path, 'agents', agent_type, plant,
                                               plant_dict['sizing']['file']),
                        plant_id=plant_id, plant_dict=plant_dict,
                        delta=pd.Timedelta(f"{(timeseries.index[1] - timeseries.index[0]) / 3}S"))
                    timeseries = timeseries.join(ts)
                except KeyError:
                    specs = None

                # Add state of charge (SOC) if applicable
                try:
                    socs[plant_id] = self.__init_vals(df=socs,
                                                      vals=round(info["sizing"]["soc"] * info["sizing"]["capacity"]))
                except KeyError:
                    pass

                # Add specs file if applicable
                if specs:
                    specs_plants[plant_id] = specs

                # Add plant information to the main dictionary
                plants_dict[plant_id] = plant_dict

            # Reset for the next entry
            plant_dict = {}

        # Add forecast columns
        forecasts = pd.DataFrame(columns=timeseries.columns, index=forecasts.index)

        return plants_ids, plants_dict, meters, timeseries, socs, specs_plants, setpoints, forecasts


    def _create_plants_for_aggregator(self, account: dict, agent_type: str, agents_account_data: dict) -> Tuple:
        """Creates the plants for the aggregator, including plant IDs, meters, time series, state of charges (SOCs), and specs.

        Args:
            plants (dict): Dictionary containing information about the plants.
            agent_type (str): Type of agent for which the plants are being created.

        Returns:
            tuple: Contains the following information:
                - Plant IDs
                - Dictionary with plant information
                - DataFrame with meter values
                - DataFrame with time series for each plant
                - DataFrame with SOCs
                - Dictionary with plant specs.

        """
        # Set time ranges
        # Length of the training period in days

        train_period = datetime.timedelta(days=0)

        for agent_type in agents_account_data:
            for agent_id, account_agent in agents_account_data[agent_type].items():
                if agent_id in ast.literal_eval(account['general']['aggregated_agents']):
                    plants = account_agent["plants"]
                    train_period_agent = self.__find_max_train_period(plants=plants)
                    train_period = max(train_period, train_period_agent)

        #train_period = self.__find_max_train_period(plants=plants)
        # Length of the forecasting period in seconds
        fcast_period = account['ems']['fcasts']['horizon']
        # Start of the simulation in UTC
        start = self.setup['time']['start'].replace(tzinfo=datetime.timezone.utc)
        # Start of the forecasting period in UTC
        start_fcast_train = start - train_period
        # End of the simulation in UTC (one day added to ensure no foreward forecasting issues)
        end = start + datetime.timedelta(days=self.setup['time']['duration'] + 1)
        # End of the first forecasting period in UTC
        end_fcast_period = start + datetime.timedelta(seconds=fcast_period)
        # Time range for the simulation
        timerange = pd.date_range(start=start, end=end, freq=f"{int(self.setup['time']['timestep'])}S")[:-1]
        # Time range for the forecasting training period
        timerange_fcast_train = pd.date_range(start=start_fcast_train, end=end,
                                              freq=f"{int(self.setup['time']['timestep'])}S")[:-1]
        # Time range for the first forecasting period (first simulation period)
        timerange_fcast_period = pd.date_range(start=start, end=end_fcast_period,
                                               freq=f"{int(self.setup['time']['timestep'])}S")[:-1]
        # Initialize data structures
        # Time series of each plant
        timeseries = pd.DataFrame(index=timerange_fcast_train)
        timeseries.index.name = c.TC_TIMESTAMP
        # Meter values
        meters = pd.DataFrame(index=timerange)
        meters.index.name = c.TC_TIMESTAMP
        # SOCs
        socs = pd.DataFrame(index=timerange)
        socs.index.name = c.TC_TIMESTAMP
        # Setpoints
        setpoints = pd.DataFrame(index=timerange_fcast_period)
        setpoints.index.name = c.TC_TIMESTAMP
        # Forecasts
        forecasts = pd.DataFrame(index=timerange_fcast_period)
        forecasts.index.name = c.TC_TIMESTAMP


        # All plant information
        plants_dict = {}
        # Plant IDs
        plants_ids = []
        # Plant specs
        specs_plants = {}

        # loop over all aggregated agents
        for agent in ast.literal_eval(account['general']['aggregated_agents']):

            # Specify the directory path
            path_to_base_directory = os.path.join(self.scenario_path, 'agents')
            base_directory = Path(path_to_base_directory)

            # Specify the target folder name
            target_folder_name = agent

            # Initialize variables to store file contents
            account_data = plants_data = meters_data = timeseries_data = socs_data = specs_data = setpoints_data = forecasts_data = None

            # Search for the target folder within the base directory
            for root, dirs, files in os.walk(base_directory):
                if target_folder_name in dirs:
                    target_folder_path = Path(root) / target_folder_name
                    break
            else:
                print(f"Target folder '{target_folder_name}' not found in '{base_directory}'")
                target_folder_path = None

            # If the target folder is found, open and store the contents of each specified file
            if target_folder_path:

                # load existing data
                account_data = f.load_file(path=os.path.join(target_folder_path, 'account.json'))
                plants_data = f.load_file(path=os.path.join(target_folder_path, 'plants.json'))
                specs_data = f.load_file(path=os.path.join(target_folder_path, 'specs.json'))
                meters_data = f.load_file(path=os.path.join(target_folder_path, 'meters.ft'), df='pandas')
                timeseries_data = f.load_file(path=os.path.join(target_folder_path, 'timeseries.ft'), df='pandas')
                socs_data = f.load_file(path=os.path.join(target_folder_path, 'socs.ft'), df='pandas')
                setpoints_data = f.load_file(path=os.path.join(target_folder_path, 'setpoints.ft'), df='pandas')

                # Add plant_ids
                plants_ids.extend(account_data["plants"])

                # Add plants
                if plants_data:
                    for plant in plants_data:
                        plants_dict[plant] = plants_data[plant]

                # Add meter data
                try:
                    for plant in meters_data.columns:
                        if plant != 'timestamp':
                            meters_data = meters_data.set_index(meters.index)
                            meters[plant] = meters_data[plant]  # TODO: Ponder if rows need to be added here

                except KeyError:
                    pass

                # Add setpoints
                try:
                    for plant in setpoints_data.columns:
                        if plant != 'timestamp':
                            setpoints_data = setpoints_data.set_index(setpoints.index)
                            setpoints[plant] = setpoints_data[plant] # TODO: Ponder if rows need to be added here
                except KeyError:
                    pass

                # Add specs
                if specs_data:
                    for plant in specs_data:
                        specs_plants[plant] = specs_data[plant]

                # Add state of scos if applicable
                try:
                    for plant in socs_data.columns:
                        if plant != 'timestamp':
                            socs_data = socs_data.set_index(socs.index)
                            socs.loc[:, plant] = socs_data[plant]

                except KeyError:
                    pass

                # Add state of timeseries if applicable
                try:
                    for plant in timeseries_data.columns:
                        if plant != 'timestamp':
                            timeseries_data = timeseries_data.set_index(timeseries.index)
                            timeseries[plant] = timeseries_data[plant]

                except KeyError:
                    pass

        # Add forecast columns
        forecasts = pd.DataFrame(columns=timeseries.columns, index=forecasts.index)

        return plants_ids, plants_dict, meters, timeseries, socs, specs_plants, setpoints, forecasts


    @staticmethod
    def _create_agent(path: str, data: dict) -> None:
        """Creates the agent files in the specified path

        Args:
            path (str): Path to the folder in which the agent files are to be stored
            data (dict): Dictionary containing the data for the agent files and the names of the files

        Returns:
            None

        """

        # Create the agent files
        for key, value in data.items():
            f.save_file(path=os.path.join(path, key), data=value)

    def __clean_indexed_info(self, data: dict, key: str, index: int, separator: str = '_') -> dict:
        """
        Retrieve indexed information from a nested dictionary and remove the index suffixes.

        Args:
            data (dict): The input data, which is a nested dictionary.
            key (str): A key specifying the key with the indexed values.
            index (int): The index to remove from the list.
            separator (str, optional): The separator used for index suffixes. Default is '_'.

        Returns:
            dict: The updated data with the index suffixes removed.

        Example:
            data = {'fcast': {'arima': {'order': '[1, 0, 0]'},
                              'average': {'days': 2, 'offset': 1},
                              'cnn': {'days': 90, 'features': "['temp', 'time']"},
                              'method': 'average'},
                    'num': 1,
                    'owner': 1,
                    'sizing': {'demand_0': 3296000, 'file_0': 'hh_3296_0.csv',
                               'demand_1': 2296000, 'file_1': 'hh_2296_0.csv'}}

            modified_data = clean_indexed_info(data, 'sizing', 0)
            print(modified_data)
            # Output: {'fcast': {'arima': {'order': '[1, 0, 0]'},
            #                   'average': {'days': 2, 'offset': 1},
            #                   'cnn': {'days': 90, 'features': "['temp', 'time']"},
            #                   'method': 'average'},
            #           'num': 1,
            #           'owner': 1,
            #           'sizing': {'demand': 3296000, 'file': 'hh_3296_0.csv'}}
        """
        if key in data:
            keys_to_remove = []
            keys_to_modify = {}

            for k, v in data[key].items():
                if k.endswith(f"{separator}{index}"):
                    # Remove the index from the key and add it to the dictionary
                    new_key = k.replace(f"{separator}{index}", "")
                    keys_to_modify[new_key] = v
                    keys_to_remove.append(k)
                elif bool(re.search(f"{separator}\d+$", k)):
                    # Remove the entry from the dictionary if it ends with the separator but a different index
                    keys_to_remove.append(k)

            for k in keys_to_remove:
                data[key].pop(k)

            data[key].update(keys_to_modify)
        else:
            # Recursively call the function for nested dictionaries
            for k, v in data.items():
                if isinstance(v, dict):
                    data[k] = self.__clean_indexed_info(v, key, index, separator)

        return data

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
        weather = f.load_file(weather_path)
        weather = weather[weather[c.TC_TIMESTAMP] == weather[c.TC_TIMESTEP]]  # remove forcasting data

        # convert time data to datetime (use utc time overall in pvlib)
        time = pd.DatetimeIndex(pd.to_datetime(weather[c.TC_TIMESTAMP], unit='s', utc=True))
        weather.index = time
        weather.index.name = 'utc_time'

        # adjust temperature data
        weather.rename(columns={c.TC_TEMPERATURE: 'temp_air'}, inplace=True)  # rename to pvlib format
        weather['temp_air'] += c.KELVIN_TO_CELSIUS  # convert unit to celsius

        # get solar position
        # test data find in https://www.suncalc.org/#/48.1364,11.5786,15/2022.02.15/16:21/1/3
        solpos = pvlib.solarposition.get_solarposition(
            time=time,
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            temperature=weather['temp_air'],
            pressure=weather[c.TC_PRESSURE],
        )

        # calculate dni with solar position
        weather.loc[:, c.TC_DNI] = (weather[c.TC_GHI] - weather[c.TC_DHI]) / np.cos(solpos['zenith'])

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
        location = self.setup['location']
        location = (location['latitude'], location['longitude'], location['name'], location['altitude'])

        # get plant orientation
        orientation = (plant['sizing']['orientation'], plant['sizing']['angle'])

        # get weather path
        weather_path = os.path.join(self.input_path, 'general', 'weather',
                                    self.setup['location']['weather'])

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
        nominal_power = specs['module']['Impo'] * specs['module']['Vmpo']

        # set time index to origin timestamp
        power.index = weather[c.TC_TIMESTAMP]
        power.index.name = c.TC_TIMESTAMP

        # rename and round data column
        power.rename(c.ET_ELECTRICITY, inplace=True)
        power = power.to_frame()

        # calculate and round power
        power = power / nominal_power * plant['sizing']['power']
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
        weather = f.load_file(weather_path)
        weather = weather[weather[c.TC_TIMESTAMP] == weather[c.TC_TIMESTEP]]  # remove forcasting data

        # convert time data to datetime
        time = pd.DatetimeIndex(pd.to_datetime(weather[c.TC_TIMESTAMP], unit='s', utc=True))
        weather.index = time
        weather.index.name = None

        # delete unnecessary columns and rename
        weather = weather[[c.TC_TIMESTAMP, c.TC_TEMPERATURE, c.TC_TEMPERATURE_FEELS_LIKE, c.TC_TEMPERATURE_MAX,
                           c.TC_TEMPERATURE_MIN, c.TC_PRESSURE, c.TC_HUMIDITY, c.TC_WIND_SPEED, c.TC_WIND_DIRECTION]]
        weather.rename(columns={c.TC_TEMPERATURE: 'temperature'}, inplace=True)

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
        weather_path = os.path.join(self.input_path, 'general', 'weather',
                                    self.setup['location']['weather'])

        # get weather data
        weather = self.__adjust_weather_data_for_wind(weather_path=weather_path)
        specs_wind = specs.copy()

        # copy specs parameter
        specs_wind = copy.deepcopy(specs)

        # get nominal power
        nominal_power = specs_wind['nominal_power']

        # convert power curve to dataframe
        specs_wind['power_curve'] = pd.DataFrame(data={
            "value": specs_wind['power_curve'],
            "wind_speed": specs_wind['wind_speed']})

        # convert power coefficient curve to dataframe
        specs_wind['power_coefficient_curve'] = pd.DataFrame(data={
            "value": specs_wind['power_coefficient_curve'],
            "wind_speed": specs_wind['wind_speed']})

        # generate a WindTurbine object from data
        turbine = WindTurbine(**specs_wind)

        # calculate turbine model
        mc_turbine = ModelChain(turbine).run_model(weather)

        # get output power
        power = mc_turbine.power_output

        # set time index to origin timestamp
        power.index = weather[c.TC_TIMESTAMP].unstack(level=0).values
        power.index.name = c.TC_TIMESTAMP

        # rename data column
        power.rename(c.ET_ELECTRICITY, inplace=True)
        power = power.to_frame()

        # calculate and round power
        power = power / nominal_power * plant['sizing']['power']
        power = power.round().astype(int)

        return power

    def __timeseries_from_specs_hp(self, specs: dict, plant: dict):
        """Creates a time series for hp config spec file using hplib model.

        Args:
            specs (dict): A dictionary of hp hardware config.
            plant (dict): A dictionary of hp plant information.

        Returns:
            power: A Pandas Dataframe object representing the time series data for the hp plant.

        """

        # obtain the model type of the heat pump predefined in agents.xlsx
        hp_model = specs['model']
        hp_type = specs['type']
        dtemp_transfer_loss = 5  # delta T of inlet and outlet in the secondary side is fixed at 5K
        # TODO: Currently temperature levels are fixed, but should be read from the config file
        supply_temp = {c.P_HEAT: 40, c.P_DHW: 55}

        # get weather path
        weather_path = os.path.join(self.input_path, 'general', 'weather',
                                    self.setup['location']['weather'])
        weather = f.load_file(weather_path)
        weather = weather[weather[c.TC_TIMESTAMP] == weather[c.TC_TIMESTEP]]  # remove forcasting data

        # convert time data to datetime
        time_index = pd.DatetimeIndex(pd.to_datetime(weather[c.TC_TIMESTAMP], unit='s', utc=True))
        weather.index = time_index
        weather.index.name = 'utc_time'

        # abstract the input data needed for hp simulation from specs and plant
        t_amb = weather[c.TC_TEMPERATURE] + c.KELVIN_TO_CELSIUS  # convert unit to celsius
        t_brine = self.calc_brine_temp(t_amb)

        # obtain the parameters of heat pumps based on reference/set temperature and thermal power
        parameters = hplib.get_parameters(model=hp_model)

        # create hp system
        system = hplib.HeatPump(parameters)

        # simulate the heat pump and attain the results
        # mode=1 -> heating, mode=2 -> cooling
        # delta T of inlet and outlet in the secondary side is fixed at dtemp_transfer_loss
        if hp_type == "Outdoor Air/Water":
            t_in = t_amb
        else:
            t_in = t_brine
        res_sh = system.simulate(t_in_primary=np.array(t_in),
                                 t_in_secondary=supply_temp[c.P_HEAT] - dtemp_transfer_loss,
                                 t_amb=np.array(t_amb), mode=1)
        res_dhw = system.simulate(t_in_primary=np.array(t_in),
                                  t_in_secondary=supply_temp[c.P_DHW] - dtemp_transfer_loss,
                                  t_amb=np.array(t_amb), mode=1)

        # Reformat the results in the right format (W for power and COP * 100 for COP)
        res_hp = {
            f'{c.S_POWER}_{c.ET_ELECTRICITY}_{c.P_HEAT}': list(res_sh['P_el']),
            f'{c.S_POWER}_{c.ET_HEAT}_{c.P_HEAT}': list(res_sh['P_th']),
            f'{c.S_COP}_{c.P_HEAT}': list(res_sh['COP'] * c.COP_TO_COP100),
            f'{c.S_POWER}_{c.ET_ELECTRICITY}_{c.P_DHW}': list(res_dhw['P_el']),
            f'{c.S_POWER}_{c.ET_HEAT}_{c.P_DHW}': list(res_dhw['P_th']),
            f'{c.S_COP}_{c.P_DHW}': list(res_dhw['COP'] * c.COP_TO_COP100)
        }
        # Create dataframe from dict
        ts_hp = pd.DataFrame(data=res_hp)
        ts_hp = ts_hp.round().astype(int)

        # set time index to origin timestamp
        ts_hp.index = weather[c.TC_TIMESTAMP]
        ts_hp.index.name = c.TC_TIMESTAMP

        return ts_hp

    @staticmethod
    def calc_brine_temp(t_avg_d: float):
        """
        Calculate the soil temperature by the average Temperature of the day.
        Source: „WP Monitor“ Feldmessung von Wärmepumpenanlagen S. 115, Frauenhofer ISE, 2014
        added 9 points at -15°C average day at 3°C soil temperature in order to prevent higher temperature of soil below -10°C.

        Parameters
        ----------
        t_avg_d : the average temperature of the day.

        Returns:
        ----------
        t_brine : the temperature of the soil/ Brine inflow temperature
        """

        t_brine = -0.0003 * t_avg_d ** 3 + 0.0086 * t_avg_d ** 2 + 0.3047 * t_avg_d + 5.0647

        return t_brine

    @staticmethod
    def __init_vals(df: pd.DataFrame, vals: int | list[int] = 0, idx: int | list[int] = 0) -> list:
        """

        Args:
            df (pd.DataFrame): The DataFrame to which the meter values are to be added.
            init (int, optional): The initial value of the meter. Default is 0.

        Returns:
            list: A list of meter values.

        """

        # Create a list of zeros with the length of the time series
        values = [0] * len(df.index)

        # Set the first value to the specified SOC
        values[idx] = vals

        return values

    @staticmethod
    def __find_max_train_period(plants: dict, keys: str | list[str] = ('offset', 'days'), period: str = 'days') -> pd.Timedelta:

        # Convert keys to a list if it is a string
        if isinstance(keys, str):
            keys = [keys]

        # Find the maximum value for the given keys
        val = 0
        for name, plant in plants.items():
            for key in keys:
                try:
                    method = plant['fcast']['method']
                    val = max(val, plant['fcast'][method][key])
                except KeyError:
                    pass

        # Convert the value to a timedelta object
        if period == 'seconds':
            val = datetime.timedelta(seconds=val)
        elif period == 'minutes':
            val = datetime.timedelta(minutes=val)
        elif period == 'hours':
            val = datetime.timedelta(hours=val)
        elif period == 'days':
            val = datetime.timedelta(days=val)
        elif period == 'weeks':
            val = datetime.timedelta(weeks=val)
        else:
            raise KeyError(f'Period {period} not recognized.')

        return val

    def __make_timeseries(self, file_path: str, plant_id: str, plant_dict: dict, delta: pd.Timedelta) \
            -> tuple[pd.DataFrame, object | None]:
        """
        Create a time series for the given plant.

        Args:
            file_path (str): The file path to the time series data.
            plant_id (str): The ID of the plant for which to create a time series.
            plant_dict (dict): A dictionary of information about the plant, including the file format and power rating.
            delta (pd.Timedelta): The time interval for resampling the time series data.

        Returns:
            tuple[pd.DataFrame, object | None]: A tuple containing the created time series DataFrame and
            additional information (e.g., specs) if applicable.

        Raises:
            KeyError: If required information is missing from the plant_dict or if a specific function is not found.
        """

        # Load required file as a series or a DataFrame
        file = f.load_file(file_path)

        # Convert index to datetime index if file is a DataFrame
        if isinstance(file, pd.DataFrame):
            file.index = pd.to_datetime(file.index, unit='s', utc=True)

        # Initialize specs as None
        specs = None

        # Multiply with power or demand if time series is per unit
        if "pu" in plant_dict['sizing']['file'].split('_')[-1]:
            try:
                file.iloc[:, 0] *= plant_dict['sizing']['power']
            except KeyError:
                try:
                    file.iloc[:, 0] *= plant_dict['sizing']['demand']
                except KeyError:
                    raise KeyError("Plant type neither has power nor demand information to multiply with the pu file.")

        # If the file is a spec file, create a time series accordingly
        if isinstance(file, dict):
            specs = file
            try:
                # Use the plant-specific specs function to create a time series from the spec data
                file = self.plants[plant_dict['type']]['specs'](specs=file, plant=plant_dict)
            except KeyError:
                # If the specs function is not available for this plant type, raise an error
                raise KeyError(f'Time series creation from spec file not available for plant type '
                               f'{plant_dict["type"]}.')

        # Apply a special function to the time series if specified
        if func_ts := self.plants[plant_dict['type']].get('func_ts'):
            file = func_ts(file, plant_dict)

        # Set appropriate column names and resample the time series data
        if len(file.columns) == 1:
            file = file.squeeze()
            file.name = f'{plant_id}_{file.name}'
        else:
            file.columns = [f'{plant_id}_{col}' for col in file.columns]

        file = file.round().astype(int)
        file = file.astype(float)

        # Resample the time series data to ensure all rows are filled
        file = self._resample_timeseries(timeseries=file, delta=delta)

        return file, specs

    def __make_timeseries_heat(self, df: pd.DataFrame, plant_dict: dict) -> pd.DataFrame:
        # TODO: Change it so that it does not change the input values if it is not a pu file that is used
        #  (e.g. as it was the problem in the paper when Soner's input got reduced)

        # Get the goal values that are to be searched for: efficiency, occupants, temperature
        goal = [self.account['general']['parameters']['efficiency'],
                self.account['general']['parameters']['occupants'],
                plant_dict['sizing']['temperature']]

        # Check if there are multiple columns
        if len(df.columns) == 1:
            df['heat'] = df.iloc[:, 0]
        else:
            # Check columns one by one, starting with the first value
            # Note: For each value, we calculate the distance between each column's numeric value for that position and
            #       the corresponding goal value. We then filter the list of columns to only include those with the
            #       smallest distance for that value. We repeat this process for each value, so that we end up with a
            #       list of columns that have the smallest distance for each value.
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
        duration = (df.index[-1] - df.index[0] + delta) / pd.Timedelta(hours=8760)  # in years

        # Scale the heat to the demand and take into account the duration of the time series
        df['heat'] *= plant_dict['sizing']['demand'] * duration / sum(df['heat'])

        # Drop all columns except the heat column and round the values
        df = df['heat'].round().astype(int).to_frame()

        return df

    @staticmethod
    def __list_to_dict(input_list: list, separator: str = '/') -> dict:
        """
        Convert a list of strings to a nested dictionary.

        Parameters:
            input_list (list): The list of strings to convert to a nested dictionary.
            separator (str, optional): The separator used to split the keys in the input_list. Defaults to '/'.

        Returns:
            dict: A nested dictionary representing the key paths from the input_list.
        """
        nested_dict = {}
        for item in input_list:
            # Split the item (string) into a list of keys using the provided separator.
            keys = item.split(separator)
            current_dict = nested_dict
            for key in keys:
                # Traverse the nested dictionary structure and create intermediate dictionaries if needed.
                if key not in current_dict:
                    current_dict[key] = {}
                current_dict = current_dict[key]
        # Return the nested dictionary representing the key paths from the input_list.
        return nested_dict

    def __dict_to_list(self, nested_dict: dict, parent_key: str = '', separator: str = '/') -> list:
        """
        Convert a nested dictionary to a list of strings.

        Parameters:
            nested_dict (dict): The nested dictionary to convert to a list.
            parent_key (str, optional): The parent key in the nested dictionary. Defaults to ''.
            separator (str, optional): The separator used to concatenate keys. Defaults to '/'.

        Returns:
            list: A list of strings, where each string represents a key path in the nested dictionary.
        """
        items = []
        for key, value in nested_dict.items():
            # Concatenate the current key with the parent key and the separator (if parent key exists).
            new_key = f"{parent_key}{separator}{key}" if parent_key else key

            if isinstance(value, dict):
                # If the value is a nested dictionary, recursively call the function to process it.
                # Extend the 'items' list with the results from the nested dictionary processing.
                items.extend(self.__dict_to_list(value, new_key, separator))
            else:
                # If the value is not a dictionary (i.e., it is a leaf node in the nested dictionary),
                # append the key path to the 'items' list.
                items.append(new_key)

        # Return the list of strings representing key paths in the nested dictionary.
        return items

    @staticmethod
    def __dataframe_to_dict(df, key_column, key_separator='/'):
        """
        Convert a DataFrame to a dictionary with nested structure.

        Each row in the DataFrame will be converted to a nested dictionary.
        The columns will be split by '/' to create a nested structure in the dictionary.
        The keys for the main dictionary will be taken from the specified column or
        a combination of columns separated by the specified separator.

        Args:
            df (pd.DataFrame): The input DataFrame.
            key_column (str or list): The column name or list of column names to be used as keys in the dictionary.
            key_separator (str): The separator to join multiple columns as a key (default is '/').

        Returns:
            dict: A dictionary where keys are specified columns or their combination separated by the specified separator,
                  and values are dictionaries representing each row in the DataFrame with nested structure based on column names.

        Example:
            Input DataFrame:
            | general/agent_id | general/name | general/comment | general/bus | general/sub_id | general/parameters/apartments_independent | general/parameters/apartments | general/parameters/occupants | general/parameters/area | general/parameters/floors | general/parameters/height | general/parameters/efficiency | general/market_participant | inflexible_load/owner | inflexible_load/num |
            |------------------|--------------|-----------------|-------------|----------------|------------------------------------------|-------------------------------|------------------------|------------------------|--------------------------|--------------------------|-----------------------------|---------------------------|-------------------------|------------------|---------------------|
            | Pf4DMIybJoUSQT7  | NaN          | NaN             | NaN         | main           | False                                    | 6                             | 3                      | 420                    | 1                        | 2.6                        | 55                           | 1                        | 0                    | 0                   |
            | Pf4DMIybJoUSQT7  | NaN          | NaN             | NaN         | ztPLtIOn4EWPNFy | False                                    | 6                             | 3                      | 70                     | 1                        | 2.6                        | 55                           | 1                        | 1                    | 1                   |

            Output with key_column='general/agent_id':
            {'Pf4DMIybJoUSQT7': {'general': {'agent_id': 'Pf4DMIybJoUSQT7', 'name': None, 'comment': None, 'bus': None, 'sub_id': 'main',
                                             'parameters': {'apartments_independent': False, 'apartments': 6, 'occupants': 3, 'area': 420,
                                                            'floors': 1, 'height': 2.6, 'efficiency': 55},
                                             'market_participant': 1},
                                 'inflexible_load': {'owner': 0, 'num': 0}},
             'Pf4DMIybJoUSQT7-ztPLtIOn4EWPNFy': {'general': {'agent_id': 'Pf4DMIybJoUSQT7', 'name': None, 'comment': None, 'bus': None, 'sub_id': 'ztPLtIOn4EWPNFy',
                                                             'parameters': {'apartments_independent': False, 'apartments': 6, 'occupants': 3, 'area': 70,
                                                                            'floors': 1, 'height': 2.6, 'efficiency': 55},
                                                             'market_participant': 1},
                                                'inflexible_load': {'owner': 1, 'num': 1}}}
        """

        # Initialize an empty dictionary to store the nested dictionaries
        result = {}

        # Iterate over the rows in the DataFrame with their indices
        for idx, row in df.iterrows():
            # Initialize an empty dictionary for each row
            nested_dict = {}

            # Create the key for the row based on the specified column(s)
            if isinstance(key_column, list):
                # If multiple columns are specified as keys
                key = key_separator.join(str(row[col]) for col in key_column)
            else:
                # If a single column is specified as key
                key = row[key_column]

            # Include the key in the result dictionary
            result[key] = nested_dict

            # Iterate over the columns in the DataFrame
            for col in df.columns:
                # Split the column name by the key separator to get the keys for nested structure
                keys = col.split(key_separator)
                # Create a temporary dictionary to traverse the nested structure
                temp_dict = nested_dict
                # Iterate over the keys (except the last one) to create nested dictionaries
                for key_part in keys[:-1]:
                    # Use setdefault to create nested dictionary if not exists
                    temp_dict = temp_dict.setdefault(key_part, {})
                # Set the value in the innermost nested dictionary
                temp_dict[keys[-1]] = row[col]

        return result

    @staticmethod
    def __group_keys_under_nested_dict(input_dict: dict, output_name: str, keys: tuple) -> dict:
        # Create the new nested dictionary with the specified name
        nested_dict = {}

        # Iterate through the keys to group and move them to the new nested dictionary
        for key in keys:
            try:
                # If the key is a string, move the key-value pair to the new nested dictionary
                nested_dict[key] = input_dict.pop(key)
            except KeyError:
                # If the key is not in the input dictionary, skip it
                pass

        # Add the new nested dictionary to the input dictionary under the specified output name
        input_dict[output_name] = nested_dict

        return input_dict

    @staticmethod
    def __add_entries(info: dict, plant_dict: dict, idx: int, entries: list) -> dict:
        """Adds specified entries from info in plant_dict"""

        for entry in entries:
            plant_dict[entry] = info[f"{entry}_{idx}"]

        return plant_dict

    def _resample_timeseries(self, timeseries: pd.DataFrame, delta: pd.Timedelta) -> pd.DataFrame:
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

        # Ensure that index is a datetime index
        timeseries.index = pd.to_datetime(timeseries.index, unit='s', utc=True)

        # Calculate the delta of the input timeseries
        original_delta = timeseries.index[1] - timeseries.index[0]

        # Resample the timeseries based on the comparison between the input delta and the time delta of the input
        if delta > original_delta:
            # Reduce the timeseries with mean values of rows
            resampled = timeseries.resample(delta).mean()
        elif delta < original_delta:
            # Expand the timeseries with linearly interpolated data
            # Add last row again to the dataframe with the index moved by original timedelta
            # Note: This is necessary as otherwise the timeseries would not be long enough for the simulation
            last_row = timeseries.iloc[-1:].copy()
            last_row.index = last_row.index + original_delta
            timeseries = pd.concat([timeseries, last_row])
            # Interpolate the timeseries
            resampled = timeseries.resample(delta).interpolate()
            # Drop the last row
            resampled = resampled[:-1]
        elif delta == original_delta:
            # Return the original timeseries if the delta is the same
            resampled = timeseries
        else:
            raise ValueError(f"Invalid delta value: {delta}")

        # Convert the data types back to the original ones
        if delta != original_delta:
            resampled = self.__restore_dtypes(resampled, timeseries)

        # Check if data types can be optimized, thus, converted to a lower data type
        resampled, _ = self.__optimize_dtypes(resampled)

        return resampled

    @staticmethod
    def __restore_dtypes(df_resampled: pd.DataFrame, df_original: pd.DataFrame) -> pd.DataFrame:
        """
        Restore original data types and optimize them if possible.

        Args:
            df_resampled (pd.DataFrame or pd.Series): The resampled data.
            df_original (pd.DataFrame or pd.Series): The original data before resampling.

        Returns:
            pd.DataFrame or pd.Series: The resampled data with restored and optimized data types.

        Raises:
            TypeError: If input is not a pandas Series or DataFrame.
        """

        # Convert the data types back to the original ones
        if isinstance(df_original, pd.DataFrame):
            for col, dtype in df_original.dtypes.items():
                df_resampled[col] = df_resampled[col].astype(dtype)
        elif isinstance(df_original, pd.Series):
            df_resampled = df_resampled.astype(df_original.dtype)
        else:
            raise TypeError(f"Type {type(df_original)} is not supported. Input should be a pandas Series or DataFrame.")

        return df_resampled

    @staticmethod
    def __optimize_dtypes(series_or_df: Union[pd.Series, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
        """
        Optimize data types of numeric columns in a pandas Series or DataFrame if possible.

        Args:
            series_or_df (pd.Series or pd.DataFrame): The input data to optimize data types.

        Returns:
            tuple[pd.DataFrame, dict]: A tuple containing the optimized DataFrame and a dictionary
            mapping column names to their optimized data types.

        Raises:
            ValueError: If input is not a pandas Series or DataFrame.
        """

        def is_int_type(dtype):
            """Check if the dtype is an integer type."""
            return np.issubdtype(dtype, np.integer)

        def is_float_type(dtype):
            """Check if the dtype is a floating-point type."""
            return np.issubdtype(dtype, np.floating)

        def is_numeric_type(dtype):
            """Check if the dtype is a numeric type."""
            return is_int_type(dtype) or is_float_type(dtype)

        def reduce_integer_dtype(dtype, series: pd.Series):
            """
            Reduce integer dtype to a smaller one if possible.

            Args:
                dtype (np.dtype): The input dtype to potentially reduce.

            Returns:
                np.dtype: The reduced dtype if applicable, otherwise the original dtype.
            """
            if is_int_type(dtype):
                max_value = series.max()
                min_value = series.min()
                if max_value <= np.iinfo(np.uint8).max and min_value >= np.iinfo(np.uint8).min:
                    return 'uint8'
                elif max_value <= np.iinfo(np.uint16).max and min_value >= np.iinfo(np.uint16).min:
                    return 'uint16'
                elif max_value <= np.iinfo(np.uint32).max and min_value >= np.iinfo(np.uint32).min:
                    return 'uint32'
                elif max_value <= np.iinfo(np.uint64).max and min_value >= np.iinfo(np.uint64).min:
                    return 'uint64'
                elif max_value <= np.iinfo(np.int8).max and min_value >= np.iinfo(np.int8).min:
                    return 'int8'
                elif max_value <= np.iinfo(np.int16).max and min_value >= np.iinfo(np.int16).min:
                    return 'int16'
                elif max_value <= np.iinfo(np.int32).max and min_value >= np.iinfo(np.int32).min:
                    return 'int32'
                elif max_value <= np.iinfo(np.int64).max and min_value >= np.iinfo(np.int64).min:
                    return 'int64'
                elif max_value == 1 and min_value == 0:
                    return 'bool'
            return dtype

        if isinstance(series_or_df, pd.Series):
            df = series_or_df.to_frame()
        elif isinstance(series_or_df, pd.DataFrame):
            df = series_or_df
        else:
            raise ValueError("Input should be a pandas Series or DataFrame.")

        new_dtypes = {}
        for col in df.columns:
            dtype = df[col].dtype
            if is_numeric_type(dtype):
                if is_float_type(dtype):
                    # Check if float can be reduced to integer or bool (if decimals are zero)
                    if (df[col] % 1 == 0).all():
                        if df[col].isin([0, 1]).all():
                            new_dtype = 'bool'
                        else:
                            new_dtype = reduce_integer_dtype(dtype, series=df[col])
                    else:
                        new_dtype = dtype
                else:
                    # Check if integer can be reduced to a smaller integer type
                    new_dtype = reduce_integer_dtype(dtype, series=df[col])
            else:
                new_dtype = dtype

            if new_dtype != dtype:
                new_dtypes[col] = new_dtype
                df[col] = df[col].astype(new_dtype)

        return df, new_dtypes

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
    def __get_types(path: str, idx: int = 0, sep: str = '_') -> list:
        """Returns a list of all types of a specific agent type"""

        # Return the unique types from the files in the directory
        return list(set([file.split(sep)[idx] for file in os.listdir(path)]))
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
        try:
            input_vals = [int(val.split("_")[input_idx]) for val in input_files]
        except IndexError:
            raise Warning('Index of value in file name not found. Check if file name is correct '
                          'and if input data is correct.')
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
        list_files = [np.nan] * len(list_type)
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

        # Change to float as np.round does not work with int
        if isinstance(vals, pd.Series):
            vals = vals.astype(float)

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

    @classmethod
    def _add_info_simple(cls, keys: list, config: dict, df: pd.DataFrame, separator: str = "/",
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
                cls._add_info_simple(keys=keys + [item], config=value, df=df, separator=separator,
                                     preface=preface, appendix=appendix)
            else:
                # If value is in item_list, add the value to the dataframe
                if f"{path_info}{preface}{item}{appendix}" in item_list:
                    try:
                        df.loc[:, f"{path_info}{preface}{item}{appendix}"] = value
                    except ValueError:
                        df.loc[:, f"{path_info}{preface}{item}{appendix}"] = str(value)

        return df

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

    @classmethod
    def get_num_from_grid(cls, df: pd.DataFrame, agent_type: str, col: str = None, unique: str = 'bus') -> int:
        """Returns the number of plants of a specific type in a grid"""

        # Assign the column to the column containing '_type' if no column is specified
        col = [col for col in df.columns if '_type' in str(col)][0] if col is None else col

        # Get the number of agents of the specified agent type
        num = df[df[col] == agent_type]

        if unique:
            num = len(num[unique].unique())

        return num

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
            return after, pos
        else:
            return before, pos

    @staticmethod
    def _get_closest(search_list: list, val: float, sorted_list: bool = False):
        """ Returns the closest value to val in search_list."""

        if sorted_list:
            return Agents._get_closest_sorted(search_list, val)

        return min(enumerate(search_list), key=lambda x: abs(x[1] - val))

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
    def _gen_new_ids_sfh(n: int = 1, length: int = 15) -> Union[str, list[str]]:
        """creates random ID"""
        ids = []
        idx = 0
        for _ in range(n):
            ids.append("sfh_" + str(idx))
            idx = idx + 1

        if len(ids) == 1:
            return ids[0]
        else:
            return ids


# Playground
if __name__ == "__main__":
    print('Not working anymore ;) Call the example function to create a scenario instead.')