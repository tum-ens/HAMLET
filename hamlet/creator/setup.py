__author__ = "TUM-Doepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "TUM-Doepfert"
__email__ = "markus.doepfert@tum.de"

import time
import shutil
import os
from ruamel.yaml import YAML
import json
import numpy as np
from tqdm import tqdm
import pandas as pd
import difflib
import pandapower as pp
from pprint import pprint
from hamlet.creator.agents.agents import Agents
from hamlet.creator.markets.markets import Markets
from hamlet.creator.grids.grids import Grids

# Principal steps:
# 1. Identify the scenario structure
# 2. Create the necessary folders for the scenario
# 3. Create the locality/region for each market -> either based on config or directly from files


class Creator:
    """Class to create a scenario from the config files

    """

    def __init__(self, path: str, name: str = None) -> None:
        """Initialize a scenario instance.

        Args:
            path (str): Path to the config folder.
            name (str, optional): Name of the scenario. Defaults to None.

        Returns:
            None
        """

        # Initialize the progress bar
        self.pbar = tqdm()

        # Initialize the YAML parser
        self.yaml = YAML()

        # Load the config and set paths
        self.path_config = os.path.abspath(path)
        self.root_config = self.path_config.rsplit(os.sep, 1)[0]
        self.config = self.load_config(path=os.path.join(self.path_config, 'config_setup.yaml'))
        self.path_input = os.path.abspath(self.config['paths']['input'])
        self.path_scenarios = os.path.abspath(self.config['paths']['scenarios'])
        self.path_results = os.path.abspath(self.config['paths']['results'])

        # Set the name of the scenario and create the scenario and folder structure
        self.name = name if name is not None else self.path_config.split(os.sep)[-1]
        self.scenario_structure = self.__get_structure(name=self.name, path=self.path_config)
        self.subfolders = self.__add_subfolders_to_dict(dict={}, path=self.path_input, max_level=1)

        # Initialize dictionaries for different elements
        self.markets = {}
        self.retailers = {}
        self.grids = {}

        # Create progress bar description
        self.progress_bar_description = {
            '__create_agent_files': 'Creating the missing agent files:',
            '__create_grid_files': 'Creating the missing grid files:',
            '__create_scenario_folders': 'Creating the folders for the scenario:',
            '__create_markets': 'Creating the markets for each region:',
            '__create_agents': 'Creating the agents for each region:',
            '__copy_grids': 'Copying the grid files from the input folder to the scenario folder:',
            '__copy_config_to_scenarios': 'Copying the files from the config folder to the scenario folder:',
            '__create_general_files': 'Copying the files from the general config file to the scenario folder:',
            '__combine_files': 'Combining the market and grid files:'
        }

        # Count the number of regions in the scenario structure
        self.region_number = self.__count_all_keys_in_dict(self.scenario_structure)

    def load_config(self, path: str) -> dict:
        """Load configuration from the given file.

        Args:
            path (str): Path to the configuration file.

        Returns:
            dict: Configuration dictionary.
        """

        with open(path, 'r') as config_file:
            config = self.yaml.load(config_file)

        return config

    def new_scenario_from_configs(self, delete: bool = True) -> None:
        """Create a new scenario using configuration files.

        Args:
            delete (bool): If True, delete the folder if it already exists.

        Returns:
            None
        """
        # Initialize progress bar
        self.pbar.reset(total=6 * self.region_number + 3)

        # Create the missing agent files
        self.pbar.set_description_str(self.progress_bar_description[self.__create_agent_files.__name__])
        self.__loop_through_dict(self.scenario_structure, path=self.path_config.rsplit(os.sep, 1)[0],
                                 func=self.__create_agent_files, update_pbar=True, method='config')

        # TODO: Create the missing grid files
        self.pbar.set_description_str(self.progress_bar_description[self.__create_grid_files.__name__])
        self.__loop_through_dict(self.scenario_structure, path=self.path_config.rsplit(os.sep, 1)[0],
                                 func=self.__create_grid_files, update_pbar=True)

        # Continue with creating a new scenario from files
        self.new_scenario_from_files(delete=delete)

    def new_scenario_from_grids(self, fill_from_config: bool = False, delete: bool = True) -> None:
        """Create a new scenario using grid files.

        Args:
            fill_from_config (bool): If True, create missing plant types from the config file.
            delete (bool): If True, delete the folder if it already exists.

        Returns:
            None
        """
        # Initialize progress bar
        self.pbar.reset(total=5 * self.region_number + 3)

        # Create the missing agent files
        self.pbar.set_description_str(self.progress_bar_description[self.__create_agent_files.__name__])
        self.__loop_through_dict(self.scenario_structure, path=self.path_config.rsplit(os.sep, 1)[0],
                                 func=self.__create_agent_files, update_pbar=True, method='grid')

        # Create the scenario from the generated files
        self.new_scenario_from_files(delete=delete)

    def new_scenario_from_files(self, delete: bool = True) -> None:
        """Creates a new scenario from the files.

        Args:
            delete (bool): If True, the folder will be deleted if it already exists.

        Returns:
            None
        """
        # Initialize progress bar if necessary
        if self.pbar.total is None:
            self.pbar.reset(total=4 * self.region_number + 3)

        # Create the folders for the scenario
        self.pbar.set_description_str(self.progress_bar_description[self.__create_scenario_folders.__name__])
        self.__create_scenario_folders(delete=delete)
        self.pbar.update(1)

        # Create the markets for each region by looping through the structure
        self.pbar.set_description_str(self.progress_bar_description[self.__create_markets.__name__])
        self.__loop_through_dict(self.scenario_structure, path=self.path_config.rsplit(os.sep, 1)[0],
                                 func=self.__create_markets, update_pbar=True)

        # Create the agents for each region by looping through the structure
        self.pbar.set_description_str(self.progress_bar_description[self.__create_agents.__name__])
        self.__loop_through_dict(self.scenario_structure, path=self.path_config.rsplit(os.sep, 1)[0],
                                 func=self.__create_agents, update_pbar=True)

        # Copy the grid files from the input folder to the scenario folder
        self.pbar.set_description_str(self.progress_bar_description[self.__copy_grids.__name__])
        self.__loop_through_dict(self.scenario_structure, path=self.path_config.rsplit(os.sep, 1)[0],
                                 func=self.__copy_grids, update_pbar=True)

        # Copy the files from the config folder to the scenario folder
        self.pbar.set_description_str(self.progress_bar_description[self.__copy_config_to_scenarios.__name__])
        self.__loop_through_dict(self.scenario_structure, path=self.path_config.rsplit(os.sep, 1)[0],
                                 func=self.__copy_config_to_scenarios, update_pbar=True)

        # Copy the files from the general config file to the scenario folder
        self.pbar.set_description_str(self.progress_bar_description[self.__create_general_files.__name__])
        self.__create_general_files()
        self.pbar.update(1)

        # Combine the market and grid files
        self.pbar.set_description_str(self.progress_bar_description[self.__combine_files.__name__])
        self.__combine_files()
        self.pbar.update(1)

        self.pbar.set_description_str('Successfully created scenario')

    def __combine_files(self) -> None:
        """Combine market, retailer, and grid files within the scenario.

        This method performs the following actions:
        1. Combines individual market and retailer files based on their name and/or type.
        2. Creates a combined market file that consolidates all the individual market files.
        3. Combines individual grid files based on their name and/or type.
        4. Creates a combined grid file that consolidates all the individual grid files.

        Returns:
            None
        """

        # Combine individual market and retailer files based on their name and/or type
        self.__loop_through_dict(self.scenario_structure, path=self.path_scenarios,
                                 func=self.__combine_market_files)

        # Create a combined market file that consolidates all individual market files
        self.__create_combined_market_file()

        # Combine individual grid files based on their name and/or type
        self.__loop_through_dict(self.scenario_structure, path=self.path_scenarios,
                                 func=self.__combine_grid_files)

        # Create a combined grid file that consolidates all individual grid files
        self.__create_combined_grid_file()

    def __get_structure(self, name: str, path: str, max_level: int = np.inf) -> dict:
        """Get the structure of the scenario from the config folder.

        This method iterates through the given path in the config folder and creates a nested dictionary that represents the
        structure of the scenario. It does so by recursively adding subfolders to the dictionary up to the specified maximum
        level. This functionality is similar to evaluating the config structure as seen in 'make_list_from_nested_dict()'.

        Args:
            name (str): The name to be used as the key for the root of the dictionary.
            path (str): The path to the config folder containing the scenario structure.
            max_level (int, optional): The maximum level of nested folders to include in the dictionary. Defaults to np.inf, meaning no limit.

        Returns:
            dict: A dictionary representing the structure of the scenario.
        """

        # Initialize the dictionary with the given name as the root
        dict_struct = {name: {}}

        # Recursively add subfolders to the dictionary, representing the structure of the scenario
        dict_struct[name] = self.__add_subfolders_to_dict(dict_struct[name], path, max_level)

        return dict_struct

    def __create_agent_files(self, path_config: str, method: str = 'config', fill_from_config: bool = False,
                             overwrite: bool = True) -> None:
        """Creates the missing agent files based on the specified method.

        Depending on the selected method, this function either creates agent files from a config file or from a grid file.
        The created files are located in the specified path, and existing files can be overwritten if desired.

        Args:
            path_config (str): Path to the config file.
            method (str): Method to create the agent files, either 'config' or 'grid'. Defaults to 'config'.
            fill_from_config (bool): If True, missing plant types will be created from the config file. Only relevant if method is 'grid'. Defaults to False.
            overwrite (bool): If True, existing agent files will be overwritten if they already exist. Defaults to True.

        Returns:
            None
        """

        # Determine the subpath of the config file relative to the root
        subpath = path_config.replace(self.root_config, '')

        # Create an instance of the Agents class, configuring paths based on the subpath
        agents = Agents(config_path=path_config,
                        config_root=os.path.join(self.root_config, subpath.split(os.sep, 2)[1]),
                        input_path=self.path_input,
                        scenario_path=os.path.join(self.path_scenarios, subpath.split(os.sep, 1)[-1]))

        # Determine the method to create the agent files and call the appropriate function
        if method == 'config':
            agents.create_agents_file_from_config(overwrite=overwrite)
        elif method == 'grid':
            agents.create_agents_file_from_grid(fill_from_config=fill_from_config, overwrite=overwrite)

    def __create_grid_files(self, path_config: str) -> None:
        """Creates the missing grid files.

        Note: This method is currently unavailable as no mechanism to artificially create grids exists yet.

        Args:
            path_config (str): Path to the config file.

        Returns:
            None
        """

        # Return early as the functionality is not yet implemented
        return

        # Uncomment the following code once the functionality to create grid files is implemented

        # Determine the subpath of the config file relative to the root
        # subpath = path_config.replace(self.root_config, '')

        # Create an instance of the Grids class, configuring paths based on the subpath
        # grids = Grids(config_path=path_config,
        #               config_root=os.path.join(self.root_config, subpath.split(os.sep, 2)[1]),
        #               input_path=self.path_input,
        #               scenario_path=os.path.join(self.path_scenarios, subpath.split(os.sep, 1)[-1]))

        # Create the grid files that are not specified as files in the config file
        # grids.create_grid_files()

    def __create_agents(self, path_config: str) -> None:
        """Creates the agent files for the scenario using the specified config file.

        This method uses the Agents class to create agent files based on the provided configuration path. The created
        agent files will be used to define the agents within the scenario.

        Args:
            path_config (str): Path to the config file used to create the agent files.

        Returns:
            None
        """

        # Determine the subpath of the config file relative to the root. This subpath is used to construct paths for
        # other files.
        subpath = path_config.replace(self.root_config, '')

        # Create an instance of the Agents class, using the path information derived from the config file's subpath.
        agents = Agents(config_path=path_config,
                        config_root=os.path.join(self.root_config, subpath.split(os.sep, 2)[1]),
                        input_path=self.path_input,
                        scenario_path=self.path_scenarios + subpath)

        # Invoke the method to create the agent files based on the provided config file. This creates the necessary
        # files to represent agents in the scenario.
        agents.create_agents_from_file()

    def __copy_grids(self, path_config: str) -> None:
        """Copies the grid files for the scenario.

        This method creates an instance of the Grids class and then copies the grid files from the input folder to the
        scenario folder, according to the specifications in the provided configuration file.

        Args:
            path_config (str): Path to the config file that specifies the grid files to copy.

        Returns:
            None
        """

        # Determine the subpath of the config file relative to the root
        # This subpath is used to construct paths for the grid files
        subpath = path_config.replace(self.root_config, '')

        # Create an instance of the Grids class, configuring paths based on the subpath
        grids = Grids(config_path=path_config,
                      config_root=os.path.join(self.root_config, subpath.split(os.sep, 2)[1]),
                      input_path=self.path_input,
                      scenario_path=os.path.join(self.path_scenarios, subpath.split(os.sep, 1)[-1]))

        # Invoke the method to copy the grid files from the input folder to the scenario folder
        # This ensures that the scenario has the necessary grid information
        grids.copy_grid_files()

    def __create_markets(self, path_config: str) -> None:
        """Creates the market files for the scenario.

        This method uses the Markets class to create market files based on the provided configuration path. The created
        market files will be used to define the markets within the scenario.

        Args:
            path_config (str): Path to the config file used to create the market files.

        Returns:
            None
        """

        # Determine the subpath of the config file relative to the root
        # This subpath is used to construct paths for other files
        subpath = path_config.replace(self.root_config, '')

        # Create an instance of the Markets class, using the path information derived from the config file's subpath
        markets = Markets(config_path=path_config,
                          config_root=os.path.join(self.root_config, subpath.split(os.sep, 2)[1]),
                          input_path=self.path_input,
                          scenario_path=self.path_scenarios + subpath)

        # Invoke the method to create the market files based on the provided config file
        # This creates the necessary files to represent markets in the scenario
        markets.create_markets()

    def __create_general_files(self):
        """Copies the files specified in the general config to the scenario folder.

        This method is responsible for creating a general file that includes the structure of the simulation.
        It also handles copying the weather file from the input directory to the scenario's general folder.

        Returns:
            None
        """

        # Initialize the general file as a dictionary
        general = {}

        # Define the structure of the simulation by flattening the scenario structure
        general['structure'] = self.flatten_dict(self.scenario_structure)

        # Save the general dictionary as a JSON file in the scenario's 'general' directory
        self._save_file(os.path.join(self.path_scenarios, self.name, 'general', 'general.json'), general)

        # Create the folder for the weather files within the scenario's 'general' directory
        self.__create_folder(os.path.join(self.path_scenarios, self.name, 'general', 'weather'))

        # Copy the specified weather file from the input directory to the scenario's 'general' folder
        shutil.copy(os.path.join(self.path_input, 'general', 'weather', self.config['location']['weather']),
                    os.path.join(self.path_scenarios, self.name, 'general', 'weather',
                                 self.config['location']['weather']))

    def __combine_market_files(self, path: str) -> tuple[dict, dict]:
        """Combines the market and retailer files of the scenario based on the provided path.

        Args:
            path: The path to the directory containing the market and retailer files.

        Returns:
            tuple[dict, dict]: A tuple containing two dictionaries representing the combined markets and retailers.

        Raises:
            TypeError: If an encountered file is not a valid type (.csv or .ft).
        """

        # Set the paths to the markets and retailers folders
        path_markets = os.path.join(path, 'markets')
        path_retailers = os.path.join(path, 'retailers')

        # Determine the region's name by comparing the path_scenarios and the given path
        region = difflib.ndiff(self.path_scenarios.split(os.sep), path.split(os.sep))
        region = os.sep.join(x.split()[-1] for x in region if x.startswith('+ '))

        # Loop through the markets directory, identifying and loading relevant files into the self.markets dictionary
        for root, dirs, files in os.walk(path_markets):
            for file in files:
                if 'timetable' in file:
                    filepath = os.path.join(root, file)
                    if file.endswith('.csv'):
                        self.markets[region] = self._load_file(filepath, index=None)
                    elif file.endswith('.ft'):
                        df = self._load_file(filepath)
                        self.markets[region] = df.set_index(df.columns[0])
                    else:
                        raise TypeError(f'File {file} is not a valid file type. Please use .csv or .ft files.')

        # Loop through the retailers directory, identifying and loading relevant files into the self.retailers dictionary
        for root, dirs, files in os.walk(path_retailers):
            for file in files:
                if 'retailer' in file:
                    filepath = os.path.join(root, file)
                    if file.endswith('.csv'):
                        self.retailers[region] = self._load_file(filepath, index=None)
                    elif file.endswith('.ft'):
                        df = self._load_file(filepath)
                        self.retailers[region] = df.set_index(df.columns[0])
                    else:
                        raise TypeError(f'File {file} is not a valid file type. Please use .csv or .ft files.')

        return self.markets, self.retailers

    def __create_combined_market_file(self, file_type: str = 'ft'):

        # Take all the market files in self.markets and concat all to one big dataframe
        markets = pd.concat(self.markets.values(), ignore_index=True)

        # Sort markets by timestamp, market, name and timestep
        markets = markets.sort_values(by=['timestamp', 'region', 'market', 'name', 'timestep'], ignore_index=True).\
            set_index(markets.columns[0], drop=True)

        # Save the file
        self._save_file(os.path.join(self.path_scenarios, self.name, 'general', f'timetable.{file_type}'), markets)

        # Take all the retailer files in self.retailers and combine to one file
        retailers = pd.concat(self.retailers.values(), ignore_index=True)

        # Sort retailers by timestamp
        retailers = retailers.sort_values(by=['timestamp'], ignore_index=True).\
            set_index(retailers.columns[0], drop=True)

        # Save the file
        self._save_file(os.path.join(self.path_scenarios, self.name, 'general', f'retailer.{file_type}'), retailers)

    def __combine_grid_files(self, path: str):

        # Set the path to the grids folder
        path_grids = os.path.join(path, 'grids')

        # Get the region's name
        region = difflib.ndiff(self.path_scenarios.split(os.sep), path.split(os.sep))
        region = os.sep.join(x.split()[-1] for x in region if x.startswith('+ '))

        # Loop through the grids and find the files to add to self.grids
        for root, dirs, files in os.walk(path_grids):
            for file in files:
                if file.endswith('.json'):
                    self.grids[region] = pp.from_json(os.path.join(root, file))
                elif file.endswith('.xlsx'):
                    self.grids[region] = pp.from_excel(os.path.join(root, file))
                else:
                    raise TypeError(f'File {file} is not a valid file type. Please use .json or .xlsx files.')

    def __create_combined_grid_file(self) -> None:
        """
        Create a combined grid by merging the main grid with the grids of the regions.

        If the main grid is not found in self.grids, the function returns without creating the combined grid.

        The combined grid is saved as a JSON file in the specified path.
        """

        # Check if the main grid's name exists in the dictionary
        if self.name not in self.grids:
            return

        # Take the main grid and expand it with the grids of the regions
        grid = self.grids[self.name]

        # Loop through the grids and add them to the main grid
        for region, grid_region in self.grids.items():
            if region != self.name:
                grid = pp.merge_nets(grid, grid_region, validate=False)

        # Save the combined grid as a JSON file
        pp.to_json(grid, os.path.join(self.path_scenarios, self.name, 'general', 'grid.json'))

    def __create_scenario_folders(self, delete: bool = True) -> None:
        """Creates the scenario folders in which the files are put for each region

        Returns:
            None
        """

        # Create the folders for each region by looping through the structure
        self.__loop_through_dict(self.scenario_structure, path=self.path_scenarios,
                                 func=self.__create_folders_from_dict, subfolders=self.subfolders, delete=delete)

    def __create_folders_from_dict(self, path: str, subfolders: dict, delete: bool = True) -> None:
        """Creates a folder at the given path

        Args:
            path: path to the folder
            delete: if True, the folder will be deleted if it already exists

        Returns:
            None
        """

        # Create main folder if does not exist
        if not os.path.exists(path):
            os.mkdir(path)

        # Loop through the subfolders and create them
        self.__loop_through_dict(subfolders, path=path, func=self.__create_folder, delete=delete)

    def __copy_config_to_scenarios(self, path_config: str, delete: bool = True) -> None:
        """Copies the config file to the scenario folder

        Returns:
            None
        """

        # Get the path of the scenario folder to which it needs to be copied
        path_scenario = path_config.replace(self.root_config, '')  # get the subpath of the config file to the root
        path_scenario = os.path.join(self.path_scenarios, path_scenario.split(os.sep, 1)[-1], 'config')

        # Copy (only) files from config to scenario folder
        self.__copy_folder(src=path_config, dst=path_scenario, only_files=True, delete=delete)

    def __count_all_keys_in_dict(self, dictionary: dict):
        """Count all keys in a multi-level dictionary.

        Args:
            dictionary: dict to be counted

        Returns:
            count: number of keys in dictionary
        """
        count = len(dictionary)
        for value in dictionary.values():
            if isinstance(value, dict):
                count += self.__count_all_keys_in_dict(dictionary=value)

        return count

    @classmethod
    def __add_subfolders_to_dict(cls, dict: dict, path: str, max_level: int = np.inf, cur_level: int = 0) -> dict:
        """adds subfolders to the dictionary

        Args:
            dict: dictionary to which the subfolders are added
            path: path of the folder

        Returns:
            dictionary with subfolders

        """

        # if max_level < np.inf:
        #     # TODO: This function is not working correctly as it does not use max_level correctly
        #     print('This function is not working correctly as it does not use max_level correctly')

        if cur_level > max_level:
            return dict

        # Loop through all subfolders and add them to the dictionary
        for subfolder in os.listdir(path):
            if os.path.isdir(f"{path}/{subfolder}"):
                cur_level += 1
                dict[subfolder] = {}
                dict[subfolder] = cls.__add_subfolders_to_dict(dict[subfolder], f"{path}/{subfolder}", max_level, cur_level)
        return dict

    def __loop_through_dict(self, nested_dict: dict, path: str, func: callable, update_pbar: bool = False, *args,
                            **kwargs) -> None:
        """loops through the dictionary and calls the function for each item

        Args:
            nested_dict: dictionary to loop through
            path: path to the folder
            func: function to call
            update_pbar: if progress bar should be updated
            *args: arguments for the function
            **kwargs: keyword arguments for the function

        Returns:
            None

        """
        pbar_desc = self.pbar.desc

        # Loop through all key-value pairs of the dictionary
        for key, value in nested_dict.items():
            if update_pbar:     # update progress bar description if it is active
                pbar_desc_new = pbar_desc + ' ' + key
                self.pbar.set_description_str(pbar_desc_new)

            # Check if value is a dictionary
            if isinstance(value, dict):
                # If value is a dictionary, Call the function again
                func(os.path.join(path, key), *args, **kwargs)
                self.__loop_through_dict(value, os.path.join(path, key), func, update_pbar, *args, **kwargs)
            else:
                # If value is not a dictionary, call the function
                func(os.path.join(path, key), *args, **kwargs)

            if update_pbar:     # update progress bar if it is active
                self.pbar.update(1)
                self.pbar.set_description_str(pbar_desc)

    @staticmethod
    def __create_folder(path: str, delete: bool = True) -> None:
        """Creates a folder at the given path

        Args:
            path: path to the folder
            delete: if True, the folder will be deleted if it already exists

        Returns:
            None
        """

        # Create main folder if does not exist
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            if delete:
                shutil.rmtree(path)
                os.makedirs(path)
        time.sleep(0.01)

    @staticmethod
    def __copy_folder(src: str, dst: str, only_files: bool = False, delete: bool = True) -> None:
        """Copies a folder to another location

        Args:
            src: path to the copy
            dst: path to the folder
            delete: if True, the folder will be deleted if it already exists

        Returns:
            None
        """

        # Check if only files should be copied
        if only_files:
            # Create the destination folder if it does not exist
            os.makedirs(dst, exist_ok=True)
            # Get a list of all files in the source folder
            files = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
            # Iterate through the list of files and copy them to the destination folder
            for file in files:
                src_file = os.path.join(src, file)
                dst_file = os.path.join(dst, file)
                shutil.copy(src_file, dst_file)
        else:
            # Check if the folder exists
            if not os.path.exists(dst):
                shutil.copytree(src, dst)
            else:
                if delete:
                    shutil.rmtree(dst)
                    shutil.copytree(src, dst)
            time.sleep(0.01)

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

    def flatten_dict(self, nested_dict, parent_key='', sep=os.sep):
        """
        Recursively flatten a nested dictionary and return a new dictionary
        with the keys as strings indicating their full position in the original
        dictionary.
        """
        items = []
        for k, v in nested_dict.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict):
                name = new_key.rsplit(os.sep, 1)[-1]
                items.append((name, new_key))
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            else:
                name = new_key.rsplit(os.sep, 1)[-1]
                items.append((name, new_key))
        return dict(items)

