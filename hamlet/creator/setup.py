__author__ = "TUM-Doepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "TUM-Doepfert"
__email__ = "markus.doepfert@tum.de"

import time
import shutil
import os
from ruamel.yaml import YAML
import numpy as np
from lama.creator.agents.agents import Agents

# TODO Plan: All functions that are necessary are briefly described so that they can be worked on one by one
# 1. Identify the scenario structure
# 2. Create the necessary folders for the scenario
# 3. Create the locality/region for each market -> either based on config or directly from files


class Scenario:
    """Class to create a scenario from the config files

    """

    def __init__(self, path: str, name: str = None) -> None:
        """Initializes the scenario

        Args:
            path: path to the config folder
            name: name of the scenario

        Returns:
            None
        """

        # Initialize the yaml parser
        self.yaml = YAML()

        # Load the config and set paths
        self.path_config = os.path.abspath(path)
        self.root_config = self.path_config.rsplit('\\', 1)[0]
        self.config = self.load_config(path=os.path.join(self.path_config, 'config_general.yaml'))
        self.path_input = os.path.abspath(self.config['simulation']['paths']['input'])
        self.path_scenarios = os.path.abspath(self.config['simulation']['paths']['scenarios'])
        self.path_results = os.path.abspath(self.config['simulation']['paths']['results'])

        # Set the name of the scenario and create the scenario and folder structure
        self.name = name if name is not None else self.path_config.split('\\')[-1]
        self.scenario_structure = self.__get_structure(name=self.name, path=self.path_config)
        self.subfolders = self.__add_subfolders_to_dict(dict=dict(), path=self.path_input, max_level=1)

    def load_config(self, path: str) -> dict:
        """Loads the config file from the given path

        Args:
            path: path to the config file

        Returns:
            config: dictionary with the config
        """

        with open(f"{path}") as config_file:
            config = self.yaml.load(config_file)

        return config

    def new_scenario_from_configs(self, delete: bool = True) -> None:
        """Creates a new scenario from the config files

        Args:
            delete: if True, the folder will be deleted if it already exists

        Returns:
            None
        """

        # TODO: Create the missing market files
        self.__loop_through_dict(self.scenario_structure, path=self.path_config.rsplit('\\', 1)[0],
                                 func=self.__create_market_files)

        # Create the missing agent files
        self.__loop_through_dict(self.scenario_structure, path=self.path_config.rsplit('\\', 1)[0],
                                 func=self.__create_agent_files, method='config')

        # TODO: Create the missing grid files
        self.__loop_through_dict(self.scenario_structure, path=self.path_config.rsplit('\\', 1)[0],
                                 func=self.__create_grid_files)

        # Create the scenario from the generated files
        self.new_scenario_from_files(delete=delete)

    def new_scenario_from_grids(self, fill_from_config: bool = False, delete: bool = True) -> None:
        """Creates a new scenario from the grid files

        Args:
            fill_from_config: if True, the missing plant types will be created from the config file
            delete: if True, the folder will be deleted if it already exists

        Returns:
            None
        """

        # Create the missing agent files
        self.__loop_through_dict(self.scenario_structure, path=self.path_config.rsplit('\\', 1)[0],
                                 func=self.__create_agent_files, method='grid')

        # Create the scenario from the generated files
        self.new_scenario_from_files(delete=delete)

    def new_scenario_from_files(self, delete: bool = True) -> None:
        """Creates a new scenario from the files

        Args:
            delete: if True, the folder will be deleted if it already exists

        Returns:
            None
        """

        # Create the folders for the scenario
        self.__create_scenario_folders()

        # Create the files for each region by looping through the structure
        self.__loop_through_dict(self.scenario_structure, path=self.path_config.rsplit('\\', 1)[0],
                                 func=self.__create_agents)

        # Copy the files from the config folder to the scenario folder
        self.__loop_through_dict(self.scenario_structure, path=self.path_config.rsplit('\\', 1)[0],
                                 func=self.__copy_config_to_scenarios)

    def __get_structure(self, name: str, path: str, max_level: int = np.inf) -> dict:
        """
        This will require something similar to what is used to evaluate the config structure, see
            'make_list_from_nested_dict()'
        """

        # Iterate through the config folder and create a dictionary with the structure of the scenario
        dict_struct = {name: {}}
        dict_struct[name] = self.__add_subfolders_to_dict(dict_struct[name], path, max_level)

        return dict_struct

    def __create_agent_files(self, path_config: str, method: str = 'config', fill_from_config: bool = False,
                             overwrite: bool = True) -> None:
        """Creates the agent files that are still missing

        Returns:
            None
        """

        # Create instance of Agents class
        subpath = path_config.replace(self.root_config, '')  # get the subpath of the config file to the root
        agents = Agents(config_path=path_config,
                        config_root=os.path.join(self.root_config, subpath.split('\\', 2)[1]),
                        input_path=self.path_input,
                        scenario_path=os.path.join(self.path_scenarios, subpath.split('\\', 1)[-1]))

        # Create the agent files from the config file
        if method == 'config':
            agents.create_agents_file_from_config(overwrite=overwrite)
        elif method == 'grid':
            agents.create_agents_file_from_grid(fill_from_config=fill_from_config, overwrite=overwrite)

    def __create_grid_files(self, path_config: str, overwrite: bool = True) -> None:
        """TODO: Creates the grid files that are still missing

        Returns:
            None
        """
        pass

    def __create_market_files(self, path_config: str, overwrite: bool = True) -> None:
        """TODO: Creates the market files that are still missing

        Returns:
            None
        """
        # Should actually not be necessary for now since market files stem from the config file
        pass

    def __create_agents(self, path_config: str, overwrite: bool = True) -> None:
        """Creates the agent files of the scenario

        Returns:
            None
        """

        # Create instance of Agents class
        subpath = path_config.replace(self.root_config, '')     # get the subpath of the config file
        agents = Agents(config_path=path_config,
                        config_root=os.path.join(self.root_config, subpath.split('\\', 2)[1]),
                        input_path=self.path_input,
                        scenario_path=self.path_scenarios + subpath)

        # Create the agent files from the config file
        agents.create_agents_from_file()

    def __create_grids(self, path_config: str, overwrite: bool = True) -> None:
        """TODO: Creates the grid files of the scenario

        Returns:
            None
        """
        pass

    def __create_markets(self, path_config: str, overwrite: bool = True) -> None:
        """TODO: Creates the market files of the scenario

        Returns:
            None
        """
        pass

    def __create_scenario_folders(self) -> None:
        """Creates the scenario folders in which the files are put for each region

        Returns:
            None
        """

        # Create the folders for each region by looping through the structure
        self.__loop_through_dict(self.scenario_structure, path=self.path_scenarios,
                                 func=self.__create_folders_from_dict, subfolders=self.subfolders)

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

    # def process_dict(data, level, func):
    #     # check if the current data is a dictionary
    #     if isinstance(data, dict):
    #         # if level is 0, run the function on all items in the dictionary
    #         if level == 0:
    #             for key, value in data.items():
    #                 process_dict(value, level, func)
    #         # if level is not 0, go deeper into the dictionary and run the function at the next level
    #         else:
    #             for key, value in data.items():
    #                 process_dict(value, level - 1, func)
    #     # if the data is not a dictionary, run the function on the data
    #     else:
    #         func(data)

    @classmethod
    def __loop_through_dict(cls, nested_dict: dict, path: str, func: callable, *args, **kwargs) -> None:
        """loops through the dictionary and calls the function for each item

        Args:
            nested_dict: dictionary to loop through
            path: path to the folder
            func: function to call
            *args: arguments for the function
            **kwargs: keyword arguments for the function

        Returns:
            None

        """

        # Loop through all key-value pairs of the dictionary
        for key, value in nested_dict.items():
            # Check if value is a dictionary
            if isinstance(value, dict):
                # If value is a dictionary, Call the function again
                func(os.path.join(path, key), *args, **kwargs)
                cls.__loop_through_dict(value, os.path.join(path, key), func, *args, **kwargs)
            else:
                # If value is not a dictionary, call the function
                print(f'func: {os.path.join(path, key)}')
                func(os.path.join(path, key), *args, **kwargs)

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

    def __copy_config_to_scenarios(self, path_config: str, delete: bool = True) -> None:
        """Copies the config file to the scenario folder

        Returns:
            None
        """

        # Get the path of the scenario folder to which it needs to be copied
        path_scenario = path_config.replace(self.root_config, '')  # get the subpath of the config file to the root
        path_scenario = os.path.join(self.path_scenarios, path_scenario.split('\\', 1)[-1], 'config')

        # Copy (only) files from config to scenario folder
        self.__copy_folder(src=path_config, dst=path_scenario, only_files=True, delete=delete)
