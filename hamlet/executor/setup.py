__author__ = "TUM-Doepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "TUM-Doepfert"
__email__ = "markus.doepfert@tum.de"

import os
import shutil
import time
import pandas as pd
from ruamel.yaml import YAML
from pprint import pprint
import json
from .. import functions as f

# TODO: Consideration for the development of the executor
# Use Callables to create a sequence for all agents in executor: https://realpython.com/python-ellipsis/?__s=dwe8kijzzlj1n7oskl9x
# Possible packages for multiprocessing: multiprocessing, joblib
# Decrease file size wherever possible (e.g. use numpy arrays instead of pandas dataframes, define data types, shorten file lengths, etc.)
# Load all files into the RAM and not read/save as in lemlab to increase performance


class Executor:

    def __init__(self, path_scenario, name: str = None, delete_results: bool = True):

        # Paths
        self.name = name if name else os.path.basename(path_scenario)  # Name of the scenario
        self.path_scenario = path_scenario  # Path to the scenario folder
        self.path_results = None  # Path to the results folder

        # Scenario general information and configuration
        self.general = None
        self.config = None

        # Scenario timetable
        self.timetable = None

        # Scenario type (sim or in the future also rts)
        self.type = None

        # Scenario structure (load from general.json 'structure')
        self.structure = None

        # Database connections (currently contains 'admin' and 'user')
        self.db_connections = {}

    def run(self):
        """Runs the simulation"""

        self.setup()

        self.execute()

        self.stop()

    def setup(self):
        """Sets up the scenario before execution"""

        self.__prepare_scenario()

        self.__create_db_connection()

        self.__setup_markets()

        self.__setup_grids()

        self.__setup_agents()

    def execute(self):
        """Executes the scenario"""

        # TODO: Iterate over timetable and execute tasks
        # From here on there would be a for-loop iterating over the timetable

        self.__execute_agents()

        self.__execute_markets()

        self.__execute_grids()

    def pause(self):
        """Pauses the simulation"""

        raise NotImplementedError("Pause functionality not implemented yet")

    def resume(self):
        """Resumes the simulation"""

        raise NotImplementedError("Resume functionality not implemented yet")

    def save_results(self):
        """Saves the (current) results of the simulation"""

        pass

    def stop(self):
        """Cleans up the scenario after execution"""

        self.save_results()

        self.__close_db_connection()

    def __prepare_scenario(self):
        """Prepares the scenario"""

        # Load general information and configuration
        self.general = f.load_file(os.path.join(self.path_scenario, 'general', 'general.json'))
        self.config = f.load_file(os.path.join(self.path_scenario, 'config', 'config_general.yaml'))

        # Load timetable
        self.timetable = f.load_file(os.path.join(self.path_scenario, 'general', 'timetable.ft'))

        # Copy the scenario folder to the results folder
        # Note: For the execution the files in the results folder are used and not the ones in the scenario folder
        self.path_results = os.path.join(self.config['simulation']['paths']['results'], self.name)
        f.copy_folder(self.path_scenario, self.path_results)

    def __create_db_connection(self):
        """Creates a database connector object"""

        # Setup database connections
        # TODO: Iterate over the database connections defined in the scenario config and create a database connection

        # Pseudocode:
        # for name, info in self.scenario_config["database"].items():
        #     self.db_connections[name] = DBConnection(info["host"], info["port"], info["user"], info["pw"], info["db"])

    def __setup_markets(self):
        """Sets up the markets"""

        # Set the market type
        self.type = self.config['simulation']['type']

        # Select the database connection
        db = self.db_connections['admin']

        # TODO: Discuss if separate registries or everything handed in bulk

        # Register markets in database
        db.register_markets()  # TODO: @Jiahe: Implement this function

        # Register retailers in database
        db.register_retailers()  # TODO: @Jiahe: Implement this function

    def __setup_grids(self):
        """Sets up the grids

        Note: For now this should not need to do anything but is kept for consistency"""
        pass

    def __setup_agents(self):
        """Sets up the agents"""

        # Select the database connection
        db = self.db_connections['user']

        # TODO: Discuss if separate registries or everything handed in bulk

        # Register agents in database (this means registering the user and the meters)
        db.register_agents()  # TODO: @Jiahe: Implement this function

        # TODO: See which other files need to be created

    def __execute_agents(self):
        pass

    def __execute_markets(self):
        pass

    def __execute_grids(self):
        pass

    def __close_db_connection(self):
        pass

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

    @staticmethod
    def _load_file(path: str, index: int = 0):
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

    @staticmethod
    def _save_file(path: str, data, index: bool = True) -> None:
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