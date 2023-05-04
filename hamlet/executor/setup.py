__author__ = "MarkusDoepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

import os
import shutil
import time
import pandas as pd
from ruamel.yaml import YAML
from pprint import pprint
import json
import polars as pl
from hamlet import functions as f
from numba import njit, jit
import pandapower as pp
from typing import Callable
from hamlet.executor.agents.agents import Agents
from hamlet.executor.markets.markets import Markets
from hamlet.executor.grids.grids import Grids
pl.StringCache()

# TODO: Considerations
# - Use Callables to create a sequence for all agents in executor: this was similarly done in the creator and should be continued for consistency
# - Possible packages for multiprocessing: multiprocessing, joblib, threading (can actually be useful when using not just pure python)
# - Decrease file size wherever possible (define data types, shorten file lengths, etc.) -> this needs to occur in the scenario creation
# - Load all files into the RAM and not read/save as in lemlab to increase performance
# - Use polars instead of pandas to increase performance --> finding: polars is faster as long as lazy evaluation is used. Otherwise pandas 2.0 can keep up well (depending on use case)
# - Always check if numba can help improve performance


class Executor:

    def __init__(self, path_scenario, name: str = None, delete_results: bool = True):

        # Paths
        self.name = name if name else os.path.basename(path_scenario)  # Name of the scenario
        self.path_scenario = os.path.abspath(path_scenario)  # Path to the scenario folder
        self.root_scenario = os.path.dirname(self.path_scenario)  # Path to the root folder of the scenario
        self.path_results = None  # Path to the results folder

        # Scenario general information and configuration
        self.general = None
        self.config = None

        # Scenario timetable
        self.timetable = None

        # Scenario type (sim or in the future also rts)
        self.type = None

        # Scenario structure
        self.structure = {}

        # Agents data (contains all information of each agent in a structured dict)
        self.agents = {}

        # Grids data (contains all grids)
        self.grids = {}

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

        # TODO: Put back in:

        # self.__setup_db_connection()
        #
        # self.__setup_markets()
        #
        self.__setup_grids()

        self.__setup_agents()

    def compare_pandas_polars(self, engine: str, runs: int = 100000) -> float:
        # TODO: To be removed. This is just for testing purposes

        # POLARS

        # create a Polars DataFrame from the pandas DataFrame
        df = self.timetable
        print(df.estimated_size()/1e6)

        df_p = df.to_pandas()
        print(df_p.info())

        start = time.time()

        if engine == 'polars':

            # group by the 'timestamp' column and get the unique keys
            for _ in range(runs):
                pltt = df.groupby('timestamp')

            for row in pltt:
                # with pl.Config() as cfg:
                #     cfg.set_tbl_cols(30)
                print(row)
                break

        elif engine == 'pandas':

            # PANDAS

            for _ in range(runs):
                pdtt = df_p.groupby('timestamp')

            for row in pdtt:
                print(row[1].head(10).to_string())
                break

        return time.time() - start

    def execute(self):
        """Executes the scenario"""

        # TODO: There is one more loop to do. For now it just goes over the timetable but we actually defined the
        #  timestep size in the config file. This either needs to be taken out or also be considered in the loop

        # TODO: set up the parallelization here

        # Loop through the timetable and execute the tasks for each market for each timestamp
        # Iterate over timetable by timestamp
        for timestamp in self.timetable.partition_by('timestamp'):
            # Iterate over timestamp by region
            for region in timestamp.partition_by('region'):
                # Iterate over region by market
                for market in region.partition_by('market'):
                    # Iterate over market by name:
                    for name in market.partition_by('name'):

                        # Execute the agents for this market
                        self.__execute_agents(tasklist=name)

                        # Execute the market
                        self.__execute_markets(tasklist=name)

            # Calculate the grids for the current timestamp (calculated together as they are connected)
            self.__execute_grids()

    def __execute_agents(self, tasklist: pl.DataFrame):
        """Executes all general tasks for all agents

        TODO: Think about how to make this step parallel (maybe even pass the agents to the agents class and from there
              start the parallelization)
        """

        # TODO: Get the data of the agents that are part of the tasklist
        data = self.agents

        # Pass info to agents class and execute its tasks
        Agents(data, tasklist).execute()

    def __execute_markets(self, tasklist: pl.DataFrame):

        # Pass info to markets class and execute its tasks
        Markets(tasklist).execute()

    def __execute_grids(self, tasklist: pl.DataFrame):

        # Pass info to grids class and execute its tasks
        Grids(tasklist).execute()

    def pause(self):
        """Pauses the simulation"""
        raise NotImplementedError("Pause functionality not implemented yet")

    def resume(self):
        """Resumes the simulation"""
        raise NotImplementedError("Resume functionality not implemented yet")

    def save_results(self):
        """Saves the (current) results of the simulation"""
        ...

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
        self.timetable = f.load_file(os.path.join(self.path_scenario, 'general', 'timetable.ft'),
                                     df='polars', method='eager')

        # Load scenario structure
        self.structure = self.general['structure']

        # Copy the scenario folder to the results folder
        # Note: For the execution the files in the results folder are used and not the ones in the scenario folder
        self.path_results = os.path.join(self.config['simulation']['paths']['results'], self.name)
        # TODO: Put back in: f.copy_folder(self.path_scenario, self.path_results)

    def __setup_db_connection(self):
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

        # TODO: Discuss if separate registries or everything handed over in bulk

        # Register markets in database
        db.register_markets()  # TODO: @Jiahe: Implement this function

        # Register retailers in database
        db.register_retailers()  # TODO: @Jiahe: Implement this function

    def __setup_grids(self):
        """Sets up the grids, i.e. loads the grid data from the general scenario folder"""

        # Load all combined grid files
        # Note: Currently this is only one file as only electricity is available. In the future this will be more and
        # required the function that was commented out
        self.grids['electricity'] = pp.from_json(os.path.join(self.path_scenario, 'general', 'grid.json'))
        # self.grids = self.__add_nested_data(path=os.path.join(self.path_scenario, 'general', 'grids'))

    def __setup_agents(self):
        """Sets up the agents"""

        # Select the database connection
        # db = self.db_connections['user'] # TODO: Put back in once function is implemented

        # TODO: Discuss if separate registries or everything handed in bulk

        # TODO: See which other files need to be created
        # Load all agent files by looping through the scenario files
        self.agents = f.loop_folder(src=self.root_scenario, struct=self.structure, folder='agents',
                                    func=f.add_nested_data, df='polars', method='lazy', parse_dates=[0])

        # Register agents in database (this means registering the user and the meters)
        # db.register_agents(self.agents)  # TODO: @Jiahe: Implement this function

    def __close_db_connection(self):
        ...
