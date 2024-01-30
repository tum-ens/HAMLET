__author__ = "MarkusDoepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

import os
import sys
from tqdm import tqdm
import shutil
import time
import warnings
import multiprocessing as mp
import pandas as pd
from ruamel.yaml import YAML
from pprint import pprint
import json
import polars as pl
pl.enable_string_cache(True)
from hamlet import functions as f
# from numba import njit, jit
import pandapower as pp
import concurrent.futures
from typing import Callable
from datetime import datetime
from hamlet.executor.agents.agent import Agent
from hamlet.executor.markets.market import Market
from hamlet.executor.grids.grid import Grid
from hamlet.executor.utilities.database.database import Database
import hamlet.constants as c
# pl.enable_string_cache(True)
from copy import copy

# TODO: Considerations
# - Use Callables to create a sequence for all agents in executor: this was similarly done in the creator_backup and should be continued for consistency
# - Possible packages for multiprocessing: multiprocessing, joblib, threading (can actually be useful when using not just pure python)
# - Decrease file size wherever possible (define data types, shorten file lengths, etc.) -> this needs to occur in the scenario creation
# - Load all files into the RAM and not read/save as in lemlab to increase performance
# - Use polars instead of pandas to increase performance --> finding: polars is faster as long as lazy evaluation is used. Otherwise pandas 2.0 can keep up well (depending on use case)
# - Always check if numba can help improve performance


class Executor:

    def __init__(self, path_scenario, name: str = None, num_workers: int = None, overwrite_sim: bool = True):

        # Progress bar
        self.pbar = tqdm()

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
        self.type = None  # set in self.__prepare_scenario()

        # Database containing all information
        self.database = Database(self.path_scenario)

        # Scenario structure
        self.structure = {}  # TODO: this will need to contain more information than just the path. Also: above and below markets to know where to look for the data

        # Number of workers for parallelization
        self.num_workers = num_workers

        # Thread pool for parallelization
        self.pool = None

        # Overwrites the results folder if it already exists
        self.overwrite = overwrite_sim

    def run(self):
        """Runs the simulation"""

        self.setup()

        self.execute()

        self.cleanup()

    def setup(self):
        """Sets up the scenario before execution"""

        self.__prepare_scenario()

        self.__setup_database()

    def execute(self):
        """Executes the scenario

        """

        # Get number of logical processors (for parallelization)
        # TODO: Benchmark with different numbers of workers
        if not self.num_workers:
            self.num_workers = os.cpu_count() - 1  # logical processors (threads) - 1
            # self.num_workers = mp.cpu_count() - 1  # physical processors - 1
            # self.num_workers = len(os.sched_getaffinity(0))  # number of usable CPUs

        # Setup up the thread pool for parallelization
        if self.num_workers > 1:
            # TODO: Benchmark with ProcessPoolExecutor
            self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers)

        # Loop through the timetable and execute the tasks for each market for each timestamp
        # Note: The design assumes that there is nothing to be gained for the simulation to run in between market
        #   timestamps. Therefore, the simulation is only executed for the market timestamps
        # Iterate over timetable by timestamp
        # Set the progress bar
        self.pbar.reset(total=len(self.timetable.partition_by('timestamp')))
        self.pbar.set_description(desc='Start execution')

        for timestamp in self.timetable.partition_by('timestamp'):
            # Wait for the timestamp to be reached if the simulation is to be carried out in real-time
            if self.type == 'rts':
                self.__wait_for_ts(timestamp.iloc[0, 0])

            # get current timestamp as string item for progress bar
            timestamp_str = str(timestamp.select(c.TC_TIMESTAMP).sample(n=1).item())

            # Iterate over timestamp by region
            for region in timestamp.partition_by(c.TC_REGION):
                # get current region as string item for progress bar
                region_str = str(region.select(c.TC_REGION).sample(n=1).item())

                # update progress bar description
                self.pbar.set_description('Executing timestamp ' + timestamp_str + ' for region ' + region_str + ': ')

                # Execute the agents and market in parallel or sequentially
                if self.pool:
                    # Execute the agents for this market
                    self.__execute_agents_parallel(tasklist=region)

                    # Execute the market
                    # TODO: Replace this with a parallel execution once it works
                    self.__execute_markets(tasklist=region)
                else:
                    # Execute the agents for this market
                    self.__execute_agents(tasklist=region)

                    # Execute the market
                    self.__execute_markets(tasklist=region)

            # Calculate the grids for the current timestamp (calculated together as they are connected)
            self.pbar.set_description('Executing timestamp ' + timestamp_str + ' for grid: ')

            self.__execute_grids()

            self.pbar.update(1)

        # Cleanup the thread pool
        if self.pool:
            self.pool.shutdown()

    def cleanup(self):
        """Cleans up the scenario after execution"""
        self.database.save_database(os.path.dirname(self.path_results))

    def pause(self):
        """Pauses the simulation"""
        raise NotImplementedError("Pause functionality not implemented yet")

    def resume(self):
        """Resumes the simulation"""
        raise NotImplementedError("Resume functionality not implemented yet")

    def __execute_agents_parallel(self, tasklist: pl.DataFrame):
        """Executes all agent tasks for all agents in parallel"""

        # Define the function to be executed in parallel
        def tasks(agent):
            # Execute the agent
            return agent.execute()

        # Get the data of the agents that are part of the tasklist
        region = tasklist.select(pl.first(c.TC_REGION)).item()
        agents = self.database.get_agent_data(region=region)

        # Create a list to store the agents
        agents_list = []

        # Iterate over the tasklist and populate the agents_list
        for agent_type, agent in agents.items():
            for agent_id, data in agent.items():
                agents_list.append(Agent(agent_type=agent_type, data=agent[agent_id], timetable=tasklist,
                                         database=self.database))

        # Submit the agents for parallel execution
        futures = [self.pool.submit(tasks, agent) for agent in agents_list]

        # Wait for all agents to complete
        concurrent.futures.wait(futures)

        # Retrieve and process results from the futures
        results = []
        for future in futures:
            try:
                result = future.result()
                results.append(result)
                # Optionally process each result as needed
                # e.g., logging or additional computations
            except Exception as e:
                # Handle exceptions (e.g., log them)
                raise f"An error occurred when retrieving agent results: {e}"

        # Post the agent data back to the database
        self.database.post_agents_to_region(region=region, agents=results)

    def __execute_market_parallel(self, tasklist: pl.DataFrame):
        """Executes the market tasks in parallel"""

        # Define the function to be executed in parallel
        def tasks(market):
            # Execute the market
            return market.execute()

        # Create a list to store the markets
        markets_list = []

        # Iterate over the tasklist and populate the markets_list
        for task in tasklist.iter_rows(named=True):
            # Get the market data for the current market
            market = self.database.get_market_data(region=task[c.TC_REGION],
                                                   market_type=task[c.TC_MARKET],
                                                   market_name=task[c.TC_NAME])
            market = copy(market)
            # Create an instance of the Market class and append it to the markets_list
            markets_list.append(Market(data=market, tasks=task, database=self.database))


        # Submit the markets for parallel execution
        futures = [self.pool.submit(tasks, market) for market in markets_list]

        # Wait for all markets to complete
        # TODO: For some reason it gets stuck here. Investigate why.
        concurrent.futures.wait(futures)

        # Retrieve and process results from the futures
        results = []
        for future in futures:
            try:
                result = future.result(timeout=60)
                results.append(result)
                # Optionally process each result as needed
                # e.g., logging or additional computations
            except Exception as e:
                # Handle exceptions (e.g., log them)
                raise f"An error occurred when retrieving market results: {e}"

        print(results)
        exit()

        # Post the agent data back to the database
        self.database.post_markets_to_region(region=task[c.TC_REGION], markets=results)

    def __execute_agents(self, tasklist: pl.DataFrame):
        """Executes all agent tasks for all agents sequentially
        """

        # Get the data of the agents that are part of the tasklist
        region = tasklist.select(pl.first(c.TC_REGION)).item()
        agents = self.database.get_agent_data(region=region)

        # Create a list to store the results
        results = []

        # Iterate over the agents and execute them sequentially
        for agent_type, agent in agents.items():
            for agent_id, data in agent.items():
                # Create an instance of the Agents class and execute its tasks
                results.append(Agent(agent_type=agent_type, data=agent[agent_id], timetable=tasklist,
                                     database=self.database).execute())

        # Post the agent data back to the database
        self.database.post_agents_to_region(region=region, agents=results)

    def __execute_markets(self, tasklist: pl.DataFrame):

        # Create a list to store the results
        results = []

        # Iterate over tasklist row by row
        for tasks in tasklist.iter_rows(named=True):
            # Get the market data for the current market
            market = self.database.get_market_data(region=tasks[c.TC_REGION],
                                                   market_type=tasks[c.TC_MARKET],
                                                   market_name=tasks[c.TC_NAME])
            # Create an instance of the Market class and execute its tasks
            results.append(Market(data=market, tasks=tasks, database=self.database).execute())

        # counter = 0
        # for result in results:
        #     with pl.Config(set_tbl_width_chars=400, set_tbl_cols=25, set_tbl_rows=100):
        #         print(result.market_transactions)
        #     counter += 1
        #     if counter > 1:
        #         break

        # Post the agent data back to the database
        self.database.post_markets_to_region(region=tasks[c.TC_REGION], markets=results)

    def __execute_grids(self):

        return
        # Pass info to grids class and execute its tasks
        Grid().execute()

    def __prepare_scenario(self):
        """Prepares the scenario"""

        # Load general information and configuration
        self.general = f.load_file(os.path.join(self.path_scenario, 'general', 'general.json'))
        self.config = f.load_file(os.path.join(self.path_scenario, 'config', 'config_setup.yaml'))

        # Load timetable
        self.timetable = f.load_file(os.path.join(self.path_scenario, 'general', 'timetable.ft'),
                                     df='polars', method='eager')

        # Load scenario structure
        self.structure = self.general['structure']

        # Set the results path
        self.path_results = os.path.join(self.config['paths']['results'], self.name)
        # Check if the results folder exists and stop simulation if overwrite is set to False
        if os.path.exists(self.path_results) and self.overwrite is False:
            raise FileExistsError(f"Results folder already exists. "
                                  f"Set overwrite to True to overwrite the results folder.")
        # Copy the scenario folder to the results folder
        # Note: For the execution the files in the results folder are used and not the ones in the scenario folder
        f.copy_folder(self.path_scenario, self.path_results)

    def __setup_database(self):
        """Creates a database connector object"""

        self.database.setup_database(self.structure)

    @staticmethod
    def __wait_for_ts(timestamp):
        """Waits until the target timestamp is reached"""

        # Get current datetime
        current_datetime = datetime.now()

        # Calculate time difference
        time_difference = (timestamp - current_datetime).total_seconds()

        # Wait until the target time is reached
        if time_difference > 0:
            time.sleep(time_difference)
        elif time_difference < 0:
            warnings.warn(f"Target time is in the past: {timestamp} vs. {current_datetime}")

        return
