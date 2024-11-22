__author__ = "MarkusDoepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

import os
import time

import polars as pl
from tqdm import tqdm

pl.enable_string_cache(True)
from hamlet import functions as f
from copy import deepcopy
# from numba import njit, jit
from datetime import datetime
from hamlet.executor.utilities.database.database import Database
import hamlet.constants as c
# pl.enable_string_cache(True)
from hamlet.executor.utilities.tasks_execution.agent_task_executioner import AgentTaskExecutioner
from hamlet.executor.utilities.tasks_execution.market_task_executioner import MarketTaskExecutioner
from hamlet.executor.grids.grid import Grid
import warnings

warnings.filterwarnings("ignore")


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

        # Initialize task executioners
        self.agent_task_executioner = AgentTaskExecutioner(self.database, num_workers)
        self.market_task_executioner = MarketTaskExecutioner(self.database, 1)  # use 1 worker only

        # Overwrites the results folder if it already exists
        self.overwrite = overwrite_sim

        # Maximal number of iterations per timesteps (when direct power control is activated)
        self.max_iteration = 1

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
        """Executes the scenario"""
        # Loop through the timetable and execute the tasks for each market for each timestamp
        # Note: The design assumes that there is nothing to be gained for the simulation to run in between market
        #   timestamps. Therefore, the simulation is only executed for the market timestamps
        # Iterate over timetable by timestamp
        # Set the progress bar
        self.pbar.reset(total=len(self.timetable.partition_by('timestamp')))
        self.pbar.set_description(desc='Start execution')

        # Update results path in task executioners
        self.agent_task_executioner.set_results_path(self.path_results)
        self.market_task_executioner.set_results_path(self.path_results)

        for timestamp in self.timetable.partition_by('timestamp'):
            # Wait for the timestamp to be reached if the simulation is to be carried out in real-time
            if self.type == 'rts':
                self.__wait_for_ts(timestamp.iloc[0, 0])

            # init variables for the grid simulation
            grid_ok = False  # init variable, grid is not simulated yet
            num_iteration = 0  # init number of iteration, max. 10

            # get initial database at timestamp, in case this ts need to be overwritten
            initial_db = deepcopy(self.database)

            while not grid_ok:  # iterate until grid is working
                num_iteration += 1

                # get current timestamp as string item for progress bar
                timestamp_str = str(timestamp.select(c.TC_TIMESTAMP).sample(n=1).item())

                # Iterate over timestamp by region_tasks
                for region_tasks in timestamp.partition_by(c.TC_REGION):
                    # get current region_tasks as string item for progress bar
                    region_name = str(region_tasks.select(c.TC_REGION).sample(n=1).item())

                    # update progress bar description
                    self.pbar.set_description(
                        'Executing timestamp ' + timestamp_str + ' for region_tasks ' + region_name + ': ')

                    # Execute agent and market tasks
                    self.agent_task_executioner.execute(region_tasks)
                    self.market_task_executioner.execute(region_tasks)

                # Calculate the grids for the current timestamp (calculated together as they are connected)
                self.pbar.set_description('Executing timestamp ' + timestamp_str + ' for grid: ')
                grid_ok = self.__execute_grids(tasklist=timestamp, initial_db=initial_db, num_iteration=num_iteration)

            self.pbar.update(1)

        # Cleanup the parallel pool
        self.agent_task_executioner.close_pool()
        self.market_task_executioner.close_pool()

    def cleanup(self):
        """Cleans up the scenario after execution"""
        self.database.save_database(os.path.dirname(self.path_results))

        self.database.concat_market_files()

        self.database.save_grid(os.path.dirname(self.path_results))

        self.pbar.set_description('Simulation finished: ')

    def pause(self):
        """Pauses the simulation"""
        raise NotImplementedError("Pause functionality not implemented yet")

    def resume(self):
        """Resumes the simulation"""
        raise NotImplementedError("Resume functionality not implemented yet")

    def __execute_grids(self, tasklist: pl.DataFrame, initial_db: Database, num_iteration: int) -> (bool, dict):
        """Execute grids for the given tasklist."""
        # Only electricity grids is implemented now
        grid_results = {}

        # get grid databases
        grids_data = self.database.get_grid_data()

        # execute grids
        grid_ok = True  # set a base variable for grid status
        for grid_type, grid_db in grids_data.items():   # iterate through all grid types
            result, single_grid_ok = Grid(grid_db=grid_db, tasks=tasklist, grid_type=c.G_ELECTRICITY,
                                          database=self.database).execute()

            grid_ok = grid_ok and single_grid_ok    # each grid should be ok

            grid_results[grid_type] = result

        # if number of iteration exceed maximal number of iteration, set grid_ok to True so that this timestep won't be
        # simulated again
        if num_iteration > self.max_iteration:
            grid_ok = True

        # if grid status is not ok, delete all simulated data for this ts, this ts needs to be simulated again
        if not grid_ok:
            self.database = deepcopy(initial_db)

        # write grid results to database
        for grid_type, grid_db in grid_results.items():
            self.database.post_grids(grid_type=grid_type, grid=grid_db)

        return grid_ok

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

        # assign maximal number of iterations from the database
        self.max_iteration = (self.database.get_general_data()[c.K_GRID][c.K_GRID][c.G_ELECTRICITY]
        ['direct_power_control']['max_iteration'])

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
