__author__ = "MarkusDoepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

import numbers
import os
import time
# Still used by `__wait_for_ts`. The blanket filter this module used to install made that
# `warnings.warn` a guaranteed no-op; removing the filter is what makes it a real warning.
import warnings

import polars as pl
from tqdm import tqdm

pl.enable_string_cache()
from hamlet import functions as f
from copy import deepcopy
# from numba import njit, jit
from datetime import datetime
from hamlet.executor.utilities.database.database import Database
from hamlet.executor.utilities import checkpoint
import hamlet.constants as c
# pl.enable_string_cache(True)
from hamlet.executor.agents.agent import Agent
from hamlet.executor.markets.market import Market
from hamlet.executor.grids.grid import Grid
from hamlet.executor.utilities.grid_restrictions.grid_restriction import GridRestriction
from hamlet.warning_policy import quiet_known_noise


class Executor:

    #: Rejected `num_workers` values get this. Stated once so the test and the message agree.
    NO_PARALLELISM = (
        "num_workers={requested}: HAMLET simulates in one process. The multiprocessing path was "
        "removed because it did not work -- on every shipped example, each worker raised inside "
        "`agent_pool.task`, whose bare `except` turned that into a None the parent then "
        "unpacked. Nothing that ran it noticed, because no run ever used it. Pass 1 or None. "
        "When parallelism returns it will be threads over agents rather than processes, which "
        "needs no state transfer at all: see ROADMAP section 7.3.")

    #: Timesteps between two checkpoints when the caller does not say. A checkpoint writes the
    #: agent tables in full and the run's state, so the interval trades write cost against how
    #: much a crash loses: at this value a crash costs at most a day of an hourly scenario. The
    #: agent tables are cheap -- measured on the Mac mini with design 6's 104 agents: 0.06 s for a
    #: month's tables (48 MB in memory), 0.42 s for a full year's (575 MB). The state pickle is the
    #: larger part: 0.4 s and 111 MB for the 4-agent example, almost all of it the weather table
    #: every forecaster shares (`checkpoint.py`); unmeasured at design 6 scale.
    DEFAULT_CHECKPOINT_INTERVAL = 24

    def __init__(self, path, name: str = None, num_workers: int = None, overwrite_sim: bool = True,
                 allow_incompatible_scenario: bool = False,
                 checkpoint_interval: int | None = DEFAULT_CHECKPOINT_INTERVAL, resume: bool = False):
        # Kept as a parameter, and refused rather than ignored. Every caller in the tree passes 1
        # already; silently running serial for someone who asked for eight would be the same class
        # of quiet wrong answer this executor has been accumulating.
        if num_workers not in (None, 1):
            raise ValueError(self.NO_PARALLELISM.format(requested=num_workers))

        # Every `checkpoint_interval` timesteps the agents' result tables are written to the results
        # folder, so a crash loses at most that many timesteps of them (#249). None or 0 turns it
        # off. Refused rather than coerced: a float or a negative would silently mean "never".
        # `Integral` rather than `int` so a numpy integer read out of a config is accepted; `bool`
        # is an `Integral` too and is not.
        if checkpoint_interval is not None and (
                isinstance(checkpoint_interval, bool)
                or not isinstance(checkpoint_interval, numbers.Integral)
                or checkpoint_interval < 0):
            raise ValueError(f"checkpoint_interval={checkpoint_interval!r}: pass a non-negative "
                             f"integer number of timesteps, or None to disable checkpointing")
        self.checkpoint_interval = int(checkpoint_interval) if checkpoint_interval else None

        # With `resume`, `path` is the *results* folder of a checkpointed run rather than a
        # scenario (#249); see `__prepare_scenario` and `__setup_database`.
        self.resume = resume
        #: After setup on a resumed run, how many timesteps the checkpoint had completed.
        self.resumed_from_step = None
        # The last timestep the checkpoint had completed; the timetable starts after it.
        self.__last_timestamp = None

        # See `pause`.
        self.__pause_requested = False
        self.paused = False

        # Progress bar
        self.pbar = tqdm()

        # Paths
        self.name = name if name else os.path.basename(path)  # Name of the scenario
        self.path_scenario = os.path.abspath(path)  # Path to the scenario folder
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
        self.database = Database(self.path_scenario, name=self.name)

        # Scenario structure
        self.structure = {}

        # Overwrites the results folder if it already exists
        self.overwrite = overwrite_sim

        # Runs a scenario whose format this version of HAMLET does not read. Off by default: the
        # mismatch it guards against changes results without raising anything (see
        # c.SCENARIO_FORMAT_VERSION)
        self.allow_incompatible_scenario = allow_incompatible_scenario

        # Maximal number of iterations per timesteps (when direct power control is activated)
        self.max_iteration = 1

        # Whether any applied grid restriction can reject a pass; decides per run whether the
        # database is copied at the top of every timestep. Set in `__setup_database`.
        self.rollback_possible = True

    def run(self):
        """Runs the simulation.

        The whole run sits inside `quiet_known_noise`, which hides the enumerated polars 0.20
        deprecations HAMLET is knowingly carrying and nothing else. It replaces the blanket
        `warnings.filterwarnings("ignore")` this module used to install at import -- see
        `hamlet/warning_policy.py` and issue #199. Entered here rather than around each stage so
        the filters are swapped once per run instead of once per timestep.
        """
        with quiet_known_noise():
            self.setup()

            self.execute()

            # A paused run is left as something to resume; see `pause`.
            if not self.paused:
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
        self.pbar.set_description(desc='Start execution')

        # A request from before this call belongs to the run that call was made on.
        self.__pause_requested = False

        # A resumed run starts after the checkpoint's last timestep, and counts its steps from the
        # checkpoint's step so that, given the same interval, the checkpoint cadence is the one the
        # uninterrupted run would have had.
        timestamps = self.timetable.partition_by('timestamp')
        first_step = 1
        if self.__last_timestamp is not None:
            timestamps = [frame for frame in timestamps
                          if frame[c.TC_TIMESTAMP][0] > self.__last_timestamp]
            first_step = self.resumed_from_step + 1
        self.pbar.reset(total=len(timestamps))

        for step, timestamp in enumerate(timestamps, start=first_step):
            # Wait for the timestamp to be reached if the simulation is to be carried out in real-time
            if self.type == 'rts':
                self.__wait_for_ts(timestamp.iloc[0, 0])

            # init variables for the grid simulation
            grid_ok = False  # init variable, grid is not simulated yet
            num_iteration = 0  # init number of iteration, max. 10

            # The copy a rejected pass rolls back to. Only taken when a restriction can reject one:
            # it is the whole database, agent tables included, so it scales with scenario length
            # rather than with anything the timestep does -- a full-year scenario holds a second
            # ~575 MB of agent tables all step long, and a third for the instant the next copy is
            # made before this name is rebound, for nothing on every design without direct power
            # control (#249).
            initial_db = deepcopy(self.database) if self.rollback_possible else None

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
                        'Executing timestamp ' + timestamp_str + ' for region_tasks ' + region_name)

                    # Execute agent and market tasks
                    self.__execute_agents(region_tasks)
                    self.__execute_markets(region_tasks)

                # Calculate the grids for the current timestamp (calculated together as they are connected)
                self.pbar.set_description('Executing timestamp ' + timestamp_str + ' for grid')
                grid_ok = self.__execute_grids(tasklist=timestamp, initial_db=initial_db, num_iteration=num_iteration)

            # The timestep is over, whether the loop converged or `__execute_grids` was told to stop
            # on the iteration cap. A direct power control command belongs to the timestep it was
            # computed for, and the next timestep's agents read `restriction_commands` before the
            # grid stage -- so anything left here is imposed on an hour it was never assessed for.
            self.database.end_timestep()

            # After the timestep is closed, never inside the loop above: a pass that is rolled back
            # restores `initial_db`, and a checkpoint taken before that would have written the
            # dispatch the rollback discards.
            # Read once: `pause` can be called from a signal handler between any two bytecodes,
            # and a request landing between two reads would stop the run without a checkpoint.
            pausing = self.__pause_requested
            due = self.checkpoint_interval and step % self.checkpoint_interval == 0
            if due or pausing:
                self.pbar.set_description('Checkpointing after timestamp ' + timestamp_str)
                self.__checkpoint(step, timestamp[c.TC_TIMESTAMP][0])

            self.pbar.update(1)

            if pausing:
                self.paused = True
                self.pbar.set_description('Paused after timestamp ' + timestamp_str)
                break

    def cleanup(self):
        """Cleans up the scenario after execution"""
        self.database.save_database(os.path.dirname(self.path_results))

        self.database.concat_market_files()

        # A finished run has nothing to resume, and the state file is large.
        checkpoint.remove(self.path_results)

        self.pbar.set_description('Simulation finished')

    def pause(self):
        """Ask the run to checkpoint and stop once the timestep in progress is over.

        Safe to call from another thread or a signal handler: it only sets a flag the loop reads
        once per timestep boundary. `run` then skips `cleanup` -- which would concatenate the
        market fragments and delete them, and a resumed run needs them intact -- so the results
        folder is exactly a checkpointed run: `Executor(<results folder>, resume=True).run()`
        finishes it. This object is done; calling `run` on it again would start over.
        """
        self.__pause_requested = True

    def __checkpoint(self, step: int, last_timestamp):
        """Write the agent tables for the analyzer, then the state for a resume (#249).

        The two are separate on purpose. The tables are results, at the paths the final save
        overwrites, so a crashed run's agents are readable without resuming; the state is the
        whole database graph, readable by nothing but `resume`.
        """
        self.database.save_agents(os.path.dirname(self.path_results))
        checkpoint.write(self.path_results, self.database, step, last_timestamp, self.checkpoint_interval)

    def __execute_agents(self, tasks: pl.DataFrame):
        """Run every agent of this region for this timestamp, and post what they produced.

        Moved here from `AgentTaskExecutioner.execute_serial` and `.postprocess_results` when the
        multiprocessing path was deleted. The two were split across a base class only so that the
        parallel branch could reuse the second half; with one branch left, the split cost a file
        and bought nothing.
        """
        region_name = str(tasks.select(c.TC_REGION).sample(n=1).item())

        # Get the data of the agents that are part of the tasklist
        agents = self.database.get_agent_data(region=region_name)

        # Get the data of the markets that are part of the tasklist
        markets = self.database.get_market_data(region=region_name)

        # Get grid restriction commands
        grid_commands = {}
        for grid_type, grid in self.database.get_grid_data().items():
            grid_commands[grid_type] = grid.restriction_commands

        # Iterate over the agents and execute them sequentially
        results = []
        for agent_type, agent in agents.items():
            for agent_id, agent_db in agent.items():
                # Update save path for agent
                agent_db.agent_save = os.path.join(self.path_results, 'agents', agent_type, agent_id)
                # Create an instance of the Agent class and execute its tasks
                results.append(Agent(agent_type=agent_type, data=agent_db, timetable=tasks, market=markets,
                                     grid_commands=grid_commands).execute())

        # Update agents data in database
        self.database.post_agents_to_region(region=region_name, agents=results)

    def __execute_markets(self, tasks: pl.DataFrame):
        """Clear every market of this region for this timestamp, and post the results.

        Moved here from `MarketTaskExecutioner`, which never had a parallel branch at all: the
        Executor constructed it with one worker and no `MarketPool` was ever built.
        """
        markets = []
        for task in tasks.iter_rows(named=True):
            market = self.database.get_market_data(region=task[c.TC_REGION],
                                                   market_type=task[c.TC_MARKET],
                                                   market_name=task[c.TC_NAME])
            markets.append(Market(data=market, tasks=task, database=self.database))

        results = []
        for market in markets:
            results.append(market.execute())

        region_name = tasks.select(pl.first(c.TC_REGION)).item()
        timestamp = tasks.select(c.TC_TIMESTAMP).sample(n=1).item()
        self.database.post_markets_to_region(region=region_name, markets=results, timestamp=timestamp,
                                             path_results=self.path_results)

    def __execute_grids(self, tasklist: pl.DataFrame, initial_db: Database | None,
                        num_iteration: int) -> (bool, dict):
        """Execute grids for the given tasklist.

        `initial_db` is None when no restriction declared it could reject a pass; a rejection is
        then an error, because continuing from the rejected state would be the silent kind.
        """
        # Only electricity grids is implemented now
        grid_results = {}

        # get grid databases
        grids_data = self.database.get_grid_data()

        # execute grids
        grid_ok = True  # set a base variable for grid status
        for grid_type, grid_db in grids_data.items():   # iterate through all grid types
            result, single_grid_ok = Grid(grid_db=grid_db, tasks=tasklist, grid_type=grid_type,
                                          database=self.database).execute()

            grid_ok = grid_ok and single_grid_ok    # each grid should be ok

            grid_results[grid_type] = result

        # if number of iteration exceed maximal number of iteration, set grid_ok to True so that this timestep won't be
        # simulated again
        # `converged` is the grid's own verdict, before the cap overrides it. The two differ on the
        # cap pass: a timestep can converge exactly there, and then the last frame is the obeyed one
        # and must stand.
        converged = grid_ok
        forced_exit = num_iteration > self.max_iteration and not converged
        if num_iteration > self.max_iteration:
            grid_ok = True

        # if grid status is not ok, delete all simulated data for this ts, this ts needs to be simulated again
        if not grid_ok:
            if initial_db is None:
                raise RuntimeError(
                    'a grid restriction rejected a pass, but every applied restriction declared '
                    'through `can_reject_a_pass` that it could not, so no copy was taken to roll '
                    'back to. Fix the declaration on the restriction class.')
            self.database = deepcopy(initial_db)

        # write grid results to database
        for grid_type, grid_db in grid_results.items():
            self.database.post_grids(grid_type=grid_type, grid=grid_db,
                                     path_results=self.path_results)

        # On a forced exit the pass just computed a command no agent will run under: the loop is
        # over and the dispatch being kept obeyed the pass before. Publish that one (#248). Not on
        # a pass that converged on its own, cap or no cap -- there the last frame is right.
        if forced_exit:
            self.database.restore_superseded_commands()

        return grid_ok

    def __prepare_scenario(self):
        """Prepares the scenario"""

        # Load general information and configuration
        self.general = f.load_file(os.path.join(self.path_scenario, 'general', 'general.json'))

        # Check the scenario format before anything is read out of the folder. A scenario written
        # by a different Creator is not read wrongly here -- it is read successfully and produces
        # different numbers, so this has to be a refusal rather than a warning
        f.check_scenario_format(self.general, self.path_scenario, self.allow_incompatible_scenario)

        self.config = f.load_file(os.path.join(self.path_scenario, 'config', 'setup.yaml'))

        # Load timetable
        self.timetable = f.load_file(os.path.join(self.path_scenario, 'general', 'timetable.ft'),
                                     df='polars', method='eager')

        # Load scenario structure
        self.structure = self.general['structure']

        if self.resume:
            # The folder given *is* the results folder: the scenario copy inside it is what the
            # run reads, and the copy that would normally be made here is what it would destroy.
            self.path_results = self.path_scenario
            return

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
        """Creates a database connector object, or restores the checkpointed one on a resume."""

        if self.resume:
            state = checkpoint.read(self.path_results)
            # The pickled database writes under its own run name, so a results folder renamed
            # before the resume would finish into a re-created folder of the old name, silently.
            if state['database'].run_name != self.name:
                raise RuntimeError(
                    f'the checkpoint under {self.path_results} belongs to a run named '
                    f'{state["database"].run_name!r}; the folder is named {self.name!r}. A results '
                    f'folder can be moved but not renamed: `general.json` and the checkpoint both '
                    f'carry its name.')
            self.database = state['database']
            self.resumed_from_step = state['step']
            self.__last_timestamp = state['last_timestamp']
            checkpoint.restore_rng(state)
            checkpoint.truncate_grid_csvs(self.path_results, state)
        else:
            self.database.setup_database(self.structure)
            if self.checkpoint_interval:
                checkpoint.check_picklable(self.database)

        # assign maximal number of iterations from the database
        self.max_iteration = (self.database.get_general_data()[c.K_GRID][c.G_ELECTRICITY]['restrictions']
                              ['max_iteration'])

        self.rollback_possible = self.__rollback_possible()

    def __rollback_possible(self) -> bool:
        """Whether any applied restriction of a registered grid can reject a pass in this run.

        Mirrors `Electricity.execute`: the restrictions it runs are `restrictions.apply` of the
        electricity grid's general config, each with its own sub-config, and `grid_ok` comes from
        nowhere else -- the heat and hydrogen grids have no `execute`.
        """
        if c.G_ELECTRICITY not in self.database.get_grid_data():
            return False
        restrictions = self.database.get_general_data()[c.K_GRID][c.G_ELECTRICITY]['restrictions']
        return any(GridRestriction.can_reject_a_pass(name, restrictions[name])
                   for name in restrictions['apply'])

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
