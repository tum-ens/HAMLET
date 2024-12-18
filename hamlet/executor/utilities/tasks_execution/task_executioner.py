__author__ = "HodaHamdy"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

import multiprocessing as mp
import os
import shutil

from hamlet import functions as f
import hamlet.constants as c
from hamlet.executor.utilities.database.agent_db import AgentDB
from hamlet.executor.utilities.forecasts.forecaster import Forecaster
from hamlet.executor.utilities.database.market_db import MarketDB

import psutil

cached_general = None
def get_general(path) -> dict:
    """Loads general information"""
    global cached_general
    if cached_general is None:
        return {'weather': f.load_file(path=os.path.join(path, 'general', 'weather', 'weather.ft'), df='polars', method='eager', memory_map=False),
                   'retailer': f.load_file(path=os.path.join(path, 'general', 'retailer.ft'),
                                           df='polars', method='eager'),
                   'tasks': f.load_file(path=os.path.join(path, 'general', 'timetable.ft'),
                                        df='polars', method='eager'),
                   'general': f.load_file(path=os.path.join(path, 'config', 'config_setup.yaml'))}
    return cached_general

def init_agentdb(agent_type, agent_id, region_tasks, region_path, agent_path):
    """Initializes agent database"""
    agent_db = AgentDB(path=agent_path,
                       agent_type=agent_type,
                       agent_id=agent_id)
    agent_db.agent_save = agent_path
    # Load data from files
    agent_db.register_agent()

    market_db, market_type = get_market(region_path, region_tasks)
    add_forecaster(agent_db, market_db, market_type, region_path)
    return agent_db

def get_market(region_path, region_tasks):
    """Gets market database"""
    market_type = str(region_tasks.select(c.TC_MARKET).sample(n=1).item())
    market_name = str(region_tasks.select(c.TC_NAME).sample(n=1).item())
    market_db = MarketDB(market_type=market_type,
                         name=market_name,
                         market_path=os.path.join(region_path, 'markets',
                                                  market_type, market_name),
                         retailer_path=os.path.join(region_path, 'retailers',
                                                    market_type, market_name))
    market_db.load_market_from_files(market_transactions_only=True)
    market_db = {market_type: {market_name: market_db}}
    return market_db, market_type

def add_forecaster(agent_db, market_db, market_type, region_path):
    """Adds agent forecaster to the agent database"""
    general = get_general(region_path)
    forecaster = Forecaster(agentDB=agent_db, marketsDB=market_db[market_type], general=general)
    forecaster.init_forecaster()
    # TODO if forecaster is updated during the simulation (e.g. by update_local_market_in_forecasters),
    #  we need to load it here correctly
    agent_db.forecaster = forecaster  # register

class TaskExecutioner:
    """
    A base class to manage tasks execution

    Attributes:
        database (Database): database instance
        num_workers (int): number of workers
        pool: None. Needs to be initialized by child classes if multiprocessing is enabled
        results_path (str): results path
    """
    MIN_GB_AVAILABLE = 35

    def __init__(self, database, num_workers):
        self.database = database
        self.num_workers = num_workers
        if not self.num_workers:
            self.num_workers = max(1, mp.cpu_count() - 1)  # physical processors
        self.pool = None
        self.results_path = None

    def execute(self, tasks):
        """Executes input tasks"""
        # Check if multiprocessing is enabled
        if self.num_workers == 1 or self.pool is None:
            results = self.execute_serial(tasks)
        else:
            # Prepare parallel tasks
            para_tasks = self.prepare_para_tasks(tasks)
            # Save database to file to allow loading inside each process
            self.database.save_database(os.path.dirname(self.results_path))
            # Update workers according to number of required parallel tasks
            self.update_num_workers(len(para_tasks))
            # Also update the pool's workers
            self.pool.update_num_workers(self.num_workers)
            # Execute multiprocessing pool
            result_dirs = self.pool.execute(para_tasks)
            results = list(map(lambda x: init_agentdb(*x), result_dirs))
            for fn in result_dirs:
                print(fn)
                shutil.rmtree(fn[4])
        # Postprocess results of tasks execution
        self.postprocess_results(tasks, results)

    def enough_memory(self):
        available_gigabytes = psutil.virtual_memory().available / (1024.0 ** 3)
        return available_gigabytes >= self.MIN_GB_AVAILABLE

    def close_pool(self):
        """Closes pool"""
        if self.pool is not None:
            self.pool.close()

    def prepare_para_tasks(self, tasks):
        """Prepares parallel tasks"""
        return tasks

    def update_num_workers(self, num_tasks):
        """Updates number of workers"""
        self.num_workers = max(1, min(num_tasks, self.num_workers))

    def execute_serial(self, tasks):
        """Executes serial tasks"""
        raise NotImplementedError

    def postprocess_results(self, tasks, results):
        """Post-processes results"""
        raise NotImplementedError

    def set_results_path(self, results_path):
        """Set results path"""
        self.results_path = results_path
