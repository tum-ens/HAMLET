__author__ = "HodaHamdy"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

import os
import pickle
import hamlet.constants as c
from hamlet import functions as f
from hamlet.executor.agents.agent import Agent
from hamlet.executor.utilities.database.agent_db import AgentDB
from hamlet.executor.utilities.database.market_db import MarketDB
from hamlet.executor.utilities.forecasts.forecaster import Forecaster
from hamlet.executor.utilities.tasks_execution.process_pool import ProcessPool


# Define the function to be executed in parallel
def task(agent_data):
    # Prepare agent data
    agent_type, agent_id, region_tasks, region_path = agent_data
    agent_db = init_agentdb(agent_type, agent_id, region_path)
    market_db, market_type = get_market(region_path, region_tasks)
    add_forecaster(agent_db, market_db, market_type, region_path)
    grid_commands = get_grid_restriction_commands(region_path)
    # Initialize and execute the agent instance
    agent = Agent(agent_type=agent_type, data=agent_db, timetable=region_tasks, market=market_db,
                  grid_commands=grid_commands)
    agent_db = agent.execute()
    return agent_db


def init_agentdb(agent_type, agent_id, region_path):
    """Initializes agent database"""
    agent_path = os.path.join(region_path, 'agents', agent_type, agent_id)
    agent_db = AgentDB(path=agent_path,
                       agent_type=agent_type,
                       agent_id=agent_id)
    agent_db.agent_save = agent_path
    # Load data from files
    agent_db.register_agent()
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
    general = load_general(region_path)
    forecaster = Forecaster(agentDB=agent_db, marketsDB=market_db[market_type], general=general)
    forecaster.init_forecaster()
    # TODO if forecaster is updated during the simulation (e.g. by update_local_market_in_forecasters),
    #  we need to load it here correctly
    with open(os.path.join(agent_db.agent_save, 'forecaster_train.pickle'), 'rb') as handle:
        forecaster.train_data = pickle.load(handle)

    agent_db.forecaster = forecaster  # register


def load_general(path):
    """Loads general information"""
    general = {'weather': f.load_file(path=os.path.join(path, 'general', 'weather',
                                                        'weather.ft'), df='polars', method='eager'),
               'retailer': f.load_file(path=os.path.join(path, 'general', 'retailer.ft'),
                                       df='polars', method='eager'),
               'tasks': f.load_file(path=os.path.join(path, 'general', 'timetable.ft'),
                                    df='polars', method='eager'),
               'general': f.load_file(path=os.path.join(path, 'config', 'config_setup.yaml'))}
    return general


def get_grid_restriction_commands(region_path):
    """Load grid restriction commands."""
    grid_restriction_commands = {}
    file_names = os.listdir(os.path.join(region_path, 'grids'))     # list all files
    file_names = [file for file in file_names if 'restriction' in file]
    for file in file_names:
        grid_type = file.split("_")[0]
        with open(os.path.join(region_path, 'grids', file), 'rb') as handle:
            grid_restriction_commands[grid_type] = pickle.load(handle)

    return grid_restriction_commands


class AgentPool(ProcessPool):
    """
    A class to manage the agents multiprocessing pool

    Attributes:
        num_workers (int): the number of processes to spawn
        task: method to execute in parallel
    """
    def __init__(self, num_workers: int):
        super().__init__(num_workers, task)
