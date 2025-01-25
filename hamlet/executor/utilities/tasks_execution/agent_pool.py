__author__ = "HodaHamdy"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

import os
import traceback

import hamlet.constants as c
from hamlet import functions as f
from hamlet.executor.agents.agent import Agent
from hamlet.executor.utilities.database.agent_db import AgentDB
from hamlet.executor.utilities.database.market_db import MarketDB
from hamlet.executor.utilities.forecasts.forecaster import Forecaster
from hamlet.executor.utilities.tasks_execution.process_pool import ProcessPool


# Define the function to be executed in parallel
def task(agent_data):
    try:
        # Prepare agent data
        agent_type, agent_id, region_tasks, region_path = agent_data
        agent_db = init_agentdb(agent_type, agent_id, region_path)
        market_db, market_type = get_market(region_path, region_tasks)
        add_forecaster(agent_db, market_db, market_type, region_path)
        # Initialize and execute the agent instance
        agent = Agent(agent_type=agent_type, data=agent_db, timetable=region_tasks, market=market_db)
        agent_db = agent.execute()

        folder = f"{agent_db.__hash__()}"
        ret_path = os.path.join(region_path, 'agents', agent_type, agent_id, folder)
        agent_db.save_agent(ret_path, save_all=True)
        return (agent_type, agent_id, region_tasks, region_path, ret_path)

    except Exception:
        # Exceptions from the function running inside the multiprocessing pool
        # are not reported by python (the process silently exists and None is
        # returned)
        # There is not much we can do about this, but at least we can print the
        # Exception so we see that something did go wrong (and also investigate
        # what did go wrong)
        print(traceback.format_exc())
        return None


def init_agentdb_full(agent_type, agent_id, region_tasks, region_path, agent_path=None):
    """fully initializes agent database (including market + forecaster)"""
    if not agent_path:
        agent_path = os.path.join(region_path, 'agents', agent_type, agent_id)

    agent_db = init_agentdb(agent_type, agent_id, region_path, agent_path)

    market_db, market_type = get_market(region_path, region_tasks)
    add_forecaster(agent_db, market_db, market_type, region_path)

    return agent_db

def init_agentdb(agent_type, agent_id, region_path, agent_path=None):
    """Initializes agent database"""
    if not agent_path:
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
    agent_db.forecaster = forecaster  # register


cached_general = None
def load_general(path) -> dict:
    """Loads general information"""
    global cached_general
    if cached_general is None:
        cached_general = {'weather': f.load_file(path=os.path.join(path, 'general', 'weather', 'weather.ft'), df='polars', method='eager', memory_map=False),
                   'retailer': f.load_file(path=os.path.join(path, 'general', 'retailer.ft'),
                                           df='polars', method='eager'),
                   'tasks': f.load_file(path=os.path.join(path, 'general', 'timetable.ft'),
                                        df='polars', method='eager'),
                   'general': f.load_file(path=os.path.join(path, 'config', 'config_setup.yaml'))}
    return cached_general


class AgentPool(ProcessPool):
    """
    A class to manage the agents multiprocessing pool

    Attributes:
        num_workers (int): the number of processes to spawn
        task: method to execute in parallel
    """
    def __init__(self, num_workers: int):
        super().__init__(num_workers, task)
