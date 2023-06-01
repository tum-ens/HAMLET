import os.path

import polars as pl
from hamlet import functions as f

class AgentDB:
    """Database contains all the information for agent.
    Should only be connected with Database class, no connection with main Executor."""
    def __init__(self, path, type):

        self.agent_path = path
        self.agent_type = type
        self.agent_id = ''
        self.sub_agents = {}
        self.account = {}
        self.plants = {}
        self.meters = pl.LazyFrame()
        self.socs = pl.LazyFrame()
        self.timeseries = pl.LazyFrame()
        self.setpoints = pl.LazyFrame()
        self.forecasts = pl.LazyFrame()

    def register_agent(self):
        """Assign class attribute from data in agent folder。"""
        self.account = f.load_file(path=os.path.join(self.agent_path, 'account.json'))
        self.plants = f.load_file(path=os.path.join(self.agent_path, 'plants.json'))
        self.meters = f.load_file(path=os.path.join(self.agent_path, 'meters.ft'), df='polars')
        self.timeseries = f.load_file(path=os.path.join(self.agent_path, 'timeseries.ft'), df='polars')
        self.socs = f.load_file(path=os.path.join(self.agent_path, 'socs.ft'), df='polars')

    def register_sub_agent(self, id, path):
        self.sub_agents[id] = AgentDB(path, self.agent_type)
        self.sub_agents[id].register_agent()
