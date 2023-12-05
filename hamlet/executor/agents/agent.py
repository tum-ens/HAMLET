__author__ = "MarkusDoepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

# This file is in charge of handling the agents in the execution of the scenario

# Imports
import polars as pl
from hamlet.executor.utilities.database.database import Database
from hamlet.executor.agents.sfh.sfh import Sfh
from hamlet.executor.agents.mfh.mfh import Mfh
from hamlet.executor.agents.ctsp.ctsp import Ctsp
from hamlet.executor.agents.industry.industry import Industry
from hamlet.executor.agents.producer.producer import Producer
from hamlet.executor.agents.storage.storage import Storage


class Agent:
    def __init__(self, agent_type: str, data: dict, timetable: pl.LazyFrame, database: Database):

        # Instance of the agent class
        self.agent = AgentFactory.create_agent(agent_type, data, timetable, database)

    def execute(self):
        return self.agent.execute()


class AgentFactory:
    @staticmethod
    def create_agent(agent_type, data, timetable, database: Database):
        types = {
            'sfh': Sfh,
            'mfh': Mfh,
            'ctsp': Ctsp,
            'industry': Industry,
            'producer': Producer,
            'storage': Storage,
        }
        return types[agent_type](data, timetable, database)

