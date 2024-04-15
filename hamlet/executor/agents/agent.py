__author__ = "MarkusDoepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

# This file is in charge of handling the agents in the execution of the scenario

# Imports
import polars as pl
from hamlet.executor.utilities.database.database import Database
from hamlet.executor.utilities.database.agent_db import AgentDB
from hamlet.executor.agents.sfh.sfh import Sfh
from hamlet.executor.agents.mfh.mfh import Mfh
from hamlet.executor.agents.ctsp.ctsp import Ctsp
from hamlet.executor.agents.industry.industry import Industry
from hamlet.executor.agents.producer.producer import Producer
from hamlet.executor.agents.storage.storage import Storage
from hamlet.executor.agents.aggregator.aggregator import Aggregator
import hamlet.constants as c


class Agent:
    """
    Initializes an Agent instance.

    Parameters
    ----------
    agent_type : str
        The type of agent.
    data : dict
        The data needed for creating the agent.
    timetable : pl.LazyFrame
        The timetable information for the agent.
    database : Database
        The database object for the agent.

    Methods
    -------
    execute() -> AgentDB
        Executes the given `Agent` and returns the resulting `AgentDB`.

    """
    def __init__(self, agent_type: str, data: dict, timetable: pl.DataFrame, database: Database):
        """
        Parameters
        ----------
        agent_type : str
            The type of agent.

        data : dict
            The data needed for creating the agent.

        timetable : pl.LazyFrame
            The timetable information for the agent.

        database : Database
            The database object for the agent.

        """
        # Instance of the agent class
        self.agent = AgentFactory.create_agent(agent_type, data, timetable, database)

    def execute(self) -> AgentDB:
        """
        Executes the given `Agent` and returns the resulting `AgentDB`.

        Parameters:
            None.

        Returns:
            AgentDB: The resulting `AgentDB` after executing the `Agent`.

        """

        return self.agent.execute()


class AgentFactory:
    """
    A factory class to create different types of agents.

    AgentFactory is responsible for creating agents based on the agent type.
    It uses a predefined mapping of agent types to the corresponding agent classes.

    Attributes:
        AGENT_MAPPING (dict): A mapping of agent types to the corresponding agent classes.

    Methods:
        create_agent: Creates an agent based on the given agent type.

    """
    AGENT_MAPPING = {
        c.A_SFH: Sfh,
        c.A_MFH: Mfh,
        c.A_CTSP: Ctsp,
        c.A_INDUSTRY: Industry,
        c.A_PRODUCER: Producer,
        c.A_STORAGE: Storage,
        c.A_AGGREGATOR: Aggregator,
    }

    @staticmethod
    def create_agent(agent_type: str, agent_data: dict, timetable: pl.DataFrame, database: Database):
        """Create an agent.

        Parameters
        ----------
        agent_type : str
            The type of the agent to create.
        agent_data : dict
            The data required to initialize the agent.
        timetable : pl.LazyFrame
            The timetable for the agent.
        database : Database
            The database for the agent.

        Returns
        -------
        Agent
            The created agent.

        """
        return AgentFactory.AGENT_MAPPING[agent_type](agent_data, timetable, database)

