import os.path
import polars as pl
from hamlet import functions as f


class AgentDB:
    """
    A class to manage information related to an agent.

    This class serves as a database that contains all the information for an agent,
    including account details, plants, meters, timeseries, and other specifications.
    It should only be connected with the Database class and have no direct connection
    with the main Executor.

    Attributes:
        agent_path (str): The file path where the agent's information is stored.
        agent_type (str): The type of agent.
        sub_agents (dict): A dictionary containing sub-agents.
        account (dict): Account information for the agent.
        plants (dict): Information about plants managed by the agent.
        specs (dict): Various specifications related to the agent.
        meters (pl.LazyFrame): Data related to meters.
        socs (pl.LazyFrame): Data related to State of Charge (SOC).
        timeseries (pl.LazyFrame): Timeseries data.
        setpoints (pl.LazyFrame): Setpoints data.
        forecasts (pl.LazyFrame): Forecast data.
    """
    def __init__(self, path: str, agent_type: str) -> None:
        """
        Initializes the AgentDB with the given path and agent type.

        Args:
            path (str): The file path where the agent's information is stored.
            agent_type (str): The type of agent.
        """

        self.agent_path = path
        self.agent_type = agent_type
        self.agent_id = ''
        self.sub_agents = {}
        self.account = {}
        self.plants = {}
        self.specs = {}
        self.meters = pl.LazyFrame()
        self.socs = pl.LazyFrame()
        self.timeseries = pl.LazyFrame()
        self.setpoints = pl.LazyFrame()
        self.forecasts = pl.LazyFrame()

    def register_agent(self) -> None:
        """
        Reads and assigns class attributes from the data files located in the agent's folder.

        The method loads various data files including account, plants, meters,
        timeseries, State of Charge (SOC), and specifications. The data is stored
        as attributes of the AgentDB instance.

        Note:
            The loading process relies on the 'hamlet' library's load_file function.
        """
        self.account = f.load_file(path=os.path.join(self.agent_path, 'account.json'))
        self.plants = f.load_file(path=os.path.join(self.agent_path, 'plants.json'))
        self.specs = f.load_file(path=os.path.join(self.agent_path, 'specs.json'))
        self.meters = f.load_file(path=os.path.join(self.agent_path, 'meters.ft'), df='polars')
        self.timeseries = f.load_file(path=os.path.join(self.agent_path, 'timeseries.ft'), df='polars')
        self.socs = f.load_file(path=os.path.join(self.agent_path, 'socs.ft'), df='polars')
        self.setpoints = f.load_file(path=os.path.join(self.agent_path, 'setpoints.ft'), df='polars')
        self.forecasts = f.load_file(path=os.path.join(self.agent_path, 'forecasts.ft'), df='polars')

    def register_sub_agent(self, id: str, path: str) -> None:
        """
        Registers a sub-agent with a given ID and path.

        The method creates an instance of the AgentDB class for the sub-agent
        and calls the register_agent method to load its information.

        Args:
            id (str): The identifier for the sub-agent.
            path (str): The file path where the sub-agent's information is stored.
        """
        self.sub_agents[id] = AgentDB(path, self.agent_type)
        self.sub_agents[id].register_agent()
