__author__ = "jiahechu"
__credits__ = "MarkusDoepfert"
__license__ = ""
__maintainer__ = "jiahechu"
__email__ = "jiahe.chu@tum.de"

import os.path
import time
import polars as pl
from polars.type_aliases import SizeUnit
from hamlet import constants as c
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
    def __init__(self, path: str, agent_type: str, agent_id: str) -> None:
        """
        Initializes the AgentDB with the given path and agent type.

        Args:
            path (str): The file path where the agent's information is stored.
            agent_type (str): The type of agent.
            agent_id (str): the agent id.
        """

        self.forecaster = None
        self.agent_path = path
        self.agent_save = None  # path to save the agent
        self.agent_type = agent_type
        self.agent_id = agent_id
        self.sub_agents = {}

        self.account = {}
        self.plants = {}
        self.specs = {}
        self.meters = pl.DataFrame()
        self.socs = pl.DataFrame()
        self.timeseries = pl.DataFrame()
        self.setpoints = pl.DataFrame()
        self.forecasts = pl.DataFrame()
        self.bids_offers = pl.DataFrame()

    def register_agent(self) -> None:
        """
        Reads and assigns class attributes from the data files located in the agent's folder.

        The method loads various data files including account, plants, meters,
        timeseries, State of Charge (SOC), and specifications. The data is stored
        as attributes of the AgentDB instance.

        Note:
            The loading process relies on the 'hamlet' library's load_file function.
        """
        # load existing data
        self.account = f.load_file(path=os.path.join(self.agent_path, 'account.json'))
        self.plants = f.load_file(path=os.path.join(self.agent_path, 'plants.json'))
        self.specs = f.load_file(path=os.path.join(self.agent_path, 'specs.json'))
        self.meters = f.load_file(path=os.path.join(self.agent_path, 'meters.ft'), df='polars', method='eager')
        self.socs = f.load_file(path=os.path.join(self.agent_path, 'socs.ft'), df='polars', method='eager')
        self.timeseries = f.load_file(path=os.path.join(self.agent_path, 'timeseries.ft'), df='polars', method='eager')
        # Note: some scenarios were written with the time column named 'index' rather than
        # 'timestamp'. Without this the forecaster fails much later with
        # "list.remove(x): x not in list", which names neither the file nor the column.
        if 'index' in self.timeseries.columns:
            self.timeseries = self.timeseries.rename({'index': c.TC_TIMESTAMP})
        self.setpoints = f.load_file(path=os.path.join(self.agent_path, 'setpoints.ft'), df='polars', method='eager')
        self.forecasts = f.load_file(path=os.path.join(self.agent_path, 'forecasts.ft'), df='polars', method='eager')

        fn = os.path.join(self.agent_path, 'bids_offers.ft')
        if os.path.exists(fn):
            self.bids_offers = f.load_file(path=fn, df='polars', method='eager')

        # initialize setpoints and forecast

    def register_sub_agent(self, id: str, path: str) -> None:
        """
        Registers a sub-agent with a given ID and path.

        The method creates an instance of the AgentDB class for the sub-agent
        and calls the register_agent method to load its information.

        Args:
            id (str): The identifier for the sub-agent.
            path (str): The file path where the sub-agent's information is stored.
        """
        self.sub_agents[id] = AgentDB(path, self.agent_type, id)
        self.sub_agents[id].register_agent()

    def save_agent(self, path: str, dtype: str = 'ft') -> None:
        """
        Saves the agent's data to the agent's folder.

        The method saves the agent's data to the agent's folder as files.
        The data is stored as files with the same name as the class attributes.

        Two things went with the multiprocessing path. `save_all` wrote `account`, `plants`,
        `specs` and `bids_offers` as well; only a worker handing an agent back to the parent ever
        passed it, and the serial save never did, because the Executor copies the scenario folder
        into the results folder and those four are unchanged there. And a pickle of the
        forecaster's training data was written next to them on *every* save, for a worker to read
        back when it rebuilt the agent -- nothing else has ever read it.
        """

        # Update agent path
        self.agent_save = os.path.abspath(path)

        # Save data. Each table goes to a sibling file first and is renamed over the real one, so
        # a crash in the middle of a write leaves the previous save intact rather than a truncated
        # feather. This is what lets the executor call it periodically (#249): without it a crash
        # *during* a checkpoint would also destroy the checkpoint before it.
        for name, table in (('meters', self.meters), ('timeseries', self.timeseries),
                            ('socs', self.socs), ('setpoints', self.setpoints),
                            ('forecasts', self.forecasts)):
            target = os.path.join(self.agent_save, f'{name}.{dtype}')
            staging = os.path.join(self.agent_save, f'{name}.tmp.{dtype}')
            f.save_file(path=staging, data=table, df='polars')
            self._replace(staging, target)

    #: How often `_replace` retries a rename Windows refused, and how long it waits between tries.
    REPLACE_ATTEMPTS = 3
    REPLACE_RETRY_SECONDS = 0.1

    @classmethod
    def _replace(cls, staging: str, target: str) -> None:
        """`os.replace`, retried on the transient refusal Windows gives for a file someone has open.

        POSIX rename is atomic and does not care who holds the old file. Windows refuses with
        `PermissionError` while any handle is open on either path -- an antivirus scan of the
        freshly written staging file, a viewer on the target. Nothing in HAMLET holds one, but the
        periodic checkpoint makes the window hundreds of times more frequent than the single save
        at the end of a run used to, so a transient refusal is retried before it kills the run.
        """
        for attempt in range(1, cls.REPLACE_ATTEMPTS + 1):
            try:
                os.replace(staging, target)
                return
            except PermissionError:
                if attempt == cls.REPLACE_ATTEMPTS:
                    raise
                time.sleep(cls.REPLACE_RETRY_SECONDS)

    def estimated_size(self, unit:SizeUnit = "kb") -> float:
        ret = 0
        ret += self.meters.estimated_size(unit)
        ret += self.socs.estimated_size(unit)
        ret += self.timeseries.estimated_size(unit)
        ret += self.setpoints.estimated_size(unit)
        ret += self.forecasts.estimated_size(unit)
        ret += self.bids_offers.estimated_size(unit)
        return ret
