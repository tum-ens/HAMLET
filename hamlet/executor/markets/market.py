__author__ = "MarkusDoepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

# This file is in charge of handling the markets in the execution of the scenario

# Imports
from copy import deepcopy, copy
from hamlet.executor.utilities.database.market_db import MarketDB
from hamlet.executor.utilities.database.database import Database
from hamlet.executor.markets.lem.lem import Lem
from hamlet.executor.markets.lfm.lfm import Lfm
from hamlet.executor.markets.lhm.lhm import Lhm
from hamlet.executor.markets.lh2m.lh2m import Lh2m
import hamlet.constants as c



class Market:
    """
    Initializes a Market instance.

    Parameters
    ----------
    data : MarketDB
        An instance of the MarketDB class containing the market data.
    tasks : dict
        A dictionary containing tasks for the market.
    database : Database
        An instance of the Database class.
    **kwargs
        Additional keyword arguments.

    Methods
    -------
    execute() -> MarketDB
        Executes the given `Market` and returns the resulting `MarketDB`.

    """
    def __init__(self, data: MarketDB, tasks: dict, database: Database, **kwargs):
        """
        Parameters
        ----------
        data : MarketDB
            An instance of the MarketDB class containing the market data.
        tasks : dict
            A dictionary containing tasks for the market.
        database : Database
            An instance of the Database class.
        **kwargs
            Additional keyword arguments.

        """
        # Create a new instance from the data as otherwise it always points to the same instance
        self.data = copy(data)

        # Create a new market instance
        self.market = MarketFactory.create_market(data=self.data, tasks=tasks, database=database)

    def execute(self) -> MarketDB:
        """
        Executes the given `Market` and returns the resulting `MarketDB`.

        Parameters:
            None.

        Returns:
            MarketDB: The resulting `MarketDB` after executing the `Market`.

        """

        result = self.market.execute()

        # Delete the data attribute to avoid memory issues
        del self.data

        return result


class MarketFactory:
    """Factory class for creating Market objects based on market type.

    This class provides a static method `create_market` that takes in market information, tasks, and a database instance and returns an instance of the Market class based on the market type
    * specified in the tasks dictionary.

    Attributes
    ----------
    MARKET_MAPPING : dict
        A dictionary mapping market types to their corresponding Market subclasses.

    Methods
    -------
    create_market(data: MarketDB, tasks: dict, database: Database) -> Market
        Creates and returns an instance of the Market class based on the market type extracted from the tasks dictionary.

    """
    MARKET_MAPPING = {
        c.MT_LEM: Lem,
        c.MT_LFM: Lfm,
        c.MT_LHM: Lhm,
        c.MT_LH2M: Lh2m,
    }

    @staticmethod
    def create_market(data: MarketDB, tasks: dict, database: Database):
        """
        Parameters
        ----------
        data: MarketDB
            An instance of the MarketDB class, representing the market information.

        tasks: dict
            A dictionary containing tasks related to the market. It should include the market type specified by the key c.TC_MARKET.

        database: Database
            An instance of the Database class, representing the database to be used.

        Returns
        -------
        Market
            An instance of the Market class, based on the market type extracted from the tasks dictionary.

        """
        market_type = tasks[c.TC_MARKET]  # extract market type by selecting the market type in tasks
        return MarketFactory.MARKET_MAPPING[market_type](data, tasks, database)
