__author__ = "MarkusDoepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

# This file is in charge of handling the markets in the execution of the scenario

# Imports
from hamlet.executor.utilities.database.market_db import MarketDB
from hamlet.executor.utilities.database.database import Database
from hamlet.executor.markets.lem.lem import Lem
# from hamlet.executor.markets.lfm import Lfm  # currently not implemented
# from hamlet.executor.markets.lhm import Lhm  # currently not implemented
# from hamlet.executor.markets.lh2m import Lh2m  # currently not implemented
import hamlet.constants as c


class Markets:

    def __init__(self, data: MarketDB, tasks: dict, database: Database, **kwargs):

        # Types of markets (add your own if others are created here)
        self.types = {
            c.MT_LEM: Lem,
            # c.MT_LFM: Lfm,
            # c.MT_LHM: Lhm,
            # c.MT_LH2M: Lh2m,
        }

        # Instance of the market class
        market_type = tasks[c.TC_MARKET]  # extract market type by selecting the first row's market value
        self.market = self.types[market_type](data, tasks, database)

    def execute(self):
        """Executes the market"""

        # Execute the market's tasks
        self.market.execute()



