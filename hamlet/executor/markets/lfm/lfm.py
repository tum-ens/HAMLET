__author__ = "MarkusDoepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

# Imports
import os
import pandas as pd
import polars as pl
import numpy as np
import time
import logging
import traceback
from datetime import datetime
import hamlet.constants as c
from hamlet.executor.utilities.database.market_db import MarketDB
from hamlet.executor.utilities.database.region_db import RegionDB
from hamlet.executor.utilities.database.database import Database
from hamlet.executor.markets.market_base import MarketBase


class Lfm(MarketBase):

    def __init__(self, market: MarketDB, tasks: dict, database: Database):

        # Call the super class
        super().__init__()

        # Market database
        self.market = market

        # Tasklist
        self.tasks = tasks

        # Database
        self.database = database
