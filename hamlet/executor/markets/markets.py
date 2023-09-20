__author__ = "MarkusDoepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

# This file is in charge of handling the markets in the execution of the scenario

# Imports
import os
import pandas as pd
import polars as pl
import numpy as np
import time
import logging
import traceback
from datetime import datetime


# TODO: Change the structure of the markets to be more similar to the agents


class Markets:

    def __init__(self, timetable: pl.LazyFrame):

        # Types of markets (add your own if others are created here)
        from hamlet.executor.markets.lem import Lem
        # from hamlet.executor.markets.lfm import Lfm  # currently not implemented
        # from hamlet.executor.markets.lhm import Lhm  # currently not implemented
        # from hamlet.executor.markets.lh2m import Lh2m  # currently not implemented
        self.types = {
            'lem': Lem,
            # 'lfm': Lfm,
            # 'lhm': Lhm,
            # 'lh2m': Lh2m,
        }

        # Instance of the market class
        market_type = timetable.collect()[0, 'market']  # extract market type by selecting the first row's market value
        self.market = self.types[market_type](timetable)

    def execute(self):
        """Executes the market"""

        # Execute the market's tasks
        self.market.execute()



