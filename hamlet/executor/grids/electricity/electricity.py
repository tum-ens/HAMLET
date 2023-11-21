__author__ = "MarkusDoepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

# This file is in charge of handling the grids in the execution of the scenario

# Imports
import os
import pandas as pd
import polars as pl
import numpy as np
import time
import logging
import traceback
from datetime import datetime
import pandapower as pp
from hamlet.executor.grids.grid_base import GridBase

# TODO: This does not work yet but merely the code structure is shown


class Electricity(GridBase):

    def __init__(self, grid: pp.pandapowerNet, trades: pl.DataFrame = None, method: str = 'dc'):

        # Call the super class
        super().__init__()

        # Grid (topology)
        self.grid = grid

        # Calculation method
        self.method = method

        # Trades
        self.trades = trades

    def execute(self):
        """Executes the grid"""

        # Obtain the energy trades if they were not provided
        if self.trades is None:
            self.trades = self.get_trades()

        # Calculate the power flows
        powerflow = self.calculate_powerflow()

        return powerflow

    def get_trades(self):
        """Obtains the energy trades (converts them from the original shape to suit the necessary format)"""
        ...

    def calculate_powerflow(self):
        """Calculates the power flows"""

        match self.method:
            case 'ac':
                return self.grid.runpp()
            case 'dc':
                return self.grid.rundc()
            case 'acopf':
                return self.grid.runopp()
            case 'dcopf':
                return self.grid.rundcopf()
