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

# TODO: Considerations
# - None so far


class Electricity:

    def __init__(self, grid: pp.pandapowerNet):
        self.grid = grid

    def execute(self):
        """Executes the grid"""

        # Obtain the energy trades
        trades = self.get_trades()

        # Calculate the power flows
        powerflow = self.calculate_powerflow(trades)

        return powerflow

    def get_trades(self):
        """Obtains the energy trades"""
        ...

    def calculate_powerflow(self, trades):
        """Calculates the power flows"""
        ...
