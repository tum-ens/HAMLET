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
import hamlet.constants as c

# TODO: Considerations
# - None so far


class Grids:

    def __init__(self, grid: pp.pandapowerNet, grid_type: str):

        # Types of grids (add your own if others are created here)
        from hamlet.executor.grids.electricity import Electricity
        # from hamlet.creator_backup.grids.heat import Heat
        # from hamlet.creator_backup.grids.hydrogen import Hydrogen
        self.types = {
            c.ET_ELECTRICITY: Electricity,
            # c.ET_HEAT: Heat,
            # c.ET_H2: Hydrogen,
        }

        # Instance of the grid class
        self.grid = self.types[grid_type](grid)

    def execute(self, data: dict, timetable: pl.DataFrame):
        """Executes the grid"""

        self.grid.execute()
