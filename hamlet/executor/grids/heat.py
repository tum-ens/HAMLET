__author__ = "MarkusDoepfert"
__credits__ = "jiahechu"
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

import polars as pl
from hamlet.executor.grids.grid_base import GridBase

# This file is in charge of handling heat grids
# Note: Not yet implemented


class Heat(GridBase):

    def __init__(self, grid_db, tasks: pl.DataFrame, database):

        # Call the super class
        super().__init__(grid_db=grid_db, tasks=tasks, database=database)
