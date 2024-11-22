__author__ = "MarkusDoepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

import polars as pl
from hamlet.executor.utilities.database.database import Database

# Used to ensure a consistent design of all grids


class GridBase:

    def __init__(self, grid_db, tasks: pl.DataFrame, database: Database):

        # Grid database
        self.grid_db = grid_db

        # Tasks dataframe
        self.tasks = tasks

        # Whole database
        self.database = database

    def execute(self):
        raise NotImplementedError('This grid type is not implemented yet but the structure is already in place.')
