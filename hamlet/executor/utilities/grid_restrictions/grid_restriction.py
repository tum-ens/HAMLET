__author__ = "jiahechu"
__credits__ = ""
__license__ = ""
__maintainer__ = "jiahechu"
__email__ = "jiahe.chu@tum.de"

import polars as pl
from hamlet.executor.utilities.database.database import Database
from hamlet.executor.utilities.grid_restrictions.enwg_14a import EnWG14a


class GridRestriction:

    def __init__(self, grid_db, tasks: pl.DataFrame, restriction_type: str, database: Database, **kwargs):

        # types of grid restrictions. If implementing new restrictions they should be added to the list.
        self.types = {
            'enwg_14a': EnWG14a
        }

        # Instance of the grid regulator class.
        self.regulator = self.types[restriction_type](grid_db=grid_db, tasks=tasks, database=database, **kwargs)

    def execute(self):
        """Executes the grid restriction. Should always return a grid database object."""

        return self.regulator.execute()
