__author__ = "jiahechu"
__credits__ = ""
__license__ = ""
__maintainer__ = "jiahechu"
__email__ = "jiahe.chu@tum.de"

import polars as pl
import hamlet.constants as c
from hamlet.executor.utilities.database.database import Database


class GridRestrictionBase:

    def __init__(self, grid_db, grid, tasks: pl.DataFrame, database: Database, **kwargs):

        # grid database object
        self.grid_db = grid_db

        # tasks dataframe
        self.tasks = tasks

        # main database object
        self.database = database

        # Grid object
        self.grid = grid

        # Current timestamp
        self.timestamp = self.tasks.select(c.TC_TIMESTAMP).sample(n=1).item()

        # Calculation method
        self.method = self.database.get_general_data()[c.K_GRID][c.G_ELECTRICITY]['powerflow']

        # Further config defined in grid json file
        self.restriction_config = kwargs

    def execute(self):
        raise NotImplementedError('This restriction type is not implemented yet but the structure is already in place.')

    @staticmethod
    def can_reject_a_pass(config: dict) -> bool:
        """Whether `execute` can ever return `grid_ok=False` under this restriction's config.

        Decides whether the executor takes the per-timestep copy a rollback needs (#249); see
        `Executor.__execute_grids` for what happens to a rejection that answered False here.
        """
        return False
