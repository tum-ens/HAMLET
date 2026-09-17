__author__ = "jiahechu"
__credits__ = ""
__license__ = ""
__maintainer__ = "jiahechu"
__email__ = "jiahe.chu@tum.de"

import polars as pl
from hamlet.executor.utilities.database.database import Database
from hamlet.executor.utilities.grid_restrictions.enwg_14a import EnWG14a


class GridRestriction:

    # types of grid restrictions. If implementing new restrictions they should be added to the list.
    TYPES = {
        'enwg_14a': EnWG14a
    }

    def __init__(self, grid_db, tasks: pl.DataFrame, restriction_type: str, database: Database, **kwargs):

        self.types = self.TYPES

        # Instance of the grid regulator class.
        self.regulator = self.types[restriction_type](grid_db=grid_db, tasks=tasks, database=database, **kwargs)

    @classmethod
    def can_reject_a_pass(cls, restriction_type: str, config: dict) -> bool:
        """Whether this restriction type can send a timestep back for another pass under `config`."""
        return cls.TYPES[restriction_type].can_reject_a_pass(config)

    def execute(self):
        """Executes the grid restriction. Should always return a grid database object."""

        return self.regulator.execute()
