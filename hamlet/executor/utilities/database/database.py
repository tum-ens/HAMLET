import pandas as pd
import sqlalchemy as db
import polars as pl
import os
import hamlet.functions as f
from hamlet.executor.utilities.database.region_db import RegionDB
from hamlet.executor.utilities.database.agent_db import AgentDB
from hamlet.executor.utilities.database.market_db import MarketDB


class Database:
    """Database connection provides all database connection methods required by HAMLET.
       In order to remain database-agnostic no other module may connect to the database directly."""

    def __init__(self, path_scenario):

        self.path_scenario = path_scenario

        self.general = {}  # dict

        self.regions = {}

    def setup_database(self, structure):
        """Main setup function"""

        self.__setup_general()

        self.__register_all_regions(structure)

    def get_agent_data(self, region):
        """Get all agents data for the given region."""
        return self.regions[region].agents

    def __setup_general(self):
        """Setup general dictionary."""
        self.general['weather'] = f.load_file(path=os.path.join(self.path_scenario, 'general', 'weather', 'weather.ft'),
                                              df='polars')

    def __register_all_regions(self, structure):
        """Register all regions."""
        for region in structure.keys():
            self.regions[region] = RegionDB(os.path.join(os.path.dirname(self.path_scenario), structure[region]))
            self.regions[region].register_region()