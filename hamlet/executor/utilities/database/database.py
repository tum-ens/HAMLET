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

        self.__regions = {}

    def setup_database(self, structure):
        """Main setup function"""

        self.__setup_general()

        self.__register_all_regions(structure)

    # agent database
    def get_agent_data(self, region, agent_id=None):
        """Get all agents data for the given region."""
        if agent_id is None:
            return self.__regions[region].agents
        else:
            return self.__regions[region].get_agent_data(agent_id)

    def edit_agent_data(self, region, agent_id, table_name, new_df):
        self.__regions[region].edit_agent_data(agent_id, table_name, new_df)

    # meter database
    def get_meters(self, region, agent_id):
        return self.__regions[region].get_meters(agent_id)

    @staticmethod
    def filter_market_data(market, by: list[str], value: list[str], inclusive: bool = False):
        """Filter market data by given columns and values.

        @Jiahe: I want to hand it a market data table and only get the rows back where the values are in the given columns.

        Args:
            market: market data table
            by: list of columns to filter by
            value: list of values to filter by
            inclusive: if True, returns rows where the values are in all the given columns,
                       if False, returns rows where the values are in at least one of the given columns

        Returns:
            filtered market data table
        """

        # TODO: @Jiahe, please implement this function (once market data actually exists)
        return market

    def __setup_general(self):
        """Setup general dictionary."""
        self.general['weather'] = f.load_file(path=os.path.join(self.path_scenario, 'general', 'weather', 'weather.ft'),
                                              df='polars')

    def __register_all_regions(self, structure):
        """Register all regions."""
        for region in structure.keys():
            self.__regions[region] = RegionDB(os.path.join(os.path.dirname(self.path_scenario), structure[region]))
            self.__regions[region].register_region()