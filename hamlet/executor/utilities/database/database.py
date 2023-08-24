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

    # initialize
    def __init__(self, scenario_path):

        self.__scenario_path = scenario_path

        self.__general = {}  # dict

        self.__regions = {}

    def setup_database(self, structure):
        """Main setup function"""

        self.__setup_general()

        self.__register_all_regions(structure)

    # general data
    def get_general_data(self) -> dict:
        return self.__general

    def get_weather_data(self):
        return self.__general['weather']

    # market data
    def get_market_data(self, region: str):
        return self.__regions[region].markets

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

    # agent data
    def get_agent_data(self, region, agent_type=None, agent_id=None):
        """Get all agents data for the given region."""
        if agent_type is None:
            return self.__regions[region].agents
        else:
            return self.__regions[region].get_agent_data(agent_type, agent_id)

    def edit_agent_data(self, region, agent_type, agent_id, table_name, new_df):
        self.__regions[region].edit_agent_data(agent_type, agent_id, table_name, new_df)

    def get_meters(self, region, agent_type, agent_id):
        return self.__regions[region].get_meters(agent_type, agent_id)

    # private functions
    def __setup_general(self):
        """Setup general dictionary."""
        self.__general['weather'] = f.load_file(path=os.path.join(self.__scenario_path, 'general', 'weather',
                                                                  'weather.ft'), df='polars')
        self.__general['retailer'] = f.load_file(path=os.path.join(self.__scenario_path, 'general', 'retailer.ft'),
                                                 df='polars')
        self.__general['timetable'] = f.load_file(path=os.path.join(self.__scenario_path, 'general', 'timetable.ft'),
                                                  df='polars')

    def __register_all_regions(self, structure):
        """Register all regions."""
        for region in structure.keys():
            self.__regions[region] = RegionDB(os.path.join(os.path.dirname(self.__scenario_path), structure[region]))
            self.__regions[region].register_region()
