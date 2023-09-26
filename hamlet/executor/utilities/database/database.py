import pandas as pd
import sqlalchemy as db
import polars as pl
import os
import hamlet.functions as f
from hamlet.executor.utilities.database.region_db import RegionDB
from hamlet.executor.utilities.database.agent_db import AgentDB
from hamlet.executor.utilities.database.market_db import MarketDB
from datetime import datetime


class Database:
    """Database connection provides all database connection methods required by HAMLET.
       In order to remain database-agnostic no other module may connect to the database directly."""

    def __init__(self, scenario_path):

        self.__scenario_path = scenario_path

        self.__general = {}  # dict

        self.__regions = {}

    ########################################## PUBLIC METHODS ##########################################

    """initialize"""

    def setup_database(self, structure):
        """Main setup function"""

        self.__setup_general()

        self.__register_all_regions(structure)

    """get data"""

    def get_general_data(self) -> dict:
        return self.__general

    def get_weather_data(self):
        return self.__general['weather']

    def get_market_data(self, region: str):
        return self.__regions[region].markets

    def get_market_data(self, region: str, market_type=None, market_name=None):
        """Get all markets data for the given region."""
        if market_type is None and market_name is None:
            return self.__regions[region].markets
        elif market_name is None:
            # TODO Jiahe: Return all markets of the given type
            pass
        else:
            return self.__regions[region].get_market_data(market_type, market_name)

    def get_bids_offers(self, region: str, market_type: str | list[str] = None, market_name: str | list[str] = None,
                        timestep: datetime | list[datetime] = None):
        """Get bids and offers for the given market and timestep."""
        # TODO Jiahe: This function should return the bids and offers of all agents in a LazyFrame
        #    market_type, market_name and timesteps are optional and if none are given all bids and offers are returned
        return self.__regions[region].get_bids_offers(market_type, market_name, timestep)

    @staticmethod
    def filter_market_data(market, by: list[str], values: list[list], inclusive: bool = False) -> list:
        """Filter market data by given columns and values.

        Args:
            market: market data table
            by: list of columns to filter by (column names)
            value: list of values to filter by, 2D list. Should be the same length as by.
            inclusive: if True, returns rows where the values are in all the given columns, (AND)
                       if False, returns rows where the values are in at least one of the given columns (OR)

        Returns:
            filtered market data table
        """
        filters = {}

        # generate filter for each column
        for i in range(len(by)):
            filters[by[i]] = False  # init an empty list, will be filled with statements
            for value in values[i]:
                filters[by[i]] = filters[by[i]] | (pl.col(by[i]) == value)

        # combine filters for all columns according to if inclusive
        if inclusive:
            filter = True
            for column in filters.keys():
                filter = filter & filters[column]
        else:
            filter = False
            for column in filters.keys():
                filter = filter | (filters[column])

        # filtering
        filtered_market = market.filter(filter)

        return filtered_market

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

    """post data"""

    def post_agents_to_region(self, region: str, agents: list):
        """Reassign the given agents to the given region."""
        for agent in agents:
            agent_id = agent.agent_id
            self.__regions[region].agents[agent_id] = agent

    def post_markets_to_region(self, region: str, markets: list):
        """Reassign the given markets to the given region."""
        for market in markets:
            market_name = market.market_name
            self.__regions[region].markets[market_name] = market

    def update_forecaster(self):
        """Also include update forecaster."""
        # calculate mean value and assign this value as local market price
        ...

    ########################################## PRIVATE METHODS ##########################################

    def __setup_general(self):
        """Setup general dictionary."""
        self.__general['weather'] = f.load_file(path=os.path.join(self.__scenario_path, 'general', 'weather',
                                                                  'weather.ft'), df='polars')
        self.__general['retailer'] = f.load_file(path=os.path.join(self.__scenario_path, 'general', 'retailer.ft'),
                                                 df='polars')
        self.__general['tasks'] = f.load_file(path=os.path.join(self.__scenario_path, 'general', 'timetable.ft'),
                                                  df='polars')
        self.__general['general'] = f.load_file(path=os.path.join(self.__scenario_path, 'config', 'config_setup.yaml'))

    def __register_all_regions(self, structure):
        """Register all regions."""
        for region in structure.keys():
            # initialize RegionDB object
            self.__regions[region] = RegionDB(os.path.join(os.path.dirname(self.__scenario_path), structure[region]))

            # register region
            self.__regions[region].register_region()

            # register agent's forecaster for agents in the region
            self.__regions[region].register_forecasters_for_agents(self.__general)
