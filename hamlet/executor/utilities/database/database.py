__author__ = "jiahechu"
__credits__ = ""
__license__ = ""
__maintainer__ = "jiahechu"
__email__ = "jiahe.chu@tum.de"

import time

import pandas as pd
import polars as pl
import os
import hamlet.functions as f
import hamlet.constants as c
from hamlet.executor.utilities.database.region_db import RegionDB
from hamlet.executor.utilities.database.agent_db import AgentDB
from hamlet.executor.utilities.database.market_db import MarketDB
from datetime import datetime
from pprint import pprint


class Database:
    """
    Database connection provides all database connection methods required by HAMLET.

    In order to remain database-agnostic, only Database should be connected to outside directly. The AgentDB and
    MarketDB objects should be obtained through "get" functions of the Database. To write an updated AgentDB or
    MarketDB object back to the Database, also use "post" functions of the Database.

    Attributes:
        scenario_path: Path where executed scenario is stored.
        __general: Dictionary contains general data.
        __regions: Dictionary contains RegionDB objects. The AgentDB and MarketDB objects of the corresponding region
        are stored in each RegionDB object.

    """

    def __init__(self, scenario_path):

        self.__scenario_path = scenario_path

        self.__general = {}  # dict

        self.__regions = {}

    ########################################## PUBLIC METHODS ##########################################

    """initialize database"""

    def setup_database(self, structure):
        """Initialize the database."""

        self.__setup_general()

        self.__register_all_regions(structure)

    """get data"""

    def get_general_data(self) -> dict:
        return self.__general

    def get_weather_data(self):
        return self.__general['weather']

    def get_agent_data(self, region, agent_type=None, agent_id=None):
        """
        Retrieve agent data for the specified region, agent type, and agent ID.

        Parameters:
            region (str): The name of the region for which to retrieve agent data.
            agent_type (str, optional): The type of agent to filter by or None to include all agent types.
            agent_id (str, optional): The ID of the agent to retrieve or None to include all agents of the specified
            type.

        Returns:
            agent_data: The agent data for the specified region, agent type, and agent ID. Returns a dictionary or a
            specific agent object based on the provided parameters.

        Note:
        - If 'agent_type' is provided without 'agent_id,' the function returns all agents of the specified type in the
        region.
        - If both 'agent_type' and 'agent_id' are provided, the function returns the specific agent object.
        - If no filter parameters are provided, the function returns all agent data for the specified region.

        Example:
        To retrieve all agents of type 'sfh' in the 'example_region' region, you can call the function as follows:
        ```
        agent_data = your_instance.get_agent_data('example_region', agent_type='sfh')
        ```

        To retrieve a specific agent with ID '12345' of type 'sfh' in the 'example_region' region, you can call the
        function as follows:
        ```
        specific_agent = your_instance.get_agent_data('example_region', agent_type='sfh', agent_id='12345')
        ```

        """
        if not agent_type:
            return self.__regions[region].agents
        elif agent_type and not agent_id:
            return self.__regions[region].agents[agent_type]
        elif agent_type and not agent_id:
            return self.__regions[region].agents[agent_type][agent_id]

    def get_market_data(self, region: str, market_type=None, market_name=None):
        """
        Retrieve market data for the specified region, market type, and market name.

        Parameters:
            region (str): The name of the region for which to retrieve market data.
            market_type (str, optional): The type of market to filter by or None to include all market types.
            market_name (str, optional): The name of the market to retrieve or None to include all markets of the
            specified type.

        Returns:
            market_data: The market data for the specified region, market type, and market name. Returns a dictionary
            or a specific market object based on the provided parameters.

        Note:
        - If 'market_type' is provided without 'market_name,' the function returns all markets of the specified type in
        the region.
        - If both 'market_type' and 'market_name' are provided, the function returns the specific market object.
        - If no filter parameters are provided, the function returns all market data for the specified region.

        Example:
        To retrieve all markets of type 'lem' in the 'example_region' region, you can call the function as follows:
        ```
        market_data = your_instance.get_market_data('example_region', market_type='lem')
        ```

        To retrieve a specific market with the name 'continuous' of type 'lem' in the 'example_region' region, you can
        call the function as follows:
        ```
        specific_market = your_instance.get_market_data('example_region', market_type='electricity',
                                                        market_name='continuous')
        ```

        """
        if not market_type:
            return self.__regions[region].markets
        elif market_type and not market_name:
            return self.__regions[region].markets[market_type]
        else:
            return self.__regions[region].markets[market_type][market_name]

    def get_bids_offers(self, region: str, market_type: str | list[str] = None, market_name: str | list[str] = None,
                        timestep: datetime | list[datetime] = None):
        """
        Retrieve bids and offers data for the specified region, market type, market name, and timestep(s).

        Args:
            region (str): The name of the region for which to retrieve data.
            market_type (str | list[str], optional): Filter by market type(s) or None to include all market types.
            market_name (str | list[str], optional): Filter by market name(s) or None to include all market names.
            timestep (datetime | list[datetime], optional): Filter by timestep(s) or None to include all timesteps.

        Returns:
            bids_offers_table (polars.LazyFrame): A lazyframe containing the bids and offers data filtered based on the
            specified parameters.

        Note:
            - If multiple filter criteria are provided, they will be combined using the 'inclusive' filter mode.
            - All filters are optional. If no filters are provided, the function returns the bids and offers data for
            all markets and timesteps in the specified region.

        """
        # get bids and offers table
        bids_offers = {}

        for agents in self.__regions[region].agents.values():
            for agent_id, agentDB in agents.items():
                if not isinstance(agentDB.account["general"]["aggregated_by"], str):
                    bids_offers[agent_id] = agentDB.bids_offers

        # combine tables
        bids_offers_table = pl.concat(bids_offers.values(), how='vertical')

        # if given, generate lists for applying self.filter_market_data function
        by = []
        values = []
        if market_type:
            by.append(c.TC_MARKET)
            values.append(market_type)
        if market_name:
            by.append(c.TC_NAME)
            values.append(market_name)
        if timestep:
            by.append(c.TC_TIMESTEP)
            values.append(timestep)

        # convert single value to a list for filtering function
        for i in range(len(values)):
            if values[i] is not list:
                values[i] = [values[i]]

        # filtering
        if by and values:
            bids_offers_table = self.filter_market_data(market=bids_offers_table, by=by, values=values, inclusive=True)

        return bids_offers_table

    """post data"""

    def post_agents_to_region(self, region: str, agents: list):
        """
        Post the given agents to the given region.

        The given agents are passed to the function as a list of AgentDB objects. The function maps each agent according
        to the AgentDB attributes: agent_id and agent_type. If an agent in agent_type with agent_id already exists in
        the region, replace it with the given new one. If not, add it to the region.

        Args:
            region: name of the region.
            agents: list of AgentDB objects to be written into Database.

        """
        for agent in agents:
            agent_id = agent.agent_id
            agent_type = agent.agent_type
            self.__regions[region].agents[agent_type][agent_id] = agent

    def post_markets_to_region(self, region: str, markets: list):
        """
        Post the given markets to the given region.

        The given markets are passed to the function as a list of MarketDB objects. The function maps each market
        according to the MarketDB attributes: market_name and market_type. If a market in market_type with market_name
        already exists in the region, replace it with the given new one. If not, add it to the region.

        Args:
            region: name of the region.
            markets: list of MarketDB objects to be written into Database.

        """
        # Dict to store all markets in the region
        region_markets = {}

        # Seperator
        item_separator = '----------------------------------------'

        counter = 0
        for market in markets:
            # Create unique key for each market
            unique_key = market.market_type + item_separator + market.market_name

            # Check if market already exists in region
            if unique_key in region_markets.keys():
                # If market already exists, extend the list of market tables
                region_markets[unique_key][c.TN_MARKET_TRANSACTIONS].append(market.market_transactions)
                region_markets[unique_key][c.TN_BIDS_CLEARED].append(market.bids_cleared)
                region_markets[unique_key][c.TN_BIDS_UNCLEARED].append(market.bids_uncleared)
                region_markets[unique_key][c.TN_OFFERS_CLEARED].append(market.offers_cleared)
                region_markets[unique_key][c.TN_OFFERS_UNCLEARED].append(market.offers_uncleared)
                region_markets[unique_key][c.TN_POSITIONS_MATCHED].append(market.positions_matched)
            else:
                # If market does not exist, add it to the region
                region_markets[unique_key] = {}
                region_markets[unique_key][c.TN_MARKET_TRANSACTIONS] = [market.market_transactions]
                region_markets[unique_key][c.TN_BIDS_CLEARED] = [market.bids_cleared]
                region_markets[unique_key][c.TN_BIDS_UNCLEARED] = [market.bids_uncleared]
                region_markets[unique_key][c.TN_OFFERS_CLEARED] = [market.offers_cleared]
                region_markets[unique_key][c.TN_OFFERS_UNCLEARED] = [market.offers_uncleared]
                region_markets[unique_key][c.TN_POSITIONS_MATCHED] = [market.positions_matched]

        # Save results to region
        for market, results in region_markets.items():
            # Split market name into market type and market name
            market_type, market_name = market.split(item_separator)
            market_db = self.__regions[region].markets.get(market_type, {}).get(market_name)

            # Save results to region
            # Tables that are to be expanded
            market_db.set_market_transactions(pl.concat([market_db.market_transactions]
                                                        + results[c.TN_MARKET_TRANSACTIONS], how='vertical'))
            market_db.set_bids_cleared(pl.concat([market_db.bids_cleared]
                                                 + results[c.TN_BIDS_CLEARED], how='vertical'))
            market_db.set_offers_cleared(pl.concat([market_db.offers_cleared]
                                                   + results[c.TN_OFFERS_CLEARED], how='vertical'))
            # Note: Positions matched are not used in the current version
            # market_db.set_positions_matched(pl.concat(results[c.TN_POSITIONS_MATCHED], how='vertical'))
            # Tables that are to be overwritten
            market_db.set_bids_uncleared(pl.concat(results[c.TN_BIDS_UNCLEARED], how='vertical'))
            market_db.set_offers_uncleared(pl.concat(results[c.TN_OFFERS_UNCLEARED], how='vertical'))

        # Update local market price in forecasters
        self.__regions[region].update_local_market_in_forecasters()

    """static methods"""

    @staticmethod
    def filter_market_data(market, by: list[str], values: list[list], inclusive: bool = False) -> list:
        """
        Filter market data by given columns and values.

        Args:
            market: The market data table to filter.
            by (list[str]): A list of column names to filter by.
            values (list[list]): A 2D list of values to filter by. The length of this list should match the length of
            the by argument.
            inclusive (bool, optional):
                If True, filters rows where the values are in all the given columns (AND).
                If False, filters rows where the values are in at least one of the given columns (OR).

        Returns:
            filtered_market: The filtered market data table.

        Notes:
            - If 'inclusive' is True, the function applies an AND condition to combine filters for multiple columns.
            - If 'inclusive' is False, the function applies an OR condition to combine filters for multiple columns.
            - If the filter values are datetime objects, the function generates new columns with adjusted datetime
            values for comparison and removes them after filtering, because the time unit and time zone need to be
            adjusted in polars.

        Example:
            To filter a market table for rows where 'column1' contains 'value1' OR 'column2' contains 'value2', you can
            call the function as follows:
            ```
            filtered_data = YourClass.filter_market_data(market_data, ['column1', 'column2'], [['value1'], ['value2']],
                                                         inclusive=False)
            ```

        """
        # TODO: Make a more performant customized implementation to get the market data in lem (get_bids_offers())
        filters = {}    # filters to be applied
        new_columns_count = 0   # number of new columns added to df
        new_columns = []    # names of new columns added to df

        # generate filter for each column
        for i in range(len(by)):
            filters[by[i]] = False  # init an empty list, will be filled with statements

            for value in values[i]:
                # check if value is a datetime object
                if isinstance(value, datetime):
                    # get time info from original dataframe
                    datetime_index = market.select(by[i])
                    dtype = datetime_index.dtypes[0]
                    time_unit = dtype.time_unit
                    time_zone = dtype.time_zone

                    # generate a new column with current timestep and adjust data type
                    column_name = 'new_columns_' + str(new_columns_count)
                    market = market.with_columns(pl.lit(value)
                                                 .alias(column_name)
                                                 .cast(pl.Datetime(time_unit=time_unit, time_zone=time_zone)))

                    # update new columns count and name
                    new_columns_count += 1
                    new_columns.append(column_name)

                    # filtering
                    filters[by[i]] = filters[by[i]] | (pl.col(by[i]) == pl.col(column_name))
                else:
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

        # delete added columns
        if new_columns:
            filtered_market = filtered_market.drop(new_columns)

        return filtered_market

    """save database"""

    def save_database(self, path: str):
        """
        Save the database to the specified path.

        Args:
            path: The path to save the database to.

        """

        # create directory if not exists
        if not os.path.exists(path):
            os.makedirs(path)

        # save region data
        for region in self.__regions.keys():
            self.__regions[region].save_region(path=os.path.join(path, region))

    ########################################## PRIVATE METHODS ##########################################

    def __setup_general(self):
        """
        Setup general dictionary.

        Get all general information from files in scenario path and write them to self.__general dict.

        """
        self.__general['weather'] = f.load_file(path=os.path.join(self.__scenario_path, 'general', 'weather',
                                                                  'weather.ft'), df='polars', method='eager')
        self.__general['retailer'] = f.load_file(path=os.path.join(self.__scenario_path, 'general', 'retailer.ft'),
                                                 df='polars', method='eager')
        self.__general['tasks'] = f.load_file(path=os.path.join(self.__scenario_path, 'general', 'timetable.ft'),
                                                  df='polars', method='eager')
        self.__general['general'] = f.load_file(path=os.path.join(self.__scenario_path, 'config', 'config_setup.yaml'))

    def __register_all_regions(self, structure):
        """
        Register all regions.

        Initialize a RegionDB object for each region and register each region. Then initialize forecasters for all
        agents in each region.

        """

        for region in structure.keys():
            # initialize RegionDB object
            self.__regions[region] = RegionDB(os.path.join(os.path.dirname(self.__scenario_path), structure[region]))

            # register region
            self.__regions[region].register_region()

            # register agent's forecaster for agents in the region
            self.__regions[region].register_forecasters_for_agents(self.__general)
