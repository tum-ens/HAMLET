__author__ = "jiahechu"
__credits__ = ""
__license__ = ""
__maintainer__ = "jiahechu"
__email__ = "jiahe.chu@tum.de"

import os

import polars as pl

from hamlet import constants as c
from hamlet import functions as f
from hamlet.executor.utilities.database.agent_db import AgentDB
from hamlet.executor.utilities.database.market_db import MarketDB
from hamlet.executor.utilities.forecasts.forecaster import Forecaster


class RegionDB:
    """Database contains all the information for region."""
    def __init__(self, path):

        self.region_path = path
        self.region_save = None  # path to save the region
        self.agents = {}
        self.markets = {}
        self.subregions = {}

    def register_region(self):
        """Register this region."""
        self.__register_all_agents()

        self.__register_all_markets()

    def save_region(self, path):
        """Save this region."""

        # Update region path
        self.region_save = os.path.abspath(path)

        self.__save_all_agents()

        self.__save_all_markets()

    def register_forecasters_for_agents(self, general: dict):
        """
        Add forecaster for each agent in the region.

        This function first summarize all markets in this region to one dictionary without market type keys. Then pass
        initialize a forecaster with summarized markets dict and given general dict for each agent. Finally, the
        initialized forecaster will be added to each AgentDB object as an attribute.

        Args:
            general: General dict of the Database.

        """
        markets = {}

        # summarize all markets in this region as a dict without market types
        for market_type in self.markets.keys():
            for market_name in self.markets[market_type].keys():
                markets[market_name] = self.markets[market_type][market_name]

        for agent_type, agents in self.agents.items():
            for agent_id, agentDB in agents.items():
                forecaster = Forecaster(agentDB=agentDB, marketsDB=markets, general=general)
                forecaster.init_forecaster()    # initialize
                self.agents[agent_type][agent_id].forecaster = forecaster   # register

    def update_local_market_in_forecasters_old(self):
        """
        Update local market train data with the current local market price forecaster for each agent in the region.

        This function should first calculate the summarized (e.g. average) local market price for each market in this
        region. Then replace a part of the train data for markets in each forecaster with the calculated market price
        according to the c.TC_TIMESTAMP. Currently only relevant for local market, because the "real" local market price
        need to be updated after each simulated timestamp.

        """
        for markets in self.markets.values():
            for market in markets.values():
                wholesale_market_key = f'{market.market_name}_{c.TT_RETAIL}'  # key of market for lookup

                # initialize
                unique_timestep_bids = market.bids_cleared.select(c.TC_TIMESTEP).unique()
                market_price = pl.concat([market.bids_cleared.select(c.TC_TIMESTEP),
                                          market.offers_cleared.select(c.TC_TIMESTEP)], how='vertical').unique()
                market_price = market_price.with_columns(pl.lit(None).alias('new_target_buy'),
                                                         pl.lit(None).alias('new_target_sell'))
                market_price = market_price.rename({c.TC_TIMESTEP: c.TC_TIMESTAMP})

                # calculate average price for each timestep
                for ts in unique_timestep_bids:
                    for ts_ in ts:
                        # buy price
                        bids_for_ts = market.bids_cleared.filter((pl.col(c.TC_TIMESTEP) == ts_) &
                                                                 (pl.col(c.TC_ID_AGENT_IN) != 'retailer'))

                        if len(bids_for_ts) != 0:
                            buy_price_for_ts = int(bids_for_ts.select(pl.col(c.TC_PRICE_IN).sum()).item() /
                                                   bids_for_ts.select(pl.col(c.TC_ENERGY_IN).sum()).item())

                            market_price = market_price.with_columns(pl.when(pl.col(c.TC_TIMESTAMP) == ts_)
                                                                     .then(buy_price_for_ts)
                                                                     .otherwise(pl.col('new_target_buy'))
                                                                     .alias('new_target_buy'))

                        # sell price
                        offers_for_ts = market.offers_cleared.filter((pl.col(c.TC_TIMESTEP) == ts_) &
                                                                     (pl.col(c.TC_ID_AGENT_OUT) != 'retailer'))

                        if len(offers_for_ts) != 0:
                            sell_price_for_ts = int(offers_for_ts.select(pl.col(c.TC_PRICE_OUT).sum()).item() /
                                                    offers_for_ts.select(pl.col(c.TC_ENERGY_OUT).sum()).item())

                            market_price = market_price.with_columns(pl.when(pl.col(c.TC_TIMESTAMP) == ts_)
                                                                     .then(sell_price_for_ts)
                                                                     .otherwise(pl.col('new_target_sell'))
                                                                     .alias('new_target_sell'))

                for agents in self.agents.values():
                    for agent in agents.values():
                        # print(f'updating local market for agent {agent.agent_id}')
                        old_target = agent.forecaster.train_data[wholesale_market_key][c.K_TARGET]

                        # replace a part of the old target with new target
                        new_target = old_target.join(market_price, on=c.TC_TIMESTAMP, how='left')
                        new_target = new_target.with_columns(pl.when(pl.col('new_target_buy').is_null())
                                                             .then(pl.col(f'{c.TC_ENERGY}_{c.TC_PRICE}_{c.PF_OUT}'))
                                                             .otherwise(pl.col('new_target_buy'))
                                                             .alias(f'{c.TC_ENERGY}_{c.TC_PRICE}_{c.PF_OUT}'))
                        new_target = new_target.with_columns(pl.when(pl.col('new_target_sell').is_null())
                                                             .then(pl.col(f'{c.TC_ENERGY}_{c.TC_PRICE}_{c.PF_IN}'))
                                                             .otherwise(pl.col('new_target_sell'))
                                                             .alias(f'{c.TC_ENERGY}_{c.TC_PRICE}_{c.PF_IN}'))

                        # delete unnecessary column
                        new_target = new_target.drop('new_target_buy', 'new_target_sell')

                        # update forecaster
                        agent.forecaster.update_forecaster(id=wholesale_market_key, dataframe=new_target, target=True)

    def update_local_market_in_forecasters(self):
        """
        Update local market train data with current local market price forecasters efficiently.
        Uses vectorized operations and optimized data transformations for better performance.
        """

        def calculate_market_prices(market, market_key: str) -> tuple[pl.DataFrame, dict]:
            """Calculate market prices using vectorized operations."""
            # Pre-filter retailer transactions once
            bids = market.bids_cleared.lazy().filter(pl.col(c.TC_ID_AGENT_IN) != 'retailer')
            offers = market.offers_cleared.lazy().filter(pl.col(c.TC_ID_AGENT_OUT) != 'retailer')

            # Calculate buy prices with proper sequencing to avoid overflow
            buy_prices = (
                bids.groupby(c.TC_TIMESTEP)
                .agg([
                    pl.col(c.TC_PRICE_IN).sum().alias('price_sum'),
                    pl.col(c.TC_ENERGY_IN).sum().alias('energy_sum')
                ])
                .filter(pl.col('energy_sum') != 0)
                .with_columns([
                    (pl.col('price_sum') / pl.col('energy_sum')).round().cast(pl.Int32).alias('price')
                ])
                .select([c.TC_TIMESTEP, 'price'])
            )

            # Calculate sell prices with proper sequencing
            sell_prices = (
                offers.groupby(c.TC_TIMESTEP)
                .agg([
                    pl.col(c.TC_PRICE_OUT).sum().alias('price_sum'),
                    pl.col(c.TC_ENERGY_OUT).sum().alias('energy_sum')
                ])
                .filter(pl.col('energy_sum') != 0)
                .with_columns([
                    (pl.col('price_sum') / pl.col('energy_sum')).round().cast(pl.Int32).alias('price')
                ])
                .select([c.TC_TIMESTEP, 'price'])
            )

            # Create base price DataFrame with all timesteps
            all_timesteps = (
                pl.concat([
                    bids.select(c.TC_TIMESTEP),
                    offers.select(c.TC_TIMESTEP)
                ])
                .unique()
                .rename({c.TC_TIMESTEP: c.TC_TIMESTAMP})
            )

            # Join buy and sell prices efficiently
            market_prices = (
                all_timesteps
                .join(
                    buy_prices.rename({c.TC_TIMESTEP: c.TC_TIMESTAMP, 'price': 'new_target_buy'}),
                    on=c.TC_TIMESTAMP,
                    how='left'
                )
                .join(
                    sell_prices.rename({c.TC_TIMESTEP: c.TC_TIMESTAMP, 'price': 'new_target_sell'}),
                    on=c.TC_TIMESTAMP,
                    how='left'
                )
            ).collect()

            column_mapping = {
                'new_target_buy': f'{c.TC_ENERGY}_{c.TC_PRICE}_{c.PF_OUT}',
                'new_target_sell': f'{c.TC_ENERGY}_{c.TC_PRICE}_{c.PF_IN}'
            }

            return market_prices, column_mapping

        def update_forecasters(market_prices: pl.DataFrame, column_mapping: dict,
                               agents_dict: dict, market_key: str) -> None:
            """Update forecasters for all agents in batch."""
            for agents in agents_dict.values():
                for agent in agents.values():
                    old_target = agent.forecaster.train_data[market_key][c.K_TARGET]

                    # Efficient join and update
                    new_target = (
                        old_target.join(market_prices, on=c.TC_TIMESTAMP, how='left')
                        .with_columns([
                            pl.when(pl.col(temp_col).is_null())
                            .then(pl.col(final_col))
                            .otherwise(pl.col(temp_col))
                            .alias(final_col)
                            for temp_col, final_col in column_mapping.items()
                        ])
                        .drop(list(column_mapping.keys()))
                    )

                    agent.forecaster.update_forecaster(
                        id=market_key,
                        dataframe=new_target,
                        target=True
                    )

        # Main processing loop with optimized batch operations
        for markets in self.markets.values():
            for market in markets.values():
                market_key = f'{market.market_name}_{c.TT_RETAIL}'

                # Calculate prices once for all agents
                market_prices, column_mapping = calculate_market_prices(market, market_key)

                # Update all agents' forecasters
                update_forecasters(market_prices, column_mapping, self.agents, market_key)

    def __register_all_agents(self):
        """
        Register all agents for this region.

        This function loop through the agents path of this region and write each agent and its data to an AgentDB
        object. The AgentDB objects will be stored in the dict self.agents with first level keys agent types and
        second level keys agent ids.

        """
        agents_types = f.get_all_subdirectories(os.path.join(self.region_path, 'agents'))
        for agents_type in agents_types:
            # register agents for each type
            self.agents[agents_type] = {}
            agents = f.get_all_subdirectories(os.path.join(self.region_path, 'agents', agents_type))
            if agents:
                for agent in agents:
                    sub_agents = f.get_all_subdirectories(os.path.join(self.region_path, 'agents', agents_type, agent))
                    self.agents[agents_type][agent] = AgentDB(
                        path=os.path.join(self.region_path, 'agents', agents_type, agent),
                        agent_type=agents_type,
                        agent_id=agent)
                    if not sub_agents:
                        self.agents[agents_type][agent].register_agent()
                    else:
                        for sub_agent in sub_agents:
                            self.agents[agents_type][agent].register_sub_agent(id=sub_agent,
                                                                               path=os.path.join(self.region_path,
                                                                                                 'agents', agents_type,
                                                                                                 agent, sub_agent))

    def __register_all_markets(self):
        """
        Register all markets for this region.

        This function loop through the markets path of this region and write each market and its data to an MarketDB
        object. The MarketDB objects will be stored in the dict self.markets with first level keys market types and
        second level keys market names.

        """
        markets_types = f.get_all_subdirectories(os.path.join(self.region_path, 'markets'))
        for markets_type in markets_types:
            self.markets[markets_type] = {}
            markets = f.get_all_subdirectories(os.path.join(self.region_path, 'markets', markets_type))
            for market in markets:
                self.markets[markets_type][market] = MarketDB(market_type=markets_type, name=market,
                                                              market_path=os.path.join(self.region_path, 'markets',
                                                                                       markets_type, market),
                                                              retailer_path=os.path.join(self.region_path, 'retailers',
                                                                                         markets_type, market))
                self.markets[markets_type][market].register_market()

    def __save_all_agents(self):
        """
        Save all agents for this region.

        This function loop through the agents dict of this region and save each AgentDB object to the corresponding
        path.

        """

        for agents_type, agents in self.agents.items():
            for agent_id, agentDB in agents.items():
                # Path to save results to
                path = os.path.join(self.region_save, 'agents', agents_type, agent_id)

                # Save agent data
                agentDB.save_agent(path)
                # TODO: Add subagent functionality

    def __save_all_markets(self):
        """
        Save all markets for this region.

        This function loop through the markets dict of this region and save each MarketDB object to the corresponding
        path.

        """
        for markets_type, markets in self.markets.items():
            for market_name, marketDB in markets.items():
                # Path to save results to
                path = os.path.join(self.region_save, 'markets', markets_type, market_name)

                # Save market data
                marketDB.save_market(path, save_all=True)
