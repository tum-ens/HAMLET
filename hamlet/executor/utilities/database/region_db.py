__author__ = "jiahechu"
__credits__ = ""
__license__ = ""
__maintainer__ = "jiahechu"
__email__ = "jiahe.chu@tum.de"

import polars as pl
import os
from datetime import datetime
from hamlet import functions as f
from hamlet import constants as c
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

    def update_local_market_in_forecasters(self):
        """
        Update local market train data with the current local market price forecaster for each agent in the region.

        This function should first calculate the summarized (e.g. average) local market price for each market in this
        region. Then replace a part of the train data for markets in each forecaster with the calculated market price
        according to the c.TC_TIMESTAMP. Currently only relevant for local market, because the "real" local market price
        need to be updated after each simulated timestamp.

        """
        for markets in self.markets.values():
            for market in markets.values():
                local_market_key = market.market_name + '_local'    # keys of local market for lookup in forecaster

                # TODO: calculate the new market price, the result should be in this format:
                # 'new_target' column should contain market price
                market_price = pl.DataFrame(schema={c.TC_TIMESTAMP: pl.Datetime(time_unit='ns', time_zone='UTC'),
                                                    'new_target': pl.Int64})

                for agents in self.agents.values():
                    for agent in agents.values():
                        # print(f'updating local market for agent {agent.agent_id}')
                        old_target = agent.forecaster.train_data[local_market_key][c.K_TARGET]

                        # get column name of the old target
                        column_name = old_target.columns
                        column_name.remove(c.TC_TIMESTAMP)
                        column_name = column_name[0]
                        # print('hi')

                        # replace a part of the old target with new target
                        # NOTICE: adjust dataframe to dataframe if necessary, polars concat is sometimes tricky
                        new_target = pl.concat([old_target, market_price], how='diagonal')
                        new_target = new_target.with_columns(pl.when(pl.col('new_target').is_null())
                                                             .then(pl.col(column_name))
                                                             .otherwise(pl.col('new_target')).alias(column_name))
                        # print('hi2')

                        # delete unnecessary column
                        new_target = new_target.drop('new_target')

                        # update forecaster bzw. models
                        # NOTICE: new_target should be dataframe now
                        agent.forecaster.update_forecaster(id=local_market_key, dataframe=new_target, target=True)

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
                    if sub_agents is None:
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

                if isinstance(agentDB.account["general"]["aggregated_by"], str):
                    # get the aggregator
                    aggregator = agentDB.account["general"]["aggregated_by"]

                    # iterate over all agents/results
                    for agents_type_aggregated, agents_aggregated in self.agents.items():
                        for agent_id_aggregated, agentDB_aggregated in agents_aggregated.items():

                            # check if the agent is the aggregator
                            if agent_id_aggregated == aggregator:

                                # update meters
                                agentDB.meters = agentDB.meters.update(agentDB_aggregated.meters)

                                # update socs
                                agentDB.socs = agentDB.socs.update(agentDB_aggregated.socs)

                                # update setpoints
                                agentDB.setpoints = agentDB.setpoints.update(agentDB_aggregated.setpoints)

                                # update forecasts
                                agentDB.forecasts = agentDB.forecasts.update(agentDB_aggregated.forecasts)

                                ## update forecasts_all
                                #if hasattr(agentDB, 'forecasts_all'):
                                #    agentDB.forecasts_all = agentDB.forecasts_all.update(agentDB_aggregated.forecasts_all)
                                #else:
                                #    agentDB.forecasts_all = agentDB_aggregated.forecasts_all

                                # update timeseries
                                agentDB.timeseries = agentDB.timeseries.update(agentDB_aggregated.timeseries)


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
                marketDB.save_market(path)