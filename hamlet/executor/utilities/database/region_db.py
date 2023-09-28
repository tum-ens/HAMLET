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
        self.agents = {}
        self.markets = {}
        self.subregions = {}

    def register_region(self):
        """Register this region."""
        self.__register_all_agents()

        self.__register_all_markets()

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
                    self.agents[agents_type][agent] = AgentDB(path=os.path.join(self.region_path, 'agents', agents_type,
                                                                                agent), agent_type=agents_type,
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
                self.markets[markets_type][market] = MarketDB(type=markets_type, name=market,
                                                              market_path=os.path.join(self.region_path, 'markets',
                                                                                       markets_type, market),
                                                              retailer_path=os.path.join(self.region_path, 'retailers',
                                                                                         markets_type, market))
                self.markets[markets_type][market].register_market()

