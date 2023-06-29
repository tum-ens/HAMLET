import polars as pl
import os
from hamlet import functions as f
from hamlet.executor.utilities.database.agent_db import AgentDB
from hamlet.executor.utilities.database.market_db import MarketDB

class RegionDB:
    """Database contains all the information for region."""
    def __init__(self, path):

        self.path_region = path
        self.agents = {}
        self.markets = {}
        self.subregions = {}

    def register_region(self):
        """Register this region."""
        self.__register_all_agents()

        self.__register_all_markets()

    def __register_all_agents(self):
        """Register all agents for this region."""
        agents_types = f.get_all_subdirectories(os.path.join(self.path_region, 'agents'))
        for agents_type in agents_types:
            # register agents for each type
            agents = f.get_all_subdirectories(os.path.join(self.path_region, 'agents', agents_type))
            for agent in agents:
                sub_agents = f.get_all_subdirectories(os.path.join(self.path_region, 'agents', agents_type, agent))
                self.agents[agent] = AgentDB(path=os.path.join(self.path_region, 'agents', agents_type, agent),
                                             type=agents_type)
                if sub_agents is None:
                    self.agents[agent].register_agent()
                else:
                    for sub_agent in sub_agents:
                        self.agents[agent].register_sub_agent(id=sub_agent, path=os.path.join(self.path_region,
                                                                                              'agents', agents_type,
                                                                                              agent, sub_agent))

    def __register_all_markets(self):
        """Register all markets for this region."""
        markets_types = f.get_all_subdirectories(os.path.join(self.path_region, 'markets'))
        for markets_type in markets_types:
            markets = f.get_all_subdirectories(os.path.join(self.path_region, 'markets', markets_type))
            for market in markets:
                self.markets[market] = MarketDB(path=os.path.join(self.path_region, 'markets', markets_type, market),
                                                type=markets_type)
                self.markets[market].register_market()
