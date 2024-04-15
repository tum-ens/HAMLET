__author__ = "MarkusDoepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

# This file is in charge of handling the agents in the execution of the scenario

# Imports
import polars as pl
from hamlet.executor.utilities.controller.controller import Controller
from hamlet.executor.utilities.database.database import Database
from hamlet.executor.utilities.database.agent_db import AgentDB
from hamlet.executor.utilities.trading.trading import Trading
from hamlet import constants as c


class AgentBase:
    """Base class for all agents. It provides a default implementation of the run method."""

    def __init__(self, agent_type: str, agent: AgentDB, timetable: pl.DataFrame, database: Database):

        # Type of agent
        self.agent_type = agent_type

        # Data
        self.agent = agent  # agent dataframe

        # Timetable
        self.timetable = timetable  # part of tasks for one timestep

        # Database
        self.db = database

        # Market data
        self.market = None

    def execute(self):
        """Executes the agent"""

        # Get the market data from the database
        self.get_market_data()

        # Get the grid data from the database
        self.get_grid_data()

        if not isinstance(self.agent.account["general"]["aggregated_by"], str):

            # Get forecasts
            self.get_forecasts()

            # Set controllers
            self.set_controllers()

            # Create bids and offers based on trading strategy
            self.create_bids_offers()

        return self.agent

    def get_market_data(self):
        """Gets the market data from the database"""

        # Get the market data
        self.market = self.db.get_market_data(region=self.timetable[c.TC_REGION].head(1).to_list()[0])

    def get_grid_data(self):
        """Gets the grid data from the database"""
        ...

    def get_forecasts(self):
        """Gets the predictions for the agent"""
        # Get the forecasts
        self.agent.forecasts = self.agent.forecaster.make_all_forecasts(self.timetable)

        return self.agent

    def set_controllers(self):
        """Sets the controller for the agent"""

        # Get the required data
        ems = self.agent.account[c.K_EMS]

        # Loop through the ems controllers
        for controller, params in ems['controller'].items():
            # Skip if method is None
            # TODO: Check if this needs to be changed since it might be that the method is None but still something
            #  needs to be done for the tables.
            if params['method'] is None:
                continue

            # Get the controller
            controller = Controller(controller_type=controller, **params).create_instance()

            # Run the controller
            self.agent = controller.run(agent=self.agent, timetable=self.timetable, market=self.market)

        return self.agent

    def create_bids_offers(self):
        """Create the bids and offers to the market based on the trading strategy"""

        # Get the required data
        market_info = self.agent.account[c.K_EMS][c.K_MARKET]

        # Get the markets of the region from the timetable by selecting the unique market types and names
        unique_types_names = self.timetable.unique(subset=[c.TC_MARKET, c.TC_NAME])
        market_types = unique_types_names.select(c.TC_MARKET).to_series().to_list()
        market_names = unique_types_names.select(c.TC_NAME).to_series().to_list()

        # Loop through the markets
        for m_type, m_name in zip(market_types, market_names):
            # Get the strategy
            strategy = Trading(strategy=market_info['strategy'], timetable=self.timetable,
                               market=m_name, market_data=self.market[m_type][m_name], agent=self.agent).create_instance()

            # Create bids and offers
            self.agent = strategy.create_bids_offers()

        return self.agent