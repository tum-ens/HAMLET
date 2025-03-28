__author__ = "MarkusDoepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

# This file is in charge of handling the agents in the execution of the scenario

# Imports
import polars as pl

from hamlet import constants as c
from hamlet.executor.utilities.controller.controller import Controller
from hamlet.executor.utilities.database.agent_db import AgentDB
from hamlet.executor.utilities.trading.trading import Trading


class AgentBase:
    """Base class for all agents. It provides a default implementation of the run method."""

    def __init__(self, agent_type: str, agent: AgentDB, timetable: pl.DataFrame, market: dict, grid_commands: dict):

        # Type of agent
        self.agent_type = agent_type

        # Data
        self.agent = agent  # agent dataframe

        # Timetable
        self.timetable = timetable  # part of tasks for one timestep

        # Market data
        self.market = market

        # Grid data
        self.grid_commands = grid_commands

    def execute(self):
        """Executes the agent"""
        # Apply grid commands
        self.apply_grid_commands()

        # Get forecasts
        self.get_forecasts()

        # Set controllers
        self.set_controllers()

        # Create bids and offers based on trading strategy
        self.create_bids_offers()

        return self.agent

    def apply_grid_commands(self):
        """Adjust data or parameter of agent to apply grid restriction commands."""
        if self.grid_commands.get(c.G_ELECTRICITY, {}).get('current_variable_grid_fees'):
            self.__update_variable_grid_fees()

    def get_forecasts(self):
        """Gets the predictions for the agent"""
        # Get the forecasts
        self.agent.forecasts = self.agent.forecaster.make_all_forecasts(self.timetable)

        return self.agent

    def set_controllers(self):
        """Sets the controller for the agent"""

        # Get the required data
        controllers = self.agent.account[c.K_EMS][c.C_CONTROLLER]

        # Loop through the ems controllers
        for controller, params in controllers.items():
            # Skip if method is None
            if params['method'] is None:
                continue

            # Get the controller
            controller = Controller(controller_type=controller, **params).create_instance()

            # Run the controller
            self.agent = controller.run(agent=self.agent, timetable=self.timetable, market=self.market,
                                        grid_commands=self.grid_commands)

        return self.agent

    def create_bids_offers(self):
        """Create the bids and offers to the market based on the trading strategy"""

        # Get the required data
        market_info = self.agent.account[c.K_EMS][c.K_MARKET]

        # Get the markets of the region from the timetable by selecting the unique market types and names
        unique_values = self.timetable.unique(subset=[c.TC_MARKET, c.TC_NAME, c.TC_CLEARING_METHOD])
        market_types = unique_values.select(c.TC_MARKET).to_series().to_list()
        market_names = unique_values.select(c.TC_NAME).to_series().to_list()
        market_clearing_methods = unique_values.select(c.TC_CLEARING_METHOD).to_series().to_list()

        # Loop through the markets
        for m_type, m_name, m_method in zip(market_types, market_names, market_clearing_methods):

            # Check if there is a market clearing, if not do not apply the strategy but use the default strategy
            if m_method.capitalize() == 'None' or m_method is None:
                # Get the strategy
                strategy = (Trading(timetable=self.timetable, market=m_name, market_data=self.market[m_type][m_name],
                                    agent=self.agent).create_instance())
            else:
                strategy = (Trading(strategy=market_info['strategy'], timetable=self.timetable,
                                    market=m_name, market_data=self.market[m_type][m_name], agent=self.agent)
                            .create_instance())

            # Create bids and offers
            self.agent = strategy.create_bids_offers()

        return self.agent

    def __update_variable_grid_fees(self):
        """Update grid fees for agent in forecast (§14a EnWG regulation)."""
        # get variable grid fees factor at the bus
        bus_id = self.agent.account[c.K_GENERAL]['bus']
        variable_grid_fees = self.grid_commands[c.G_ELECTRICITY]['current_variable_grid_fees'][bus_id]

        # get target grid data in forecaster
        for key, data in self.agent.forecaster.train_data.items():
            train_data = data[c.K_TARGET]
            train_data = (train_data.join(variable_grid_fees, on=c.TC_TIMESTAMP, how='left'))

            # adjust grid fees by replacing original grid fees with variable grid fees
            if f'{c.TT_GRID}_{c.TT_MARKET}_{c.PF_IN}' in train_data.columns:
                train_data = train_data.with_columns(pl.when(pl.col(str(bus_id)).is_null())
                                                       .then(pl.col(f'{c.TT_GRID}_{c.TT_MARKET}_{c.PF_IN}'))
                                                       .otherwise(pl.col(str(bus_id)))
                                                       .alias(f'{c.TT_GRID}_{c.TT_MARKET}_{c.PF_IN}').cast(pl.Int32))

            if f'{c.TT_GRID}_{c.TT_RETAIL}_{c.PF_IN}' in train_data.columns:
                train_data = train_data.with_columns(pl.when(pl.col(str(bus_id)).is_null())
                                                       .then(pl.col(f'{c.TT_GRID}_{c.TT_RETAIL}_{c.PF_IN}'))
                                                       .otherwise(pl.col(str(bus_id)))
                                                       .alias(f'{c.TT_GRID}_{c.TT_RETAIL}_{c.PF_IN}').cast(pl.Int32))

            train_data = train_data.drop(str(bus_id))

            # write new grid fees to agent db object
            self.agent.forecaster.update_forecaster(id=key, dataframe=train_data, target=True)
