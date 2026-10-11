__author__ = "MarkusDoepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

# This file is in charge of handling the agents in the execution of the scenario

# Imports
import warnings

import pandas as pd
import polars as pl

from hamlet import constants as c
import hamlet.functions as f
from hamlet.executor.utilities.controller.controller import Controller
from hamlet.executor.utilities.database.agent_db import AgentDB
from hamlet.executor.utilities.trading.trading import Trading


class ConfiguredStrategyUnused(UserWarning):
    """A market with no clearing discarded an agent's configured trading strategy.

    Its own category rather than a bare `UserWarning` for two reasons. A user who has decided the
    substitution is fine for their scenario can silence exactly this and nothing else, which a
    bare category would not allow without silencing every warning HAMLET raises. And a test can
    assert this warning was or was not raised without a polars deprecation from the same call
    satisfying it -- which, measured, is what happens: one `create_bids_offers` raises nine of
    them.

    See `AgentBase.create_bids_offers` for why it is a warning and not an error.
    """


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
            # Skip if no method is configured
            # Note: an empty cell in agents.xlsx arrives as NaN rather than None, so checking for
            # None alone let a disabled controller run anyway
            if params['method'] is None or pd.isna(params['method']):
                continue

            # Get the controller
            controller = Controller(controller_type=controller, **params).create_instance()

            # Run the controller
            self.agent = controller.run(agent=self.agent, timetable=self.timetable, market=self.market,
                                        grid_commands=self.grid_commands)

        return self.agent

    #: Whether a market's `method:` names an actual clearing. Delegated to `hamlet.functions` so
    #: that this stage and `ElectricityMarket`'s method registry cannot disagree about what
    #: "no clearing" is spelled like -- they did, and `method: none` was accepted here and refused
    #: there. Kept as a name on the class because that is how the tests and the loop below read.
    clears_a_market = staticmethod(f.clears_a_market)

    def create_bids_offers(self):
        """Create the bids and offers to the market based on the trading strategy.

        **A market with no clearing overrides the agent's configured strategy, on purpose.** With
        `method: None` the retailer is the only counterparty there is, so a price ladder aimed at
        other agents has nothing to aim at, and `agents.yaml` documents `retailer` as "the default
        strategy when there is no clearing". The substitution is correct; what it was not is
        *reported*. `tests/e2e/scenarios/ctsp_industry` declared `strategy: linear` in both its
        config and its workbook and had only ever run `retailer`, and nothing said so -- which is
        how a fixture came to imply coverage of a strategy that had never executed anywhere in the
        suite (#223).

        So it warns rather than raising. Raising would be the house answer for a silently ignored
        setting (#199, #234), but this setting is not ignored by accident: it is documented
        behaviour that existing configs rely on, and `simple_scenario` and `grid_golden` already
        spell the coherent combination (`method: None` with `strategy: retailer`), which is what a
        user should be nudged towards rather than broken over.

        `warnings.warn` deduplicates per call site and message text, so a simulation reports each
        (market, strategy) pair once rather than once per agent per timestep.
        """

        # Get the required data
        market_info = self.agent.account[c.K_EMS][c.K_MARKET]
        configured = market_info['strategy']

        # Get the markets of the region from the timetable by selecting the unique market types and names
        # Note: `maintain_order=True` is load-bearing rather than tidiness. polars' `unique`
        #  defaults to an unspecified order, so a region with more than one market handed them to
        #  the loop below in a **different order on every call** -- and each iteration overwrites
        #  `self.agent.bids_offers`, so which market's book survived was random. No shipped
        #  scenario declares two markets in one region, so this was latent; it was found because
        #  the test for the loop caught its own mutation only 9 times in 13. #223.
        unique_values = self.timetable.unique(subset=[c.TC_MARKET, c.TC_NAME, c.TC_CLEARING_METHOD],
                                              maintain_order=True)
        market_types = unique_values.select(c.TC_MARKET).to_series().to_list()
        market_names = unique_values.select(c.TC_NAME).to_series().to_list()
        market_clearing_methods = unique_values.select(c.TC_CLEARING_METHOD).to_series().to_list()

        # Loop through the markets
        for m_type, m_name, m_method in zip(market_types, market_names, market_clearing_methods):

            # Check if there is a market clearing, if not do not apply the strategy but use the default strategy
            if self.clears_a_market(m_method):
                chosen = configured
            else:
                chosen = c.MTS_RETAILER
                if str(configured).lower() != c.MTS_RETAILER:
                    warnings.warn(
                        f"Market '{m_name}' has no clearing (method: {m_method}), so the "
                        f"configured trading strategy '{configured}' cannot be used and "
                        f"'{c.MTS_RETAILER}' is used instead. Set `strategy: {c.MTS_RETAILER}` "
                        f"for this agent, or give the market a clearing method, so that the "
                        f"config says what the simulation does.",
                        ConfiguredStrategyUnused, stacklevel=2)

            # Get the strategy. Named explicitly in both branches rather than letting the
            # no-clearing case fall through to `Trading`'s keyword default, so the substitution
            # above is a decision this method makes rather than one buried two modules away.
            strategy = (Trading(strategy=chosen, timetable=self.timetable, market=m_name,
                                market_data=self.market[m_type][m_name], agent=self.agent)
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
            # Note: the section 14a fee applies to what the agent draws from the grid. Retailer
            # columns are named from the retailer's point of view, so that is the `_out` column.
            if f'{c.TT_GRID}_{c.TT_MARKET}_{c.PF_OUT}' in train_data.columns:
                train_data = train_data.with_columns(pl.when(pl.col(str(bus_id)).is_null())
                                                       .then(pl.col(f'{c.TT_GRID}_{c.TT_MARKET}_{c.PF_OUT}'))
                                                       .otherwise(pl.col(str(bus_id)))
                                                       .alias(f'{c.TT_GRID}_{c.TT_MARKET}_{c.PF_OUT}').cast(pl.Int32))

            if f'{c.TT_GRID}_{c.TT_RETAIL}_{c.PF_OUT}' in train_data.columns:
                train_data = train_data.with_columns(pl.when(pl.col(str(bus_id)).is_null())
                                                       .then(pl.col(f'{c.TT_GRID}_{c.TT_RETAIL}_{c.PF_OUT}'))
                                                       .otherwise(pl.col(str(bus_id)))
                                                       .alias(f'{c.TT_GRID}_{c.TT_RETAIL}_{c.PF_OUT}').cast(pl.Int32))

            train_data = train_data.drop(str(bus_id))

            # write new grid fees to agent db object
            self.agent.forecaster.update_forecaster(id=key, dataframe=train_data, target=True)
