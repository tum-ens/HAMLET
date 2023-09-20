__author__ = "MarkusDoepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

# This file is in charge of handling the agents in the execution of the scenario

# Imports
import os
import pandas as pd
import polars as pl
import numpy as np
import time
import logging
import traceback
from datetime import datetime
from hamlet.executor.utilities.forecasts.forecaster import Forecaster
from hamlet.executor.utilities.controller.controller import Controller
from hamlet.executor.utilities.database.database import Database
from hamlet import constants as c
from pprint import pprint


class Agent:
    def __init__(self, data: dict, timetable: pl.LazyFrame, agent_type: str, database):

        # Instance of the agent class
        self.agent = AgentFactory.create_agent(agent_type, data, timetable, database)

    def execute(self):
        return self.agent.execute()


class AgentFactory:
    @staticmethod
    def create_agent(agent_type, data, timetable, database):
        from hamlet.executor.agents.sfh import Sfh
        from hamlet.executor.agents.mfh import Mfh
        from hamlet.executor.agents.ctsp import Ctsp
        from hamlet.executor.agents.industry import Industry
        from hamlet.executor.agents.producer import Producer
        from hamlet.executor.agents.storage import Storage
        types = {
            'sfh': Sfh,
            'mfh': Mfh,
            'ctsp': Ctsp,
            'industry': Industry,
            'producer': Producer,
            'storage': Storage,
        }
        return types[agent_type](data, timetable, database)


class AgentBase:
    """Base class for all agents. It provides a default implementation of the run method."""

    def __init__(self, agent_type: str, agent, timetable: pd.DataFrame, database):

        # Type of agent
        self.agent_type = agent_type

        # Data
        self.agentDB = agent  # agent dataframe

        # Timetable
        self.timetable = timetable  # part of timetable for one timestep

        # Database
        self.db = database

        # Market data
        self.market = pl.LazyFrame()  # TODO: Replace with the market results

    def execute(self):
        """Executes the agent"""

        # Get the market data (database)
        # self.get_market_data()
        # Get the grid data (database)
        self.get_grid_data()
        # Get forecasts (train models if needed)
        self.get_forecasts()
        # Controller (RTC, MPC, etc.)
        self.set_controllers()
        # Log data
        self.log_data()
        # Post bids and offers
        self.post_to_market()

    def get_market_data(self):
        """Gets the market data from the database"""

        # Get the market data
        self.market = self.db.get_market_data()

    def get_grid_data(self):
        """Gets the grid data from the database"""
        ...

    def get_forecasts(self):
        """Gets the predictions for the agent"""
        # Get the forecasts
        self.agentDB.forecasts = self.agentDB.forecaster.make_all_forecasts(self.timetable)

        return self.agentDB

    def set_controllers(self):
        """Sets the controller for the agent"""

        # Get the required data
        # ems = self.data[c.K_ACCOUNT][c.K_EMS]

        ems = {
        "controller": {
            "rtc": {
                "method": "linopy"
            },
            "mpc": {
                "method": "linopy",
                "horizon": 86400
            }
        },
        "market": {
            "strategy": ['linear'],
            "horizon": [10800, 21600, 32400, 43200],
            "fcast": {
                "local": "naive",
                "wholesale": "naive"
            }
        },
        "fcasts": {
            "retraining_period": 86400,
            "update_period": 3600
        }
    }

        # Loop through the ems controllers
        for controller, params in ems['controller'].items():
            # Get the method

            # Skip if method is None
            if params['method'] is None:
                continue

            # Get the controller
            controller = Controller(controller_type=controller, **params).create_instance()

            # Run the controller
            self.agent = controller.run(agent=self.agent, timetable=self.timetable, market=self.market)

    def log_data(self):
        """Logs the data of the agent"""
        ...

    def post_to_market(self):
        """Posts the bids and offers to the market"""
        ...


