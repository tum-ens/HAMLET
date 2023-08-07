__author__ = "MarkusDoepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

# Imports
import os
import pandas as pd
import polars as pl
import numpy as np
import time
import logging
import traceback
from datetime import datetime
from hamlet.executor.utilities.forecasts import Forecasts
from hamlet.executor.utilities.controller import Optimization
import pyomo.environ as pyo
from jump import Model, Variable, Constraint, Objective, Binary
from jump.scale import NonNegativeReals
from linopy import LinearProgram, Variable, Constraint, Objective


class Sfh:

    def __init__(self, data, timetable):

        # Data
        self.data = data

        # Timetable
        self.timetable = timetable

        # Market data
        self.market = None

        # Available controllers (see agent config)  # TODO: Move to agent class (import controller class to obtain functions)
        self.rtc = {'rtc': self._controller_rtc,
                    'rtc_max_pv': self._controller_rtc_max_pv,}
        self.mpc = {'mpc': self._controller_mpc,
                    'mpc_rtc': self._controller_mpc_rtc,}

    def execute(self):
        """Executes the agent"""

        # Get the market data (database)
        self.get_market_data()
        # Get the grid data (database)
        self.get_grid_data()
        # Get forecasts (train models if needed)
        self.get_forecasts()
        # Controller (RTC, MPC, etc.)
        self.set_controller()
        # Log data
        self.log_data()
        # Post bids and offers
        self.post_to_market()

    def get_market_data(self):
        """Gets the market data from the database"""

        # Create the database object
        db = Database()

        # Get the market data
        self.market = db.get_market_data()

    def get_grid_data(self):
        """Gets the grid data from the database"""
        ...

    def get_forecasts(self):
        """Gets the predictions for the agent"""

        # Create the forecasts object
        fc = Forecasts()

        # Get the forecasts
        self.data['forecasts'] = fc.get_forecasts(self.data, self.timetable, self.market)

    def set_controller(self):
        """Sets the controller for the agent"""
        # TODO: This needs to be adjusted to new format

        hems = self.data['account']['hems']

        # Calls the respective controller
        self.data['setpoints'] = self.controllers[hems['strategy']](data=self.data, timetable=self.timetable, market=self.market)

    def log_data(self):
        """Logs the data of the agent"""
        ...

    def post_to_market(self):
        """Posts the bids and offers to the market"""
        ...

    @staticmethod
    def __get_plants(plants, plant_types) -> list:

        # Ensure that plant_types is a list
        plant_types = plant_types if isinstance(plant_types, list) else list(plant_types)

        # Return list of plants of the specified type
        list_plants = []
        for plant_id, plant in plants.items():
            # Check if plant is of the correct type
            if plant.get("type") in plant_types:
                list_plants.append(plant_id)

        return list_plants

    @staticmethod
    def find_column(df, search_string):
        for column in df.columns:
            if search_string in column:
                return column
        return None