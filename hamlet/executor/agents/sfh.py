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


class Sfh:

    def __init__(self, data, timetable):
        self.data = data
        self.timetable = timetable

    def execute(self):
        """Executes the agent"""

        # Get the market data (database)
        self.get_market_data()
        # Get the grid data (database)
        self.get_grid_data()
        # Get predictions (train models if needed)
        self.get_predictions()
        # Controller (RTC, MPC, etc.)
        self.set_controller()
        # Log data
        self.log_data()
        # Post bids and offers
        self.post_to_market()

    def get_market_data(self):
        """Gets the market data from the database"""
        ...

    def get_grid_data(self):
        """Gets the grid data from the database"""
        ...

    def get_predictions(self):
        """Gets the predictions for the agent"""
        # Also includes the training of the models
        ...

    def set_controller(self):
        """Sets the controller for the agent"""
        # Calls the respective controller
        ...

    def log_data(self):
        """Logs the data of the agent"""
        ...

    def post_to_market(self):
        """Posts the bids and offers to the market"""
        ...
