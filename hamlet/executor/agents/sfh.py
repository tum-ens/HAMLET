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
from hamlet.executor.utilities.forecasts.forecasts import Forecaster
from hamlet.executor.utilities.controller.controller import Controller
from hamlet import constants as c
from hamlet.executor.utilities.database.database import Database
from hamlet.executor.agents.agent import AgentBase


class Sfh(AgentBase):

    def __init__(self, data, timetable, database):

        # Type of agent
        super().__init__(agent_type='sfh', data=data, timetable=timetable, database=database)
