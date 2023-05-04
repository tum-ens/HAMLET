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


# TODO: Considerations
# - Use polars instead of pandas to increase performance
# - Use linopy instead of pyomo

# Plan
# - Create a class for each agent type
# - Focus on non-mfh at first


class Agents:

    def __init__(self, data: dict, timetable: pl.DataFrame):

        # Types of agents (add your own if others are created here)
        from hamlet.executor.agents.sfh import Sfh
        from hamlet.executor.agents.mfh import Mfh
        from hamlet.executor.agents.ctsp import Ctsp
        from hamlet.executor.agents.industry import Industry
        from hamlet.executor.agents.producer import Producer
        from hamlet.executor.agents.storage import Storage
        self.types = {
            'sfh': Sfh,
            'mfh': Mfh,
            'ctsp': Ctsp,
            'industry': Industry,
            'producer': Producer,
            'storage': Storage,
        }

        # Instance of the agent class
        self.agent = self.types['type'](data, timetable)

    def execute(self):
        """Executes the agent"""

        # Execute the agent's tasks
        self.agent.execute()
