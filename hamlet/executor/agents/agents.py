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