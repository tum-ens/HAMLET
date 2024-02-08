__author__ = "TUM-Doepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "TUM-Doepfert"
__email__ = "markus.doepfert@tum.de"
__status__ = "Development"

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib import rc
from matplotlib import rcParams
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
import datetime
import pickle
import tqdm
import re
import copy
import warnings
import itertools


class Analyzer:

    def __init__(self, path: str):
        """Initializes the analyzer object."""
        self.path = path

    def plot_general_analysis(self):
        """Plots the general analysis."""
        raise NotImplementedError