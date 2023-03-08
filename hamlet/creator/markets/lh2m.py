__author__ = "TUM-Doepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "TUM-Doepfert"
__email__ = "markus.doepfert@tum.de"

import pandas as pd
from markets import Markets


class Lh2m(Markets):

    def __init__(self):
        super().__init__()