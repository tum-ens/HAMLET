__author__ = "TUM-Doepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "TUM-Doepfert"
__email__ = "markus.doepfert@tum.de"

import pandas as pd
from markets import Markets


class Lem(Markets):

    def __init__(self, config_path: str, input_path: str, scenario_path: str, config_root):
        super().__init__(config_path, input_path, scenario_path, config_root)
