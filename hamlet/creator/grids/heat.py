__author__ = "MarkusDoepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

from hamlet.creator.grids.grids import Grids


class Heat(Grids):

    def __init__(self, grid: dict, config_path: str, input_path: str, scenario_path: str, config_root):
        super().__init__(config_path, input_path, scenario_path, config_root)

        self.grid = grid
        self.config_path = config_path