__author__ = "TUM-Doepfert"
__credits__ = "jiahechu"
__license__ = ""
__maintainer__ = "TUM-Doepfert"
__email__ = "markus.doepfert@tum.de"

from hamlet.creator.grids.grids import Grids


class Electricity(Grids):

    def __init__(self, grid: dict, config_path: str, input_path: str, scenario_path: str, config_root):
        super().__init__(config_path, input_path, scenario_path, config_root)

        self.grid = grid
        self.config_path = config_path

        self.methods = ['file']

    def create_grids(self) -> None:
        """
        Creates the grids from the configuration file.

        Currently, the grid doesn't really need to be created, only grid file need to be copied. So this function does
        nothing now. If e.g. synthetic grid generation is implemented in the future, this function need to be adjusted
        accordingly.

        Returns:
            None
        """

        pass
