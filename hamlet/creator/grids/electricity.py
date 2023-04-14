__author__ = "TUM-Doepfert"
__credits__ = ""
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
        """Creates the grids from the configuration file

        Returns:
            None
        """

        if self.grid['method'] == 'file':
            return None
        else:
            raise ValueError(f'Method "{self.grid["method"]}" not available. Available methods are: {self.methods}')

        # return grid