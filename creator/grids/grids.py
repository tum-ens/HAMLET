__author__ = "TUM-Doepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "TUM-Doepfert"
__email__ = "markus.doepfert@tum.de"

import os
import pandas as pd
from ruamel.yaml import YAML, timestamp
import json
import datetime
import time
import shutil
from pprint import pprint
import pandapower as pp


class Grids:

    def __init__(self, config_path: str, input_path: str, scenario_path: str, config_root: str = None):
        # Set paths and names
        self.config_path = config_path
        self.config_root = config_root if config_root is not None else config_path
        self.input_path = input_path
        self.scenario_path = scenario_path


        # Load setup plus configuration and/or agent file
        self.setup = self._load_file(path=os.path.join(self.config_root, 'config_setup.yaml'))
        # TODO: Change to config_grids.yaml once the files have been renamed
        self.config = self._load_file(path=os.path.join(self.config_path, 'config_grid.yaml'))

        # Available types of markets
        from hamlet.creator.grids.electricity import Electricity
        # from hamlet.creator.grids.heat import Heat
        # from hamlet.creator.grids.hydrogen import Hydrogen
        self.types = {
            'electricity': Electricity,
            # 'heat': Heat,
            # 'hydrogen': Hydrogen,
        }

    def create_grid_files(self) -> None:
        """Creates the grids from the configuration file

        Returns:
            None
        """

        # Dictionary to store the grid information
        dict_grids = {}

        # Create grids
        for grid, config in self.config['grids'].items():
            if grid in self.types:
                if config['active']:
                    if config['method'] == 'file':
                        continue
                    else:
                        # Create grid
                        grid = self.types[grid](config=config, config_path=self.config_path, input_path=self.input_path,
                                                scenario_path=self.scenario_path, config_root=self.config_root)

                        # Create grid
                        dict_grids[grid] = grid.create_grid()
            else:
                raise ValueError(f'Grid type "{grid}" not available. Available types are: {list(self.types.keys())}')

        # Save grids
        for name, grid in dict_grids.items():
            self._save_file(path=os.path.join(self.config_path, f'{name}.xlsx'), data=grid)

    def copy_grid_files(self) -> None:
        """Copies the grid files from the input folder to the scenario folder

        Returns:
            None

        Note: This file is already adapted to more grids but needs to be changed once more than pandapower grid files
        are used.
        """

        # Copy grids
        for grid, config in self.config['grids'].items():
            if config['active']:
                path = os.path.join(self.scenario_path, 'grids', grid)
                self.__create_folder(path=path, delete=False)
                if config['method'] == 'file':
                    # Load grid file and save as json if it is an Excel file
                    if config['file']['file'].split('.')[-1] == 'xlsx':
                        net = pp.from_excel(os.path.join(self.config_path, config['file']['file']))
                        pp.to_json(net, os.path.join(path, f'{grid}.json'))
                    # Copy file
                    else:
                        shutil.copy(os.path.join(self.config_path, config['file']['file']),
                                    os.path.join(path, config['file']['file']))
                else:
                    # Load grid file and save as json
                    file = self._load_file(path=os.path.join(self.config_path, f'{grid}.xlsx'), index=0)
                    self._save_file(path=os.path.join(path, f'{grid}.xlsx'), data=file)

            else:
                pass

    @classmethod
    def _load_file(cls, path: str, index: int = 0):
        file_type = path.rsplit('.', 1)[-1]
        if file_type == 'yaml' or file_type == 'yml':
            with open(path) as file:
                file = YAML().load(file)
        elif file_type == 'json':
            with open(path) as file:
                file = json.load(file)
        elif file_type == 'csv':
            file = pd.read_csv(path, index_col=index)
        elif file_type == 'xlsx':
            file = pd.ExcelFile(path)
        elif file_type == 'ft':
            file = pd.read_feather(path)
        else:
            raise ValueError(f'File type "{file_type}" not supported')

        return file

    @classmethod
    def _save_file(cls, path: str, data, index: bool = True) -> None:
        file_type = path.rsplit('.', 1)[-1]

        if file_type == 'yaml' or file_type == 'yml':
            with open(path, 'w') as file:
                YAML().dump(data, file)
        elif file_type == 'json':
            with open(path, 'w') as file:
                json.dump(data, file, indent=4)
        elif file_type == 'csv':
            data.to_csv(path, index=index)
        elif file_type == 'xlsx':
            data.to_excel(path, index=index)
        elif file_type == 'ft':
            data.reset_index(inplace=True)
            data.to_feather(path)
        else:
            raise ValueError(f'File type "{file_type}" not supported')

    @classmethod
    def __create_folder(cls, path: str, delete: bool = True) -> None:
        """Creates a folder at the given path

        Args:
            path: path to the folder
            delete: if True, the folder will be deleted if it already exists

        Returns:
            None
        """

        # Create main folder if it does not exist
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            if delete:
                shutil.rmtree(path)
                os.makedirs(path)
        time.sleep(0.0001)
