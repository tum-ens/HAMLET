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
# from lfm import Lfm
# from lhm import Lhm
# from lh2m import Lh2m


class Markets:

    def __init__(self, config_path: str, input_path: str, scenario_path: str, config_root: str = None):
        # Set paths and names
        self.config_path = config_path
        self.config_root = config_root if config_root is not None else config_path
        self.input_path = input_path
        self.scenario_path = scenario_path
        self.region = self.config_path.rsplit(os.sep, 1)[1]

        # Load setup plus configuration and/or agent file
        self.setup = self._load_file(path=os.path.join(self.config_root, 'config_setup.yaml'))
        self.config = self._load_file(path=os.path.join(self.config_path, 'config_markets.yaml'))

        # Available types of markets
        from hamlet.creator.markets.lem import Lem
        # from hamlet.creator.markets.lfm import Lfm  # currently not implemented
        # from hamlet.creator.markets.lhm import Lhm  # currently not implemented
        # from hamlet.creator.markets.lh2m import Lh2m  # currently not implemented
        self.types = {
            'lem': Lem,
            # 'lfm': Lfm,
            # 'lhm': Lhm,
            # 'lh2m': Lh2m,
        }

    def create_markets(self, file_type: str = 'ft'):
        """Create markets from configuration file and create the tasks for each market"""

        # Dictionary to store the market information
        dict_markets = {}

        # Loop over all markets
        for key, config in self.config.items():
            # Get market type and name
            market_type = config['type']
            try:
                name = key
            except IndexError:
                name = None
            if market_type in self.types:
                if config['active']:
                    # Create market dict
                    dict_markets[key] = {}

                    # Create market
                    market = self.types[market_type](market=config, name=name,
                                                     config_path=self.config_path,
                                                     input_path=self.input_path,
                                                     scenario_path=self.scenario_path,
                                                     config_root=self.config_root)

                    # Create market timetable
                    dict_markets[key]['timetable'] = market.create_market_from_config()

                    # Create retailers
                    dict_markets[key]['retailers'] = market.create_retailers_from_config(timetable=dict_markets[key]['timetable'])

                    # Save original configuration
                    dict_markets[key]['config'] = config
            else:
                raise ValueError(f'Market type "{market_type}" not available. Available types are: {self.types.keys()}')

        # Concatenate all timetables and sort by timestamp and timestep
        timetable = pd.concat([val['timetable'] for _, val in dict_markets.items()], axis=1)
        timetable.sort_values(by=['timestamp', 'timestep'], inplace=True)

        # Save concatenated timetable
        self._save_file(path=os.path.join(self.scenario_path, 'markets', f'timetable.{file_type}'),
                        data=timetable, index=False)

        # Concatenate all retailers and sort by timestamp and timestep (needs loop in loop)
        retailers = pd.concat([val['retailers'] for _, val in dict_markets.items()], axis=1)
        retailers.sort_values(by=['timestamp'], inplace=True)

        # Save concatenated retailers
        self._save_file(path=os.path.join(self.scenario_path, 'retailers', f'retailers.{file_type}'),
                        data=retailers, index=False)

        # Save individual information
        for name, market in dict_markets.items():
            # Get market type from name
            market_type = name.split('_', 1)[0]

            # Path of the folder in which all the market's files are to be stored
            path = os.path.join(self.scenario_path, 'markets', market_type, name)

            # Create folder for the market (if it does not exist)
            self.__create_folder(path, delete=False)

            # Save individual timetable
            self._save_file(path=os.path.join(path, f'timetable.{file_type}'), data=market['timetable'], index=False)

            # Save individual configuration
            self._save_file(path=os.path.join(path, 'config.json'), data=market['config'])

            # Save individual retailer
            # Path of the folder in which all the retailer's files are to be stored
            path = os.path.join(self.scenario_path, 'retailers', market_type, name)

            # Create folder for the retailer (if it does not exist)
            self.__create_folder(path, delete=False)

            # Save individual retailer
            self._save_file(path=os.path.join(path, f'retailer.{file_type}'), data=market['retailers'], index=False)

        return timetable, dict_markets

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
                os.mkdirs(path)
        time.sleep(0.0001)


