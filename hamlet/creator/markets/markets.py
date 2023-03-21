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
# from lem import Lem
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
        self.region = self.config_path.rsplit('\\', 1)[1]

        # Load setup plus configuration and/or agent file
        self.setup = self._load_file(path=os.path.join(self.config_root, 'config_general.yaml'))
        self.config = self._load_file(path=os.path.join(self.config_path, 'config_markets.yaml'))

        # Available types of markets
        self.types = {
            'lem': Lem,
            # 'lfm': Lfm,
            # 'lhm': Lhm,
            # 'lh2m': Lh2m,
        }

    def create_markets_from_config(self):
        """Create markets from configuration file and create the timetable for each market"""

        # Dictionary to store the market information
        dict_markets = {}

        # Loop over all markets
        for key, config in self.config.items():
            market_type = key.split('_', 1)[0]
            try:
                name = key.split('_', 1)[1]
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
                raise ValueError(f'Market type "{market_type}" not available')

        # Concatenate all timetables and sort by timestamp and timestep
        timetable = pd.concat([val['timetable'] for _, val in dict_markets.items()], axis=1)
        timetable.sort_values(by=['timestamp', 'timestep'], inplace=True)

        # Save concatenated timetable
        self._save_file(path=os.path.join(self.scenario_path, 'markets', 'timetable.ft'), data=timetable, index=False)

        # Concatenate all retailers and sort by timestamp and timestep (needs loop in loop)
        retailers = pd.concat([val['retailers'] for _, val in dict_markets.items()], axis=1)
        retailers.sort_values(by=['timestamp'], inplace=True)

        # Save concatenated retailers
        self._save_file(path=os.path.join(self.scenario_path, 'retailers', 'retailers.ft'), data=retailers, index=False)

        # Save individual information
        for name, market in dict_markets.items():
            # Get market type from name
            market_type = name.split('_', 1)[0]

            # Path of the folder in which all the market's files are to be stored
            path = os.path.join(self.scenario_path, 'markets', market_type, name)

            # Create folder for the market (if it does not exist)
            self.__create_folder(path, delete=False)

            # Save individual timetable
            self._save_file(path=os.path.join(path, 'timetable.ft'), data=market['timetable'], index=False)

            # Save individual configuration
            self._save_file(path=os.path.join(path, 'config.json'), data=market['config'])

            # Save individual retailer
            # Path of the folder in which all the retailer's files are to be stored
            path = os.path.join(self.scenario_path, 'retailers', market_type, name)

            # Create folder for the retailer (if it does not exist)
            self.__create_folder(path, delete=False)

            # Save individual retailer
            self._save_file(path=os.path.join(path, 'retailer.ft'), data=market['retailers'], index=False)

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


class Lem(Markets):

    def __init__(self, market: dict, config_path: str, input_path: str, scenario_path: str, config_root,
                 name: str = None):
        super().__init__(config_path, input_path, scenario_path, config_root)

        self.market = market
        self.market_type = 'lem'
        self.name = name if name else f'{self.market["clearing"]["type"]}' \
                                      f'_{self.market["clearing"]["method"]}' \
                                      f'_{self.market["clearing"]["pricing"]}'

        # Available types of clearing
        self.clearing_types = {
            'ex-ante': {
                'timetable': self._create_timetable_ex_ante,
            },
            # 'ex-post': {
            #     'timetable': self._create_timetable_ex_post(),
            #     'settling': self._create_settling_ex_post(),
            # },
        }

    def create_market_from_config(self):

        # Create timetable
        timetable = self._create_timetable()

        return timetable

    def _create_timetable(self) -> pd.DataFrame:
        """Create timetable for the market"""

        # Call the respective function depending on the clearing type to create the timetable
        if self.market['clearing']['type'] in self.clearing_types:
            return self.clearing_types[self.market['clearing']['type']]['timetable']()
        else:
            raise ValueError(f'Clearing type "{self.market["clearing"]["type"]}" not available')

    def _create_timetable_ex_ante(self) -> pd.DataFrame:
        """Create timetable for ex-ante clearing"""

        # Get clearing and timing
        clearing = self.market['clearing']
        timing = clearing['timing']

        # Get start and end time of the market simulation
        # start is either a timestamp or a timedelta
        start = timing['start'] if type(timing['start']) == timestamp.TimeStamp \
            else self.setup['simulation']['sim']['start'] + pd.Timedelta(timing['start'], unit='seconds')
        start = start.replace(tzinfo=datetime.timezone.utc)  # needed to obtain correct time zone
        # end is the end of the simulation
        end = self.setup['simulation']['sim']['start'] + pd.Timedelta(self.setup['simulation']['sim']['duration'], unit='days')
        end = end.replace(tzinfo=datetime.timezone.utc)  # needed to obtain correct time zone

        # Create timetable template and main template
        tt = pd.DataFrame(columns=['timestamp', 'timestep', 'region', 'market', 'name', 'action'])  # template
        timetable = tt.copy()  # main template that will contain all timetables created in the following loop
        # tt['timestamp'] = pd.date_range(start=start, end=end, freq=f'{timing["frequency"]}S')

        # Loop over all time steps (each market opening)
        time_opening = start
        while time_opening < end:
            # Create timetable for entire clearing period
            tt_opening = tt.copy()

            # Set the time frequency to the opening time
            time_frequency = time_opening

            # The duration of time_frequency depends on the frequency and opening of the market
            if timing['frequency'] == timing['opening']:
                # Last time step is the last time step before the next market opening
                end_opening = time_opening + pd.Timedelta(timing['opening'], unit='seconds')
            elif timing['frequency'] < timing['opening']:
                # Last time step is the last time step of the market horizon
                end_opening = time_opening + pd.Timedelta(timing['horizon'][1], unit='seconds')
            else:
                raise ValueError(f'Frequency ({timing["frequency"]}) must be smaller than opening ({timing["opening"]})')

            # Loop over each frequency time step
            while time_frequency < end_opening:
                # Create timetable for each frequency time step
                tt_frequency = tt.copy()

                # Add the time steps where actions are to be executed
                # Starting time is either the current time or the first timestamp of the horizon
                start_frequency = max(time_opening + pd.Timedelta(timing['horizon'][0], unit='seconds'), time_frequency)
                tt_frequency['timestep'] = pd.date_range(
                    start=start_frequency,
                    end=time_opening + pd.Timedelta(timing['horizon'][1], unit='seconds'),
                    freq=f'{timing["duration"]}S',
                    inclusive='left')  # 'left' as the end time step is not included

                # Add timestamp (at which time are all actions to be executed)
                tt_frequency['timestamp'] = time_frequency

                # Begin: Add actions
                # Note: the actions depends on the timing parameters
                # Note: actions are separated by a comma (e.g. 'clear,settle')
                # Advice: The creation of the correct task sequence took several days due to its complexity. Do not be
                #   discouraged if it seems complicated at first (because it is). I recommend to try to model a few
                #   different market types to understand the use of each parameter.

                # At first, it is assumed that all markets are to be cleared (corrections are done subsequently
                # depending on the parameter 'closing')
                tt_frequency['action'] = 'clear'

                # Check for time steps that are also to be settled
                if timing['settling'] == 'continuous':
                    # Settle all time steps whose settling time is smaller/equal to current time (frequency)
                    tt_frequency.loc[tt_frequency['timestep'] <= time_frequency, 'action'] += ',settle'
                elif timing['settling'] == 'periodic':
                    # Settle all time steps once the time frequency plus closing is greater/equal than the time step
                    if any(tt_frequency['timestep']
                           <= time_frequency + pd.Timedelta(timing['closing'], unit='seconds')):
                        tt_frequency['action'] += ',settle'

                # Check for time steps that are only to be settled (no clearing)
                # Check if the closing time is to be applied continously 'c' or for the entire period/horizon 'p'
                if timing['settling'] == 'continuous':
                    # Get all time steps whose closing time is before the current timestamp
                    tt_frequency.loc[tt_frequency['timestep'] - tt_frequency['timestamp']
                                     < pd.Timedelta(timing['closing'], unit='seconds'), 'action'] = 'settle'
                elif timing['settling'] == 'periodic':
                    # Set all time steps to be settled once first value (i.e. any value) becomes True
                    if any(tt_frequency['timestep'] - tt_frequency['timestamp'] < pd.Timedelta(timing['closing'],
                                                                                           unit='seconds')):
                        tt_frequency['action'] = 'settle'
                else:
                    raise ValueError(f'Closing type "{timing["closing"]}" not available')

                # End: Add actions

                # Add to timetable of opening
                tt_opening = tt_opening.append(tt_frequency, ignore_index=True)

                # Add time until the next frequency time step
                time_frequency += pd.Timedelta(timing['frequency'], unit='seconds')

            # Append to main timetable
            timetable = timetable.append(tt_opening, ignore_index=True)

            # Add time until the next opening of market
            time_opening += pd.Timedelta(timing['opening'], unit='seconds')

        # Add market type and name
        timetable['region'] = self.region
        timetable['market'] = self.market_type
        timetable['name'] = self.name

        # Add remaining clearing information
        timetable['type'] = clearing['type']
        timetable['method'] = clearing['method']
        timetable['pricing'] = clearing['pricing']
        timetable['coupling'] = clearing['coupling']

        # Sort timetable by timestamp and timestep
        timetable.sort_values(by=['timestamp', 'timestep'], inplace=True)

        # Change timestamps and timesteps to seconds
        timetable['timestamp'] = timetable['timestamp'].apply(lambda x: int(x.timestamp()))
        timetable['timestep'] = timetable['timestep'].apply(lambda x: int(x.timestamp()))

        return timetable

    def create_retailers_from_config(self, timetable) -> pd.DataFrame:

        # Note: This is already prepared to create multiple retailers from the configuration file but currently only
        # one is possible

        # Dictionary to store the retailers
        dict_retailers = {}

        # Create retailers
        for retailer, config in self.market['pricing'].items():
            dict_retailers[retailer] = self._create_retailer(name=retailer, config=config, timetable=timetable)

        return dict_retailers[retailer]

    def _create_retailer(self, name: str, config: dict, timetable: pd.DataFrame) -> pd.DataFrame:
        """Create retailer prices from configuration file"""

        # Create price series by copying the timetable
        prices = timetable['timestamp'].copy()

        # Get only unique timestamp values in the dataframe
        prices = prices.drop_duplicates().to_frame()

        # Add region, market and name of market
        prices['region'] = self.region
        prices['market'] = self.market_type
        prices['name'] = self.name

        # Add retailer name
        prices['retailer'] = name

        # Add columns of the retailer for each cost component (e.g. energy, grid, etc.)
        for key, info in config.items():
            prices = self._add_columns(df=prices, component=key, config=info)

        return prices

    def _add_columns(self, df, component, config):
        """Add columns for each cost component"""

        # Add prices and quantities based on the method
        if config['method'] == 'fixed':
            df = self._create_fixed_cols(df=df, config=config['fixed'], prefix=component)
        elif config['method'] == 'file':
            df = self._create_file_cols(df=df, config=config['file'])
        else:
            raise ValueError(f'Pricing method "{config["method"]}" not available')

        return df

    @staticmethod
    def _create_fixed_cols(df: pd.DataFrame, config: dict, prefix: str):
        """Create fixed prices"""

        # Add price information
        for key, val in config.items():
            if isinstance(val, list):
                df[f'{prefix}_{key}_sell'] = val[0]
                df[f'{prefix}_{key}_buy'] = val[1]
            else:
                df[f'{prefix}_{key}'] = val

        return df

    def _create_file_cols(self, df: pd.DataFrame, config: dict):
        """Create prices from file"""

        # Read file
        file = pd.read_csv(os.path.join(self.input_path, 'retailers', self.market_type, config['file']))

        # Add price information from file
        df = df.join(file.set_index('timestamp'), on='timestamp', how='left')

        return df

