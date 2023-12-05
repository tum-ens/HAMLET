__author__ = "TUM-Doepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "TUM-Doepfert"
__email__ = "markus.doepfert@tum.de"

import pandas as pd
from hamlet.creator.markets.markets import Markets
import os
import pandas as pd
from ruamel.yaml import YAML, timestamp
import json
import datetime
import time
import shutil
import hamlet.constants as c


class Lem(Markets):

    def __init__(self, market: dict, config_path: str, input_path: str, scenario_path: str, config_root,
                 name: str = None):
        super().__init__(config_path, input_path, scenario_path, config_root)

        self.market = market
        self.market_type = 'lem'
        self.name = name if name else f'{self.market["clearing"]["type"]}' \
                                      f'_{self.market["clearing"]["method"]}' \
                                      f'_{self.market["clearing"]["pricing"]}'
        self.energy_type = c.TRADED_ENERGY.get(self.market_type)

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
            else self.setup['time']['start'] + pd.Timedelta(timing['start'], unit='seconds')
        start = start.replace(tzinfo=datetime.timezone.utc)  # needed to obtain correct time zone
        # end is the end of the simulation
        # Make sure that fractions are properly read
        if isinstance(self.setup['time']['duration'], str):
            numerator, denominator = map(int, self.setup['time']['duration'].split('/'))
            self.setup['time']['duration'] = numerator / denominator
        end = self.setup['time']['start'] + pd.Timedelta(self.setup['time']['duration'], unit='days')
        end = end.replace(tzinfo=datetime.timezone.utc)  # needed to obtain correct time zone

        # Create timetable template and main template
        tt = pd.DataFrame(columns=[c.TC_TIMESTAMP, c.TC_TIMESTEP, c.TC_REGION, c.TC_MARKET, c.TC_NAME, c.TC_ENERGY_TYPE,
                                   c.TC_ACTIONS])
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
                tt_frequency[c.TC_TIMESTEP] = pd.date_range(
                    start=start_frequency,
                    end=time_opening + pd.Timedelta(timing['horizon'][1], unit='seconds'),
                    freq=f'{timing["duration"]}S',
                    inclusive='left')  # 'left' as the end time step is not included

                # Add timestamp (at which time are all actions to be executed)
                tt_frequency[c.TC_TIMESTAMP] = time_frequency

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
                    tt_frequency.loc[tt_frequency[c.TC_TIMESTEP] <= time_frequency, 'action'] += ',settle'
                elif timing['settling'] == 'periodic':
                    # Settle all time steps once the time frequency plus closing is greater/equal than the time step
                    if any(tt_frequency[c.TC_TIMESTEP]
                           <= time_frequency + pd.Timedelta(timing['closing'], unit='seconds')):
                        tt_frequency['action'] += ',settle'

                # Check for time steps that are only to be settled (no clearing)
                # Check if the closing time is to be applied continously 'c' or for the entire period/horizon 'p'
                if timing['settling'] == 'continuous':
                    # Get all time steps whose closing time is before the current timestamp
                    tt_frequency.loc[tt_frequency[c.TC_TIMESTEP] - tt_frequency[c.TC_TIMESTAMP]
                                     < pd.Timedelta(timing['closing'], unit='seconds'), 'action'] = 'settle'
                elif timing['settling'] == 'periodic':
                    # Set all time steps to be settled once first value (i.e. any value) becomes True
                    if any(tt_frequency[c.TC_TIMESTEP] - tt_frequency[c.TC_TIMESTAMP] < pd.Timedelta(timing['closing'],
                                                                                           unit='seconds')):
                        tt_frequency['action'] = 'settle'
                else:
                    raise ValueError(f'Closing type "{timing["closing"]}" not available')

                # End: Add actions

                # Add to timetable of opening
                tt_opening = pd.concat([tt_opening, tt_frequency], ignore_index=True)


                # Add time until the next frequency time step
                time_frequency += pd.Timedelta(timing['frequency'], unit='seconds')

            # Append to main timetable
            timetable = pd.concat([timetable, tt_opening], ignore_index=True)

            # Add time until the next opening of market
            time_opening += pd.Timedelta(timing['opening'], unit='seconds')

        # Add market type and name
        timetable[c.TC_REGION] = self.region
        timetable[c.TC_MARKET] = self.market_type
        timetable[c.TC_NAME] = self.name
        timetable[c.TC_ENERGY_TYPE] = self.energy_type

        # Add remaining clearing information
        timetable[c.TC_CLEARING_TYPE] = clearing['type']
        timetable[c.TC_CLEARING_METHOD] = clearing['method']
        timetable[c.TC_CLEARING_PRICING] = clearing['pricing']
        timetable[c.TC_COUPLING] = clearing['coupling']

        # Sort timetable by timestamp and timestep
        timetable.sort_values(by=[c.TC_TIMESTAMP, c.TC_TIMESTEP], inplace=True)

        # Change dtypes for all columns
        timetable = timetable.astype({
            c.TC_TIMESTAMP: 'datetime64[ns, UTC]',
            c.TC_TIMESTEP: 'datetime64[ns, UTC]',
            c.TC_REGION: 'category',
            c.TC_MARKET: 'category',
            c.TC_NAME: 'category',
            c.TC_ENERGY_TYPE: 'category',
            c.TC_ACTIONS: 'category',
            c.TC_CLEARING_TYPE: 'category',
            c.TC_CLEARING_METHOD: 'category',
            c.TC_CLEARING_PRICING: 'category',
            c.TC_COUPLING: 'category',
        })

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

        # Create price series by copying the timetable's timesteps
        prices = timetable[c.TC_TIMESTEP].copy()

        # Get only unique timestamp values in the dataframe
        prices = prices.drop_duplicates().to_frame()

        # Rename column and reset index
        prices.rename(columns={c.TC_TIMESTEP: c.TC_TIMESTAMP}, inplace=True)
        prices.reset_index(drop=True, inplace=True)

        # Add region, market and name of market
        prices[c.TC_REGION] = self.region
        prices[c.TC_MARKET] = self.market_type
        prices[c.TC_NAME] = self.name

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
                df[f'{prefix}_{key}_sell'] = int(val[0] * c.EUR_KWH_TO_EURe7_WH)
                df[f'{prefix}_{key}_buy'] = int(val[1] * c.EUR_KWH_TO_EURe7_WH)
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
