__author__ = "jiahechu"
__credits__ = ""
__license__ = ""
__maintainer__ = "jiahechu"
__email__ = "jiahe.chu@tum.de"

import polars as pl
import pytz
from datetime import datetime
import inspect
import ast
from hamlet import constants as c
import hamlet.executor.utilities.forecasts.models as models
import hamlet.functions as f
from pprint import pprint


class Forecaster:
    """
    A class to make forecast for the given agent.

    This class contains all the functions related to forecasting. The forecasting models are defined in a separate file
    model.py and will be collected in this class. Each agent class initializes a forecaster for itself and calls the
    forecast models through this class. There's no direct connection between forecast models and agent class.

    Attributes:
        config_dict: dictionary contains forecast config
        train_data: data to feed into models
        all_models: all available forecast models
        used_models: chosen models for forecasting
        weather: weather dataframe
        length_to_predict: length to predict everytime when calling forecast. Unit: s
        start_ts: timestamp when simulation starts
        refit_period: period, after which models should be refitted. Unit: s

    Methods:

    """

    def __init__(self, agentDB, marketsDB: dict, general: dict):
        """
        Initialize the Forecaster object.

        Args:
            agentDB: AgentDB object.
            marketsDB: Dictionary containing all MarketDB objects in the region where the agent is.
            general: General data dictionary.

        """
        self.agentDB = agentDB  # AgentDB object
        self.marketsDB = marketsDB  # MarketDB object
        self.general = general  # general data

        # empty variables, to be initialized
        self.config_dict = {}   # dictionary contains forecast config
        self.train_data = {}    # data to feed into models
        self.all_models = {}    # all available forecast models
        self.used_models = {}   # chosen models for forecasting
        self.weather = pl.LazyFrame()   # weather dataframe
        self.length_to_predict = 0  # length to predict everytime when calling forecast
        self.start_ts = datetime.now()   # timestamp when simulation starts
        self.refit_period = 0   # period, after which models should be refitted

    ########################################## PUBLIC METHODS ##########################################

    def init_forecaster(self):
        """
        Initialize forecaster with corresponding model and train data for each plant and market.
        """
        # initialize config dict
        self.__assign_config_dict()

        # initialize general data
        self.__init_general_data()

        # generate full train data
        self.__prepare_train_data()

        # initialize forecast models
        self.__assign_models()

    def update_forecaster(self, id, dataframe, target=True):
        """
        Update training data for e.g. local market. Will replace the given dataframe to the corresponding part of
        train data.

        Args:
            id: Identifier for the plant or market.
            dataframe: Updated data for the plant or market.
            target: Boolean indicating if the updated data is a target variable.

        """
        # new train data with new target
        if target:
            self.train_data[id][c.K_TARGET] = dataframe
        else:
            self.train_data[id][c.K_FEATURES] = dataframe

        # update model
        self.used_models[id].update_train_data(self.train_data[id])

    def make_all_forecasts(self, timetable):
        """
        Make forecasts for all plants. Re-fit model if current time step is at the beginning of a refitting period.

        Args:
            timetable: Timetable lazyframe at current timestep for managing time-related and other necessary data.

        Returns:
            forecasts: DataFrame (lazyframe) containing all forecast results.

        """

        forecasts = {}   # empty dict to store all forecast results
        current_ts = timetable.select(c.TC_TIMESTAMP).item(0, 0)      # get current timestep from timetable

        # make forecast for each plant and assign results to the empty dict
        for id in self.config_dict.keys():
            forecast = self.__make_forecast_for_id(current_ts, id)
            forecasts[id] = forecast

        # summarize all forecasts to a dataframe (lazyframe)
        forecasts = self.__summarize_forecasts_to_df(forecasts, current_ts)

        # Return the forecasts dataframe (lazyframe)
        return forecasts

    ########################################## PRIVATE METHODS ##########################################

    """relevant for initialization"""

    def __init_general_data(self):
        """
        Initialize general variables and dataframes.

        Take information and data from the general dict and agent account and assign them to the class attributes.

        """
        # assign weather data
        self.weather = self.general['weather']

        # get start time
        self.start_ts = self.general['general']['time']['start']

        # get length to predict
        self.length_to_predict = self.agentDB.account['ems']['fcasts']['horizon']

        # get refit period
        self.refit_period = self.agentDB.account['ems']['fcasts']['retraining']

    def __assign_config_dict(self):
        """
        Summarize all forecast config into one dictionary.

        Assign market config and plant config separately. The markets' configs are hard-coded with only two methods:
        naive and perfect. The plants' configs are taken from agentDB.

        """
        # assign market config
        market_config = self.agentDB.account['ems'][c.TC_MARKET]['fcast']  # get forecast config for market
        for market_name in self.marketsDB.keys():   # assign market config dict for each market
            wholesale_id = market_name + '_wholesale'
            local_id = market_name + '_local'

            # add to config dict
            self.config_dict[wholesale_id] = market_config['wholesale']
            self.config_dict[local_id] = market_config['local']

            # if no additional parameters given for the chosen model, assign an empty dict
            chosen_model_wholesale = market_config['wholesale']['method']
            chosen_model_local = market_config['local']['method']

            if chosen_model_wholesale not in list(market_config['wholesale'].keys()):
                self.config_dict[wholesale_id][chosen_model_wholesale] = {}

            if chosen_model_local not in list(market_config['local'].keys()):
                self.config_dict[local_id][chosen_model_local] = {}

        # assign plants config
        for plant_id in self.agentDB.plants.keys():
            if 'fcast' in self.agentDB.plants[plant_id].keys():     # only take plants which need to be forecasted
                self.config_dict[plant_id] = self.agentDB.plants[plant_id]['fcast']
                self.config_dict[plant_id]['type'] = self.agentDB.plants[plant_id]['type']
                chosen_model = self.config_dict[plant_id]['method']

                # if no additional parameters given for the chosen model, assign an empty dict
                if chosen_model not in list(self.config_dict[plant_id].keys()):
                    self.config_dict[plant_id][chosen_model] = {}

    def __assign_models(self):
        """
        Assign related forecast models.

        Get all available models from models module and assign them to a dict according to the given keywords (name
        attribute), then assign the chosen forecast model for each plant / market. Initialize the chosen forecast
        models using given parameters in config file.

        SPECIAL CASE FOR WEATHER MODEL: more data will be passed to the weather model. All columns in weather data will
        be taken as features, another key 'config' will be added to the train data with value of specs dict.

        """
        # get all models from imported module
        # Note: change code here if models are separated into separate files
        all_models = inspect.getmembers(models, inspect.isclass)
        for model in all_models:
            if hasattr(model[1], c.TC_NAME):  # include only models defined in the imported module with right format
                self.all_models[model[1].name] = model[1]

        # assign and initialize the chosen models
        for id in self.config_dict.keys():
            chosen_model = self.config_dict[id]['method']  # keyword of the chosen model as string
            # check if there's extra keyword arguments for model initialization
            self.used_models[id] = self.all_models[chosen_model](self.train_data[id],
                                                                 **self.config_dict[id][chosen_model])

            # SPECIAL CASE FOR WEATHER MODEL
            if chosen_model == 'weather':
                # add other necessary data to train data
                self.train_data[id]['specs'] = self.agentDB.specs[id]
                self.train_data[id]['plant_config'] = self.agentDB.plants[id]
                self.train_data[id]['general_config'] = self.general['general']
                self.train_data[id][c.K_FEATURES] = self.weather

    def __prepare_train_data(self):
        """
        Prepare train data according to the given features.

        The training data is a dictionary with two keys: features and target. The value of each key is a polars
        lazyframe.

        """
        self.__prepare_markets_target_data()

        self.__prepare_plants_target_data()

        self.__prepare_features()

    def __prepare_markets_target_data(self):
        """
        Prepare target data for markets.

        Each market has a wholesale and local market. The data for wholesale market are taken from retailer. The local
        market firstly takes retailer data and will be updated during the simulation. The data for one day before the
        beginning of the simulation will be added to the target data for naive method.

        """
        for market_name, marketDB in self.marketsDB.items():  # assign market config dict for each market
            # get retailer data
            target_wholesale = marketDB.retailer

            # pre-processing for retailer data
            # calculate market data resolution
            resolution = f.calculate_time_resolution(target_wholesale)

            # add offset day(s) before simulation with the same data
            offset = self.config_dict[market_name + '_wholesale']['naive']['offset']    # here method is hard-coded

            day_before = f.slice_dataframe_between_times(target_df=target_wholesale, reference_ts=self.start_ts,
                                                         duration=c.DAYS_TO_SECONDS * offset + resolution)

            day_before = day_before.with_columns((pl.col(c.TC_TIMESTAMP) - pl.duration(days=offset, seconds=resolution))
                                                 .alias(c.TC_TIMESTAMP)
                                                 .cast(pl.Datetime(time_unit='ns', time_zone='UTC')))

            target_wholesale = pl.concat([day_before, target_wholesale], how='vertical')

            # drop unnecessary columns
            # TODO: update column name in constants
            target_wholesale = target_wholesale.drop('index', c.TC_MARKET, c.TC_NAME, c.TC_REGION, 'retailer')

            # initial prepare for the wholesale market
            self.train_data[market_name + '_wholesale'] = {c.K_TARGET: target_wholesale}

            # initial prepare for the local market
            target_local = target_wholesale.select(c.TC_TIMESTAMP, 'energy_price_sell')\
                                           .rename({'energy_price_sell': 'energy_price_local'})
            self.train_data[market_name + '_local'] = {c.K_TARGET: target_local}

    def __prepare_plants_target_data(self):
        """
        Prepare target data for plants.

        Go through the timeseries and take each column as target data for corresponding plant.

        """
        # get plants' timeseries
        plants_timeseries = self.agentDB.timeseries

        # get all related plants except timestamps
        plants_cols = plants_timeseries.columns
        plants_cols.remove(c.TC_TIMESTAMP)
        # get plants id
        plants_ids = list(set([col.split('_')[0] for col in plants_cols]))

        # get target data
        for plant_id in plants_ids:
            # combine timestamp and all columns for the plant using the plant id
            columns = [c.TC_TIMESTAMP] + [col for col in plants_cols if col.split('_')[0] == plant_id]
            # assign target data to dict
            self.train_data[plant_id] = {c.K_TARGET: plants_timeseries.select(columns)}

    def __prepare_features(self):
        """
        Prepare feature data for each plant or market.

        If there's features assigned to the plant with given id, collect them from weather data or generate them
        (relevant for 'time' features, will generate two columns: 'hour' representing daily fluctuation and 'month'
        representing seasonal fluctuation). Otherwise, only initialize an empty lazyframe. Add them to the train data
        of corresponding plant or market with key c.K_FEATURES as a lazyframe.

        """
        for id in self.train_data.keys():
            # if given, get features
            chosen_model = self.config_dict[id]['method']  # keyword of the chosen model as string

            # check if features columns need to be added
            if (c.K_FEATURES in self.config_dict[id][chosen_model].keys() and
                    ast.literal_eval(self.config_dict[id][chosen_model][c.K_FEATURES])):
                # get features name as strings
                features_name = ast.literal_eval(self.config_dict[id][chosen_model][c.K_FEATURES])

                # get features data from weather file
                features = self.weather

                # add time features
                if 'time' in features_name:
                    features_name.remove('time')
                    features = features.with_columns(pl.col(c.TC_TIMESTAMP).dt.hour().alias('hour'))
                    features = features.with_columns(pl.col(c.TC_TIMESTAMP).dt.month().alias('month'))
                    features_name += ['hour', 'month']

                # add other weather features and time steps for indexing
                features_name += [c.TC_TIMESTAMP, c.TC_TIMESTEP]
                features = features.select(features_name)
            else:
                features = pl.LazyFrame()

            self.train_data[id][c.K_FEATURES] = features

    """relevant for data processing"""

    def __make_forecast_for_id(self, current_ts, id):
        """
        Make forecast for the plant with given id. Re-fit model before forecasting if needed.

        Args:
            current_ts: Current timestamp for the forecast.
            id: Identifier for the plant or market.

        Returns:
            forecast: Lazyframe containing the forecast result for the given plant.

        """
        chosen_model = self.config_dict[id]['method']  # keyword of the chosen model as string

        # refit model if needed
        offset = (current_ts - self.start_ts.replace(tzinfo=pytz.UTC)).seconds  # offset between current ts and start ts
        # check offset % refit period to see if the current time is exactly the beginning of a new refitting period
        if offset % self.refit_period == 0:
            self.used_models[id].fit(current_ts=current_ts, length_to_predict=self.length_to_predict,
                                     **self.config_dict[id][chosen_model])

        forecast = self.used_models[id].predict(current_ts=current_ts, length_to_predict=self.length_to_predict,
                                                **self.config_dict[id][chosen_model])  # predict

        return forecast

    def __summarize_forecasts_to_df(self, forecasts: dict, current_ts) -> pl.DataFrame:
        """
        Summarize all forecasts from dictionary to one polars lazyframe. All forecasts should have the same length and
        stored as a lazyframe.

        Args:
            forecasts: Dictionary containing forecast results for each plant or market.
            current_ts: Current timestamp for the forecasts.

        Returns:
            forecasts_df: Dataframe containing the summarized forecast results.

        """

        # generate a column contains time index
        timestamps = f.slice_dataframe_between_times(target_df=self.agentDB.timeseries, reference_ts=current_ts,
                                                     duration=self.length_to_predict).select(c.TC_TIMESTAMP)
        # add timestep column
        timestamps = timestamps.with_columns(pl.col(c.TC_TIMESTAMP).alias(c.TC_TIMESTEP))

        # get time info from original dataframe
        datetime_index = timestamps.select(c.TC_TIMESTAMP)
        dtype = datetime_index.dtypes[0]
        time_unit = dtype.time_unit
        time_zone = dtype.time_zone

        # adjust timestamp column to current time and keep the datatype
        timestamps = timestamps.with_columns(pl.lit(current_ts)
                                             .alias(c.TC_TIMESTAMP)
                                             .cast(pl.Datetime(time_unit=time_unit, time_zone=time_zone)))

        forecasts_list = [timestamps]     # list which will contain all forecast dataframes

        for user_id, forecast in forecasts.items():
            # remove time column(s) for all forecasts
            if c.TC_TIMESTAMP in forecast.columns:
                forecast = forecast.drop(c.TC_TIMESTAMP)
            if c.TC_TIMESTEP in forecast.columns:
                forecast = forecast.drop(c.TC_TIMESTEP)

            # change data type of each forecast, should be the same as the target data
            for column in forecast.columns:
                dtype = self.train_data[user_id][c.K_TARGET].select(column).dtypes

                # each column should only have one data type
                forecast = forecast.with_columns(pl.col(column).cast(dtype[0]))

            # add forecast to the list
            forecasts_list.append(forecast)

        # summarize everything together
        forecasts_df = pl.concat(forecasts_list, how='horizontal')

        return forecasts_df
