# Similar to optimization that it contains the class and calls the classes with functions
import sktime
import darts
import copy
import pytz
import json
import pandas as pd
import polars as pl
# from keras import optimizers
# from keras.layers import Input, Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout
# from keras.models import Model
from sktime.forecasting.arima import ARIMA
from darts.models.forecasting.rnn_model import RNNModel
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from hamlet import constants as c


class Forecaster:
    def __init__(self, agent, plants, ems, timeseries, market, weather, database):
        self.agent = agent  # Contains all files related to the agent and should therefore be returned with the results
                            # is this an AgentDB object?
                            # plant is under agent, but market not. change structure to one forecaster overall?
        self.plants_config = plants     # dict
        self.ems_config = ems       # dict
        self.timeseries = timeseries    # dataframe
        self.market = market    # dict
        self.weather = weather      # dataframe
        self.database = database    # database object
        self.forecasts = agent[c.K_FORECASTS]   # ?
        self.fit_ts = {}

        # pre-defined models from packages
        self.__random_forest_regressor = RandomForestRegressor()
        self.__random_forest_classifier = RandomForestClassifier()
        self.__arima = ARIMA()
        self.__rnn = RNNModel()

        self.models = {}

    # public methods
    def init_forecaster(self):
        """Initialize forecaster with corresponding model for each plant, initial fitting of models."""
        for plant_id in self.plants_config.keys:
            # always get an object with fit and predict method
            if 'fcast' in self.plants_config[plant_id].keys:
                method = self.plants_config[plant_id]['fcast']['method']
                self.models[plant_id] = getattr(self, method)(**self.plants_config[plant_id]['fcast']['method'])

                # fit to the corresponding plant data
                if hasattr(self.models[plant_id], 'fit'):
                    self.models[plant_id].fit(self.timeseries[plant_id], **self.plants_config[plant_id]['fcast']['method'])

                    # generate a list of timesteps where model need to be fitted again
                    self.fit_ts[plant_id] = []  # to-be generated

    def make_all_forecasts(self):
        """Make forecasts for all plants."""
        # TODO: @Jiahe, please implement this method

        # Loop through all forecast columns (probably best to sort this by plant ID and market type)

        # Check if the model needs to be re-fitted

        # Make the forecast

        # Return the forecasts dataframe (i.e. self.forecasts)
        return self.forecasts

    def make_forecast(self, plant_id, current_ts, length_to_predict):
        """Make forecast for the given timestep. Re-fit model before forecasting if needed."""
        if current_ts in self.fit_ts[plant_id]:
            # re-fit
            self.models[plant_id].fit(self.timeseries[plant_id])

        return self.models[plant_id].forecast(current_ts, length_to_predict,
                                              **self.plants_config[plant_id]['fcast']['predict'])

    # pre-defined model objects
    class __naive:
        def __init__(self,  **kwargs):
            self.past_df = pl.dataframe()

        def fit(self, target, **kwargs):
            self.past_df = target

        def predict(self, length_to_predict, **kwargs):
            pass

    # class __cnn:
    #     def __init__(self, seq_length, feature_length, **kwargs):
    #         inputs = Input(shape=(seq_length, feature_length))
    #         x = Conv1D(32, 3, activation='relu')(inputs)
    #         x = MaxPooling1D(2)(x)
    #         x = Conv1D(64, 3, activation='relu')(x)
    #         x = MaxPooling1D(2)(x)
    #         x = Flatten()(x)
    #         outputs = Dense(1, activation='sigmoid')(x)
    #
    #         self.cnn_model = Model(inputs=inputs, outputs=outputs)
    #
    #     def fit(self, features, target, epoch, **kwargs):
    #         self.cnn_model.fit(target, features)
    #
    #     def predict(self, features, **kwargs):
    #         self.cnn_model.predict()

    """
    user can also define own model-object here. some basic rules:
    1. the object must contain predict() method, fit() is optional
    2. all the functions need to contain **kwargs to make overall structure working
    3. if extra parameters need, define in config file with exactly same name
        some general parameters include: features, target, length_to_predict
    4. object must be private ("__") and name should be exactly same as the string defined in method in config file
    """
    # user-defined model objects
    class __my_model:
        """This is just an example."""
        def __init__(self, **kwargs):
            self.my_data = pl.dataframe

        def fit(self, target, **kwargs):
            self.my_data = target

        def predict(self, length_to_predict, **kwargs):
            return self.my_data[:length_to_predict]

    # static methods
    @staticmethod
    def __prepare_training_data(data, t_current, length_to_predict, target, features):
        data_prepared = copy.deepcopy(data[:data.loc[data['timestamp'] == t_current].index[0] + 1])

        # convert epoch timestamps to datetime objects in UTC timezone
        utc_time = pd.to_datetime(data['timestamp'], unit='s')

        # format Berlin time as string in desired format (e.g. '2022-03-21 18:45:00')
        data_prepared['hour'] = utc_time.dt.hour
        data_prepared['month'] = utc_time.dt.month

        return data_prepared






