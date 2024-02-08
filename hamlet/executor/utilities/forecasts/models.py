__author__ = "jiahechu"
__credits__ = ""
__license__ = ""
__maintainer__ = "jiahechu"
__email__ = "jiahe.chu@tum.de"

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # turn off onednn for tensorflow
from datetime import timedelta
import ast
import polars as pl
import pandas as pd
import numpy as np
from keras.layers import Input, Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.models import Model
from sktime.forecasting.arima import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pvlib
from pvlib.pvsystem import PVSystem
from pvlib.location import Location
from windpowerlib import ModelChain, WindTurbine
from hamlet import constants as c
from hamlet import functions as f


def forecast_model(name):
    """Decorator to match model with the given name. All forecast models should use this decorator."""
    def decorator(cls):
        cls.name = name
        return cls
    return decorator


class ModelBase:
    """
    Base class for all forecast models, all forecast models should inherit from this class. This class contains the
    basic necessary attributes and methods.

    Attributes:
        train_data: dictionary contains 2 keys: features and target. Target contains perfect data of plant / market
        using this model. Features contain the data from weather file with user-defined columns. Column names of
        features should be assigned in config file under 'features' and should be identical to the column names in
        weather file.

    Methods:
        fit: fitting the forecast model. Relevant for e.g. ML or DL models.
        predict: make forecast and return the resulting forecast data as a Dataframe.
        update_train_data: replace the train data with a new train data.

    """
    def __init__(self, train_data: dict, **kwarg):
        self.train_data = train_data

    def fit(self, current_ts, length_to_predict, **kwargs):
        """
        Implement this function when necessary, e.g. for machine learning or deep learning models. If fitting is not
        necessary (e.g. statistical models), do nothing. 2 args will be passed to this method for all models (but users
        can decide if they want to use them or not):

        Args:
            current_ts: Current timestep when making the fitting.
            length_to_predict: How long in the future should be covered in the resulting forecast of this model. Unit:
            seconds.
        """
        pass

    def predict(self, current_ts, length_to_predict, **kwargs):
        """
        This function should be implemented for all models. The function should return a polars Dataframe contains
        forecasting results with column name equals target column name. 2 args will be passed to this method for all
        models (but users can decide if they want to use them or not):

        Args:
            current_ts: Current timestep when making the prediction.
            length_to_predict: How long in the future should be covered in the resulting forecast of this model. Unit:
            seconds.

        """
        raise NotImplementedError('The forecast model must have the \'predict\' method.')

    def update_train_data(self, new_train_data):
        """
        Replace train data with the given new train data. This function should be called in the Forecaster. Currently
        only relevant for local market, because the "real" local market price need to be updated after each simulated
        timestamp.
        """
        self.train_data = new_train_data


# pre-defined model objects
@forecast_model(name='perfect')
class PerfectModel(ModelBase):
    """Perfect forecast of the future."""
    def predict(self, current_ts, length_to_predict, **kwargs):
        """
        Simply take the actual data from target as forecast.

        Args:
            current_ts: Current timestep when making the prediction.
            length_to_predict: How long in the future should be covered in the resulting forecast of this model. Unit:
            seconds.

        Returns:
            forecast: Forecast result as Dataframe.

        """
        # predict
        forecast = f.slice_dataframe_between_times(target_df=self.train_data[c.K_TARGET], reference_ts=current_ts,
                                                   duration=length_to_predict, unit='second')
        return forecast


@forecast_model(name='naive')
class NaiveModel(ModelBase):
    """Today will be the same as last day with offset."""
    def predict(self, current_ts, length_to_predict, offset, **kwargs):
        """
        Take the data starting at the same time step as current_ts from offset day(s) before.

        Args:
            current_ts: Current timestep when making the prediction.
            length_to_predict: How long in the future should be covered in the resulting forecast of this model. Unit:
            seconds.
            offset:  Offset in days to the current day. Unit: days.

        Returns:
            forecast: Forecast result as Dataframe.

        """
        reference_ts = current_ts - timedelta(days=offset)  # calculate reference time step
        forecast = f.slice_dataframe_between_times(target_df=self.train_data[c.K_TARGET], reference_ts=reference_ts,
                                                   duration=length_to_predict, unit='second')    # predict

        return forecast


@forecast_model(name='average')
class AverageModel(ModelBase):
    """Today will be the same as the average of the last n days with offset."""
    def predict(self, current_ts, length_to_predict, offset, days, **kwargs):
        """
        Calculate the average of the last days of given number (days) with offset (offset) to current timestep, use the
        result as forecast.

        Args:
            current_ts: Current timestep when making the prediction.
            length_to_predict: How long in the future should be covered in the resulting forecast of this model. Unit:
            seconds.
            offset:  Offset in days to the current day. Unit: days.
            days: Number of days to be used for averaging. Unit: days.

        Returns:
            forecast: Forecast result as Dataframe.

        """
        # get column name
        plant_id = self.train_data[c.K_TARGET].drop(c.TC_TIMESTAMP).columns[0]

        # generate dataframe with data for last n days
        past_data = []  # empty list, will contain past data for each past day
        for day in range(days):
            reference_ts = current_ts - timedelta(days=(offset + day))
            forecast = f.slice_dataframe_between_times(target_df=self.train_data[c.K_TARGET], reference_ts=reference_ts,
                                                       duration=length_to_predict, unit='second')
            past_data.append(forecast.select(plant_id).rename({plant_id: str(day)}))

        # initialize a forecast Dataframe
        forecast = pl.concat(past_data, how='horizontal')

        # calculate average of past data
        forecast = forecast.with_columns((pl.sum_horizontal(forecast.columns) / len(forecast.columns)).alias(plant_id))

        return forecast.select(plant_id)


@forecast_model(name='smoothed')
class SmoothedModel(ModelBase):
    """Prediction value is a moving mean of the future values with a specified window width."""
    def predict(self, current_ts, length_to_predict, steps, **kwargs):
        """
        Calculate the moving average in the future as prediction.

        For each to be predicted time step, calculate the average of the next "steps" timesteps and use the average as
        the prediction for this to be predicted time step.

        Args:
            current_ts: Current timestep when making the prediction.
            length_to_predict: How long in the future should be covered in the resulting forecast of this model. Unit:
            seconds.
            steps: Number of future time steps to be used for smoothing (T-ts). Unit: time steps

        Returns:
            forecast: Forecast result as Dataframe.

        """
        # calculate train data resolution first
        resolution = f.calculate_time_resolution(self.train_data[c.K_TARGET])

        # calculate the moving average for each timestep to predict
        forecast = []   # empty list, will contain the moving average for each horizon
        for timestep in range(1, int(length_to_predict / resolution) + 1):
            reference_ts = current_ts + timestep * resolution
            horizon = f.slice_dataframe_between_times(target_df=self.train_data[c.K_TARGET], reference_ts=reference_ts,
                                                      duration=steps * resolution, unit='second')  # get horizon

            # calculate average for the horizon
            forecast.append(horizon.mean())

        # summarize all moving averages to one Dataframe
        forecast = pl.concat(forecast, how='vertical')

        return forecast


@forecast_model(name='sarma')
class SARMAModel(ModelBase):
    """Seasonal autoregressive moving average model."""
    def __init__(self, train_data, **kwargs):
        super().__init__(train_data, **kwargs)
        raise NotImplementedError('SARMA model needs to be implemented!')

    def predict(self, **kwargs):
        ...


@forecast_model(name='rfr')
class RandomForest(ModelBase):
    """Random forest regressor."""
    def __init__(self, train_data, **kwargs):
        super().__init__(train_data, **kwargs)
        self.model = RandomForestRegressor()

    def fit(self, current_ts, days, **kwargs):
        """
        Prepare features and targets for fitting and fit the regressor.

        The past actual (c.TC_TIMESTAMP == c.TC_TIMESTEP) feature data except c.TC_TIMESTAMP and c.TC_TIMESTEP columns
        will be taken as features. The past target data except c.TC_TIMESTAMP column will be taken as target. Both
        target and features data will be converted to pandas.DataFrame since sklearn does not support polars yet.

        Args:
            current_ts: Current timestep when making the fitting.
            days: Past days that are used to fit the random forest regressor.

        """
        # slice target for training
        target = f.slice_dataframe_between_times(target_df=self.train_data[c.K_TARGET], reference_ts=current_ts,
                                                 duration=(-days), unit='day')

        # slice features for training
        filter_condition = (pl.col(c.TC_TIMESTAMP) == pl.col(c.TC_TIMESTEP))    # take only actual past weather data
        features_all = self.train_data[c.K_FEATURES].filter(filter_condition)
        features = f.slice_dataframe_between_times(target_df=features_all, reference_ts=current_ts, duration=(-days),
                                                   unit='day')

        # delete time columns for fitting
        target = target.drop(c.TC_TIMESTAMP)
        features = features.drop(c.TC_TIMESTAMP, c.TC_TIMESTEP)

        # convert data into pandas, since sklearn does not support polars yet
        target = target.to_pandas()
        features = features.to_pandas()

        # fitting
        self.model.fit(X=features, y=target)

    def predict(self, current_ts, length_to_predict, **kwargs):
        """
        Make prediction using the random forest regressor.

        The forecasted (take c.TC_TIMESTEP as time index) features data will be used as features and fed into the
        regressor. The forecasting result from the regressor will be converted to a polars.DataFrame as return.

        Args:
            current_ts: Current timestep when making the prediction.
            length_to_predict: How long in the future should be covered in the resulting forecast of this model. Unit:
            seconds.

        Returns:
            forecast: Forecast result as Dataframe.
        """
        # slice features for prediction
        features = f.slice_dataframe_between_times(target_df=self.train_data[c.K_FEATURES], reference_ts=current_ts,
                                                   duration=0)   # get data at current ts
        # set future time steps as 'index'
        features = features.with_columns(pl.col(c.TC_TIMESTEP).alias(c.TC_TIMESTAMP))
        features = f.slice_dataframe_between_times(target_df=features, reference_ts=current_ts,
                                                   duration=length_to_predict, unit='second')
        features = features.drop(c.TC_TIMESTAMP, c.TC_TIMESTEP)     # delete time columns for prediction

        # convert data into pandas, since sklearn only takes pandas for now
        features = features.to_pandas()

        # predict and convert result to polars Dataframe
        forecast = self.model.predict(X=features)
        plant_id = self.train_data[c.K_TARGET].columns  # get column name for result
        plant_id.remove(c.TC_TIMESTAMP)
        forecast = pl.DataFrame({plant_id[0]: forecast.ravel()})

        # forecasting
        return forecast


@forecast_model(name='cnn')
class CNNModel(ModelBase):
    """Convolutional neural network."""
    def __init__(self, train_data, window_length, **kwargs):
        """
        Initialize the CNNModel.

        Parameters:
            train_data (dict): The training data containing features and target.
            window_length (int): The length of the input window for the neural network.
            **kwargs: Additional keyword arguments for model initialization.

        This constructor sets up the architecture of the neural network model and compiles it.

        It calculates the number of features, defines the model architecture, and compiles the model with
        the mean squared error loss function and the Adam optimizer.

        """
        super().__init__(train_data, **kwargs)

        # calculate number of features
        features = self.train_data[c.K_FEATURES].columns
        if c.TC_TIMESTAMP in features:
            features.remove(c.TC_TIMESTAMP)
        if c.TC_TIMESTEP in features:
            features.remove(c.TC_TIMESTEP)
        features_number = len(features)

        # define model
        inputs = Input(shape=(window_length, features_number))
        x = Conv1D(64, 3, activation='relu')(inputs)
        x = MaxPooling1D(2)(x)
        x = Conv1D(32, 3, activation='relu')(x)
        x = MaxPooling1D(2)(x)
        x = Flatten()(x)
        outputs = Dense(1, activation='relu')(x)

        self.cnn_model = Model(inputs=inputs, outputs=outputs)
        self.cnn_model.compile(loss='mse', optimizer='adam')

    @staticmethod
    def __prepare_train_data(window_length, X_train, y_train):
        """
        Prepare training data for a neural network.

        This function splits the provided training data into training and validation sets and organizes them into
        sequences with the given window length suitable for training a CNN or RNN model. For each to be predicted
        timestep, a window is used as features. The resulting sequences and targets are converted to numpy arrays with
        float32 data type.

        Parameters:
            window_length (int): The length of the input sequences as features.
            X_train (pandas.DataFrame): The training input data, typically a DataFrame.
            y_train (pandas.Series): The training target data, typically a Series.

        Returns:
            train_sequences (numpy.ndarray): Sequences of training data with shape (num_samples, window_length,
            num_features).
            train_targets (numpy.ndarray): Target values corresponding to the training sequences.
            val_sequences (numpy.ndarray): Sequences of validation data with shape (num_samples, window_length,
            num_features).
            val_targets (numpy.ndarray): Target values corresponding to the validation sequences.
        """
        # split train and test data
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

        seq_length = window_length
        target_length = 1

        train_sequences = []
        train_targets = []
        for i in range(seq_length, len(X_train) - target_length + 1):
            train_sequences.append(X_train.iloc[i - seq_length:i, :])
            train_targets.append(y_train.iloc[i:i + target_length])

        val_sequences = []
        val_targets = []
        for i in range(seq_length, len(X_test) - target_length + 1):
            val_sequences.append(X_test.iloc[i - seq_length:i, :])
            val_targets.append(y_test.iloc[i:i + target_length])

        train_sequences = np.array([np.array(train_sequence) for train_sequence in train_sequences]).astype('float32')
        train_targets = np.array([np.array(train_target) for train_target in train_targets]).astype('float32')
        val_sequences = np.array([np.array(val_sequence) for val_sequence in val_sequences]).astype('float32')
        val_targets = np.array([np.array(val_target) for val_target in val_targets]).astype('float32')

        return train_sequences, train_targets, val_sequences, val_targets

    @staticmethod
    def __prepare_predict_data(window_length, X_predict):
        """
        Prepare prediction data for a neural network. Similar to self.__prepare_train_data, but only for features.

        Parameters:
            window_length (int): The length of the input sequences as features.
            X_predict (pandas.DataFrame): The input data for prediction, typically a DataFrame.

        Returns:
            predict_sequences (numpy.ndarray): Sequences of input data for prediction with shape (num_samples,
            window_length, num_features).

        """
        seq_length = window_length
        target_length = 1

        predict_sequences = []
        for i in range(seq_length, len(X_predict) - target_length + 1):
            predict_sequences.append(X_predict.iloc[i - seq_length:i, :])

        predict_sequences = np.array([np.array(predict_sequence) for predict_sequence in predict_sequences])\
                              .astype('float32')

        return predict_sequences

    def fit(self, current_ts, days, window_length, epoch, **kwargs):
        """
        Prepare features and targets for fitting and fit the neural network.

        The past actual (c.TC_TIMESTAMP == c.TC_TIMESTEP) feature data except c.TC_TIMESTAMP and c.TC_TIMESTEP columns
        will be taken as features. The past target data except c.TC_TIMESTAMP column will be taken as target. Both
        target and features data will be converted to sequences with the given window length in form of numpy arrays
        (since keras does not support polars yet), each features' sequence matches one target timestep. The length of
        the taken features data from self.train_data[c.K_FEATURES] is "window_length" longer than the taken target, the
        actual feature sequences for fitting returned from self.__prepare_train_data is as long as the returned target
        for fitting.

        Args:
            current_ts: Current timestep when making the fitting.
            days: Past days that are used to fit the random forest regressor.
            window_length: The length of the input sequences as features.
            epoch: Number of epochs for fitting.

        """
        # slice target for training
        target = f.slice_dataframe_between_times(target_df=self.train_data[c.K_TARGET], reference_ts=current_ts,
                                                 duration=(-days), unit='day')
        # slice features for training
        filter_condition = (pl.col(c.TC_TIMESTAMP) == pl.col(c.TC_TIMESTEP))  # take only actual past weather data
        features_all = self.train_data[c.K_FEATURES].filter(filter_condition)
        features = f.slice_dataframe_between_times(target_df=features_all, reference_ts=current_ts, duration=(-days),
                                                   unit='day')

        # delete time columns for fitting
        target = target.drop(c.TC_TIMESTAMP)
        features = features.drop(c.TC_TIMESTAMP, c.TC_TIMESTEP)

        # convert data into pandas, since sklearn only takes pandas for now
        target = target.to_pandas()
        features = features.to_pandas()

        # convert data dimension to 3d
        train_sequences, train_targets, val_sequences, val_targets = self.__prepare_train_data(window_length, features,
                                                                                               target)

        # fitting
        self.cnn_model.fit(train_sequences, train_targets, epochs=epoch, validation_data=(val_sequences, val_targets))

    def predict(self, current_ts, length_to_predict, window_length, **kwargs):
        """
        Make prediction using the neural network.

        The forecasted (take c.TC_TIMESTEP as time index) features data will be used as features and converted into
        sequences like in self.fit function. The forecasting result from the neural network will be converted to a
        polars.DataFrame as return.

        Args:
            current_ts: Current timestep when making the prediction.
            length_to_predict: How long in the future should be covered in the resulting forecast of this model. Unit:
            seconds.
            window_length: The length of the input sequences as features.

        Returns:
            forecast: Forecast result as Dataframe.

        """
        # calculate train data resolution first
        resolution = f.calculate_time_resolution(self.train_data[c.K_TARGET])

        # slice features for prediction
        features = f.slice_dataframe_between_times(target_df=self.train_data[c.K_FEATURES], reference_ts=current_ts,
                                                   duration=0)  # get data at current ts

        # set future time steps as 'index'
        features = features.with_columns(pl.col(c.TC_TIMESTEP).alias(c.TC_TIMESTAMP))
        features = f.slice_dataframe_between_times(target_df=features, reference_ts=current_ts,
                                                   duration=length_to_predict + resolution * window_length)
        features = features.drop(c.TC_TIMESTAMP, c.TC_TIMESTEP)  # delete time columns for prediction

        # convert data into pandas, since sklearn only takes pandas for now
        features = features.to_pandas()

        # convert features to 3d array
        predict_sequences = self.__prepare_predict_data(window_length, features)

        # predict and convert result to polars Dataframe
        forecast = self.rnn_model.predict(predict_sequences)
        plant_id = self.train_data[c.K_TARGET].columns  # get column name for result
        plant_id.remove(c.TC_TIMESTAMP)
        forecast = pl.DataFrame({plant_id[0]: forecast.ravel()})

        return forecast


@forecast_model(name='rnn')
class RNNModel(ModelBase):
    """Recurrent neural network."""
    def __init__(self, train_data, window_length, **kwargs):
        """
        Initialize the RNNModel.

        Parameters:
            train_data (dict): The training data containing features and target.
            window_length (int): The length of the input window for the neural network.
            **kwargs: Additional keyword arguments for model initialization.

        This constructor sets up the architecture of the neural network model and compiles it.

        It calculates the number of features, defines the model architecture, and compiles the model with
        the mean squared error loss function and the Adam optimizer.
        """
        super().__init__(train_data, **kwargs)

        # calculate number of features
        features = self.train_data[c.K_FEATURES].columns
        if c.TC_TIMESTAMP in features:
            features.remove(c.TC_TIMESTAMP)
        if c.TC_TIMESTEP in features:
            features.remove(c.TC_TIMESTEP)
        features_number = len(features)

        # define model
        inputs = Input(shape=(window_length, features_number))
        x = LSTM(64, activation='relu')(inputs)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(16, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1, activation='relu')(x)

        self.rnn_model = Model(inputs=inputs, outputs=outputs)
        self.rnn_model.compile(loss='mse', optimizer='adam')

    @staticmethod
    def __prepare_train_data(window_length, X_train, y_train):
        """
        Prepare training data for a neural network.

        This function splits the provided training data into training and validation sets and organizes them into
        sequences with the given window length suitable for training a CNN or RNN model. For each to be predicted
        timestep, a window is used as features. The resulting sequences and targets are converted to numpy arrays with
        float32 data type.

        Parameters:
            window_length (int): The length of the input sequences as features.
            X_train (pandas.DataFrame): The training input data, typically a DataFrame.
            y_train (pandas.Series): The training target data, typically a Series.

        Returns:
            train_sequences (numpy.ndarray): Sequences of training data with shape (num_samples, window_length,
            num_features).
            train_targets (numpy.ndarray): Target values corresponding to the training sequences.
            val_sequences (numpy.ndarray): Sequences of validation data with shape (num_samples, window_length,
            num_features).
            val_targets (numpy.ndarray): Target values corresponding to the validation sequences.
        """
        # split train and test data
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

        seq_length = window_length
        target_length = 1

        train_sequences = []
        train_targets = []
        for i in range(seq_length, len(X_train) - target_length + 1):
            train_sequences.append(X_train.iloc[i - seq_length:i, :])
            train_targets.append(y_train.iloc[i:i + target_length])

        val_sequences = []
        val_targets = []
        for i in range(seq_length, len(X_test) - target_length + 1):
            val_sequences.append(X_test.iloc[i - seq_length:i, :])
            val_targets.append(y_test.iloc[i:i + target_length])

        train_sequences = np.array([np.array(train_sequence) for train_sequence in train_sequences]).astype('float32')
        train_targets = np.array([np.array(train_target) for train_target in train_targets]).astype('float32')
        val_sequences = np.array([np.array(val_sequence) for val_sequence in val_sequences]).astype('float32')
        val_targets = np.array([np.array(val_target) for val_target in val_targets]).astype('float32')

        return train_sequences, train_targets, val_sequences, val_targets

    @staticmethod
    def __prepare_predict_data(window_length, X_predict):
        """
        Prepare prediction data for a neural network. Similar to self.__prepare_train_data, but only for features.

        Parameters:
            window_length (int): The length of the input sequences as features.
            X_predict (pandas.DataFrame): The input data for prediction, typically a DataFrame.

        Returns:
            predict_sequences (numpy.ndarray): Sequences of input data for prediction with shape (num_samples,
            window_length, num_features).

        """
        seq_length = window_length
        target_length = 1

        predict_sequences = []
        for i in range(seq_length, len(X_predict) - target_length + 1):
            predict_sequences.append(X_predict.iloc[i - seq_length:i, :])

        predict_sequences = np.array([np.array(predict_sequence) for predict_sequence in predict_sequences])\
                              .astype('float32')

        return predict_sequences

    def fit(self, current_ts, days, window_length, epoch, **kwargs):
        """
        Prepare features and targets for fitting and fit the neural network.

        The past actual (c.TC_TIMESTAMP == c.TC_TIMESTEP) feature data except c.TC_TIMESTAMP and c.TC_TIMESTEP columns
        will be taken as features. The past target data except c.TC_TIMESTAMP column will be taken as target. Both
        target and features data will be converted to sequences with the given window length in form of numpy arrays
        (since keras does not support polars yet), each features' sequence matches one target timestep. The length of
        the taken features data from self.train_data[c.K_FEATURES] is "window_length" longer than the taken target, the
        actual feature sequences for fitting returned from self.__prepare_train_data is as long as the returned target
        for fitting.

        Args:
            current_ts: Current timestep when making the fitting.
            days: Past days that are used to fit the random forest regressor.
            window_length: The length of the input sequences as features.
            epoch: Number of epochs for fitting.

        """
        # slice target for training
        target = f.slice_dataframe_between_times(target_df=self.train_data[c.K_TARGET], reference_ts=current_ts,
                                                 duration=(-days), unit='day')
        # slice features for training
        filter_condition = (pl.col(c.TC_TIMESTAMP) == pl.col(c.TC_TIMESTEP))  # take only actual past weather data
        features_all = self.train_data[c.K_FEATURES].filter(filter_condition)
        features = f.slice_dataframe_between_times(target_df=features_all, reference_ts=current_ts, duration=(-days),
                                                   unit='day')

        # delete time columns for fitting
        target = target.drop(c.TC_TIMESTAMP)
        features = features.drop(c.TC_TIMESTAMP, c.TC_TIMESTEP)

        # convert data into pandas, since sklearn only takes pandas for now
        target = target.to_pandas()
        features = features.to_pandas()

        # convert data dimension to 3d
        train_sequences, train_targets, val_sequences, val_targets = self.__prepare_train_data(window_length, features,
                                                                                               target)

        # fitting
        self.rnn_model.fit(train_sequences, train_targets, epochs=epoch, validation_data=(val_sequences, val_targets))

    def predict(self, current_ts, length_to_predict, window_length, **kwargs):
        """
        Make prediction using the neural network.

        The forecasted (take c.TC_TIMESTEP as time index) features data will be used as features and converted into
        sequences like in self.fit function. The forecasting result from the neural network will be converted to a
        polars.DataFrame as return.

        Args:
            current_ts: Current timestep when making the prediction.
            length_to_predict: How long in the future should be covered in the resulting forecast of this model. Unit:
            seconds.
            window_length: The length of the input sequences as features.

        Returns:
            forecast: Forecast result as Dataframe.

        """
        # calculate train data resolution first
        resolution = f.calculate_time_resolution(self.train_data[c.K_TARGET])

        # slice features for prediction
        features = f.slice_dataframe_between_times(target_df=self.train_data[c.K_FEATURES], reference_ts=current_ts,
                                                   duration=0)  # get data at current ts

        # set future time steps as 'index'
        features = features.with_columns(pl.col(c.TC_TIMESTEP).alias(c.TC_TIMESTAMP))
        features = f.slice_dataframe_between_times(target_df=features, reference_ts=current_ts,
                                                   duration=length_to_predict + resolution * window_length)
        features = features.drop(c.TC_TIMESTAMP, c.TC_TIMESTEP)  # delete time columns for prediction

        # convert data into pandas, since sklearn only takes pandas for now
        features = features.to_pandas()

        # convert features to 3d array
        predict_sequences = self.__prepare_predict_data(window_length, features)

        # predict and convert result to polars Dataframe
        forecast = self.rnn_model.predict(predict_sequences)
        plant_id = self.train_data[c.K_TARGET].columns    # get column name for result
        plant_id.remove(c.TC_TIMESTAMP)
        forecast = pl.DataFrame({plant_id[0]: forecast.ravel()})

        return forecast


@forecast_model(name='arima')
class ARIMAModel(ModelBase):
    """Autoregressive integrated moving average model."""
    def __init__(self, train_data, order, **kwargs):
        super().__init__(train_data, **kwargs)
        self.fit_ts = None
        self.arima = ARIMA(order=ast.literal_eval(order))
        raise NotImplementedError('ARIMA model needs to be fixed!')

    def fit(self, current_ts, length_to_predict, days, **kwargs):
        # slice target for training
        target = f.slice_dataframe_between_times(target_df=self.train_data[c.K_TARGET], reference_ts=current_ts,
                                                 duration=(-length_to_predict))

        # save fitting timeseries to attribute
        self.fit_ts = target.select(c.TC_TIMESTAMP)

        # delete time columns for fitting
        target = target.drop(c.TC_TIMESTAMP)

        # convert data into pandas, since sklearn only takes pandas for now
        target = target.to_pandas()

        # fitting
        self.arima.fit(y=target)

    def predict(self, current_ts, length_to_predict, **kwargs):
        # get forecast horizon
        target = f.slice_dataframe_between_times(target_df=self.train_data[c.K_TARGET], reference_ts=current_ts,
                                                 duration=length_to_predict)
        forecast_ts = target.with_columns(pl.col(c.TC_TIMESTAMP).cast(pl.Int64)).select(c.TC_TIMESTAMP)
        fit_ts = self.fit_ts.with_columns(pl.col(c.TC_TIMESTAMP).cast(pl.Int64)).select(c.TC_TIMESTAMP)

        forecast_horizon = forecast_ts.to_numpy().ravel() - fit_ts.to_numpy().ravel()

        # predict and convert result to polars Dataframe
        forecast = self.arima.predict(fh=forecast_horizon)
        plant_id = self.train_data[c.K_TARGET].columns  # get column name for result
        plant_id.remove(c.TC_TIMESTAMP)
        forecast = pl.DataFrame({plant_id[0]: forecast.ravel()})

        # forecasting
        return forecast


@forecast_model(name='weather')
class WeatherModel(ModelBase):
    """
    Forecast based on weather forecast ("specs" only).

    This model is currently implemented using pandas. The functions are taken from the Creator. For detailed
    documentations check hamlet/creator/agents/agents.py

    """
    def __pv_model(self, current_ts, length_to_predict):
        # get pv orientation
        plant = self.train_data['plant_config']
        surface_tilt = plant['sizing']['orientation']
        surface_azimuth = plant['sizing']['angle']

        # get hardware data
        config = self.train_data['specs']
        module = pd.Series(config['module'])
        inverter = pd.Series(config['inverter'])

        # set temperature model
        temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

        # generate pv system
        system = PVSystem(
            surface_tilt=surface_tilt,
            surface_azimuth=surface_azimuth,
            module_parameters=module,
            inverter_parameters=inverter,
            temperature_model_parameters=temperature_model_parameters
        )

        # get location data
        location = self.train_data['general_config']['location']
        latitude = location['latitude']
        longitude = location['longitude']
        name = location[c.TC_NAME]
        altitude = location['altitude']

        # get weather data from csv
        weather = f.slice_dataframe_between_times(target_df=self.train_data[c.K_FEATURES], reference_ts=current_ts,
                                                  duration=0)
        weather = weather.with_columns(pl.col(c.TC_TIMESTEP).alias(c.TC_TIMESTAMP))
        weather = f.slice_dataframe_between_times(target_df=weather, reference_ts=current_ts,
                                                  duration=length_to_predict)
        weather = weather.to_pandas()

        # get real dhi data
        filter_condition = (pl.col(c.TC_TIMESTAMP) == pl.col(c.TC_TIMESTEP))  # take only actual past weather data
        weather_real = self.train_data[c.K_FEATURES].filter(filter_condition)
        dhi_real = f.slice_dataframe_between_times(target_df=weather_real, reference_ts=current_ts,
                                                   duration=length_to_predict).select(c.TC_DHI).to_pandas()

        # replace dhi with real dhi
        weather[c.TC_DHI] = dhi_real[c.TC_DHI]

        # convert time data to datetime (use utc time overall in pvlib)
        time = pd.DatetimeIndex(pd.to_datetime(weather[c.TC_TIMESTAMP], unit='s', utc=True))
        weather.index = time
        weather.index.name = 'utc_time'

        # adjust temperature data
        weather.rename(columns={c.TC_TEMPERATURE: 'temp_air'}, inplace=True)  # rename to pvlib format
        weather['temp_air'] -= 273.15  # convert unit to celsius

        # get solar position
        # test data find in https://www.suncalc.org/#/48.1364,11.5786,15/2022.02.15/16:21/1/3
        solpos = pvlib.solarposition.get_solarposition(
            time=time,
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            temperature=weather['temp_air'],
            pressure=weather[c.TC_PRESSURE],
        )

        # calculate dni with solar position
        weather.loc[:, c.TC_DNI] = (weather[c.TC_GHI] - weather[c.TC_DHI]) / np.cos(solpos['zenith'])

        # get location data and create corresponding pvlib Location object
        location = Location(
            latitude,
            longitude,
            name=name,
            altitude=altitude
        )

        # create calculation model for the given pv system and location
        mc = pvlib.modelchain.ModelChain(system, location)

        # calculate model under given weather data and get output ac power from it
        mc.run_model(weather)
        power = mc.results.ac

        # calculate nominal power
        nominal_power = module['Impo'] * module['Vmpo']

        # set time index to origin timestamp
        power.index = weather[c.TC_TIMESTAMP]
        power.index.name = c.TC_TIMESTAMP

        # rename and round data column
        power.rename(c.ET_ELECTRICITY, inplace=True)
        power = power.to_frame()

        # calculate and round power
        power = power / nominal_power * plant['sizing']['power']
        power = power.round().astype(int)

        # replace all negative values
        power[power < 0] = 0

        # get column name
        plant_column = self.train_data[c.K_TARGET].columns
        plant_column.remove(c.TC_TIMESTAMP)
        column_name = plant_column[0]

        return pl.DataFrame(power).rename({'power': column_name})

    def __wind_model(self, current_ts, length_to_predict):
        # get spec file
        specs = self.train_data['specs']

        # get forecasted weather data
        weather = f.slice_dataframe_between_times(target_df=self.train_data[c.K_FEATURES], reference_ts=current_ts,
                                                  duration=0)
        weather = weather.with_columns(pl.col(c.TC_TIMESTEP).alias(c.TC_TIMESTAMP))
        weather = f.slice_dataframe_between_times(target_df=weather, reference_ts=current_ts,
                                                  duration=length_to_predict)
        weather = weather.to_pandas()

        # convert time data to datetime
        time = pd.DatetimeIndex(pd.to_datetime(weather[c.TC_TIMESTAMP], unit='s', utc=True))
        weather.index = time
        weather.index.name = None

        # delete unnecessary columns and rename
        weather = weather[[c.TC_TIMESTAMP, c.TC_TEMPERATURE, c.TC_TEMPERATURE_FEELS_LIKE, c.TC_PRESSURE, c.TC_HUMIDITY,
                           c.TC_WIND_SPEED, c.TC_WIND_DIRECTION]]
        weather.rename(columns={c.TC_TEMPERATURE: 'temperature'}, inplace=True)

        if 'roughness_length' not in weather.columns:
            weather['roughness_length'] = 0.15

        # generate height level hard-coded
        weather.columns = pd.MultiIndex.from_tuples(tuple(zip(weather.columns, [2, 2, 2, 2, 2, 2, 2, 10, 10, 2])),
                                                    names=('', 'height'))

        # get nominal power
        nominal_power = specs['nominal_power']

        # convert power curve to dataframe
        data_pc = {"value": specs['power_curve'], "wind_speed": specs['wind_speed']}
        # This needs to be done in some instances as sometimes the specs are read incorrectly (workaround)
        if isinstance(data_pc['value'], pd.DataFrame):
            data_pc['value'] = data_pc['value']['value'].values
        specs['power_curve'] = pd.DataFrame(data=data_pc)

        # convert power coefficient curve to dataframe
        data_pcc = {"value": specs['power_coefficient_curve'], "wind_speed": specs['wind_speed']}
        # This needs to be done in some instances as sometimes the specs are read incorrectly (workaround)
        if isinstance(data_pcc['value'], pd.DataFrame):
            data_pcc['value'] = data_pcc['value']['value'].values
        specs['power_coefficient_curve'] = pd.DataFrame(data=data_pcc)

        # generate a WindTurbine object from data
        turbine = WindTurbine(**specs)

        # calculate turbine model
        mc_turbine = ModelChain(turbine).run_model(weather)

        # get output power
        power = mc_turbine.power_output

        # set time index to origin timestamp
        power.index = weather[c.TC_TIMESTAMP].unstack(level=0).values
        power.index.name = c.TC_TIMESTAMP

        # rename data column
        power.rename(c.ET_ELECTRICITY, inplace=True)
        power = power.to_frame()

        # calculate and round power
        power = power / nominal_power * self.train_data['plant_config']['sizing']['power']
        power = power.round().astype(int)

        # get column name
        plant_column = self.train_data[c.K_TARGET].columns
        plant_column.remove(c.TC_TIMESTAMP)
        column_name = plant_column[0]

        return pl.DataFrame(power).rename({'power': column_name})

    def __hp_model(self, current_ts, length_to_predict):

        raise NotImplementedError('HP model cannot forecast using weather yet.')
        # This code was pasted from PV as inspiration.
        # get pv orientation
        plant = self.train_data['plant_config']
        surface_tilt = plant['sizing']['orientation']
        surface_azimuth = plant['sizing']['angle']

        # get hardware data
        config = self.train_data['specs']
        module = pd.Series(config['module'])
        inverter = pd.Series(config['inverter'])

        # set temperature model
        temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

        # generate pv system
        system = PVSystem(
            surface_tilt=surface_tilt,
            surface_azimuth=surface_azimuth,
            module_parameters=module,
            inverter_parameters=inverter,
            temperature_model_parameters=temperature_model_parameters
        )

        # get location data
        location = self.train_data['general_config']['location']
        latitude = location['latitude']
        longitude = location['longitude']
        name = location[c.TC_NAME]
        altitude = location['altitude']

        # get weather data from csv
        weather = f.slice_dataframe_between_times(target_df=self.train_data[c.K_FEATURES], reference_ts=current_ts,
                                                  duration=0)
        weather = weather.with_columns(pl.col(c.TC_TIMESTEP).alias(c.TC_TIMESTAMP))
        weather = f.slice_dataframe_between_times(target_df=weather, reference_ts=current_ts,
                                                  duration=length_to_predict)
        weather = weather.to_pandas()

        # get real dhi data
        filter_condition = (pl.col(c.TC_TIMESTAMP) == pl.col(c.TC_TIMESTEP))  # take only actual past weather data
        weather_real = self.train_data[c.K_FEATURES].filter(filter_condition)
        dhi_real = f.slice_dataframe_between_times(target_df=weather_real, reference_ts=current_ts,
                                                   duration=length_to_predict).select(c.TC_DHI).to_pandas()

        # replace dhi with real dhi
        weather[c.TC_DHI] = dhi_real[c.TC_DHI]

        # convert time data to datetime (use utc time overall in pvlib)
        time = pd.DatetimeIndex(pd.to_datetime(weather[c.TC_TIMESTAMP], unit='s', utc=True))
        weather.index = time
        weather.index.name = 'utc_time'

        # adjust temperature data
        weather.rename(columns={c.TC_TEMPERATURE: 'temp_air'}, inplace=True)  # rename to pvlib format
        weather['temp_air'] -= 273.15  # convert unit to celsius

        # get solar position
        # test data find in https://www.suncalc.org/#/48.1364,11.5786,15/2022.02.15/16:21/1/3
        solpos = pvlib.solarposition.get_solarposition(
            time=time,
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            temperature=weather['temp_air'],
            pressure=weather[c.TC_PRESSURE],
        )

        # calculate dni with solar position
        weather.loc[:, c.TC_DNI] = (weather[c.TC_GHI] - weather[c.TC_DHI]) / np.cos(solpos['zenith'])

        # get location data and create corresponding pvlib Location object
        location = Location(
            latitude,
            longitude,
            name=name,
            altitude=altitude
        )

        # create calculation model for the given pv system and location
        mc = pvlib.modelchain.ModelChain(system, location)

        # calculate model under given weather data and get output ac power from it
        mc.run_model(weather)
        power = mc.results.ac

        # calculate nominal power
        nominal_power = module['Impo'] * module['Vmpo']

        # set time index to origin timestamp
        power.index = weather[c.TC_TIMESTAMP]
        power.index.name = c.TC_TIMESTAMP

        # rename and round data column
        power.rename(c.ET_ELECTRICITY, inplace=True)
        power = power.to_frame()

        # calculate and round power
        power = power / nominal_power * plant['sizing']['power']
        power = power.round().astype(int)

        # replace all negative values
        power[power < 0] = 0

        # get column name
        plant_column = self.train_data[c.K_TARGET].columns
        plant_column.remove(c.TC_TIMESTAMP)
        column_name = plant_column[0]

        return pl.DataFrame(power).rename({'power': column_name})

    def predict(self, current_ts, length_to_predict, **kwargs):
        # check which model is to predicted
        if self.train_data['specs']['type'] == c.P_PV:
            forecast = self.__pv_model(current_ts, length_to_predict)
        elif self.train_data['specs']['type'] == c.P_WIND:
            forecast = self.__wind_model(current_ts, length_to_predict)
        elif self.train_data['specs']['type'] == c.P_HP:
            forecast = self.__hp_model(current_ts, length_to_predict)
        else:
            raise KeyError(f'The plant type provided in the specs file is not supported: '
                           f'{self.train_data["specs"]["type"]}')

        return forecast


@forecast_model(name='arrival')
class ArrivalModel(ModelBase):
    """Arrival model specific for EV."""
    def __init__(self, train_data, **kwargs):
        super().__init__(train_data, **kwargs)
        # get ev plant id
        ev_id = train_data[c.K_TARGET].columns
        ev_id.remove(c.TC_TIMESTAMP)
        self.ev_id = ev_id[0].split('_')[0]

    def predict(self, current_ts, length_to_predict, **kwargs):
        """
        Predict availability and energy consumption for the EV.

        The forecast result from this model depends on the EV availability at the current timestep. If EV is available
        (id_availability == 1) at current_ts, the model will return a perfect forecast for both availability and
        energy_consumed. If EV is not available at current_ts (id_availability == 0), the prediction for both
        availability and energy_consumed will be all 0.

        Parameters:
            current_ts (datetime): The current timestep for prediction.
            length_to_predict (int): The duration (in seconds) for which to make predictions.

        Returns:
            forecast (polars.DataFrame): A DataFrame containing the forecasted availability and energy consumption
            values for the EV plant.

        """
        # get availability at current timestep
        current_availability = f.calculate_timedelta(target_df=self.train_data[c.K_TARGET], reference_ts=current_ts)\
                                .filter(pl.col('timedelta') == 0).select(self.ev_id + '_availability')\
                                .item()

        # first, get perfect forecast
        forecast = f.slice_dataframe_between_times(target_df=self.train_data[c.K_TARGET], reference_ts=current_ts,
                                                   duration=length_to_predict, unit='second')
        if current_availability == 0:   # if current availability is 0, replace forecast with 0s
            forecast = forecast.with_columns(pl.lit(0).alias(self.ev_id + '_availability'))
            forecast = forecast.with_columns(pl.lit(0).alias(self.ev_id + '_energy_consumed'))

        return forecast


# TODO: write more detailed documentation about user-defined models!
"""
user can also define own model-object (class) here. some basic rules:
1. The model object (class) must use the @forecast_model decorator. The c.TC_NAME argument should be the same string as 
defined in config file to identify models.
2. The model object (class) must inherits from class ModelBase, which contains the train data to be forecasted as 
attribute. The train data is a dictionary consists of two keys: c.K_TARGET and c.K_FEATURES. The values are both polars
Dataframes.
3. The predict() method of the object must be implemented, fit() is optional. Two args will be passed to both methods 
for all models:
    Args:
        current_ts: Current timestep when making the fitting / prediction.
        length_to_predict: How long in the future should be covered in the resulting forecast of this model. Unit:
        seconds.
4. If extra args needed, define in config file with exact same names. 
5. All functions need to contain **kwargs to make overall structure working.
6. The predict() method needs to return a polars Dataframe with the forecasting result in the same column name as the
target column name.
"""
