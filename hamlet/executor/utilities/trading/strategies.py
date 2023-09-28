__author__ = "MarkusDoepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

# This file is in charge of the trading strategy for the markets

# Imports
import os
import pandas as pd
import polars as pl
import numpy as np
import time
import logging
import traceback
from datetime import datetime
from hamlet.executor.utilities.forecasts.forecaster import Forecaster
from hamlet.executor.utilities.controller.controller import Controller
from hamlet.executor.utilities.database.database import Database
from hamlet import constants as c
from pprint import pprint
import random

# TODO: The trading needs to know much more about the market design. This just works for the continuous market for now


class TradingBase:

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.timetable = kwargs['timetable']
        self.market = kwargs[c.TC_MARKET]
        self.market_data = kwargs['market_data']
        self.agent = kwargs['agent']
        self.bids_offers = None

        # Get agent id and trading horizon
        self.agent_id = self.agent.account[c.K_GENERAL]['agent_id']
        self.trading_horizon = pd.Timedelta(seconds=self.agent.account[c.K_EMS][c.K_MARKET]['horizon'])

        # Get the market data for the given market and agent
        # TODO: Include once it is possible to get the market data. It should be subtracted from the setpoints
        # self.market_data = self.market_data.filter((pl.col(c.TC_MARKET) == self.market)
        #                                            & (pl.col(c.TC_ID_AGENT) == self.agent)
        #                                            & (pl.col(c.TC_TIMESTEP) >= self.timetable[c.TC_TIMESTEP].first())
        #                                            & (pl.col(c.TC_TIMESTEP) <= self.timetable[c.TC_TIMESTEP].last()))

        # Create some dummy values for now
        # TODO: Delete once market data is available
        self.market_data = self.timetable.clone()
        self.market_data = self.market_data.with_columns(
            [
                pl.Series([self.agent_id] * len(self.market_data.collect())).alias(c.TC_ID_AGENT),
                pl.Series(random.sample(range(-100, 100), len(self.market_data.collect()))).alias(c.TC_ENERGY),
            ]
        )
        self.market_data = self.market_data.with_columns(
            [
                pl.col(c.TC_ENERGY).apply(lambda x: abs(x) if x > 0 else 0).alias(c.TC_ENERGY_IN),
                pl.col(c.TC_ENERGY).apply(lambda x: abs(x) if x < 0 else 0).alias(c.TC_ENERGY_OUT),
            ]
        )

        # Group the market data by the timestep
        self.market_data = self.market_data.group_by(c.TC_TIMESTEP).agg(pl.col(c.TC_ENERGY_IN, c.TC_ENERGY_OUT).sum())
        self.market_data = self.market_data.with_columns((pl.col(c.TC_ENERGY_IN) - pl.col(c.TC_ENERGY_OUT)).alias(c.TC_ENERGY))

        # Get the energy type for the given market type
        # self.energy_type = self.market_data.select(c.TC_ENERGY_TYPE).first()[0] TODO: Put back in once market data is available
        self.energy_type = c.ET_ELECTRICITY

        # Get the setpoints for the given market and agent
        self.setpoints = self.agent.setpoints.select([c.TC_TIMESTAMP, f'{self.market}_{self.energy_type}'])
        self.setpoints = self.setpoints.rename({c.TC_TIMESTAMP: c.TC_TIMESTEP})

        # Get the forecast of the prices
        # TODO: Include once available
        #self.forecast = self.agent.forecasts
        # Dummy values for now
        # TODO: Delete once forecast is available
        self.forecast = self.timetable.clone()
        self.forecast = self.forecast.with_columns(
            [
                pl.Series([0.14] * len(self.market_data.collect())).alias('energy_sell'),
                pl.Series([0.06] * len(self.market_data.collect())).alias('energy_buy'),
            ]
        )
        # Drop unnecessary columns
        self.forecast = self.forecast.drop([c.TC_TIMESTAMP, c.TC_REGION, c.TC_MARKET, c.TC_NAME, c.TC_ENERGY_TYPE, c.TC_ACTIONS, c.TC_CLEARING_TYPE, c.TC_CLEARING_METHOD, c.TC_CLEARING_PRICING, c.TC_COUPLING])

    def create_bids_offers(self):
        raise NotImplementedError('This method has yet to be implemented.')

    def _preprocess_bids_offers(self):

        # Combine setpoints and market data to know how much energy is needed
        self.market_data = self.market_data.join(self.setpoints, on=c.TC_TIMESTEP, how='left')
        self.market_data = self.market_data.with_columns(
            (pl.col(c.TC_ENERGY) - pl.col(f'{self.market}_{self.energy_type}')).alias('buy_sell'))
        self.market_data = self.market_data.with_columns(
            [
                pl.col('buy_sell').apply(lambda x: abs(x) if x > 0 else 0).alias(c.TC_ENERGY_IN),
                pl.col('buy_sell').apply(lambda x: abs(x) if x < 0 else 0).alias(c.TC_ENERGY_OUT),
            ]
        )

        # Drop unnecessary columns
        self.market_data = self.market_data.drop([c.TC_ENERGY, 'buy_sell', f'{self.market}_{self.energy_type}'])

        # Create the dataframe for the bids and offers
        # TODO: In the future this will be based on the market data and the setpoints
        self.bids_offers = self.timetable.select([c.TC_TIMESTAMP, c.TC_TIMESTEP, c.TC_REGION, c.TC_MARKET, c.TC_NAME,
                                                  c.TC_ENERGY_TYPE])

        # Compute length of table
        len_table = len(self.bids_offers.collect())

        self.bids_offers = self.bids_offers.with_columns(
            [
                pl.Series([self.agent_id] * len_table).alias(c.TC_ID_AGENT),
            ]
        )

        # Add energy in and out
        self.bids_offers = self.bids_offers.join(self.market_data, on=c.TC_TIMESTEP, how='left')

        # Add forecast values
        self.bids_offers = self.bids_offers.join(self.forecast, on=c.TC_TIMESTEP, how='left')

        # Check if it is within the horizon and how much time is left to trade
        self.bids_offers = self.bids_offers.with_columns(
            [
                (pl.col(c.TC_TIMESTEP) - pl.col(c.TC_TIMESTAMP))
                .apply(lambda x: True if x < self.trading_horizon else False).alias('within_horizon'),
                (pl.col(c.TC_TIMESTEP) - pl.col(c.TC_TIMESTAMP)).alias('time_to_trade'),
            ]
        )

        # Drop row where time_to_trade is 0 or negative
        self.bids_offers = self.bids_offers.filter(pl.col('time_to_trade') > 0)

        return self.bids_offers

    def _postprocess_bids_offers(self):

        # Filter out rows that are not within the trading horizon
        self.bids_offers = self.bids_offers.filter(pl.col('within_horizon') == True)
        # Filter out rows where energy_in and energy_out are zero
        self.bids_offers = self.bids_offers.filter((pl.col(c.TC_ENERGY_IN) != 0) | (pl.col(c.TC_ENERGY_OUT) != 0))

        # Drop unnecessary columns
        self.bids_offers = self.bids_offers.drop(['within_horizon', 'time_to_trade', 'energy_sell', 'energy_buy'])

        return self.bids_offers


class Linear(TradingBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_bids_offers(self):

        # Preprocess the bids and offers table
        self.bids_offers = self._preprocess_bids_offers()

        # Get the length of the table
        len_table = len(self.bids_offers.collect())

        # Add column that contains the number of the row
        self.bids_offers = self.bids_offers.with_columns(
            [
                pl.Series(range(0, len_table)).alias('row_number'),
            ]
        )

        # Add columns for the price per unit
        self.bids_offers = self.bids_offers.with_columns(
            [
                (((pl.col('energy_sell') - pl.col('energy_buy')) / len_table * (len_table - pl.col('row_number')))
                 + pl.col('energy_buy'))
                .alias(c.TC_PRICE_PU_IN),
                (pl.col('energy_buy')
                 + ((pl.col('energy_sell') - pl.col('energy_buy')) / len_table * pl.col('row_number')))
                .alias(c.TC_PRICE_PU_OUT),
            ]
        )

        # Remove the row number column
        self.bids_offers = self.bids_offers.drop('row_number')

        # Postprocess the bids and offers table
        self.bids_offers = self._postprocess_bids_offers()

        # Print output
        # with pl.Config() as cfg:
        #     cfg.set_tbl_cols(20)
        #     cfg.set_tbl_width_chars(200)
        #     print(self.bids_offers.collect())

        # Update agent information
        self.agent.bids_offers = self.bids_offers

        return self.agent


class Zi(TradingBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_bids_offers(self):

        # Preprocess the bids and offers table
        self.bids_offers = self._preprocess_bids_offers()

        # Get the length of the table
        len_table = len(self.bids_offers.collect())

        # Add columns for the price per unit
        self.bids_offers = self.bids_offers.with_columns(
            [
                (((pl.col('energy_sell') - pl.col('energy_buy'))
                  * pl.Series([random.random() for _ in range(len_table)]))
                 + pl.col('energy_buy'))
                .alias(c.TC_PRICE_PU_IN),
                (((pl.col('energy_sell') - pl.col('energy_buy'))
                  * pl.Series([random.random() for _ in range(len_table)]))
                 + pl.col('energy_buy'))
                .alias(c.TC_PRICE_PU_OUT),
            ]
        )

        # Postprocess the bids and offers table
        self.bids_offers = self._postprocess_bids_offers()

        # Print output
        # with pl.Config() as cfg:
        #     cfg.set_tbl_cols(20)
        #     cfg.set_tbl_width_chars(200)
        #     print(self.bids_offers.collect())

        # Update agent information
        self.agent.bids_offers = self.bids_offers

        return self.agent