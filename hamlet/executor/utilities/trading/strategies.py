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

        # Get the kwarguments
        self.kwargs = kwargs

        # Get the timetable and timestep
        self.timetable = kwargs['timetable']
        self.dt = self.timetable[1, c.TC_TIMESTEP] - self.timetable[0, c.TC_TIMESTEP]  # in datetime format
        self.dt_hours = self.dt / pd.Timedelta(hours=1)  # in hours

        # Get the energy type
        self.energy_type = self.timetable.row(0, named=True)[c.TC_ENERGY_TYPE]

        # Get the market data, name, type and transactions
        self.market_data = kwargs['market_data']
        self.market_name = self.market_data.market_name
        self.market_type = self.market_data.market_type
        self.market_transactions = self.market_data.market_transactions

        # Get the agent data, id
        self.agent = kwargs['agent']
        self.agent_id = self.agent.account[c.K_GENERAL]['agent_id']

        # Get the trading horizon and the strategy parameters
        self.trading_horizon = pd.Timedelta(seconds=self.agent.account[c.K_EMS][c.K_MARKET]['horizon'])
        self.strategy = self.agent.account[c.K_EMS][c.K_MARKET]['strategy']
        try:
            self.strategy_params = self.agent.account[c.K_EMS][c.K_MARKET][self.strategy]
        except KeyError:
            self.strategy_params = None

        # Create the bids and offers table
        self.bids_offers = pl.DataFrame(schema=c.TS_BIDS_OFFERS)

        # Get the market transactions for the given market and agent for the given time horizon
        self.market_transactions = self.market_transactions.filter((pl.col(c.TC_MARKET) == self.market_type)
                                                                   & (pl.col(c.TC_NAME) == self.market_name)
                                                                   & (pl.col(c.TC_TYPE_TRANSACTION) == c.TT_MARKET)
                                                                   & (pl.col(c.TC_ID_AGENT) == self.agent_id)
                                                                   & (pl.col(c.TC_TIMESTEP) >= self.timetable.select(pl.first(c.TC_TIMESTEP)))
                                                                   & (pl.col(c.TC_TIMESTEP) <= self.timetable.select(pl.last(c.TC_TIMESTEP)))
                                                                   )
        # Group the market data by the timestep      
        self.market_transactions = self.market_transactions.groupby(c.TC_TIMESTEP).agg(pl.col(c.TC_ENERGY_IN, c.TC_ENERGY_OUT).sum())
        # Fill all empty columns with zero
        self.market_transactions = self.market_transactions.fill_null(0)
        # Compute the net energy
        self.market_transactions = self.market_transactions.with_columns((pl.col(c.TC_ENERGY_IN).cast(pl.Int64) - pl.col(c.TC_ENERGY_OUT).cast(pl.Int64)).alias(c.TC_ENERGY))

        # Get the setpoints for the given market and agent and convert them to energy (unit: Wh)
        self.setpoints = self.agent.setpoints.select([c.TC_TIMESTAMP, f'{self.market_name}_{self.energy_type}'])
        self.setpoints = self.setpoints.rename({c.TC_TIMESTAMP: c.TC_TIMESTEP})
        self.setpoints = self.setpoints.with_columns((pl.col(f'{self.market_name}_{self.energy_type}') * self.dt_hours).cast(pl.Int32))

        # Get the forecast of the prices
        self.forecast = self.agent.forecasts
        # Reduce the table to only include the relevant columns (energy price sell and buy)
        relevant_cols = [c.TC_TIMESTEP, 'energy_price_sell', 'energy_price_buy']
        self.forecast = self.forecast.select(relevant_cols)

    def create_bids_offers(self):
        raise NotImplementedError('This method has yet to be implemented.')

    def _preprocess_bids_offers(self):
        # Combine setpoints and market data to know how much energy is needed
        self.market_transactions = self.market_transactions.join(self.setpoints, on=c.TC_TIMESTEP, how='outer')

        # Fill all empty columns with zero
        self.market_transactions = self.market_transactions.fill_null(0)

        # Check if there is a difference between the setpoints and the previous market results
        self.market_transactions = self.market_transactions.with_columns(
            (pl.col(f'{self.market_name}_{self.energy_type}') - pl.col(c.TC_ENERGY)).alias('buy_sell').cast(pl.Int32))

        # Split the buy and sell values
        self.market_transactions = self.market_transactions.with_columns(
            [
                pl.col('buy_sell').apply(lambda x: abs(x) if x > 0 else 0).alias(c.TC_ENERGY_IN).cast(pl.UInt64),
                pl.col('buy_sell').apply(lambda x: abs(x) if x < 0 else 0).alias(c.TC_ENERGY_OUT).cast(pl.UInt64),
            ]
        )

        # Drop unnecessary columns
        self.market_transactions = self.market_transactions.drop([c.TC_ENERGY, 'buy_sell', f'{self.market_name}_{self.energy_type}'])

        # Create the dataframe for the bids and offers
        self.bids_offers = self.timetable.select([c.TC_TIMESTAMP, c.TC_TIMESTEP, c.TC_REGION, c.TC_MARKET, c.TC_NAME,
                                                  c.TC_ENERGY_TYPE])

        # Compute length of table
        len_table = len(self.bids_offers)

        self.bids_offers = self.bids_offers.with_columns(
            [
                pl.Series([self.agent_id] * len_table).alias(c.TC_ID_AGENT).cast(pl.Categorical),
            ]
        )

        # Add energy in and out
        self.bids_offers = self.bids_offers.join(self.market_transactions, on=c.TC_TIMESTEP, how='left')

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

        # Drop row where time_to_trade is smaller than 0
        self.bids_offers = self.bids_offers.filter(pl.col('time_to_trade') >= 0)

        return self.bids_offers

    def _postprocess_bids_offers(self):

        # Filter out rows that are not within the trading horizon
        self.bids_offers = self.bids_offers.filter(pl.col('within_horizon') == True)
        # Filter out rows where energy_in and energy_out are zero
        self.bids_offers = self.bids_offers.filter((pl.col(c.TC_ENERGY_IN) != 0) | (pl.col(c.TC_ENERGY_OUT) != 0))

        # Drop unnecessary columns
        self.bids_offers = self.bids_offers.drop(['within_horizon', 'time_to_trade',
                                                  'energy_price_sell', 'energy_price_buy'])

        return self.bids_offers


class Linear(TradingBase):
    """This class implements the linear trading strategy.
    It means that the agent's offers/bids will linearly increase/'."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_bids_offers(self):

        # Names for the internal columns (will be removed at the end of the function)
        c_factor = 'multiplication_factor'

        # Preprocess the bids and offers table
        self.bids_offers = self._preprocess_bids_offers()

        # with pl.Config(set_tbl_cols=20, set_tbl_rows=100, set_tbl_width_chars=400):
        #     print(self.bids_offers)

        # Filter out rows that are not within the trading horizon
        self.bids_offers = self.bids_offers.filter(pl.col('within_horizon') == True)

        # Get the length of the table
        len_table = len(self.bids_offers)

        # Add column that contains multiplication factor for the price per unit
        # Note: This value depends on how many iterations before the maximum price is reached and the inclination of
        #  the price increase/decrease
        steps_final = self.strategy_params['steps_to_final']
        steps_initial = self.strategy_params['steps_from_init']
        mul_factor = np.linspace(0, len_table, len_table - steps_final - steps_initial)
        mul_factor = np.concatenate(([0] * steps_final, mul_factor, [len_table] * steps_initial))
        self.bids_offers = self.bids_offers.with_columns(pl.Series(mul_factor).alias(c_factor))

        # Add columns for the price per unit
        self.bids_offers = self.bids_offers.with_columns(
            [
                (pl.col('energy_price_buy')
                 + ((pl.col('energy_price_sell') - pl.col('energy_price_buy')) / len_table * pl.col(c_factor)))
                .alias(c.TC_PRICE_PU_IN).round().cast(pl.Int32),
                (((pl.col('energy_price_sell') - pl.col('energy_price_buy')) / len_table
                  * (len_table - pl.col(c_factor))) + pl.col('energy_price_buy'))
                .alias(c.TC_PRICE_PU_OUT).round().cast(pl.Int32),
            ]
        )

        # Remove the row number column
        self.bids_offers = self.bids_offers.drop(c_factor)

        # Postprocess the bids and offers table
        self.bids_offers = self._postprocess_bids_offers()

        # Update agent information
        self.agent.bids_offers = self.bids_offers

        return self.agent


class Zi(TradingBase):
    """This class implements the zero intelligence trading strategy.
    It means that the agent will choose a random value."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_bids_offers(self):

        # Preprocess the bids and offers table
        self.bids_offers = self._preprocess_bids_offers()

        # Get the length of the table
        len_table = len(self.bids_offers)

        # Add columns for the price per unit
        self.bids_offers = self.bids_offers.with_columns(
            [
                (((pl.col('energy_price_sell') - pl.col('energy_price_buy'))
                  * pl.Series([random.random() for _ in range(len_table)]))
                 + pl.col('energy_price_buy'))
                .alias(c.TC_PRICE_PU_IN),
                (((pl.col('energy_price_sell') - pl.col('energy_price_buy'))
                  * pl.Series([random.random() for _ in range(len_table)]))
                 + pl.col('energy_price_buy'))
                .alias(c.TC_PRICE_PU_OUT),
            ]
        )

        # Postprocess the bids and offers table
        self.bids_offers = self._postprocess_bids_offers()

        # Update agent information
        self.agent.bids_offers = self.bids_offers

        return self.agent