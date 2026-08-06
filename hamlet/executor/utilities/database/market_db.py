__author__ = "jiahechu"
__credits__ = ""
__license__ = ""
__maintainer__ = "jiahechu"
__email__ = "jiahe.chu@tum.de"

import datetime
import os
import shutil

import polars as pl

from hamlet import constants as c
from hamlet import functions as f


class MarketDB:
    """Database contains all the information for markets.
    Should only be connected with Database class, no connection with main Executor."""

    def __init__(self, market_type, name, market_path, retailer_path):
        self.market_type = market_type
        self.market_name = name
        self.market_path = market_path
        self.market_config = f.load_file(path=os.path.join(market_path, 'config.json'))
        self.market_save = None  # path to save the market
        self.retailer_path = retailer_path
        self.market_transactions = pl.DataFrame()
        self.bids_cleared = pl.DataFrame()
        self.bids_uncleared = pl.DataFrame()
        self.offers_cleared = pl.DataFrame()
        self.offers_uncleared = pl.DataFrame()
        self.positions_matched = pl.DataFrame()
        self.retailer = pl.DataFrame()

        # Tuples of (file name, file schema)
        self.files = [(f'{c.TN_MARKET_TRANSACTIONS}.ft', c.TS_MARKET_TRANSACTIONS),
                      (f'{c.TN_BIDS_CLEARED}.ft', c.TS_BIDS_CLEARED),
                      (f'{c.TN_BIDS_UNCLEARED}.ft', c.TS_BIDS_UNCLEARED),
                      (f'{c.TN_OFFERS_CLEARED}.ft', c.TS_OFFERS_CLEARED),
                      (f'{c.TN_OFFERS_UNCLEARED}.ft', c.TS_OFFERS_UNCLEARED)]

    def register_market(self):
        """Assign class attribute from data in market folder."""
        self.retailer = f.load_file(path=os.path.join(self.retailer_path, 'retailer.ft'), df='polars', method='eager')
        self.market_transactions = pl.DataFrame(schema=c.TS_MARKET_TRANSACTIONS)
        self.bids_cleared = pl.DataFrame(schema=c.TS_BIDS_CLEARED)
        self.bids_uncleared = pl.DataFrame(schema=c.TS_BIDS_UNCLEARED)
        self.offers_cleared = pl.DataFrame(schema=c.TS_OFFERS_CLEARED)
        self.offers_uncleared = pl.DataFrame(schema=c.TS_OFFERS_UNCLEARED)
        self.positions_matched = pl.DataFrame(schema=c.TS_POSITIONS_MATCHED)

    def load_market_from_files(self, market_transactions_only=True):
        """Load market market data from files"""
        self.retailer = f.load_file(path=os.path.join(self.retailer_path, f'retailer.ft'), df='polars', method='eager')
        for file_name, schema in self.files:
            # Skip other files except market transactions
            if market_transactions_only and not file_name.startswith(c.TN_MARKET_TRANSACTIONS):
                continue
            if os.path.exists(os.path.join(self.market_path, file_name)):
                # load file
                df: pl.DataFrame = f.load_file(path=os.path.join(self.market_path, file_name), df='polars',
                                               method='eager', parse_dates=True)
                # cast dataframe to the given schema
                df = df.cast(schema)
                # get database name
                attr_name = file_name.rsplit('.', 1)[0]
                # update class attribute with dataframe
                setattr(self, attr_name, df)

    def save_market(self, path, save_all: bool = False):
        """Save market data to given path."""

        # Update market path
        self.market_save = os.path.abspath(path)

        f.save_file(path=os.path.join(path, 'market_transactions.ft'), data=self.market_transactions, df='polars')
        # TODO: put back in when the data is available. If there is no use for the table, remove it and create it
        #  in the analyzer
        # f.save_file(path=os.path.join(path, 'positions_matched.csv'), data=self.positions_matched
        #             , df='polars')
        f.save_file(path=os.path.join(path, 'retailer.ft'), data=self.retailer, df='polars')

        # Data is not saved if save_all is False
        if save_all:
            f.save_file(path=os.path.join(path, 'bids_cleared.ft'), data=self.bids_cleared, df='polars')
            f.save_file(path=os.path.join(path, 'bids_uncleared.ft'), data=self.bids_uncleared, df='polars')
            f.save_file(path=os.path.join(path, 'offers_cleared.ft'), data=self.offers_cleared, df='polars')
            f.save_file(path=os.path.join(path, 'offers_uncleared.ft'), data=self.offers_uncleared, df='polars')

    def save_and_drop_past_records(self, timestamp, path_results):
        """Save market data out of horizon to files and drop the past records.

        Runs on every table on every timestep, over tables that grow through the run, so the
        cheap paths matter more than the clever ones:

        1. Most calls have nothing to drop at all, because the clearing horizon is usually
           longer than the interval at which this runs. One pass for the minimum timestep
           settles that, and costs no allocation.
        2. When there is something to drop, the membership test is evaluated once and reused
           for both sides of the split, rather than scanning the frame twice with complementary
           filters.

        These tables are deliberately *not* assumed to be sorted by timestep. They are appended
        in clearing order, and each clearing writes rows for every delivery timestep in its
        horizon, so successive blocks overlap. Measured on the shipped example, only the very
        first call sees a sorted table and the remaining 22 do not, which is why there is no
        binary-search path here: it would cost a sortedness scan to almost never apply.

        The output folder is only created when there is something to write; it used to be
        created for every table on every timestep, each call sleeping 10 ms.
        """
        horizon_range = self.market_config['clearing']['timing']['horizon'][1]
        start_horizon_ts = timestamp - datetime.timedelta(seconds=horizon_range)

        for file_name, schema in self.files:
            attr_name, extension = file_name.rsplit('.', 1)
            df = getattr(self, attr_name)

            if df.is_empty():
                continue

            # Skip the whole table if nothing has fallen out of the horizon yet
            oldest = df.get_column(c.TC_TIMESTEP).min()
            if oldest is not None and oldest >= start_horizon_ts:
                continue

            # One membership test, used for both sides of the split
            # Note: a null timestep counts as not-past, so malformed rows are retained rather
            # than silently dropped from both sides, which is what comparing them would do
            is_past = (df.get_column(c.TC_TIMESTEP) < start_horizon_ts).fill_null(False)
            past_data = df.filter(is_past)

            if past_data.is_empty():
                continue

            path = os.path.join(path_results, 'markets', self.market_type, self.market_name,
                                'past_data', attr_name)
            f.create_folder(path, delete=False)
            f.save_file(path=os.path.join(path, f'{attr_name}_{start_horizon_ts.timestamp()}.{extension}'),
                        data=past_data, df='polars')

            setattr(self, attr_name, df.filter(~is_past))

    def concat_past_data(self, delete_dir=True):
        """Concatenates past data saved in files"""
        # Files to be processed. Tuples of (file name, file schema)
        for file_name, schema in self.files:
            # Get attribute name
            attr_name = file_name.rsplit('.', 1)[0]
            # Get attribute path
            path = os.path.join(self.market_save, 'past_data', attr_name)
            # Collect dataframe parts
            df_parts = [getattr(self, attr_name)]  # initialize with current data
            # Add all saved data in files
            # Note: the folder only exists if records were actually dropped for this table, so a
            # missing folder simply means everything is still held in memory
            for file_part in os.listdir(path) if os.path.isdir(path) else []:
                df = f.load_file(path=os.path.join(path, file_part), df='polars', method='eager', parse_dates=True)
                df = df.cast(schema)
                df_parts.append(df)
            # If list is nonempty, concatenate and save dataframe
            if df_parts:
                df = pl.concat(df_parts)
                df = df.sort(by=[c.TC_TIMESTAMP, c.TC_TIMESTEP]).to_pandas()
                f.save_file(path=os.path.join(self.market_save, file_name), data=df, df='pandas')
            # Optionally delete the past data
            if delete_dir and os.path.isdir(path):
                shutil.rmtree(path)

    def set_market_transactions(self, data):
        self.market_transactions = data

    def set_bids_cleared(self, data):
        self.bids_cleared = data

    def set_bids_uncleared(self, data):
        self.bids_uncleared = data

    def set_offers_cleared(self, data):
        self.offers_cleared = data

    def set_offers_uncleared(self, data):
        self.offers_uncleared = data

    def set_positions_matched(self, data):
        self.positions_matched = data
