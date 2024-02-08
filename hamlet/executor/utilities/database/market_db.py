__author__ = "jiahechu"
__credits__ = ""
__license__ = ""
__maintainer__ = "jiahechu"
__email__ = "jiahe.chu@tum.de"

import os
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
        self.market_save = None  # path to save the market
        self.retailer_path = retailer_path
        self.market_transactions = pl.DataFrame()
        self.bids_cleared = pl.DataFrame()
        self.bids_uncleared = pl.DataFrame()
        self.offers_cleared = pl.DataFrame()
        self.offers_uncleared = pl.DataFrame()
        self.positions_matched = pl.DataFrame()
        self.retailer = pl.DataFrame()

    def register_market(self):
        """Assign class attribute from data in market folder."""
        self.retailer = f.load_file(path=os.path.join(self.retailer_path, 'retailer.ft'), df='polars', method='eager')
        self.market_transactions = pl.DataFrame(schema=c.TS_MARKET_TRANSACTIONS)
        self.bids_cleared = pl.DataFrame(schema=c.TS_BIDS_CLEARED)
        self.bids_uncleared = pl.DataFrame(schema=c.TS_BIDS_UNCLEARED)
        self.offers_cleared = pl.DataFrame(schema=c.TS_OFFERS_CLEARED)
        self.offers_uncleared = pl.DataFrame(schema=c.TS_OFFERS_UNCLEARED)
        self.positions_matched = pl.DataFrame(schema=c.TS_POSITIONS_MATCHED)

    def save_market(self, path, save_all: bool = False):
        """Save market data to given path."""

        # Update market path
        self.market_save = os.path.abspath(path)

        f.save_file(path=os.path.join(path, 'market_transactions.csv'), data=self.market_transactions, df='polars')
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
