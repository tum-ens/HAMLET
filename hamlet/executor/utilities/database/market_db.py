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

    def __init__(self, type, name, market_path, retailer_path):
        self.market_type = type
        self.market_name = name
        self.market_path = market_path
        self.market_save = None  # path to save the market
        self.retailer_path = retailer_path
        self.market_transactions = pl.LazyFrame()
        self.bids_cleared = pl.LazyFrame()
        self.bids_uncleared = pl.LazyFrame()
        self.offers_cleared = pl.LazyFrame()
        self.offers_uncleared = pl.LazyFrame()
        self.positions_matched = pl.LazyFrame()
        self.retailer = pl.LazyFrame()

    def register_market(self):
        """Assign class attribute from data in market folder."""
        self.retailer = f.load_file(path=os.path.join(self.retailer_path, 'retailer.ft'), df='polars')
        self.market_transactions = pl.LazyFrame(schema=c.TS_MARKET_TRANSACTIONS)
        self.bids_cleared = pl.LazyFrame(schema=c.TS_BIDS_CLEARED)
        self.bids_uncleared = pl.LazyFrame(schema=c.TS_BIDS_UNCLEARED)
        self.offers_cleared = pl.LazyFrame(schema=c.TS_OFFERS_CLEARED)
        self.offers_uncleared = pl.LazyFrame(schema=c.TS_OFFERS_UNCLEARED)
        self.positions_matched = pl.LazyFrame(schema=c.TS_POSITIONS_MATCHED)

    def save_market(self, path, save_all: bool = False):
        """Save market data to given path."""

        # Update market path
        self.market_save = os.path.abspath(path)

        f.save_file(path=os.path.join(path, 'market_transactions.csv'), data=self.market_transactions.collect()
                    , df='polars')
        f.save_file(path=os.path.join(path, 'positions_matched.csv'), data=self.positions_matched.collect()
                    , df='polars')

        # Data is not saved if save_all is False
        if save_all:
            f.save_file(path=os.path.join(path, 'bids_cleared.ft'), data=self.bids_cleared.collect()
                        , df='polars')
            f.save_file(path=os.path.join(path, 'bids_uncleared.ft'), data=self.bids_uncleared.collect()
                        , df='polars')
            f.save_file(path=os.path.join(path, 'offers_cleared.ft'), data=self.offers_cleared.collect()
                        , df='polars')
            f.save_file(path=os.path.join(path, 'offers_uncleared.ft'), data=self.offers_uncleared.collect()
                        , df='polars')
        f.save_file(path=os.path.join(path, 'retailer.ft'), data=self.retailer.collect()
                    , df='polars')

