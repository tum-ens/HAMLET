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
        self.market_transactions = pl.LazyFrame({c.TC_TIMESTAMP: [],
                                                 c.TC_TIMESTEP: [],
                                                 c.TC_REGION: [],
                                                 c.TC_MARKET: [],
                                                 c.TC_NAME: [],
                                                 c.TC_TYPE_TRANSACTION: [],
                                                 c.TC_ID_AGENT: [],
                                                 c.TC_ENERGY_IN: [],
                                                 c.TC_ENERGY_OUT: [],
                                                 c.TC_PRICE_PU_IN: [],
                                                 c.TC_PRICE_PU_OUT: [],
                                                 c.TC_PRICE_IN: [],
                                                 c.TC_PRICE_OUT: [],
                                                 c.TC_SHARE_QUALITY: []})
        self.bids_cleared = pl.LazyFrame({c.TC_TIMESTAMP: [],
                                          c.TC_TIMESTEP: [],
                                          c.TC_REGION: [],
                                          c.TC_MARKET: [],
                                          c.TC_NAME: [],
                                          c.TC_TYPE_TRANSACTION: [],
                                          c.TC_ID_AGENT: [],
                                          c.TC_ENERGY_IN: [],
                                          c.TC_PRICE_PU_IN: [],
                                          c.TC_PRICE_IN: []})
        self.bids_uncleared = pl.LazyFrame({c.TC_TIMESTAMP: [],
                                            c.TC_TIMESTEP: [],
                                            c.TC_REGION: [],
                                            c.TC_MARKET: [],
                                            c.TC_NAME: [],
                                            c.TC_TYPE_TRANSACTION: [],
                                            c.TC_ID_AGENT: [],
                                            c.TC_ENERGY_IN: [],
                                            c.TC_PRICE_PU_IN: [],
                                            c.TC_PRICE_IN: []})
        self.offers_cleared = pl.LazyFrame({c.TC_TIMESTAMP: [],
                                            c.TC_TIMESTEP: [],
                                            c.TC_REGION: [],
                                            c.TC_MARKET: [],
                                            c.TC_NAME: [],
                                            c.TC_TYPE_TRANSACTION: [],
                                            c.TC_ID_AGENT: [],
                                            c.TC_ENERGY_OUT: [],
                                            c.TC_PRICE_PU_OUT: [],
                                            c.TC_PRICE_OUT: [],
                                            c.TC_QUALITY: []})
        self.offers_uncleared = pl.LazyFrame({c.TC_TIMESTAMP: [],
                                              c.TC_TIMESTEP: [],
                                              c.TC_REGION: [],
                                              c.TC_MARKET: [],
                                              c.TC_NAME: [],
                                              c.TC_TYPE_TRANSACTION: [],
                                              c.TC_ID_AGENT: [],
                                              c.TC_ENERGY_OUT: [],
                                              c.TC_PRICE_PU_OUT: [],
                                              c.TC_PRICE_OUT: [],
                                              c.TC_QUALITY: []})
        self.positions_matched = pl.LazyFrame({c.TC_TIMESTAMP: [],
                                               c.TC_TIMESTEP: [],
                                               c.TC_REGION: [],
                                               c.TC_MARKET: [],
                                               c.TC_NAME: [],
                                               c.TC_TYPE_TRANSACTION: [],
                                               c.TC_ID_AGENT_IN: [],
                                               c.TC_ID_AGENT_OUT: [],
                                               c.TC_ENERGY: [],
                                               c.TC_PRICE_PU: [],
                                               c.TC_PRICE: [],
                                               c.TC_QUALITY: []})
