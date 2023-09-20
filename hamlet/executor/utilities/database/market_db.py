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
                                                 'timestep': [],
                                                 'region': [],
                                                 'market': [],
                                                 'name': [],
                                                 'type_transaction': [],
                                                 'id_agent': [],
                                                 'energy_in': [],
                                                 'energy_out': [],
                                                 'price_pu_in': [],
                                                 'price_pu_out': [],
                                                 'price_in': [],
                                                 'price_out': [],
                                                 'share_quality_XXX': []})
        self.bids_cleared = pl.LazyFrame({c.TC_TIMESTAMP: [],
                                          'timestep': [],
                                          'region': [],
                                          'market': [],
                                          'name': [],
                                          'type_transaction': [],
                                          'id_agent': [],
                                          'energy_in': [],
                                          'price_pu_in': [],
                                          'price_in': []})
        self.bids_uncleared = pl.LazyFrame({c.TC_TIMESTAMP: [],
                                            'timestep': [],
                                            'region': [],
                                            'market': [],
                                            'name': [],
                                            'type_transaction': [],
                                            'id_agent': [],
                                            'energy_in': [],
                                            'price_pu_in': [],
                                            'price_in': []})
        self.offers_cleared = pl.LazyFrame({c.TC_TIMESTAMP: [],
                                            'timestep': [],
                                            'region': [],
                                            'market': [],
                                            'name': [],
                                            'type_transaction': [],
                                            'id_agent': [],
                                            'energy_out': [],
                                            'price_pu_out': [],
                                            'price_out': [],
                                            'quality': []})
        self.offers_uncleared = pl.LazyFrame({c.TC_TIMESTAMP: [],
                                              'timestep': [],
                                              'region': [],
                                              'market': [],
                                              'name': [],
                                              'type_transaction': [],
                                              'id_agent': [],
                                              'energy_out': [],
                                              'price_pu_out': [],
                                              'price_out': [],
                                              'quality': []})
        self.positions_matched = pl.LazyFrame({c.TC_TIMESTAMP: [],
                                               'timestep': [],
                                               'region': [],
                                               'market': [],
                                               'name': [],
                                               'type_transaction': [],
                                               'id_agent_in': [],
                                               'id_agent_out': [],
                                               'energy': [],
                                               'price_pu': [],
                                               'price': [],
                                               'quality': []})
