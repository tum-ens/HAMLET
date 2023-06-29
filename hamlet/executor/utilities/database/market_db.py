import polars as pl
from hamlet import functions as f

class MarketDB:
    """Database contains all the information for markets.
    Should only be connected with Database class, no connection with main Executor."""

    def __init__(self, path, type):
        self.market_path = path
        self.market_type = type
        self.market_transactions = pl.LazyFrame()
        self.bids_cleared = pl.LazyFrame()
        self.bids_uncleared = pl.LazyFrame()
        self.offers_cleared = pl.LazyFrame()
        self.offers_uncleared = pl.LazyFrame()
        self.positions_matched = pl.LazyFrame()
        self.retailers = pl.LazyFrame()

    def register_market(self):
        """Assign class attribute from data in market folder."""
        ...