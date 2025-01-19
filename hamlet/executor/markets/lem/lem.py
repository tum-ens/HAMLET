__author__ = "MarkusDoepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

# Imports
import time

import polars as pl
from pprint import pprint

import hamlet.constants as c
from hamlet.executor.utilities.database.market_db import MarketDB
from hamlet.executor.utilities.database.database import Database
from hamlet.executor.markets.market_base import MarketBase
import hamlet.functions as f

# Definition of temporary column names (only used in this file)
C_ENERGY_CUMSUM = 'energy_cumsum'


class Lem(MarketBase):

    def __init__(self, market: MarketDB, tasks: dict, database: Database):

        # Call the super class
        super().__init__()

        # Market database
        self.market = market

        # Tasklist
        self.tasks = tasks

        # Database
        self.database = database

        # Get bids and offers
        self.bids_offers = self.database.get_bids_offers(region=self.tasks[c.TC_REGION],
                                                         market_type=self.tasks[c.TC_MARKET],
                                                         market_name=self.tasks[c.TC_NAME],
                                                         timestep=self.tasks[c.TC_TIMESTEP])

        # Get the tables from the market database and clear them
        self.bids_cleared = self.market.bids_cleared.clear()
        self.offers_cleared = self.market.offers_cleared.clear()
        self.bids_uncleared = self.market.bids_uncleared.clear()
        self.offers_uncleared = self.market.offers_uncleared.clear()
        self.transactions = self.market.market_transactions.clear()

        # Get the previous transactions for the given timestep when the action is 'settle'
        if c.MA_SETTLE in self.tasks[c.TC_ACTIONS]:
            self.transactions_prev = (self.market.market_transactions
                                      .filter(pl.col(c.TC_TIMESTEP) == self.tasks[c.TC_TIMESTEP]))
        else:
            self.transactions_prev = None

        # Get the retailer offers
        self.retailer = self.market.retailer.filter(pl.col(c.TC_TIMESTAMP) == self.tasks[c.TC_TIMESTEP])

        # Available actions (see market config)
        self.actions = {
            c.MA_CLEAR: self.__action_clear,
            c.MA_SETTLE: self.__action_settle,
        }

        # Available clearing types (see market config)
        self.types = {
            c.MCT_EX_ANTE: self.__type_ex_ante,
            c.MCT_EX_POST: self.__type_ex_post,
        }

        # Available clearing methods (see market config)
        self.methods = {
            c.MCM_NONE: self.__method_none,  # no local clearing, all energy traded with retailer
            c.MCM_PDA: self.__method_pda,  # periodic double auction
            c.MCM_COMMUNITY: self.__method_community,  # community-based clearing
        }

        # Available pricing methods (see market config)
        self.pricing = {
            c.MP_UNIFORM: self.__pricing_uniform,  # uniform pricing
            c.MP_DISCRIMINATORY: self.__pricing_discriminatory,  # discriminatory pricing
        }

        # Available coupling methods (see market config)
        # Note: This probably means that the upper market draws the offers and bids from the lower market (ponder)
        # TODO: This needs to change. The creator will either have a value or not there and the market just executes.
        #  In its current form there would be a more functionality in the executor that should be in the creator.
        self.coupling = {
            None: self.__return_data,  # no coupling
            c.MC_ABOVE: self.__coupling_above,  # post offers and bids on market above
            c.MC_BELOW: self.__coupling_below,  # post offers and bids on market below
        }

    def execute(self):
        """Executes all the actions of the LEM defined in the tasks"""

        # Generated with co-pilot so might be not quite right
        # Get the actions to be executed
        actions = self.tasks[c.TC_ACTIONS].split(',')
        # Get the clearing type
        clearing_type = self.tasks[c.TC_CLEARING_TYPE]
        # Get the clearing method
        clearing_method = self.tasks[c.TC_CLEARING_METHOD]
        # Get the pricing method
        pricing_method = self.tasks[c.TC_CLEARING_PRICING]
        # Get the coupling method
        coupling_method = self.tasks[c.TC_COUPLING]

        # Execute the actions
        for action in actions:
            self.actions[action](clearing_type, clearing_method, pricing_method,
                                 actions=actions)

        # Couple market
        # Note: This is not part of the actions, but is executed after the actions
        self.__couple_markets(clearing_type, clearing_method, pricing_method, coupling_method)

        return self.market

    def __action_clear(self, clearing_type: str = None, clearing_method: str = None, pricing_method: str = None,
                       **kwargs) -> pl.DataFrame:
        """Clears the market based on the given clearing type and method
        """

        # Define the action
        action = c.MA_CLEAR

        if clearing_type in self.types:
            # Call the clearing method
            self.transactions = self.types[clearing_type](clearing_method, pricing_method, action, **kwargs)
        else:
            raise ValueError(f'Clearing type "{clearing_type}" not available. '
                             f'Available types are: {self.types.keys()}')

        return self.transactions

    def __action_settle(self, clearing_type, clearing_method, pricing_method, **kwargs):
        """Settles the market"""
        # At this point the trades that occured get settled thus balancing energy is determined
        # as well as levies and taxes are applied

        # Set the action
        action = c.MA_SETTLE

        if clearing_type in self.types:
            # Call the clearing method
            self.transactions = self.types[clearing_type](clearing_method, pricing_method, action, **kwargs)
        else:
            raise ValueError(f'Clearing type "{clearing_type}" not available. '
                             f'Available types are: {self.types.keys()}')

        return self.transactions

    def __type_ex_ante(self, clearing_method: str = None, pricing_method: str = None, action: str = c.MA_CLEAR,
                       **kwargs) -> pl.DataFrame:
        """Clears the market ex-ante"""

        match action:
            case c.MA_CLEAR:
                # Clear the market
                self.transactions = self.__clear_ex_ante(clearing_method, pricing_method, **kwargs)
            case c.MA_SETTLE:
                # Settle the market
                self.transactions = self.__settle_ex_ante(clearing_method, pricing_method, **kwargs)
            case _:
                raise ValueError(f'Action "{action}" not available (for this market type). '
                                 f'Available actions are: {self.actions.keys()}')

        return self.transactions

    def __type_ex_post(self, clearing_method: str = None, pricing_method: str = None, action: str = c.MA_CLEAR,
                       **kwargs) -> pl.DataFrame:
        """Clears the market ex-post"""
        # match action:
        #     case c.MA_SETTLE:
        #         # Settle the market
        #         self.transactions = self.__settle_ex_post(clearing_method, pricing_method, **kwargs)
        #     case _:
        #         raise ValueError(f'Action "{action}" not available (for this market type). '
        #                          f'Available actions are: {self.actions.keys()}')
        #
        # return self.transactions
        raise NotImplementedError('Ex-post clearing not implemented yet')

    def __clear_ex_ante(self, clearing_method, pricing_method, **kwargs):

        # Check if there is anything to clear otherwise return
        if self.bids_offers.is_empty():
            self.__update_database()
            return (self.transactions, self.offers_uncleared, self.bids_uncleared, self.offers_cleared,
                    self.bids_cleared)

        # Create the bids and offers table from the bids and offers of the agents and the retailers
        bids_offers, retailer = self.__create_bids_offers()

        # Split the bids and offers into separate bids and offers tables
        bids, offers = self.__split_bids_offers(bids_offers)

        # Clear the bids and offers
        trades_cleared, trades_uncleared = self.__clear_bids_offers(bids, offers, clearing_method, pricing_method,
                                                                    retailer)

        # Create the tables about the market results
        bids_cleared, offers_cleared, bids_uncleared, offers_uncleared, transactions = (
            self.__create_market_tables(bids, offers, trades_cleared, trades_uncleared, retailer))

        # Update the tables and market database
        self.__update_database(bids_cleared=bids_cleared, offers_cleared=offers_cleared,
                               bids_uncleared=bids_uncleared, offers_uncleared=offers_uncleared,
                               transactions=transactions)

        return self.transactions

    def __settle_ex_ante(self, clearing_method, pricing_method, **kwargs):

        # Get the actions that are to be executed for this timestep
        all_actions = kwargs['actions']

        # Check if clearing has occurred, otherwise create the correct uncleared bids and offers tables
        if c.MA_CLEAR not in all_actions and not self.bids_offers.is_empty():
            # Create the bids and offers table from the bids and offers of the agents and the retailers
            bids_offers, _ = self.__create_bids_offers(include_retailer=False)
            # Split the bids and offers into separate bids and offers tables
            bids_uncleared, offers_uncleared = self.__split_bids_offers(bids_offers, add_cumsum=False)
        else:
            bids_uncleared = self.bids_uncleared
            offers_uncleared = self.offers_uncleared

        # Determine balancing energy
        transactions, _, _ = self.__determine_balancing_energy(bids_uncleared, offers_uncleared)

        # Apply levies and taxes
        grid, levies = self.__apply_grid_levies(transactions)

        # Update the tables and market database
        transactions = pl.concat([grid, levies, transactions], how='diagonal')

        self.__update_database(transactions=transactions)

        return self.transactions

    def __settle_ex_post(self, clearing_method, pricing_method, param, **kwargs):
        """Settles the market ex-post"""
        raise NotImplementedError('Ex-post clearing not implemented yet')

    def __method_none(self, bids, offers, pricing_method):
        """Clears the market with no clearing method, i.e. only the retailer acts as trading partner"""

        # Merge bids and offers on the energy_cumsum column
        bids_offers = pl.concat([bids, offers], how='diagonal')

        # Sort the bids and offers by the energy_cumsum
        bids_offers = bids_offers.sort(C_ENERGY_CUMSUM, descending=False)

        # Fill the NaN values with the retailer
        bids_offers = bids_offers.with_columns([
            pl.when(pl.col(c.TC_ID_AGENT_IN).is_not_null()).then(pl.col(c.TC_ID_AGENT_IN))
            .otherwise(pl.lit('retailer')).alias(c.TC_ID_AGENT_IN),
            pl.when(pl.col(c.TC_ID_AGENT_OUT).is_not_null()).then(pl.col(c.TC_ID_AGENT_OUT))
            .otherwise(pl.lit('retailer')).alias(c.TC_ID_AGENT_OUT),
        ])

        # Create bids and offers table where retailer is the only trading partner
        # Note: This currently works only for one retailer named 'retailer' in the future it needs to first obtain the
        #  names of the retailers and exclude them instead
        bids = bids_offers.filter((pl.col(c.TC_ID_AGENT_IN) != 'retailer') | (pl.col(c.TC_ID_AGENT_OUT) == 'retailer'))
        offers = bids_offers.filter(
            (pl.col(c.TC_ID_AGENT_IN) == 'retailer') | (pl.col(c.TC_ID_AGENT_OUT) != 'retailer'))

        # Fill the NaN values with the last value to know who trades with whom
        bids = bids.fill_null(strategy='backward')
        offers = offers.fill_null(strategy='backward')

        # Filter rows that have the same agent id in the in and out column (cannot trade with themselves)
        bids = bids.filter(pl.col(c.TC_ID_AGENT_IN) != pl.col(c.TC_ID_AGENT_OUT))
        offers = offers.filter(pl.col(c.TC_ID_AGENT_IN) != pl.col(c.TC_ID_AGENT_OUT))

        # Clear bids and offers
        bids_cleared = bids.filter(pl.col(c.TC_PRICE_PU_IN) >= pl.col(c.TC_PRICE_PU_OUT))
        offers_cleared = offers.filter(pl.col(c.TC_PRICE_PU_IN) >= pl.col(c.TC_PRICE_PU_OUT))

        # Calculate the pu price of the trades
        bids_cleared = self.pricing[pricing_method](bids_cleared)
        offers_cleared = self.pricing[pricing_method](offers_cleared)

        # Create new dataframe with the cleared bids and offers
        trades_cleared = pl.concat([bids_cleared, offers_cleared], how='vertical')

        # Calculate the price and energy of the trades
        trades_cleared = trades_cleared.with_columns(
            (trades_cleared.select([c.TC_ENERGY_IN, c.TC_ENERGY_OUT]).min(axis=1).alias(c.TC_ENERGY)),
        )
        trades_cleared = trades_cleared.with_columns(
            (pl.col(c.TC_PRICE_PU) * pl.col(c.TC_ENERGY)).alias(c.TC_PRICE).cast(pl.Int64),
        )

        # Create new dataframe with the uncleared bids and offers
        bids_uncleared = bids.filter(pl.col(c.TC_PRICE_PU_IN) < pl.col(c.TC_PRICE_PU_OUT))
        offers_uncleared = offers.filter(pl.col(c.TC_PRICE_PU_IN) < pl.col(c.TC_PRICE_PU_OUT))
        trades_uncleared = pl.concat([bids_uncleared, offers_uncleared], how='vertical')

        return trades_cleared, trades_uncleared

    def __method_community(self):
        """Clears the market with the community-based clearing method"""
        raise NotImplementedError('Community-based clearing not implemented yet')

    def __create_bids_offers(self, include_retailer=True):
        """Creates the bids and offers table from the bids and offers of the agents and the retailers"""

        # Check if the retailer should be included
        if include_retailer:
            # Find all columns that are relevant for energy
            energy_cols = [col for col in self.retailer.columns if col.startswith(c.TT_ENERGY)]
            # Add bid and offer by the retailers
            retailer = self.retailer.select(pl.col(c.TC_TIMESTAMP),
                                            pl.col(c.TC_REGION),
                                            pl.col(c.TC_MARKET),
                                            pl.col(c.TC_NAME),
                                            pl.col(c.TC_ID_AGENT),
                                            pl.col(c.TC_TYPE_ENERGY),
                                            *[pl.col(col) for col in energy_cols])
            # Add timestep column
            retailer = retailer.with_columns(
                [
                    pl.col(c.TC_TIMESTAMP).alias(c.TC_TIMESTEP),
                ]
            )
            # Remove starting 'energy' from all columns
            retailer = retailer.rename({col: col.split('_', 1)[1] for col in energy_cols})

            # Change price columns to price pu columns
            retailer = retailer.rename({c.TC_PRICE_IN: c.TC_PRICE_PU_IN, c.TC_PRICE_OUT: c.TC_PRICE_PU_OUT,})

            # Select only the relevant columns
            retailer = retailer.select(self.bids_offers.columns)

            # Cast the columns to the correct types
            retailer = f.enforce_schema(c.SCHEMA, retailer)
            # TODO: Create a general function that enforces dtypes (e.g. from a schema)
            # retailer = retailer.with_columns(
            #     [
            #         pl.col(c.TC_ID_AGENT).cast(pl.Categorical, strict=False),
            #         pl.col(c.TC_ENERGY_IN).cast(pl.UInt64, strict=False),
            #         pl.col(c.TC_ENERGY_OUT).cast(pl.UInt64, strict=False),
            #         pl.col(c.TC_PRICE_PU_IN).cast(pl.Int32, strict=False),
            #         pl.col(c.TC_PRICE_PU_OUT).cast(pl.Int32, strict=False),
            #     ]
            # )

            # Add the retailer to the bids and offers
            bids_offers = pl.concat([self.bids_offers, retailer], how='vertical')
        else:
            bids_offers = self.bids_offers
            retailer = pl.DataFrame()

        # Fill all empty values using ffill
        bids_offers = bids_offers.fill_null(strategy='forward')

        return bids_offers, retailer

    def __split_bids_offers(self, bids_offers, add_cumsum=True):
        """Splits the bids and offers into separate tables"""
        # Split the bids and offers into separate bids and offers tables
        bids = bids_offers.filter(pl.col(c.TC_ENERGY_IN) > 0)
        offers = bids_offers.filter(pl.col(c.TC_ENERGY_OUT) > 0)

        # Drop the respective empty columns
        bids = bids.drop(c.TC_ENERGY_OUT, c.TC_PRICE_PU_OUT)
        offers = offers.drop(c.TC_ENERGY_IN, c.TC_PRICE_PU_IN)

        # Rename agent column
        bids = bids.rename({c.TC_ID_AGENT: c.TC_ID_AGENT_IN})
        offers = offers.rename({c.TC_ID_AGENT: c.TC_ID_AGENT_OUT})

        # Shuffle the data to avoid bias
        bids = bids.sample(fraction=1, shuffle=True)
        offers = offers.sample(fraction=1, shuffle=True)

        # Sort the bids and offers by price
        bids = bids.sort(c.TC_PRICE_PU_IN, descending=True)
        offers = offers.sort(c.TC_PRICE_PU_OUT, descending=False)

        # Add missing columns through concatenation
        bids = pl.concat([bids, self.bids_cleared], how='diagonal')
        offers = pl.concat([offers, self.offers_cleared], how='diagonal')

        if add_cumsum:
            # Add column that contains the cumsum of the energy
            bids = bids.with_columns(pl.col(c.TC_ENERGY_IN).cumsum().alias(C_ENERGY_CUMSUM))
            offers = offers.with_columns(pl.col(c.TC_ENERGY_OUT).cumsum().alias(C_ENERGY_CUMSUM))

        return bids, offers

    def __clear_bids_offers(self, bids: pl.DataFrame, offers: pl.DataFrame, clearing_method: str, pricing_method: str,
                            retailer: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Clears the bids and offers"""

        if clearing_method in self.methods:
            trades_cleared, trades_uncleared = self.methods[clearing_method](bids, offers, pricing_method)
        else:
            raise ValueError(f'Clearing method "{clearing_method}" not available (for given market type). '
                             f'Available methods are: {self.methods.keys()}')

        # Post-process the trades
        trades_cleared = self.__postprocess_trades(trades_cleared, retailer)

        return trades_cleared, trades_uncleared

    @staticmethod
    def __create_market_tables(bids, offers, trades_cleared, trades_uncleared, retailer):
        """Creates the tables about the market results"""

        # Create the cleared tables
        # Filter out the bids and offers that were cleared by checking for the same agent id
        bids_cleared = bids.join(trades_cleared, on=c.TC_ID_AGENT_IN, how='semi')
        offers_cleared = offers.join(trades_cleared, on=c.TC_ID_AGENT_OUT, how='semi')

        # Add the energy, price pu and price column from the trades_cleared table
        cols = [c.TC_ID_AGENT_IN, c.TC_ENERGY, c.TC_PRICE_PU, c.TC_PRICE]
        bids_cleared = bids_cleared.join(trades_cleared.select(cols), on=c.TC_ID_AGENT_IN, how='inner')
        cols = [c.TC_ID_AGENT_OUT, c.TC_ENERGY, c.TC_PRICE_PU, c.TC_PRICE]
        offers_cleared = offers_cleared.join(trades_cleared.select(cols), on=c.TC_ID_AGENT_OUT, how='inner')

        # Drop the unnecessary columns and rename the relevant ones to create the final tables
        bids_cleared = bids_cleared.drop(c.TC_ENERGY_IN, c.TC_PRICE_PU_IN, c.TC_PRICE_IN, C_ENERGY_CUMSUM)
        bids_cleared = bids_cleared.rename({c.TC_ENERGY: c.TC_ENERGY_IN, c.TC_PRICE_PU: c.TC_PRICE_PU_IN,
                                            c.TC_PRICE: c.TC_PRICE_IN})
        offers_cleared = offers_cleared.drop(c.TC_ENERGY_OUT, c.TC_PRICE_PU_OUT, c.TC_PRICE_OUT, C_ENERGY_CUMSUM)
        offers_cleared = offers_cleared.rename({c.TC_ENERGY: c.TC_ENERGY_OUT, c.TC_PRICE_PU: c.TC_PRICE_PU_OUT,
                                                c.TC_PRICE: c.TC_PRICE_OUT})

        # Create the uncleared tables
        # First take all bids and offers
        bids_uncleared = bids
        offers_uncleared = offers

        # Subtract the cleared bids and offers energy amount by agent id
        # Bids
        # First get the sum of the cleared energy by agent id
        bids_cleared_by_agent_id = bids_cleared.groupby(c.TC_ID_AGENT_IN).sum()
        bids_cleared_by_agent_id = bids_cleared_by_agent_id.rename({c.TC_ENERGY_IN: c.TC_ENERGY})
        bids_cleared_by_agent_id = bids_cleared_by_agent_id.select([c.TC_ID_AGENT_IN, c.TC_ENERGY])
        # Join the dataframes to have the information about the energy that was cleared
        bids_uncleared = bids_uncleared.join(bids_cleared_by_agent_id, on=c.TC_ID_AGENT_IN, how='outer')
        # Set all null values in the energy column to 0
        bids_uncleared = bids_uncleared.fill_null(0)
        # Subtract the cleared energy from the uncleared energy
        bids_uncleared = bids_uncleared.with_columns(
            (pl.col(c.TC_ENERGY_IN) - pl.col(c.TC_ENERGY)).alias(c.TC_ENERGY_IN),
        )
        # Drop the rows where the energy is smaller or equal to 0
        bids_uncleared = bids_uncleared.filter(pl.col(c.TC_ENERGY_IN) > 0)
        # Cast the energy column to UInt64 and quality is UInt8
        bids_uncleared = bids_uncleared.with_columns(
            pl.col(c.TC_ENERGY_IN).cast(pl.UInt64),
        )
        # Drop the energy, agent_in_right and energy_cumsum column
        # TODO: Replace once polars 1.0 is used
        try:
            bids_uncleared = bids_uncleared.drop([c.TC_ENERGY, C_ENERGY_CUMSUM, f'{c.TC_ID_AGENT_IN}_right'])
        except pl.ColumnNotFoundError:
            bids_uncleared = bids_uncleared.drop([c.TC_ENERGY, C_ENERGY_CUMSUM])
        # Drop all rows where the agent id is the same as the one in the retailer table
        retailer_names = retailer.select(c.TC_ID_AGENT).to_series().to_list()
        bids_uncleared = bids_uncleared.filter(~pl.col(c.TC_ID_AGENT_IN).is_in(retailer_names))
        # Offers
        # First get the sum of the cleared energy by agent id
        offers_cleared_by_agent_id = offers_cleared.groupby(c.TC_ID_AGENT_OUT).sum()
        offers_cleared_by_agent_id = offers_cleared_by_agent_id.rename({c.TC_ENERGY_OUT: c.TC_ENERGY})
        offers_cleared_by_agent_id = offers_cleared_by_agent_id.select([c.TC_ID_AGENT_OUT, c.TC_ENERGY])
        # Join the dataframes to have the information about the energy that was cleared
        offers_uncleared = offers_uncleared.join(offers_cleared_by_agent_id, on=c.TC_ID_AGENT_OUT, how='outer')
        # Set all null values in the energy column to 0
        offers_uncleared = offers_uncleared.fill_null(0)
        # Subtract the cleared energy from the uncleared energy
        offers_uncleared = offers_uncleared.with_columns(
            (pl.col(c.TC_ENERGY_OUT) - pl.col(c.TC_ENERGY)).alias(c.TC_ENERGY_OUT),
        )
        # Drop the rows where the energy is smaller or equal to 0
        offers_uncleared = offers_uncleared.filter(pl.col(c.TC_ENERGY_OUT) > 0)
        # Enforce correct dtypes
        offers_uncleared = f.enforce_schema(c.SCHEMA, offers_uncleared)
        # Drop the energy, agent_in_right and energy_cumsum column
        # TODO: Replace once polars 1.0 is used
        try:
            offers_uncleared = offers_uncleared.drop([c.TC_ENERGY, C_ENERGY_CUMSUM, f'{c.TC_ID_AGENT_OUT}_right'])
        except pl.ColumnNotFoundError:
            offers_uncleared = offers_uncleared.drop([c.TC_ENERGY, C_ENERGY_CUMSUM])
        # Drop all rows where the agent id is the same as the one in the retailer table
        offers_uncleared = offers_uncleared.filter(~pl.col(c.TC_ID_AGENT_OUT).is_in(retailer_names))

        # Find all cols for either in or out (such exclude the other ones)
        cols_in = [col for col in trades_cleared.columns if not col.endswith(f'_{c.PF_OUT}')]
        cols_out = [col for col in trades_cleared.columns if not col.endswith(f'_{c.PF_IN}')]
        df_in = trades_cleared.select(cols_in)
        df_out = trades_cleared.select(cols_out)
        # Fill the empty columns with the values
        df_in = df_in.with_columns([
            pl.col(f'{c.TC_ENERGY}').alias(f'{c.TC_ENERGY_IN}'),
            pl.col(f'{c.TC_PRICE_PU}').alias(f'{c.TC_PRICE_PU_IN}'),
            pl.col(f'{c.TC_PRICE}').alias(f'{c.TC_PRICE_IN}'),
        ])
        df_out = df_out.with_columns([
            pl.col(f'{c.TC_ENERGY}').alias(f'{c.TC_ENERGY_OUT}'),
            pl.col(f'{c.TC_PRICE_PU}').alias(f'{c.TC_PRICE_PU_OUT}'),
            pl.col(f'{c.TC_PRICE}').alias(f'{c.TC_PRICE_OUT}'),
        ])
        # Create the transactions table from the cleared trades
        transactions = pl.concat([df_in, df_out], how='diagonal')
        # Add missing columns
        transactions = transactions.with_columns([
            pl.when(pl.col(c.TC_ID_AGENT_IN).is_not_null())
            .then(pl.col(c.TC_ID_AGENT_IN)).otherwise(pl.col(c.TC_ID_AGENT_OUT)).alias(c.TC_ID_AGENT)
        ])

        transactions = transactions.drop(c.TC_ID_AGENT_IN, c.TC_ID_AGENT_OUT)

        # Sort the rows according to the schema
        schema = c.TS_MARKET_TRANSACTIONS
        # transactions = transactions.select(schema)
        transactions = transactions.select(schema)

        return bids_cleared, offers_cleared, bids_uncleared, offers_uncleared, transactions

    def __update_database(self, bids_cleared: pl.DataFrame = None, offers_cleared: pl.DataFrame = None,
                          bids_uncleared: pl.DataFrame = None, offers_uncleared: pl.DataFrame = None,
                          transactions: pl.DataFrame = None) -> MarketDB:
        """Updates the market database

        TODO: Reduce/Increase the available energy of the retailer by the amount that was bought/sold to them
        """

        # Add the trades to their corresponding tables
        if bids_cleared is not None:
            self.bids_cleared = pl.concat([self.bids_cleared, bids_cleared], how='diagonal')
        if offers_cleared is not None:
            self.offers_cleared = pl.concat([self.offers_cleared, offers_cleared], how='diagonal')
        if bids_uncleared is not None:
            self.bids_uncleared = pl.concat([self.bids_uncleared, bids_uncleared], how='diagonal')
        if offers_uncleared is not None:
            self.offers_uncleared = pl.concat([self.offers_uncleared, offers_uncleared], how='diagonal')
        if transactions is not None:
            self.transactions = pl.concat([self.transactions, transactions], how='diagonal')

        # Update the market database
        self.market.bids_cleared = self.bids_cleared
        self.market.offers_cleared = self.offers_cleared
        self.market.bids_uncleared = self.bids_uncleared
        self.market.offers_uncleared = self.offers_uncleared
        self.market.market_transactions = self.transactions

        return self.market

    def __couple_markets(self, clearing_type, clearing_method, pricing_method, coupling_method, **kwargs):
        """Couple the market"""
        ...

    def __determine_balancing_energy(self, bids_uncleared, offers_uncleared):
        """Determines the balancing energy

        TODO: For now this ignores that there is a maximum amount of energy that can be bought/sold by the retailer
        """

        # Get the retailer offers
        # Note: This currently only works for one retailer
        retailer = self.retailer.filter((pl.col(c.TC_TIMESTAMP) == self.tasks[c.TC_TIMESTEP])
                                        & (pl.col(c.TC_REGION) == self.tasks[c.TC_REGION])
                                        & (pl.col(c.TC_MARKET) == self.tasks[c.TC_MARKET])
                                        & (pl.col(c.TC_NAME) == self.tasks[c.TC_NAME])).to_dict()

        # Create new trades table that contains only the balancing transactions
        transactions = pl.concat([bids_uncleared, offers_uncleared], how='diagonal')

        # Add price pu column information from the retailer
        transactions = transactions.with_columns([
            pl.lit(retailer[f'{c.TT_BALANCING}_{c.TC_PRICE}_{c.PF_IN}'].alias(c.TC_PRICE_PU_OUT)).cast(pl.Int32),
            pl.lit(retailer[f'{c.TT_BALANCING}_{c.TC_PRICE}_{c.PF_OUT}'].alias(c.TC_PRICE_PU_IN)).cast(pl.Int32),
        ])
        # Add missing columns
        transactions = transactions.with_columns([
            # ID agent
            pl.when(pl.col(c.TC_ID_AGENT_IN).is_not_null())
            .then(pl.col(c.TC_ID_AGENT_IN)).otherwise(pl.col(c.TC_ID_AGENT_OUT)).alias(c.TC_ID_AGENT),
            # Trade type
            pl.lit(c.TT_BALANCING).alias(c.TC_TYPE_TRANSACTION).cast(pl.Categorical),
        ])
        # Calculate the total price
        transactions = transactions.with_columns([
            (pl.col(c.TC_PRICE_PU_IN) * pl.col(c.TC_ENERGY_IN)).round().alias(c.TC_PRICE_IN).cast(pl.Int64),
            (pl.col(c.TC_PRICE_PU_OUT) * pl.col(c.TC_ENERGY_OUT)).round().alias(c.TC_PRICE_OUT).cast(pl.Int64),
        ])

        # Drop unnecessary columns
        transactions = transactions.drop(c.TC_ID_AGENT_IN, c.TC_ID_AGENT_OUT)

        # Delete the rows of the bids and offers
        self.bids_uncleared = bids_uncleared.clear()
        self.offers_uncleared = offers_uncleared.clear()

        return transactions, self.bids_uncleared, self.offers_uncleared

    def __apply_grid_levies(self, transactions):
        """Applies levies and taxes to the market"""
        # Needs to discriminate between the different types of levies and taxes (wholesale or local)

        c_TOTAL = 'total'
        c_RATIO = 'ratio'

        # Get the retailer offers
        # Note: This currently only works for one retailer
        retailer = self.retailer.filter((pl.col(c.TC_TIMESTAMP) == self.tasks[c.TC_TIMESTEP])
                                        & (pl.col(c.TC_REGION) == self.tasks[c.TC_REGION])
                                        & (pl.col(c.TC_MARKET) == self.tasks[c.TC_MARKET])
                                        & (pl.col(c.TC_NAME) == self.tasks[c.TC_NAME]))
        retailer = retailer.to_dict()

        # Concat all transactions for the given timestep to calculate the net energy
        transactions = pl.concat([self.transactions_prev, transactions], how='diagonal')
        # Remove retailer from the transactions (as they are not subject to levies and taxes)
        transactions = transactions.filter(~pl.col(c.TC_ID_AGENT).is_in(retailer[c.TC_ID_AGENT].to_list()))

        # Sum energy_in and energy_out for each type of transaction per agent
        transaction_sums = (
            transactions.group_by([c.TC_ID_AGENT, c.TC_TYPE_TRANSACTION])
            .agg([
                pl.col(c.TC_ENERGY_IN).sum().alias(f'{c_TOTAL}_{c.TC_ENERGY_IN}'),
                pl.col(c.TC_ENERGY_OUT).sum().alias(f'{c_TOTAL}_{c.TC_ENERGY_OUT}'),
            ])
            .pivot(values=[f'{c_TOTAL}_{c.TC_ENERGY_IN}', f'{c_TOTAL}_{c.TC_ENERGY_OUT}'],
                   index=c.TC_ID_AGENT,
                   columns=c.TC_TYPE_TRANSACTION,
                   aggregate_function='first')
            .fill_null(0)
        )

        # Calculate total energy in and out per agent
        transaction_types = transactions[c.TC_TYPE_TRANSACTION].unique()
        in_columns_to_sum = [f'{c_TOTAL}_{c.TC_ENERGY_IN}_{c.TC_TYPE_TRANSACTION}_{tx}' for tx in transaction_types]
        out_columns_to_sum = [f'{c_TOTAL}_{c.TC_ENERGY_OUT}_{c.TC_TYPE_TRANSACTION}_{tx}' for tx in transaction_types]
        transaction_sums = transaction_sums.with_columns([
            pl.sum_horizontal([pl.col(col) for col in in_columns_to_sum]).alias(f'{c_TOTAL}_{c.TC_ENERGY_IN}'),
            pl.sum_horizontal([pl.col(col) for col in out_columns_to_sum]).alias(f'{c_TOTAL}_{c.TC_ENERGY_OUT}'),
        ])

        # Calculate net energy on which the ratios will be based on
        transaction_sums = transaction_sums.with_columns([
            pl.when(pl.col(f'{c_TOTAL}_{c.TC_ENERGY_IN}') - pl.col(f'{c_TOTAL}_{c.TC_ENERGY_OUT}') > 0)
            .then(pl.col(f'{c_TOTAL}_{c.TC_ENERGY_IN}') - pl.col(f'{c_TOTAL}_{c.TC_ENERGY_OUT}'))
            .otherwise(0)
            .alias(c.TC_ENERGY_IN),
            pl.when(pl.col(f'{c_TOTAL}_{c.TC_ENERGY_IN}') - pl.col(f'{c_TOTAL}_{c.TC_ENERGY_OUT}') < 0)
            .then(pl.col(f'{c_TOTAL}_{c.TC_ENERGY_OUT}') - pl.col(f'{c_TOTAL}_{c.TC_ENERGY_IN}'))
            .otherwise(0)
            .alias(c.TC_ENERGY_OUT),
        ])

        # Calculate the ratio for each transaction type
        transaction_sums = transaction_sums.with_columns([
                 (pl.col(f'{c_TOTAL}_{c.TC_ENERGY_IN}_{c.TC_TYPE_TRANSACTION}_{tx}')
                  / pl.col(f'{c_TOTAL}_{c.TC_ENERGY_IN}'))
             .alias(f'{c_RATIO}_{tx}_{c.PF_IN}')
                 for tx in transaction_types
             ] + [
                 (pl.col(
                     f'{c_TOTAL}_{c.TC_ENERGY_OUT}_{c.TC_TYPE_TRANSACTION}_{tx}') / pl.col(
                     f'{c_TOTAL}_{c.TC_ENERGY_OUT}')).alias(f'{c_RATIO}_{tx}_{c.PF_OUT}')
                 for tx in transaction_types
         ])

        # Compute ratios based on market ratio
        # TODO: Simplify once polars is upgraded to 1.0 (get_column has a default value from that version on)
        if f'{c_RATIO}_{c.TT_MARKET}_{c.PF_IN}' in transaction_sums.columns:  # out is therefore also in columns
            transaction_sums = transaction_sums.with_columns(
                transaction_sums.get_column(f'{c_RATIO}_{c.TT_MARKET}_{c.PF_IN}').alias(f'{c_RATIO}_{c.PF_IN}'),
                transaction_sums.get_column(f'{c_RATIO}_{c.TT_MARKET}_{c.PF_OUT}').alias(f'{c_RATIO}_{c.PF_OUT}')
            )
        else:
            transaction_sums = transaction_sums.with_columns(
                pl.lit(0).alias(f'{c_RATIO}_{c.PF_IN}'),
                pl.lit(0).alias(f'{c_RATIO}_{c.PF_OUT}')
            )

        # Select relevant columns
        to_join = transaction_sums.select([c.TC_ID_AGENT, c.TC_ENERGY_IN, c.TC_ENERGY_OUT,
                                           f'{c_RATIO}_{c.PF_IN}', f'{c_RATIO}_{c.PF_OUT}'])

        # Select only the first row of each agent in transactions
        transactions = transactions.unique(c.TC_ID_AGENT)
        # Join the dataframes to have the information about the net energy
        suffix = '_right'
        # TODO: Change this to how='right' once polars 1.0 is used
        transactions = transactions.join(to_join, on=c.TC_ID_AGENT, how='left', suffix=suffix, coalesce=True)
        # Replace the new energy columns with the old ones
        transactions = transactions.with_columns([
            pl.col(f'{c.TC_ENERGY_IN}{suffix}').alias(c.TC_ENERGY_IN).cast(pl.UInt64),
            pl.col(f'{c.TC_ENERGY_OUT}{suffix}').alias(c.TC_ENERGY_OUT).cast(pl.UInt64),
        ])
        # Drop the unnecessary columns to finally have the transactions that are relevant for the grid fees and levies
        transactions = transactions.drop([f'{c.TC_ENERGY_IN}{suffix}', f'{c.TC_ENERGY_OUT}{suffix}'])

        # Copy the transactions table to apply the grid fees
        grid = transactions.clone()

        # Add temporary columns
        grid = grid.with_columns([
            pl.lit(retailer[f'{c.TT_GRID}_{c.TT_MARKET}_{c.PF_OUT}'].alias(f'{c.TT_MARKET}_{c.TC_PRICE_PU_OUT}')),
            pl.lit(retailer[f'{c.TT_GRID}_{c.TT_MARKET}_{c.PF_IN}'].alias(f'{c.TT_MARKET}_{c.TC_PRICE_PU_IN}')),
            pl.lit(retailer[f'{c.TT_GRID}_{c.TT_RETAIL}_{c.PF_OUT}'].alias(f'{c.TT_RETAIL}_{c.TC_PRICE_PU_OUT}')),
            pl.lit(retailer[f'{c.TT_GRID}_{c.TT_RETAIL}_{c.PF_IN}'].alias(f'{c.TT_RETAIL}_{c.TC_PRICE_PU_IN}')),
        ])

        # Replace price with agent forecast for variable grid fees
        for agent_id in grid.select(c.TC_ID_AGENT).unique().to_series().to_list():
            # get agent database
            agent_db = self.database.get_agent_data(region=self.tasks[c.TC_REGION], agent_id=agent_id)
            agent_retailer_data = agent_db.forecaster.train_data[f'{self.market.market_name}_{c.TT_RETAIL}'][c.K_TARGET]

            # iterate through different trade type and different power flow direction
            for column_name in [[c.TT_RETAIL, c.PF_IN, c.TC_PRICE_PU_IN],[c.TT_RETAIL, c.PF_OUT, c.TC_PRICE_PU_OUT],
                                [c.TT_MARKET, c.PF_IN, c.TC_PRICE_PU_IN],[c.TT_MARKET, c.PF_OUT, c.TC_PRICE_PU_OUT]]:
                # get grid fee value from forecaster
                grid_fee = agent_retailer_data.filter(pl.col(c.TC_TIMESTAMP) == self.tasks[c.TC_TIMESTEP]).select(
                    pl.col(f'{c.TT_GRID}_{column_name[0]}_{column_name[1]}')).item()
                # write to grid transaction df
                grid = grid.with_columns(
                    # Energy pu prices
                    pl.when(pl.col(c.TC_ID_AGENT) == agent_id).then(pl.lit(grid_fee))
                    .otherwise(pl.col(f'{column_name[0]}_{column_name[2]}'))
                    .alias(f'{column_name[0]}_{column_name[2]}').cast(pl.Int32)
                )

        # Calculate the price pu columns (market * ratio + retail * (1 - ratio))
        grid = grid.with_columns([
            (pl.col(f'{c.TT_MARKET}_{c.TC_PRICE_PU_IN}') * pl.col(f'{c_RATIO}_{c.PF_IN}')
             + pl.col(f'{c.TT_RETAIL}_{c.TC_PRICE_PU_IN}') * (1 - pl.col(f'{c_RATIO}_{c.PF_IN}')))
            .fill_nan(0).cast(pl.Float32).round().alias(c.TC_PRICE_PU_IN).cast(pl.Int32),
            (pl.col(f'{c.TT_MARKET}_{c.TC_PRICE_PU_OUT}') * pl.col(f'{c_RATIO}_{c.PF_OUT}')
             + pl.col(f'{c.TT_RETAIL}_{c.TC_PRICE_PU_OUT}') * (1 - pl.col(f'{c_RATIO}_{c.PF_OUT}')))
            .fill_nan(0).cast(pl.Float32).round().alias(c.TC_PRICE_PU_OUT).cast(pl.Int32),
        ])
        # Set trade type
        grid = grid.with_columns([
            pl.lit(c.TT_GRID).alias(c.TC_TYPE_TRANSACTION).cast(pl.Categorical),
        ])
        # Calculate the total price
        grid = grid.with_columns([
            (pl.col(c.TC_PRICE_PU_IN) * pl.col(c.TC_ENERGY_IN)).alias(c.TC_PRICE_IN).cast(pl.Int64),
            (pl.col(c.TC_PRICE_PU_OUT) * pl.col(c.TC_ENERGY_OUT)).alias(c.TC_PRICE_OUT).cast(pl.Int64),
        ])
        # Drop unnecessary columns
        grid = grid.drop(f'{c.TT_MARKET}_{c.TC_PRICE_PU_OUT}', f'{c.TT_MARKET}_{c.TC_PRICE_PU_IN}',
                         f'{c.TT_RETAIL}_{c.TC_PRICE_PU_OUT}', f'{c.TT_RETAIL}_{c.TC_PRICE_PU_IN}',
                         f'{c_RATIO}_{c.PF_IN}', f'{c_RATIO}_{c.PF_OUT}')

        # Copy the transactions table to apply the levies
        transactions = transactions.drop(f'{c_RATIO}_{c.PF_IN}', f'{c_RATIO}_{c.PF_OUT}')
        levies = transactions.clone()
        # Adjust the price and trade type columns
        levies = levies.with_columns([
            # Energy pu prices
            pl.lit(retailer[f'{c.TT_LEVIES}_{c.TC_PRICE_OUT}'].alias(c.TC_PRICE_PU_OUT)).cast(pl.Int32),
            pl.lit(retailer[f'{c.TT_LEVIES}_{c.TC_PRICE_IN}'].alias(c.TC_PRICE_PU_IN)).cast(pl.Int32),
            # Trade type
            pl.lit(c.TT_LEVIES).alias(c.TC_TYPE_TRANSACTION).cast(pl.Categorical),
        ])
        # Calculate the total price
        levies = levies.with_columns([
            (pl.col(c.TC_PRICE_PU_IN) * pl.col(c.TC_ENERGY_IN)).alias(c.TC_PRICE_IN).cast(pl.Int64),
            (pl.col(c.TC_PRICE_PU_OUT) * pl.col(c.TC_ENERGY_OUT)).alias(c.TC_PRICE_OUT).cast(pl.Int64),
        ])

        # Update timestamp
        grid = grid.with_columns(pl.lit(self.tasks[c.TC_TIMESTAMP]).alias(c.TC_TIMESTAMP))
        levies = levies.with_columns(pl.lit(self.tasks[c.TC_TIMESTAMP]).alias(c.TC_TIMESTAMP))

        # Enforce schema
        grid = f.enforce_schema(c.SCHEMA, grid)
        levies = f.enforce_schema(c.SCHEMA, levies)

        return grid, levies

    def __method_pda(self, bids, offers, pricing_method):
        """Clears the market with the periodic double auction method"""

        # Merge bids and offers on the energy_cumsum column
        bids_offers = bids.join(offers, on=C_ENERGY_CUMSUM, how='outer', coalesce=True)

        # Sort the bids and offers by the energy_cumsum
        bids_offers = bids_offers.sort(C_ENERGY_CUMSUM, descending=False)  # .fill_null(strategy='backward')

        # Remove all columns that end on _right
        double_cols = [col for col in bids_offers.columns if col.endswith('_right')]
        for col in double_cols:
            orig_col = col.rsplit('_', 1)[0]
            bids_offers = bids_offers.with_columns(pl.coalesce([orig_col, col]).alias(orig_col))
        bids_offers = bids_offers.drop(double_cols)

        # Fill the NaN values with the last value
        bids_offers = bids_offers.fill_null(strategy='backward')

        # Create new dataframe with the cleared bids and offers
        trades_cleared = bids_offers.filter(pl.col(c.TC_PRICE_PU_IN) >= pl.col(c.TC_PRICE_PU_OUT))

        # Calculate the pu price of the trades
        trades_cleared = self.pricing[pricing_method](trades_cleared)

        # Calculate the price and energy of the trades
        trades_cleared = trades_cleared.with_columns(
            (trades_cleared.select([c.TC_ENERGY_IN, c.TC_ENERGY_OUT]).min(axis=1).alias(c.TC_ENERGY)),
        )
        trades_cleared = trades_cleared.with_columns(
            (pl.col(c.TC_PRICE_PU) * pl.col(c.TC_ENERGY)).alias(c.TC_PRICE).cast(pl.Int64),
        )

        # Create new dataframe with the uncleared bids and offers
        # Note: this tables includes the trades that were not cleared and the ones that were only partially cleared
        trades_uncleared = bids_offers.filter(pl.col(c.TC_PRICE_PU_IN) < pl.col(c.TC_PRICE_PU_OUT))

        return trades_cleared, trades_uncleared

    @staticmethod
    def __pricing_uniform(trades):
        """Prices the market with the uniform pricing method, thus everyone gets the same price which is the average
        of the last value of the price_pu_in and price_pu_out"""

        # Calculate the uniform pu price
        uniform_price = (trades.select([
            pl.col(c.TC_PRICE_PU_OUT).tail(1).sum(),
            pl.col(c.TC_PRICE_PU_IN).tail(1).sum()
        ]).sum(axis=1) / 2).round().cast(pl.Int32)[0]

        # Assign the uniform price to the trades
        trades = trades.with_columns(pl.lit(uniform_price).alias(c.TC_PRICE_PU))

        return trades

    @staticmethod
    def __pricing_discriminatory(trades):
        """Prices the market with the discriminatory pricing method"""
        # OLD (lemlab code)
        # Calculate discriminative prices if demanded
        # if 'discriminatory' == config_lem['types_pricing_ex_ante'][i]:
        #    positions_cleared.loc[:, db_obj.db_param.PRICE_ENERGY_MARKET_ + type_pricing] = \
        #        ((positions_cleared[db_obj.db_param.PRICE_ENERGY_OFFER] +
        #          positions_cleared[db_obj.db_param.PRICE_ENERGY_BID].iloc[:]) / 2).astype(int)
        raise NotImplementedError('Discriminatory pricing not implemented yet')

    def __coupling_above(self):
        """Coupling with the market above"""
        raise NotImplementedError('Market coupling not implemented yet')

    def __coupling_below(self):
        """Coupling with the market below"""
        raise NotImplementedError('Market coupling not implemented yet')

    @staticmethod
    def __return_data(data):
        return data

    def __postprocess_trades(self, trades: pl.DataFrame, retailer: pl.DataFrame):
        """Postprocesses the trades table"""

        # Add the type of transaction depending if the retailer is involved or not
        trades = trades.with_columns([
            pl.when(
                (pl.col(c.TC_ID_AGENT_IN).is_in(retailer.select(c.TC_ID_AGENT))) |
                (pl.col(c.TC_ID_AGENT_OUT).is_in(retailer.select(c.TC_ID_AGENT)))
            ).then(pl.lit(c.TT_RETAIL))
            .otherwise(pl.lit(c.TT_MARKET))
            .alias(c.TC_TYPE_TRANSACTION)
            .cast(pl.Categorical)
        ])

        # Add unique trade id to every trade
        values = f.gen_ids(n=len(trades), length=10,
                           prefix=f'{self.tasks[c.TC_TIMESTAMP]}&{self.tasks[c.TC_TIMESTEP]}&')
        if isinstance(values, str):
            values = [values]  # Ensure that it is a list (occurs when there is only one trade)
        ser_ids = pl.Series(name=c.TC_ID_TRADE,
                            values=values,
                            dtype=pl.String)
        trades = trades.with_columns(ser_ids)

        return trades

