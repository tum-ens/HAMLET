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

    #: Transaction types that count towards an agent's net traded energy. `market_transactions`
    #: also carries `grid` and `levies` rows, and those are *clones* of the netted transactions
    #: carrying identical energy -- measured on the shipped example: retail 926 rows / 320,525 Wh
    #: in, grid 87 / 24,298, levies 87 / 24,298. Summing without this filter roughly triple-counts
    #: traded energy for any agent subject to fees, which is why the paper branch's `_net_cache`
    #: was not ported as it stood. See ROADMAP item #2.
    NET_TRANSACTION_TYPES = (c.TT_RETAIL, c.TT_MARKET, c.TT_BALANCING)

    #: Columns of the net-energy cache, and of what `get_net_energy` returns.
    NET_SCHEMA = {c.TC_TIMESTEP: c.TS_MARKET_TRANSACTIONS[c.TC_TIMESTEP],
                  c.TC_ID_AGENT: c.TS_MARKET_TRANSACTIONS[c.TC_ID_AGENT],
                  c.TC_ENERGY_IN: pl.Int64,
                  c.TC_ENERGY_OUT: pl.Int64}

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

        # Running per (timestep, agent) sums. Two of them, because the trading strategy and the
        # RTC define traded energy differently -- see `_rtc_energy_from`.
        self._net_cache = pl.DataFrame(schema=self.NET_SCHEMA)
        self._rtc_cache = pl.DataFrame(schema=self.NET_SCHEMA)

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

        # The net-energy cache follows the tables out of the horizon. Entries for dropped
        # timesteps are already unreachable -- `get_net_energy` is only ever asked about the
        # trading horizon, which is ahead of `start_horizon_ts` -- so this is about keeping the
        # cache bounded rather than correct. Without it the cache would carry one row per
        # (timestep, agent) for the whole run.
        for attribute, cache in (('_net_cache', self.net_cache), ('_rtc_cache', self.rtc_cache)):
            if cache is not None and not cache.is_empty():
                setattr(self, attribute, cache.filter(
                    (pl.col(c.TC_TIMESTEP) >= start_horizon_ts).fill_null(True)))

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

    def set_market_transactions(self, data, new_rows=None):
        """Replace the transactions table, folding `new_rows` into both energy caches.

        `new_rows` is the *addition* rather than the whole table. Passing it keeps the caches
        incremental; omitting it drops them, so a caller that replaces the table wholesale cannot
        leave a stale one behind. Correctness never depends on which happens -- both getters
        recompute when there is no cache.
        """
        self.market_transactions = data
        if new_rows is None:
            self._net_cache = None
            self._rtc_cache = None
        else:
            self._fold_into_net_cache(new_rows)
            self._fold_into_rtc_cache(new_rows)

    @property
    def net_cache(self):
        """The net-energy cache, or None when there is not one.

        Read through this rather than the attribute. A `MarketDB` can legitimately exist without
        having run `__init__` -- `tests/integration/executor/test_market_db.py` builds them that
        way, deliberately, to test the file handling without a folder behind it -- and a cache
        that is merely absent must behave exactly like a cache that was dropped. The whole design
        rests on the cache never being a second source of truth.
        """
        return getattr(self, '_net_cache', None)

    def _fold_into_net_cache(self, new_rows):
        """Add `new_rows`' netted energy to the running per (timestep, agent) sums."""
        cache = self.net_cache
        if cache is None:
            return
        folded = self._net_energy_from(new_rows)
        if folded.is_empty():
            return
        self._net_cache = (pl.concat([cache, folded], how='vertical')
                           .groupby([c.TC_TIMESTEP, c.TC_ID_AGENT])
                           .agg(pl.col(c.TC_ENERGY_IN, c.TC_ENERGY_OUT).sum()))

    def _net_energy_from(self, transactions):
        """Net energy per (timestep, agent) for this market, out of `transactions`.

        This is the single definition of "netted energy", used both to build the cache and to
        answer a query when there is none. `NET_TRANSACTION_TYPES` is applied here and nowhere
        else, so the cached and uncached answers cannot diverge by drifting filters.
        """
        if transactions is None or transactions.is_empty():
            return pl.DataFrame(schema=self.NET_SCHEMA)

        return (transactions
                .filter((pl.col(c.TC_MARKET) == self.market_type)
                        & (pl.col(c.TC_NAME) == self.market_name)
                        & pl.col(c.TC_TYPE_TRANSACTION).is_in(list(self.NET_TRANSACTION_TYPES)))
                .groupby([c.TC_TIMESTEP, c.TC_ID_AGENT])
                .agg(pl.col(c.TC_ENERGY_IN, c.TC_ENERGY_OUT).sum())
                .with_columns([pl.col(c.TC_ENERGY_IN).cast(pl.Int64),
                               pl.col(c.TC_ENERGY_OUT).cast(pl.Int64)]))

    """rtc market results -- a second, deliberately different definition"""

    @property
    def rtc_cache(self):
        """The RTC's energy cache, or None. Same contract as `net_cache`."""
        return getattr(self, '_rtc_cache', None)

    def _fold_into_rtc_cache(self, new_rows):
        """Add `new_rows` to the running per (timestep, agent) sums the RTC reads."""
        cache = self.rtc_cache
        if cache is None:
            return
        folded = self._rtc_energy_from(new_rows)
        if folded.is_empty():
            return
        self._rtc_cache = (pl.concat([cache, folded], how='vertical')
                           .groupby([c.TC_TIMESTEP, c.TC_ID_AGENT])
                           .agg(pl.col(c.TC_ENERGY_IN, c.TC_ENERGY_OUT).sum()))

    def _rtc_energy_from(self, transactions):
        """Per (timestep, agent) sums for the RTC. Same type filter as above; no market filter.

        `RtcBase._get_market_results` applied **no transaction-type filter** -- it took this
        market's whole table, kept the agent's rows for one timestep and summed everything it
        found, so `grid` and `levies` rows, which clone the netted transactions and hold identical
        energy, were counted as traded energy. `strategies.py` has always excluded them.

        **The filter is applied here now, and it is inert.** It was measured before being added,
        because a change that is meant to move nothing has to be shown to move nothing rather than
        argued to: instrumenting every call, the filtered and unfiltered sums agreed on **96 of 96
        calls on the shipped example and 1040 of 1040 on the paper's design 6** -- a scenario whose
        table does contain 890 `grid` and 890 `levies` rows. The RTC never sees one.

        The mechanism is the ordering. Fees are written when a delivery timestep is *settled*, and
        within a timestep the executor runs agents before markets, so when the RTC asks about
        timestep T there is nothing for T but ex-ante trades. The filter is therefore not a fix but
        a guard: without it, that invariant is unstated, unenforced, and silently load-bearing --
        move settlement earlier and the RTC starts counting fees with nothing to notice.

        The market/name filter that `_net_energy_from` applies is deliberately *not* copied here.
        It was not in the code this replaces, and unlike the type filter it has not been measured.
        """
        if transactions is None or transactions.is_empty():
            return pl.DataFrame(schema=self.NET_SCHEMA)

        return (transactions
                .filter(pl.col(c.TC_TYPE_TRANSACTION).is_in(list(self.NET_TRANSACTION_TYPES)))
                .groupby([c.TC_TIMESTEP, c.TC_ID_AGENT])
                .agg(pl.col(c.TC_ENERGY_IN, c.TC_ENERGY_OUT).sum())
                .with_columns([pl.col(c.TC_ENERGY_IN).cast(pl.Int64),
                               pl.col(c.TC_ENERGY_OUT).cast(pl.Int64)]))

    def get_rtc_market_result(self, agent_id, timestep):
        """One agent's `energy_in - energy_out` at one timestep, as the RTC defines it.

        Replaces the second whole-table scan: every agent did this on every timestep too, and it
        is the larger of the two. Measured on design 6 (104 agents, three months), the RTC and FBC
        phase grew 2.6 s to 7.9 s per timestep over 140 steps purely because of it, while the
        trading phase -- already served from `get_net_energy` -- stayed at 0.25 s.

        Returns a plain int, matching what the scan returned, including 0 for an agent with no
        transactions at that timestep.
        """
        cache = self.rtc_cache
        source = cache if cache is not None else self._rtc_energy_from(self.market_transactions)

        rows = source.filter((pl.col(c.TC_ID_AGENT) == agent_id)
                             & (pl.col(c.TC_TIMESTEP) == timestep))
        if rows.is_empty():
            return 0
        return int(rows.select(pl.sum(c.TC_ENERGY_IN) - pl.sum(c.TC_ENERGY_OUT)).item())

    def get_net_energy(self, agent_id, first_timestep, last_timestep):
        """One agent's netted energy per timestep, over the closed window given.

        Replaces a scan of the whole `market_transactions` table, which every agent did on every
        timestep. That table is horizon-bounded but still large -- measured on a 104-agent design
        it reaches ~10 MB within 70 timesteps -- so the scan is O(table) per agent per timestep,
        and the agent stage grew 3.7 s to 10.4 s over 80 steps because of it.

        Falls back to computing from the full table when there is no cache, which is what makes
        the cache an optimisation rather than a second source of truth.
        """
        cache = self.net_cache
        source = cache if cache is not None else self._net_energy_from(self.market_transactions)

        return (source
                .filter((pl.col(c.TC_ID_AGENT) == agent_id)
                        & (pl.col(c.TC_TIMESTEP) >= first_timestep)
                        & (pl.col(c.TC_TIMESTEP) <= last_timestep))
                .select([c.TC_TIMESTEP, c.TC_ENERGY_IN, c.TC_ENERGY_OUT]))

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
