"""Integration -- the Analyzer's per-agent balancing across more than one market.

`MarketDataProcessor.process_agent_balancing` accumulates across markets: the first market
initialises the frame and every later one is added to it. **No scenario anywhere in the repository
has more than one market**, so `tests/e2e/test_analyzer_processors.py` cannot reach the `+=` and a
review panel changed it to `=` with the whole e2e suite green -- the per-agent balancing figure
would then have shown only whichever market `os.listdir` yielded last.

Built from two hand-written `market_transactions.ft` files rather than from a second scenario,
because the branch is about arithmetic across markets and a scenario run costs minutes to reach
the same two lines. `_get_market_transactions_for_scenario` walks the tree for that filename, so
a directory with two of them is all it takes.
"""
import pandas as pd
import pytest

import hamlet.constants as c
from hamlet.analyzer.markets.market_data_processor import MarketDataProcessor

AGENTS = ['agent_one', 'agent_two']


def write_market(folder, price_in, price_out):
    """One market's `market_transactions.ft`, one row per agent per transaction type.

    Written through pandas with `category` dtype, which is what a real results file holds --
    checked against one: `id_agent` and `type_transaction` arrive as an Arrow dictionary. That
    matters beyond fidelity, because the categorical grouping is the mechanism behind #229, and a
    fixture of plain strings would exercise a different `groupby`.
    """
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for agent in AGENTS:
        for transaction_type in (c.TT_RETAIL, c.TT_GRID):
            rows.append({c.TC_ID_AGENT: agent, c.TC_TYPE_TRANSACTION: transaction_type,
                         c.TC_TIMESTEP: pd.Timestamp('2021-03-23 23:00', tz='UTC'),
                         c.TC_PRICE_IN: price_in, c.TC_PRICE_OUT: price_out,
                         c.TC_ENERGY_IN: 100})
    frame = pd.DataFrame(rows)
    for column in (c.TC_ID_AGENT, c.TC_TYPE_TRANSACTION):
        frame[column] = frame[column].astype('category')
    frame.to_feather(folder / 'market_transactions.ft')


@pytest.fixture
def agent_balancing(tmp_path):
    """Per-agent balancing over a results tree holding `count` markets with distinct prices."""
    def build(count):
        markets = tmp_path / 'run' / 'markets' / 'electricity'
        for index in range(count):
            write_market(markets / f'market_{index}', price_in=10 ** index, price_out=2 * 10 ** index)
        processor = MarketDataProcessor(path={'run': str(tmp_path / 'run')},
                                        config={'markets': {'run': {}}})
        return processor.process_agent_balancing()['run']

    return build


def test_one_market_is_the_baseline(agent_balancing):
    """The single-market path, so the two-market assertion below has something to differ from."""
    balancing = agent_balancing(1)

    assert balancing is not None, 'no market transactions were read at all'
    assert len(balancing) == len(AGENTS) * 2
    assert balancing[c.TC_PRICE_IN].sum() == pytest.approx(4 * 1)


def test_every_market_is_added_rather_than_the_last_one_winning(agent_balancing):
    """The `+=` branch. With `=` the total is the last market's alone, not the sum.

    Three markets rather than two so that a mutation to `=` is distinguishable from one that keeps
    only the *first* -- with two markets and prices 1 and 10 both defects give a wrong total, but
    only a third market makes it obvious which.
    """
    balancing = agent_balancing(3)

    # 4 rows per market, prices 1 / 10 / 100 -> 4 * 111 summed across markets.
    assert balancing[c.TC_PRICE_IN].sum() == pytest.approx(4 * 111), (
        'the per-agent balancing did not accumulate across markets, so the figure shows one '
        'market rather than the scenario')
    assert balancing[c.TC_PRICE_OUT].sum() == pytest.approx(2 * 4 * 111)
    assert len(balancing) == len(AGENTS) * 2, (
        'accumulating across markets must not change the shape -- one row per agent and '
        'transaction type, however many markets were summed')
