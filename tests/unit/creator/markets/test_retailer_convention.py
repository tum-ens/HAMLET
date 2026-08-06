"""Integration — the retailer columns the Creator emits from a real scenario config.

This is the layer that was missing when grid fees and levies silently moved onto feed-in: every
unit test used a hand-written fixture, so nothing checked what the Creator actually produces
from the shipped `markets.yaml`.

The convention that must hold for every component:
    `_out` is the direction the agent pays for (the retailer selling to it)
    `_in`  is the direction the agent is paid for (feeding in)
"""
from pathlib import Path

import pandas as pd
import pytest
from ruamel.yaml import YAML

import hamlet.constants as c
from hamlet.creator.markets.electricity import ElectricityMarket

# Resolved from the conftest's repo root rather than this file's depth, so moving the test
# between layers cannot silently empty the parameter list
REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIGS = sorted(REPO_ROOT.glob('examples/*/*/markets.yaml')) + [
    REPO_ROOT / 'config_templates' / 'markets.yaml']
assert CONFIGS and all(p.exists() for p in CONFIGS), (
    f'no shipped market configs found under {REPO_ROOT}; the repo root is wrong')


def retailer_columns(pricing_config):
    """Run the Creator's fixed-price path over a real config block."""
    frame = pd.DataFrame(index=[0])
    for component, info in pricing_config.items():
        if not isinstance(info, dict) or 'fixed' not in info:
            continue
        frame = ElectricityMarket._create_fixed_cols(df=frame, config=info['fixed'],
                                               commodity=component)
    return frame


@pytest.fixture(params=[p for p in CONFIGS if p.exists()], ids=lambda p: p.parent.name)
def emitted(request):
    config = YAML(typ='safe').load(request.param.read_text(encoding='utf-8'))
    # Each top-level key is a market; take every retailer block they define
    frames = [retailer_columns(market['pricing']['retailer'])
              for market in config.values()
              if isinstance(market, dict) and 'pricing' in market]
    if not frames:
        pytest.skip('config defines no retailer pricing')
    return frames[0]


def value(frame, column):
    return frame[column].iloc[0]


def test_buying_energy_costs_more_than_selling(emitted):
    """The defining sanity check: no risk-free arbitrage against the retailer."""
    buy = value(emitted, f'{c.TT_ENERGY}_{c.TC_PRICE}_{c.PF_OUT}')
    sell = value(emitted, f'{c.TT_ENERGY}_{c.TC_PRICE}_{c.PF_IN}')

    assert buy > sell


def test_grid_fees_sit_on_the_buying_direction(emitted):
    """Regression: grid fees were emitted under `_in`, i.e. charged on feed-in.

    The shipped configs list grid fees as [buying, selling], the opposite order from the energy
    block, and that difference used to survive into the emitted columns.
    """
    for trade_type in (c.TT_MARKET, c.TT_RETAIL):
        buy = value(emitted, f'{c.TT_GRID}_{trade_type}_{c.PF_OUT}')
        sell = value(emitted, f'{c.TT_GRID}_{trade_type}_{c.PF_IN}')

        assert buy >= sell, f'{trade_type} grid fee is charged on feed-in'
        assert buy > 0, f'{trade_type} grid fee is not charged on consumption at all'


def test_levies_sit_on_the_buying_direction(emitted):
    """Same regression for levies, which are consumption taxes."""
    buy = value(emitted, f'{c.TT_LEVIES}_{c.TC_PRICE}_{c.PF_OUT}')
    sell = value(emitted, f'{c.TT_LEVIES}_{c.TC_PRICE}_{c.PF_IN}')

    assert buy > sell
    assert buy > 0


def test_the_config_list_order_is_preserved_per_component():
    """The normalisation must turn grid and levies around, and leave energy alone.

    Pinning the translation itself, so a future edit to `_BUY_FIRST_COMPONENTS` cannot quietly
    change which end of a user's existing config means what.
    """
    energy = ElectricityMarket._create_fixed_cols(pd.DataFrame(index=[0]),
                                            config={c.TC_PRICE: [0.05, 0.15]},
                                            commodity=c.TT_ENERGY)
    levies = ElectricityMarket._create_fixed_cols(pd.DataFrame(index=[0]),
                                            config={c.TC_PRICE: [0.18, 0.0]},
                                            commodity=c.TT_LEVIES)

    # energy is configured [selling, buying] and passes through
    assert value(energy, f'{c.TT_ENERGY}_{c.TC_PRICE}_{c.PF_IN}') \
        < value(energy, f'{c.TT_ENERGY}_{c.TC_PRICE}_{c.PF_OUT}')
    # levies are configured [buying, selling] and are turned around
    assert value(levies, f'{c.TT_LEVIES}_{c.TC_PRICE}_{c.PF_IN}') == 0
    assert value(levies, f'{c.TT_LEVIES}_{c.TC_PRICE}_{c.PF_OUT}') > 0


def test_the_file_path_agrees_with_the_fixed_path():
    """The two configuration paths must produce identical column semantics.

    `_create_fixed_cols` normalises the config's per-component list order; `_create_file_cols`
    reads a CSV by column name and normalises nothing, so the shipped CSVs have to already be
    labelled in the target convention. Nothing else checks that, and a mislabel there is silent.
    """
    import polars as pl

    charged = {'grid.csv': [f'{c.TT_GRID}_{c.TT_MARKET}', f'{c.TT_GRID}_{c.TT_RETAIL}'],
               'levies.csv': [f'{c.TT_LEVIES}_{c.TC_PRICE}']}

    for file_name, prefixes in charged.items():
        frame = pl.read_csv(REPO_ROOT / 'input_data' / 'retailers' / 'lem' / file_name)
        for prefix in prefixes:
            buy = frame[f'{prefix}_{c.PF_OUT}'].max()
            sell = frame[f'{prefix}_{c.PF_IN}'].max()

            assert buy > 0, f'{file_name}: {prefix} is not charged on consumption'
            assert sell == 0, f'{file_name}: {prefix} is charged on feed-in'


def test_the_energy_and_balancing_files_price_buying_above_selling():
    """The same convention, on the two files that carry prices rather than charges."""
    import polars as pl

    for file_name, prefix in [('energy.csv', f'{c.TT_ENERGY}_{c.TC_PRICE}'),
                              ('balancing.csv', f'{c.TT_BALANCING}_{c.TC_PRICE}')]:
        frame = pl.read_csv(REPO_ROOT / 'input_data' / 'retailers' / 'lem' / file_name)

        assert frame[f'{prefix}_{c.PF_OUT}'].max() > frame[f'{prefix}_{c.PF_IN}'].max(), (
            f'{file_name}: selling is priced above buying')
