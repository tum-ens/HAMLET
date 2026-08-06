"""Unit — the market reads prices from exactly one retailer row.

Prices are read as `retailer[column][0]`. On a longer frame that silently applies the first
retailer's prices to every agent. Before the scalar read was introduced, polars raised a
ShapeError instead, so the loud failure has to be restored deliberately.
"""
import polars as pl
import pytest

import hamlet.constants as c
from hamlet.executor.markets.electricity import ElectricityMarket


def retailer(rows):
    """A retailer frame in the `to_dict()` shape the market works with."""
    return pl.DataFrame({
        f'{c.TT_BALANCING}_{c.TC_PRICE}_{c.PF_IN}': [400] * rows,
        f'{c.TT_BALANCING}_{c.TC_PRICE}_{c.PF_OUT}': [1600] * rows,
    }).to_dict()


def test_one_row_is_accepted():
    assert ElectricityMarket._single_retailer(retailer(1)) is not None


@pytest.mark.parametrize('rows', [0, 2, 5])
def test_anything_else_fails_loudly(rows):
    """Regression: reading row 0 of a longer frame priced everyone off the first retailer."""
    with pytest.raises(ValueError, match='exactly one retailer row'):
        ElectricityMarket._single_retailer(retailer(rows))


def test_the_error_says_what_to_look_for():
    """A duplicated timestamp in the retailer input is the realistic cause."""
    with pytest.raises(ValueError, match='duplicate timestamps'):
        ElectricityMarket._single_retailer(retailer(3))


def test_it_also_accepts_a_dataframe():
    """The market holds the retailer as a frame in places and as a dict in others."""
    frame = pl.DataFrame({'a': [1]})

    assert ElectricityMarket._single_retailer(frame) is frame
    with pytest.raises(ValueError):
        ElectricityMarket._single_retailer(pl.DataFrame({'a': [1, 2]}))
