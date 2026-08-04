"""Regression — retailer prices must be read as scalars, not broadcast as columns.

The market code wrote `pl.lit(series.alias(name))`, which applies the alias to the Series
*inside* the literal. With a single-row retailer frame that happens to produce the right
answer, so the defect is latent; with more than one row polars broadcasts the whole column
onto the transactions frame and each transaction gets a different price.
"""
import polars as pl
import pytest


def broadcast_pattern(retailer, target):
    """The old expression: alias applied inside pl.lit."""
    return target.with_columns([pl.lit(retailer['price'].alias('price_pu')).cast(pl.Int32)])


def scalar_pattern(retailer, target):
    """The corrected expression: take the scalar first, then alias."""
    return target.with_columns([pl.lit(retailer['price'][0]).alias('price_pu').cast(pl.Int32)])


@pytest.fixture
def transactions():
    return pl.DataFrame({'energy_in': [10, 20, 30]})


def test_single_row_retailer_is_unaffected(transactions):
    """Why the defect went unnoticed: with one retailer row both forms agree."""
    retailer = pl.DataFrame({'price': [2500]})

    assert (broadcast_pattern(retailer, transactions)['price_pu'].to_list()
            == scalar_pattern(retailer, transactions)['price_pu'].to_list())


def test_multi_row_retailer_must_not_vary_the_price_per_transaction(transactions):
    """Regression: a multi-row retailer frame leaked a different price into each row.

    Every transaction in a timestep faces the same retailer price. The old expression instead
    handed row i of the retailer frame to transaction i.
    """
    retailer = pl.DataFrame({'price': [2500, 9999, 1]})

    prices = scalar_pattern(retailer, transactions)['price_pu'].to_list()

    assert prices == [2500, 2500, 2500]
    # ... and this is what the old expression did instead
    assert broadcast_pattern(retailer, transactions)['price_pu'].to_list() == [2500, 9999, 1]
