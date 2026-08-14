"""Integration -- `load_file` must read an XLSX as either frame type.

The polars branch (`hamlet/functions.py:189`) was dead on arrival in every environment this
repository has ever shipped. `pl.read_excel` needs `xlsx2csv`, which was in neither `env.yml` nor
the conda environment developers actually used, so the call raised

    ModuleNotFoundError: required package 'xlsx2csv' not found

instead of reading anything. Nothing in the test suite touched it, so it stayed broken silently.

`xlsx2csv` is now a declared dependency. This is the test that says so, using a shipped example
file rather than a fixture, because the point is that a real config parses.
"""
import pandas as pd
import polars as pl
import pytest

from hamlet.functions import load_file

EXAMPLE_XLSX = ('examples', 'create_scenario_with_grid', 'scenario_with_grid', 'agents.xlsx')


@pytest.fixture
def agents_xlsx(repo_root):
    path = repo_root.joinpath(*EXAMPLE_XLSX)
    if not path.is_file():
        pytest.skip(f'shipped example not present: {path}')
    return path


def test_load_file_reads_xlsx_as_polars(agents_xlsx):
    """The branch that used to raise. A DataFrame with rows and columns, not an exception."""
    frame = load_file(str(agents_xlsx), df='polars')

    assert isinstance(frame, pl.DataFrame)
    assert frame.height > 0, 'read an empty frame -- the engine silently produced nothing'
    assert frame.width > 0


def test_load_file_reads_xlsx_as_pandas(agents_xlsx):
    """The branch that always worked, asserted alongside so a regression names which one broke."""
    assert isinstance(load_file(str(agents_xlsx), df='pandas'), pd.ExcelFile)


def test_load_file_rejects_an_unknown_frame_type(agents_xlsx):
    with pytest.raises(ValueError):
        load_file(str(agents_xlsx), df='numpy')
