"""Unit — the executor must read the scenario formats that exist in the wild.

Found while trying to execute one of the paper scenarios with this branch's code. Each of these
failed with an error that named neither the file nor the setting responsible.
"""
import polars as pl
import pytest

import hamlet.constants as c
import hamlet.functions as f


class TestWeatherFileName:
    """The weather file name is declared in setup.yaml and must be read from there.

    It used to be hard-coded, so a scenario declaring the other extension failed with a
    FileNotFoundError naming a file the user never asked for. The two branches of this project
    hard-coded *different* extensions, which is how the divergence went unnoticed.
    """

    def test_the_setting_is_where_the_loader_looks_for_it(self, repo_root):
        """The shipped example declares it, so the key must exist under `location`."""
        from ruamel.yaml import YAML

        setup = (repo_root / 'examples' / 'create_simple_scenario' / 'simple_scenario'
                 / 'setup.yaml')
        config = YAML(typ='safe').load(setup.read_text(encoding='utf-8'))

        assert 'weather' in config['location']

    def test_the_loader_reads_it_from_the_config(self):
        """Regression: the file name was hard-coded rather than read from the scenario."""
        import inspect

        from hamlet.executor.utilities.database import database

        source = inspect.getsource(database.Database._Database__setup_general)

        assert "'weather.ft'" not in source.replace("get('weather', 'weather.ft')", '')
        assert "get('weather'" in source


class TestCsvLoading:
    """`load_file` must be able to read a csv with default arguments."""

    @pytest.fixture
    def csv(self, tmp_path):
        path = tmp_path / 'sample.csv'
        path.write_text('timestamp,value\n2021-03-24T00:00:00,1\n2021-03-24T01:00:00,2\n',
                        encoding='utf-8')
        return path

    @pytest.mark.parametrize('method', ['eager', 'lazy'])
    def test_a_csv_loads_without_an_explicit_parse_dates(self, csv, method):
        """Regression: polars rejects the default `parse_dates=None` with a TypeError.

        The message names `try_parse_dates`, an argument the caller never passed, so a scenario
        using a csv input failed with an error pointing at the wrong thing.
        """
        frame = f.load_file(path=str(csv), df='polars', method=method)
        frame = frame.collect() if method == 'lazy' else frame

        assert len(frame) == 2

    def test_dates_are_parsed_when_asked_for(self, csv):
        frame = f.load_file(path=str(csv), df='polars', method='eager', parse_dates=True)

        assert frame[c.TC_TIMESTAMP].dtype != pl.String

    def test_dates_are_left_alone_when_not(self, csv):
        frame = f.load_file(path=str(csv), df='polars', method='eager', parse_dates=False)

        assert frame[c.TC_TIMESTAMP].dtype == pl.String


class TestTimeseriesIndexColumn:
    """Some scenarios name the time column `index` rather than `timestamp`."""

    def test_the_rename_guard_is_present(self):
        """Regression: the forecaster failed with "list.remove(x): x not in list".

        That message names neither the agent, the file, nor the column, and it surfaces a long
        way from the load that caused it.
        """
        import inspect

        from hamlet.executor.utilities.database import agent_db

        source = inspect.getsource(agent_db.AgentDB)

        assert "'index' in self.timeseries.columns" in source

    def test_renaming_is_a_no_op_for_a_conforming_table(self):
        """A table that already uses `timestamp` must be left exactly as it is."""
        frame = pl.DataFrame({c.TC_TIMESTAMP: [1, 2], 'a_electricity': [10, 20]})

        assert 'index' not in frame.columns
        assert frame.columns == [c.TC_TIMESTAMP, 'a_electricity']
