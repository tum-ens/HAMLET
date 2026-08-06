"""Integration — the Creator stamps the scenario format into `general/general.json`.

One writer, two readers: the Creator writes this file, the Executor and the Analyzer read it.
The round-trip test at the bottom is the one that matters -- it pins that what the writer emits
is what the readers accept, so the two cannot drift apart silently.
"""
import json
import shutil

import pytest

import hamlet.constants as c
import hamlet.functions as f
from hamlet.creator.setup import Creator

EXAMPLE = 'simple_scenario'


@pytest.fixture
def creator(tmp_path, repo_root):
    """A real Creator over a temp copy of the shipped example config.

    Only `__create_general_files` is run: it needs no agents, no markets and no grid, so this
    stays a fast test of the one file the readers depend on.
    """
    config = tmp_path / EXAMPLE
    shutil.copytree(repo_root / 'examples' / 'create_simple_scenario' / EXAMPLE, config)

    setup = config / 'setup.yaml'
    text = setup.read_text(encoding='utf-8')
    for old, new in (('input: ../../input_data', f'input: {(repo_root / "input_data").as_posix()}'),
                     ('scenarios: ../../scenarios', f'scenarios: {(tmp_path / "scenarios").as_posix()}'),
                     ('results: ../../results', f'results: {(tmp_path / "results").as_posix()}')):
        assert old in text, f'{old!r} not found in setup.yaml'
        text = text.replace(old, new)
    setup.write_text(text, encoding='utf-8')

    return Creator(path=str(config))


@pytest.fixture
def general(creator, tmp_path):
    """The `general.json` the Creator writes, parsed back off disk."""
    folder = tmp_path / 'scenarios' / EXAMPLE / 'general'
    folder.mkdir(parents=True)

    creator._Creator__create_general_files()

    return json.loads((folder / 'general.json').read_text(encoding='utf-8'))


class TestTheStampIsWritten:

    def test_general_json_carries_a_format_version(self, general):
        assert c.K_SCENARIO_FORMAT_VERSION in general

    def test_it_is_the_current_version(self, general):
        assert general[c.K_SCENARIO_FORMAT_VERSION] == c.SCENARIO_FORMAT_VERSION

    def test_it_is_a_plain_integer(self, general):
        """Not the tool version and not a string -- an incrementing integer compares cleanly."""
        assert isinstance(general[c.K_SCENARIO_FORMAT_VERSION], int)


class TestTheRestOfTheFileIsUnchanged:

    def test_the_structure_is_still_written(self, general):
        """The stamp is added alongside `structure`, not in place of it."""
        assert 'structure' in general
        assert general['structure']

    def test_nothing_else_was_added(self, general):
        assert set(general) == {c.K_SCENARIO_FORMAT_VERSION, 'structure'}


class TestTheReadersAcceptWhatTheCreatorWrote:
    """The point of the whole exercise: writer and readers agree, and a test says so."""

    def test_the_check_passes_on_a_freshly_written_general_file(self, general):
        assert f.check_scenario_format(general, 'scenario') == c.SCENARIO_FORMAT_VERSION
