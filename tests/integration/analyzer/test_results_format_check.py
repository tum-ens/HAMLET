"""Integration — the Analyzer refuses results it would misread.

The Executor copies the scenario folder into the results folder, so results carry the stamp of
the scenario that produced them. Plotting is where wrong numbers turn into figures, which is why
this reader refuses too rather than trusting that the Executor already checked.
"""
import json

import pytest

import hamlet.constants as c
from hamlet.analyzer.setup import Analyzer
from hamlet.functions import ScenarioFormatError


@pytest.fixture
def results(tmp_path):
    """A results folder containing only `general/general.json`."""
    def build(general, name='run'):
        folder = tmp_path / name
        (folder / 'general').mkdir(parents=True)
        (folder / 'general' / 'general.json').write_text(json.dumps(general), encoding='utf-8')
        return {name: str(folder)}

    return build


def stamped(version=c.SCENARIO_FORMAT_VERSION):
    return {c.K_SCENARIO_FORMAT_VERSION: version, 'structure': {}}


class TestUnreadableResultsAreRefused:

    def test_results_without_a_stamp_are_refused(self, results):
        with pytest.raises(ScenarioFormatError):
            Analyzer(path=results({'structure': {}}))

    @pytest.mark.parametrize('version', [c.SCENARIO_FORMAT_VERSION - 1,
                                         c.SCENARIO_FORMAT_VERSION + 1])
    def test_mismatched_results_are_refused(self, results, version):
        with pytest.raises(ScenarioFormatError):
            Analyzer(path=results(stamped(version)))

    def test_the_message_names_the_remedy(self, results):
        with pytest.raises(ScenarioFormatError) as error:
            Analyzer(path=results({'structure': {}}))

        assert 're-create this scenario' in str(error.value).lower()


class TestMatchingResultsAreRead:

    def test_the_check_passes_and_loading_continues(self, results):
        """No `config/setup.yaml` in the fixture, so getting that far proves the check passed."""
        with pytest.raises(FileNotFoundError) as error:
            Analyzer(path=results(stamped()))

        assert 'setup.yaml' in str(error.value)


class TestEveryResultsFolderIsChecked:
    """The Analyzer takes several folders at once and compares them against each other."""

    def test_one_bad_folder_among_good_ones_is_still_caught(self, results):
        paths = {**results(stamped(), name='good'),
                 **results({'structure': {}}, name='old')}

        with pytest.raises(ScenarioFormatError) as error:
            Analyzer(path=paths)

        assert 'old' in str(error.value)


class TestTheOverride:

    def test_incompatible_results_are_plotted_when_the_caller_insists(self, results):
        with pytest.raises(FileNotFoundError):
            Analyzer(path=results({'structure': {}}), allow_incompatible_scenario=True)
