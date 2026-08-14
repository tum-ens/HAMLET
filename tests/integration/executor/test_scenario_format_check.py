"""Integration — the Executor refuses a scenario folder it would misread.

The check has to happen before anything else is read out of the folder, because everything else
in the folder loads perfectly well; it is the *meaning* of what is loaded that differs.
"""
import json

import pytest

import hamlet.constants as c
from hamlet.executor.setup import Executor
from hamlet.functions import ScenarioFormatError


@pytest.fixture
def scenario(tmp_path):
    """A scenario folder containing only `general/general.json`.

    Enough to reach the format check and nothing more, so a test that gets past it fails on the
    next missing file -- which is how these tests tell "accepted" from "rejected".
    """
    def build(general):
        folder = tmp_path / 'scenario'
        (folder / 'general').mkdir(parents=True)
        (folder / 'general' / 'general.json').write_text(json.dumps(general), encoding='utf-8')
        return folder

    return build


def prepare(folder, **kwargs):
    """Run the Executor far enough to load `general.json` and check it."""
    executor = Executor(str(folder), **kwargs)
    executor._Executor__prepare_scenario()


class TestAnUnstampedScenarioIsRefused:
    """The pre-versioning case: a scenario generated before the in/out convention was fixed."""

    def test_it_raises(self, scenario):
        folder = scenario({'structure': {}})

        with pytest.raises(ScenarioFormatError):
            prepare(folder)

    def test_the_message_names_the_scenario_and_the_remedy(self, scenario):
        folder = scenario({'structure': {}})

        with pytest.raises(ScenarioFormatError) as error:
            prepare(folder)

        assert 'scenario' in str(error.value)
        assert 're-create this scenario' in str(error.value).lower()


class TestAMismatchedScenarioIsRefused:

    @pytest.mark.parametrize('version', [c.SCENARIO_FORMAT_VERSION - 1,
                                         c.SCENARIO_FORMAT_VERSION + 1])
    def test_neither_direction_is_read(self, scenario, version):
        folder = scenario({c.K_SCENARIO_FORMAT_VERSION: version, 'structure': {}})

        with pytest.raises(ScenarioFormatError):
            prepare(folder)


class TestAMatchingScenarioIsRead:

    def test_the_check_passes_and_loading_continues(self, scenario):
        """It must get past the check and on to the rest of the scenario.

        The folder has no `config/setup.yaml`, so the FileNotFoundError is the proof that the
        format check accepted it rather than short-circuiting the load.
        """
        folder = scenario({c.K_SCENARIO_FORMAT_VERSION: c.SCENARIO_FORMAT_VERSION,
                           'structure': {}})

        with pytest.raises(FileNotFoundError) as error:
            prepare(folder)

        assert 'setup.yaml' in str(error.value)


class TestTheOverride:

    def test_an_incompatible_scenario_runs_when_the_caller_insists(self, scenario):
        """`allow_incompatible_scenario=True` is a keyword argument, not an environment variable.

        It therefore appears in the user's run script and in review, where a suppressed
        correctness check belongs.
        """
        folder = scenario({'structure': {}})

        with pytest.raises(FileNotFoundError):
            prepare(folder, allow_incompatible_scenario=True)

    def test_the_default_is_to_refuse(self, scenario):
        folder = scenario({'structure': {}})

        with pytest.raises(ScenarioFormatError):
            prepare(folder, allow_incompatible_scenario=False)
