"""Does every `grids.yaml` this repository ships still load under the §14a strategy reader (#245)?

`direct_power_control.reduction` and `ramp_steps` are additive: a scenario that names neither is
`required`, and one that names them must name something the reader accepts. That is a property of
the shipped files, so it is checked against the shipped files. What the reader accepts and refuses
is pinned in `tests/unit/executor/test_enwg_14a_reduction_strategy.py`.

Here rather than in `unit/` because it reads files, which `tests/README.md` makes the criterion.
"""
from types import SimpleNamespace

import pytest
import yaml

from hamlet.executor.utilities.grid_restrictions.enwg_14a import EnWG14a
from tests.integration.executor.test_variable_grid_fee_config_horizons import (
    _repo_root, _tracked_grids_yaml)


def _direct_power_control_blocks():
    """Every tracked `grids.yaml`'s `direct_power_control` block, and what the reader must say."""
    for tracked in _tracked_grids_yaml():
        grids = _repo_root() / tracked
        for energy_type, section in (yaml.safe_load(grids.read_text(encoding='utf-8'))
                                     or {}).items():
            block = ((section or {}).get('restrictions', {}) or {}) \
                .get('enwg_14a', {}).get('direct_power_control')
            if block is not None:
                yield f'{tracked} [{energy_type}]', block


BLOCKS = list(_direct_power_control_blocks())


def reader(block):
    enwg = EnWG14a.__new__(EnWG14a)
    enwg.restriction_config = {'direct_power_control': block}
    enwg.grid_db = SimpleNamespace(restriction_commands={})
    return enwg._EnWG14a__reduction_strategy


def test_the_config_sweep_found_something_to_check():
    assert BLOCKS, 'no shipped grids.yaml carries a direct_power_control block'


@pytest.mark.parametrize('label, block', BLOCKS, ids=[label for label, _ in BLOCKS])
def test_every_shipped_config_is_accepted(label, block):
    reader(block)()


@pytest.mark.parametrize('label, block', BLOCKS, ids=[label for label, _ in BLOCKS])
def test_every_shipped_config_runs_the_default_strategy(label, block):
    """Nothing shipped opts in. If one does, the golden reference for it moves, and that is the
    signal to say so in the CHANGELOG rather than let this test be edited to agree."""
    assert reader(block)() == ('required', None), (
        f'{label} selects a non-default reduction strategy')


@pytest.mark.parametrize('label, block', BLOCKS, ids=[label for label, _ in BLOCKS])
def test_a_config_without_the_keys_reads_as_the_default(label, block):
    """The scenario format was not bumped, so a pre-#245 file must keep loading as `required`."""
    stripped = {key: value for key, value in block.items() if key not in ('reduction',
                                                                            'ramp_steps')}
    assert reader(stripped)() == ('required', None)
