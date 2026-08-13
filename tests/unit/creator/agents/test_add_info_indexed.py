"""Unit — `_add_info_indexed` must descend into nested config, and dispatch per leaf.

The Creator wrote `NaN` into every nested `charging_scheme` parameter for `ctsp` and `industry`
agents **and reported success** (#219). `_add_info_indexed` iterated `config.items()` at the top
level only, so a dict value matched no column and was skipped in silence; `_add_info_simple`
(`agents.py:1941`) recurses, and `sfh.py:1041` uses that one for exactly this key.

**Both obvious repairs are wrong, and each is pinned below.** The block the two classes pass is the
only *mixed* one in the repository -- `method` is a per-agent distribution list and the nested
leaves are scalars:

    method                     ['full', 'min_soc']   <- indexed, one draw per agent
    min_soc/val                0.5                   <- scalar, the same for everyone

so recursion alone reaches `len(0.5)` and raises, and copying `sfh`'s `_add_info_simple` call
writes the *list itself* into every row of `method` -- replacing a silent NaN with a silent
`"['full', 'min_soc']"`. `test_a_distribution_list_is_still_drawn_per_agent` is the one that fails
for that second repair, and it is the reason this file exists rather than a one-line diff.

Audited before the helper was changed: across every `agents.yaml` in the repository the other 33
call sites pass `sizing` (501 leaves) and `parameters` (52 leaves), all of them lists and none of
them nested -- so recursion and the type dispatch are both provable no-ops for them. That audit is
re-derived here by `test_no_other_call_site_passes_a_nested_or_scalar_config` rather than restated,
because a statement in a docstring cannot fail.
"""
import pathlib

import numpy as np
import pandas as pd
import pytest
from ruamel.yaml import YAML

from hamlet.creator.agents.agents import Agents

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]

#: The `charging_scheme` block `industry` passes, verbatim from the fixture. Mixed on purpose.
SCHEME = {
    'method': ['full', 'min_soc'],
    'price_sensitive': {'threshold': 0.05},
    'min_soc': {'val': 0.5},
    'min_soc_time': {'val': 0.8, 'time': 8.5},
}

#: The columns the Creator emits for it -- read off the fixture's workbook, nested leaves and all.
COLUMNS = ['ev/charging_scheme/method',
           'ev/charging_scheme/price_sensitive/threshold',
           'ev/charging_scheme/min_soc/val',
           'ev/charging_scheme/min_soc_time/val',
           'ev/charging_scheme/min_soc_time/time']

#: The nested ones, which are what #219 left empty.
NESTED = [column for column in COLUMNS if column.count('/') > 2]


class _Probe(Agents):
    """The helpers under test are inherited; nothing in `Agents.__init__` is needed to drive them."""

    def __init__(self):
        pass


@pytest.fixture
def probe():
    return _Probe()


@pytest.fixture
def frame():
    return pd.DataFrame(index=range(3), columns=COLUMNS, dtype=object)


def filled(frame, column):
    """The values written to `column`, with NaN spelled out rather than compared with `==`."""
    return [None if isinstance(value, float) and np.isnan(value) else value
            for value in frame[column]]


@pytest.mark.parametrize('column', NESTED)
def test_every_nested_parameter_is_written(probe, frame, column):
    """Regression for #219, asserted per column.

    Per column rather than as a count: the defect skipped four of them at once, so "some nested
    parameter was written" would have been satisfied by any single one of them landing.
    """
    probe._add_info_indexed(keys=['ev', 'charging_scheme'], config=SCHEME,
                            idx_list=[0, 1, 0], df=frame)

    assert None not in filled(frame, column), (
        f'{column} was left NaN; the Creator would report success and the Executor would then '
        f'raise IntCastingNaNError on it (#219)')


def test_the_nested_values_are_the_ones_the_config_asked_for(probe, frame):
    """And they are the *right* values -- writing zeroes would satisfy the test above."""
    probe._add_info_indexed(keys=['ev', 'charging_scheme'], config=SCHEME,
                            idx_list=[0, 1, 0], df=frame)

    assert filled(frame, 'ev/charging_scheme/price_sensitive/threshold') == [0.05] * 3
    assert filled(frame, 'ev/charging_scheme/min_soc/val') == [0.5] * 3
    assert filled(frame, 'ev/charging_scheme/min_soc_time/val') == [0.8] * 3
    assert filled(frame, 'ev/charging_scheme/min_soc_time/time') == [8.5] * 3


def test_a_distribution_list_is_still_drawn_per_agent(probe, frame):
    """The top-level list keeps indexed semantics -- this is what the `_add_info_simple` swap breaks.

    Copying `sfh.py:1041` fixes the nested columns and silently stringifies this one into
    `"['full', 'min_soc']"` for every agent, which is the same class of defect as #219 one level
    down. The three agents here draw indices 0, 1, 0, so the answer has to vary between them.
    """
    probe._add_info_indexed(keys=['ev', 'charging_scheme'], config=SCHEME,
                            idx_list=[0, 1, 0], df=frame)

    assert filled(frame, 'ev/charging_scheme/method') == ['full', 'min_soc', 'full']


def test_a_scalar_leaf_respects_ownership(probe):
    """A non-owner keeps NaN, as it does for every other column this helper writes.

    `idx_list` carries NaN for agents without the device. Broadcasting the scalar instead would
    hand `min_soc/val` to an agent that owns no EV while `method` beside it stayed NaN -- half a
    row, and the kind of inconsistency that is found much later.
    """
    frame = pd.DataFrame(index=range(3), columns=COLUMNS, dtype=object)

    probe._add_info_indexed(keys=['ev', 'charging_scheme'], config=SCHEME,
                            idx_list=[0, np.nan, 1], df=frame)

    assert filled(frame, 'ev/charging_scheme/min_soc/val') == [0.5, None, 0.5]
    assert filled(frame, 'ev/charging_scheme/method') == ['full', None, 'min_soc']


def test_a_column_that_does_not_exist_is_still_skipped(probe):
    """The membership check is load-bearing and recursion must not bypass it."""
    frame = pd.DataFrame(index=range(3), columns=['ev/charging_scheme/method'], dtype=object)

    probe._add_info_indexed(keys=['ev', 'charging_scheme'], config=SCHEME,
                            idx_list=[0, 1, 0], df=frame)

    assert list(frame.columns) == ['ev/charging_scheme/method']


def test_the_appendix_still_reaches_the_leaf(probe):
    """`sizing` calls pass `appendix=f'_{num}'`; recursion must carry it through, not drop it."""
    frame = pd.DataFrame(index=range(2), columns=['ev/charging_scheme/min_soc/val_0'], dtype=object)

    probe._add_info_indexed(keys=['ev', 'charging_scheme'], config={'min_soc': {'val': 0.5}},
                            idx_list=[0, 0], df=frame, appendix='_0')

    assert filled(frame, 'ev/charging_scheme/min_soc/val_0') == [0.5, 0.5]


def test_no_other_call_site_passes_a_nested_or_scalar_config():
    """The Part 1 audit, re-derived from the tree so it fails when the tree stops agreeing.

    The 35 `_add_info_indexed` call sites read exactly three config subkeys. `charging_scheme` is
    the only one that is ever nested, and the only one with scalar leaves; `sizing` and
    `parameters` are lists throughout. That is *why* changing the shared helper could not move a
    golden reference, so it is checked rather than believed -- if a future config nests a `sizing`
    block, this fails and the blast-radius argument gets re-made instead of assumed.

    Derived by walking every `agents.yaml`, not from a list of files: a list would pass by
    omission the moment someone adds a scenario.
    """
    yaml = YAML(typ='safe')
    offenders = {'nested': [], 'scalar': []}

    def walk(node, path, source):
        if isinstance(node, dict):
            for key, value in node.items():
                here = path + [str(key)]
                if key in ('sizing', 'parameters') and isinstance(value, dict):
                    for leaf, leaf_value in value.items():
                        where = f'{source}:{"/".join(here + [str(leaf)])}'
                        if isinstance(leaf_value, dict):
                            offenders['nested'].append(where)
                        elif not isinstance(leaf_value, list):
                            offenders['scalar'].append(where)
                walk(value, here, source)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, path + [f'[{index}]'], source)

    configs = sorted(path for path in REPO_ROOT.rglob('agents.yaml') if '.git' not in path.parts)
    assert configs, 'no agents.yaml found; this test would pass by finding nothing'

    for config in configs:
        walk(yaml.load(config.read_text(encoding='utf-8')), [], config.relative_to(REPO_ROOT).as_posix())

    assert offenders == {'nested': [], 'scalar': []}, (
        'a `sizing` or `parameters` block is no longer a flat list of distributions, so the #219 '
        'audit no longer holds: changing `_add_info_indexed` can now move these scenarios. '
        f'{offenders}')
