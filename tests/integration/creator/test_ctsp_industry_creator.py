"""The Creator's `Ctsp` and `Industry` classes, which nothing else executes.

**This exists because the e2e fixture does not cover them, and it is easy to believe otherwise.**
`tests/e2e/scenarios/ctsp_industry/` is built with `new_scenario_from_files`, and that entry point
reads `agents.xlsx` through `Agents.create_agents_from_file`, which never consults `Agents.types` —
the registry that holds the Creator's per-agent-type classes (`agents.py:139`, used only at `:208`
and `:267`). Traced on a real run of that fixture: the classes instantiated are
`executor.Ctsp` and `executor.Industry` and **no Creator class at all**. So the e2e fixture answers
"do these agent types simulate", and this file answers "does the Creator build them", and neither
substitutes for the other.

Which matters because `hamlet/creator/agents/ctsp.py` and `industry.py` are ~950 lines each and
near-identical -- 48 changed lines across 13 hunks (#213) -- with four behavioural divergences and
one defect found by reading (#213, #212).
A deduplication needs an oracle, and this is the cheap half of one: it runs the Creator over a
config declaring both types and pins the **shape** of the workbook each class writes — which
columns, per sheet — in the style of `test_scenario_format_shape.py`.

Deliberately shape, not values. The sizings come from a seeded RNG and from `_pick_files`, and a
test that fails for two unrelated reasons tells you less than two that each fail for one. What this
does pin is that both classes ran, that each wrote its own sheet, and that the two sheets differ
only where the classes are known to differ.
"""
import random
import shutil

import numpy as np
import pandas as pd
import pytest

from tests.scenario_run import REPO_ROOT, SEED

FIXTURE = REPO_ROOT / 'tests' / 'e2e' / 'scenarios' / 'ctsp_industry'

#: The agent types the fixture declares, and so the sheets the Creator must write.
TYPES = ('ctsp', 'industry')


@pytest.fixture(scope='module')
def created(tmp_path_factory):
    """Run the Creator over a copy of the fixture config, recording which classes it instantiated.

    `new_scenario_from_configs` rather than `new_scenario_from_files`: only that entry point builds
    the agents *from the YAML through the per-type classes*, which is the code under test here. It
    also rewrites `agents.xlsx` in the config folder, which is why this runs against a copy — see
    `.ai/context.md` on the shipped examples being dirtied by their own Creator.

    **Which classes ran is recorded by wrapping them, not inferred from the output.** A shape
    assertion alone would pass if some other code path happened to write the same columns, and the
    claim this file exists to support is specifically that *these two classes* execute.
    """
    from hamlet import Creator
    import hamlet.creator.agents.ctsp as ctsp_module
    import hamlet.creator.agents.industry as industry_module

    base = tmp_path_factory.mktemp('ctsp_industry_creator')
    config = base / 'ctsp_industry'
    shutil.copytree(FIXTURE, config)
    (base / 'scenarios').mkdir()
    (base / 'results').mkdir()

    setup = config / 'setup.yaml'
    text = setup.read_text(encoding='utf-8')
    for old, new in (('input: ../../input_data', f'input: {(REPO_ROOT / "input_data").as_posix()}'),
                     ('scenarios: ../../scenarios', f'scenarios: {(base / "scenarios").as_posix()}'),
                     ('results: ../../results', f'results: {(base / "results").as_posix()}')):
        assert old in text, f'{old!r} not found in setup.yaml'
        text = text.replace(old, new)
    setup.write_text(text, encoding='utf-8')

    instantiated = set()
    originals = {}
    for name, cls in (('ctsp', ctsp_module.Ctsp), ('industry', industry_module.Industry)):
        originals[cls] = cls.__init__

        def wrapped(self, *args, _name=name, _original=originals[cls], **kwargs):
            instantiated.add(_name)
            return _original(self, *args, **kwargs)

        cls.__init__ = wrapped

    try:
        random.seed(SEED)
        np.random.seed(SEED)
        Creator(path=str(config)).new_scenario_from_configs()
    finally:
        for cls, original in originals.items():
            cls.__init__ = original

    return instantiated, config / 'agents.xlsx'


def sheet_columns(path):
    """Each sheet's column list, read the way `create_agents_from_file` reads the workbook."""
    columns = {}
    with pd.ExcelFile(path) as book:
        for sheet in book.sheet_names:
            columns[sheet] = list(book.parse(sheet, index_col=0).columns)
    return columns


def test_the_creator_instantiated_both_classes(created):
    """The claim this file exists for, recorded by the classes themselves.

    Asserted per type rather than as a count: the two classes are 92 % identical, so "two agent
    classes ran" would be satisfied by the same one running twice, and telling those apart is the
    entire reason this coverage is wanted before a dedupe.
    """
    instantiated, _ = created

    assert instantiated == set(TYPES), (
        f'the Creator instantiated {sorted(instantiated) or "no per-type class"}; expected both '
        f'{list(TYPES)}. Without both, this file pins nothing about the class that did not run')


def test_each_type_got_its_own_sheet(created):
    """One sheet per agent type — `create_agents_file_from_config` writes them that way."""
    _, workbook = created

    assert workbook.exists(), 'the Creator wrote no agents.xlsx'
    assert sorted(sheet_columns(workbook)) == sorted(TYPES)


@pytest.mark.parametrize('agent_type', TYPES)
def test_the_sheet_carries_the_columns_this_agent_type_is_built_from(created, agent_type):
    """Each class wrote a usable sheet, checked per type and per column group.

    Named column groups rather than a count: a count is satisfied by any 100 columns, and the
    failure this guards against is one class quietly dropping a device group during a dedupe.
    """
    _, workbook = created
    columns = sheet_columns(workbook)[agent_type]

    missing = [prefix for prefix in ('general/agent_id', 'general/parameters/area',
                                     'inflexible-load/owner', 'inflexible-load/sizing/demand_0',
                                     'pv/owner', 'battery/owner',
                                     'ems/controller/rtc/optimization/framework')
               if not any(str(column).startswith(prefix) for column in columns)]

    assert not missing, f'the {agent_type} sheet is missing {missing}; it has {len(columns)} columns'


#: Columns one sheet has and the other does not, with the reason for each. **This is a record of
#: known divergence, not an approval of it.** Every entry was found by this test failing.
#:
#: It used to hold five more entries per sheet, all of them the "the change landed in two of three
#: copies" pattern of #212: `config_templates`' **ctsp** block carried the pre-nesting flat
#: `charging_scheme` sub-keys (`min_soc_val`) where `sfh` and `industry` carry
#: `min_soc: {val: ...}` — which is the form the Executor reads — and named its EV forecast
#: sub-block `random_forest_classifier:` where the registered model is `rfr`. Both were **#218**,
#: and fixing them is what let this fixture turn its EV share on at all. `ctsp` is now empty, and
#: keeping the key rather than dropping it is deliberate: it is where the next ctsp-only column
#: shows up.
#:
#: What is left is a genuine config choice: `industry` allows two EVs per agent (`num: [1, 2]`), so
#: it gets a second set of `ev/sizing/*_1` columns.
#:
#: A dedupe of these two classes has to decide each of these explicitly (#213). Until then, this
#: constant is what makes the difference visible in review instead of invisible in a diff.
EXPECTED_SHEET_DIFFERENCE = {
    'ctsp': [],
    'industry': [
        'ev/sizing/capacity_1',
        'ev/sizing/charging_AC_1',
        'ev/sizing/charging_DC_1',
        'ev/sizing/charging_efficiency_1',
        'ev/sizing/charging_home_1',
        'ev/sizing/file_1',
        'ev/sizing/soc_1',
        'ev/sizing/v2g_1',
        'ev/sizing/v2h_1',
    ],
}


def test_the_two_sheets_differ_only_where_they_are_known_to(created):
    """The dedupe oracle: the column sets are compared and the difference is pinned as data.

    `ctsp` and `industry` are near-identical (#213) and a dedupe has to preserve whatever is
    genuinely different while fixing what is not. Pinning the difference means a collapse that quietly moves
    a column shows up as a changed expectation in the same commit, rather than as nothing at all.

    **Every entry in `EXPECTED_SHEET_DIFFERENCE` was discovered by this assertion failing**, which
    is the argument for writing it this way round: an equality assertion would have been "fixed" by
    deleting it. Two of the entries are defects — see the constant.
    """
    columns = sheet_columns(created[1])
    difference = {'ctsp': sorted(set(columns['ctsp']) - set(columns['industry'])),
                  'industry': sorted(set(columns['industry']) - set(columns['ctsp']))}

    assert difference == EXPECTED_SHEET_DIFFERENCE, (
        'the two sheets differ somewhere new, or no longer differ where they did. Both classes are '
        'built from the same device groups, so a change here is either a config edit or a '
        'divergence between the classes — decide which, and update the constant with the reason '
        '(#213).\n'
        f'  found:    {difference}\n'
        f'  expected: {EXPECTED_SHEET_DIFFERENCE}')


def test_the_two_sheets_still_agree_on_everything_else(created):
    """And the shared majority is asserted separately, so the pinned difference cannot grow silently.

    Without this, adding an entry to `EXPECTED_SHEET_DIFFERENCE` is a way to make any future
    divergence pass. The shared column count is the counterweight: it has to stay large, and a
    dedupe that started dropping columns from one sheet would show up here rather than as a longer
    exception list.
    """
    columns = sheet_columns(created[1])
    shared = set(columns['ctsp']) & set(columns['industry'])

    assert len(shared) >= 140, (
        f'the two sheets now share only {len(shared)} columns, against 145 when this was written. '
        f'A dedupe should increase this number, not reduce it')


def test_the_generated_workbook_matches_the_committed_one_in_shape(created):
    """The committed fixture workbook is what the Creator would write from the committed YAML.

    Without this the two can drift: `agents.yaml` is edited, the workbook is not regenerated, and
    the e2e fixture goes on running a configuration its own YAML no longer describes. That is #214
    inside the fixture — and `test_shipped_configs_agree_with_their_workbooks.py` only compares the
    backend keys, not the shape.

    Shape only. The committed workbook holds seeded-random agent ids and file choices that a fresh
    run redraws, and pinning those would make this fail for reasons that are not drift.
    """
    _, generated = created

    assert sheet_columns(generated) == sheet_columns(FIXTURE / 'agents.xlsx'), (
        'the committed tests/e2e/scenarios/ctsp_industry/agents.xlsx no longer has the shape the '
        'Creator produces from that folder\'s agents.yaml. Regenerate it, or explain the drift')
