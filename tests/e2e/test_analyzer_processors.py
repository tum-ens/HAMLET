"""End-to-end -- the Analyzer's data processors, against committed reference numbers.

The Analyzer is the third of HAMLET's three top-level components and the only one whose output
reaches a paper. Nothing anywhere asserted on what it *computes*: the one test that ran it
(`test_simple_scenario.py`) checked that the process printed `E2E_OK`, and the one that
constructs it in the fast tier (`integration/analyzer/test_results_format_check.py`) asserts only
on its refusal to read an incompatible scenario format. A regression that plotted the wrong
column, dropped an agent or rescaled a series was caught by nothing.

    python -m pytest tests -m e2e

**The data processors are pinned; the plotters are not.** The `process_*` methods return the
numbers that become the figures, so pinning them turns a wrong figure into a red test. Pinning
matplotlib output instead means image comparison -- slow, brittle, and broken by a colour change,
which teaches people to re-baseline without looking. The plotters keep the coverage they have and
this file does not extend it; note that is less than it sounds, since `test_simple_scenario` calls
two of them and `plot_all` is called by no test at all.

**Neither scenario is run for this file.** Both requests are byte-identical to ones the `e2e` job
already makes -- `scenario_with_grid` to `test_grid_examples`'s first parameter, `grid_golden` to
`test_grid_restrictions` -- so `scenario_runs` hands over the existing run and this module adds no
scenario execution to CI. `e2e` rather than `golden` because only the `e2e` job already runs both;
the `golden` job runs `grid_golden` but not `scenario_with_grid`, so this module would pay for one
run there.

**`simple_scenario` is deliberately not pinned here**, although it is the example the Analyzer is
demonstrated on: it sets `electricity.active: False`, so both grid processors return `{}`, and a
reference recording that as coverage is the vacuity `emptiness_complaints` exists to reject.

**When it fails**, the message names the processor, the output and the statistic that moved. If
the change was intended, regenerate and commit the reference *with* it, so the review sees the
numbers move. Name the scenario -- `1` rewrites both, so a re-baseline aimed at one change
silently commits any unrelated movement in the other:

    HAMLET_UPDATE_ANALYZER=grid_golden python -m pytest tests -m e2e
"""
import json
import os
from pathlib import Path
from typing import NamedTuple

import pytest

from tests.analyzer_outputs import emptiness_complaints, processor_names, run_processors
from tests.scenario_run import REPO_ROOT

#: Tolerance on every recorded number. Matches the golden master's, and for the same reason: loose
#: enough to survive a HiGHS or polars patch release, far tighter than any real modelling change.
RELATIVE_TOLERANCE = 1e-6


class AnalyzerScenario(NamedTuple):
    """One scenario, pinned against one committed reference.

    `needs_receipt` and `creator_method` are not free choices -- they are what make the request
    identical to the one an existing module already makes, which is what stops this file paying
    for a run of its own. `test_the_pinned_runs_are_shared_with_an_existing_module` pins that
    correspondence so it cannot drift silently into two extra example runs per pipeline.
    """

    container: str
    name: str
    creator_method: str = 'new_scenario_from_configs'
    needs_receipt: bool = False

    @property
    def config_dir(self):
        return REPO_ROOT.joinpath(*self.container.split('/'))

    @property
    def reference(self):
        return Path(__file__).parent / 'analyzer' / f'{self.name}.json'


#: Every scenario whose Analyzer output is pinned. Both have an active electricity grid, so all
#: six processors return data for both -- which is what lets non-emptiness be required rather than
#: negotiated per scenario. They differ in the grid *generation method* (`file` against
#: `topology`), and that difference is load-bearing: `process_electricity_grid_topology` reads the
#: saved network by name, and pinning only one convention is how it came to be unable to read the
#: other without anything noticing.
SCENARIOS = [
    AnalyzerScenario(container='examples/create_scenario_with_grid', name='scenario_with_grid',
                     creator_method='new_scenario_from_grids', needs_receipt=True),
    AnalyzerScenario(container='tests/e2e/scenarios', name='grid_golden',
                     creator_method='new_scenario_from_files'),
]


@pytest.fixture(scope='module', params=SCENARIOS, ids=lambda pinned: pinned.name)
def scenario(request):
    return request.param


@pytest.fixture(scope='module')
def actual(scenario, scenario_runs):
    """The six processors, run against the scenario's results and reduced.

    Requested through `scenario_runs` with the *same* arguments an existing module uses, so the
    run is shared rather than repeated. `MPLBACKEND` is set in `tests/analyzer_outputs.py`, which
    has to do it at import time -- by the time this fixture runs, collection has already imported
    matplotlib through the analyzer.
    """
    from hamlet.analyzer.setup import Analyzer

    entry = scenario_runs.run(scenario.config_dir, scenario.name,
                              creator_method=scenario.creator_method,
                              needs_receipt=scenario.needs_receipt)
    results = str(entry.results)
    return run_processors(Analyzer(path={scenario.name: results}), results)


@pytest.fixture(scope='module')
def expected(scenario, actual):
    """The committed reference, regenerated in place only when explicitly asked for."""
    reference = scenario.reference
    update = os.environ.get('HAMLET_UPDATE_ANALYZER')
    if update and update in ('1', 'all', scenario.name):
        reference.parent.mkdir(parents=True, exist_ok=True)
        reference.write_text(json.dumps(actual, indent=2, sort_keys=True) + '\n',
                             encoding='utf-8')
        pytest.skip(f'reference regenerated at {reference.relative_to(REPO_ROOT)}; '
                    f'review the diff and commit it with the change that caused it')
    if update:
        pytest.skip(f'HAMLET_UPDATE_ANALYZER={update} does not name this scenario '
                    f'({scenario.name}), so its reference was left alone')

    assert reference.exists(), (
        f'no analyzer reference at {reference.relative_to(REPO_ROOT)}. Create one with '
        f'HAMLET_UPDATE_ANALYZER=1 python -m pytest tests -m e2e')

    return json.loads(reference.read_text(encoding='utf-8'))


# --------------------------------------------------------------------------------------------
# Non-emptiness, asserted first and on its own. See `emptiness_complaints`.
# --------------------------------------------------------------------------------------------

@pytest.mark.e2e
def test_every_processor_actually_returned_data(scenario, actual):
    """A processor returning nothing must fail here rather than pass everything below by vacuity.

    Separate from the comparison on purpose. If `process_*` returns `{}` because the results tree
    lacked a table, every statistic assertion over its rows is satisfied trivially -- the family
    this repository has already met as an empty `parametrize`, an empty `SCENARIOS` and an empty
    allowlist. This is the assertion that has an opinion about it.
    """
    complaints = emptiness_complaints(actual)

    assert not complaints, (
        f'{scenario.name}: the Analyzer produced nothing to compare:\n  '
        + '\n  '.join(complaints))


def test_the_empty_return_is_not_evidence():
    """The fast-tier arm: the committed references are checked for vacuity too.

    Without this, a reference regenerated against a broken run -- one where a processor returned
    `{}` -- would be compared against an equally empty live run and agree forever. It runs in the
    default suite because that is where someone re-reading a reference would want to be told.
    """
    for pinned in SCENARIOS:
        if not pinned.reference.exists():
            continue
        reference = json.loads(pinned.reference.read_text(encoding='utf-8'))
        complaints = emptiness_complaints(reference)

        assert not complaints, (
            f'the committed reference {pinned.reference.name} records no usable data:\n  '
            + '\n  '.join(complaints)
            + '\n\nIt was probably regenerated from a run that produced nothing.')


# --------------------------------------------------------------------------------------------
# The comparison.
# --------------------------------------------------------------------------------------------

@pytest.mark.e2e
def test_the_same_outputs_are_produced(scenario, actual, expected):
    """An output appearing or disappearing is a result change like any other."""
    differences = []
    for processor in sorted(set(actual) | set(expected)):
        produced, reference = set(actual.get(processor, {})), set(expected.get(processor, {}))
        if produced != reference:
            differences.append(f'{processor}: new={sorted(produced - reference)} '
                               f'missing={sorted(reference - produced)}')

    assert not differences, f'{scenario.name}:\n  ' + '\n  '.join(differences)


@pytest.mark.e2e
def test_the_numbers_match(scenario, actual, expected):
    """The substance: every numeric column's total, minimum and maximum, and every text column.

    Reported all at once rather than failing on the first, because when the Analyzer changes it is
    the shape of the difference across processors that says whether it was intended.
    """
    differences = []
    for processor, outputs in sorted(actual.items()):
        for label, entry in sorted(outputs.items()):
            reference = expected.get(processor, {}).get(label)
            if reference is None:
                continue  # reported by test_the_same_outputs_are_produced
            if entry['rows'] != reference['rows']:
                differences.append(
                    f"{processor}{label}: {entry['rows']} rows, expected {reference['rows']}")
            if entry.get('index') != reference.get('index'):
                differences.append(
                    f"{processor}{label}: index {entry.get('index')} "
                    f"!= {reference.get('index')}")
            for column, statistics in sorted(entry['columns'].items()):
                other = reference['columns'].get(column)
                if other is None:
                    continue  # reported by test_the_same_columns_are_produced
                differences.extend(
                    f'{processor}{label}:{column}.{note}' for note in
                    _column_differences(statistics, other))

    assert not differences, (
        f'{scenario.name}: the Analyzer now produces different numbers:\n  '
        + '\n  '.join(differences[:40])
        + (f'\n  ... and {len(differences) - 40} more' if len(differences) > 40 else '')
        + '\n\nIf this change was intended, regenerate the reference with '
          f'HAMLET_UPDATE_ANALYZER={scenario.name} and commit it alongside the change.')


def _column_differences(statistics, reference):
    """Every way one column's statistics differ from the reference's, as readable notes."""
    if statistics.get('kind') != reference.get('kind'):
        return [f"kind {statistics.get('kind')} != {reference.get('kind')}"]

    notes = []

    # Exact counts and identities, not measurements, so they are compared exactly. Running `nulls`
    # through the float tolerance would hide an off-by-one once the count passed 1e6.
    for name in ('dtype', 'value_types', 'nulls'):
        if name in statistics or name in reference:
            if statistics.get(name) != reference.get(name):
                notes.append(f'{name} {statistics.get(name)!r} != {reference.get(name)!r}')

    if statistics.get('kind') == 'numeric':
        for name in ('sum', 'min', 'max', 'ordered'):
            value, other = statistics.get(name), reference.get(name)
            if value is None or other is None:
                if value != other:
                    notes.append(f'{name} {value} != {other}')
                continue
            if abs(value - other) > RELATIVE_TOLERANCE * max(1.0, abs(other)):
                notes.append(f'{name} {value:,.6f} != {other:,.6f} (delta {value - other:+,.6f})')
        return notes

    for name in ('distinct', 'values', 'digest', 'ordered_digest'):
        if statistics.get(name) != reference.get(name):
            notes.append(f'{name} {statistics.get(name)!r} != {reference.get(name)!r}')
    return notes


@pytest.mark.e2e
def test_the_same_columns_are_produced(scenario, actual, expected):
    """Both directions. A column appearing is as much a change as one disappearing.

    The dropped direction is easy to miss when only the columns present are compared. The *added*
    direction is easier still: `test_the_numbers_match` skips a column with no reference rather
    than failing on it, so without this half a brand-new column reaching a plotter is reported by
    nothing at all.
    """
    differences = []
    for processor, outputs in sorted(expected.items()):
        for label, entry in sorted(outputs.items()):
            produced = actual.get(processor, {}).get(label)
            if produced is None:
                continue  # reported by test_the_same_outputs_are_produced
            missing = sorted(set(entry['columns']) - set(produced['columns']))
            added = sorted(set(produced['columns']) - set(entry['columns']))
            if missing:
                differences.append(f'{processor}{label}: columns no longer produced: {missing}')
            if added:
                differences.append(f'{processor}{label}: new columns: {added}')

    assert not differences, f'{scenario.name}:\n  ' + '\n  '.join(differences)


# --------------------------------------------------------------------------------------------
# Guards on the pinning itself, in the fast tier so they hold without an example run.
# --------------------------------------------------------------------------------------------

def test_every_processor_the_analyzer_has_is_pinned():
    """The processor set is discovered from the package, so a new one is unpinned until it is.

    A `process_*` method added to any data-processor class -- or a whole new data-processor class
    -- fails here rather than joining the Analyzer unnoticed. See `tests/analyzer_outputs.py` for
    why nothing in that discovery is enumerated.
    """
    discovered = processor_names()

    assert discovered, 'no process_* methods were discovered at all, so this file pins nothing'

    for pinned in SCENARIOS:
        if not pinned.reference.exists():
            continue
        recorded = set(json.loads(pinned.reference.read_text(encoding='utf-8')))
        assert recorded == discovered, (
            f'{pinned.reference.name} pins {sorted(recorded)} but the Analyzer has '
            f'{sorted(discovered)}. Unpinned: {sorted(discovered - recorded)}; stale: '
            f'{sorted(recorded - discovered)}. Regenerate with HAMLET_UPDATE_ANALYZER=1.')


def check_every_committed_reference_is_still_pinned():
    """`SCENARIOS` and `analyzer/*.json` must agree exactly, and neither may be empty.

    `pytest.ini` sets `empty_parameter_set_mark = fail_at_collect`, so `SCENARIOS = []` errors at
    collection rather than skipping; this is the second layer, and the one that also catches a
    reference committed for a scenario nobody pins any more, or deleted while its scenario stays.

    Run from two tests, deliberately. Markers decide jobs and a single test cannot sit in both the
    fast tier and the `e2e` job -- and this needs to be in both, because the `e2e` job is the one
    someone re-runs in isolation. The golden master splits its equivalent for the same reason.
    """
    pinned = {scenario.name for scenario in SCENARIOS}
    committed = {path.stem for path in (Path(__file__).parent / 'analyzer').glob('*.json')}

    assert pinned, 'SCENARIOS is empty, so this module pins nothing and every test skips'
    assert committed == pinned, (
        f'SCENARIOS and the committed references disagree. Pinned with no reference: '
        f'{sorted(pinned - committed)} (create it with HAMLET_UPDATE_ANALYZER=1). Reference with '
        f'nothing pinning it: {sorted(committed - pinned)} (delete it in the same commit if that '
        f'is intended)')


def test_every_committed_reference_is_still_pinned():
    """The fast-tier arm. See `check_every_committed_reference_is_still_pinned`."""
    check_every_committed_reference_is_still_pinned()


@pytest.mark.e2e
def test_every_committed_reference_is_still_pinned_in_the_e2e_job():
    """The `e2e`-job arm, so that job is self-sufficient when run alone."""
    check_every_committed_reference_is_still_pinned()


def test_the_pinned_runs_are_shared_with_an_existing_module():
    """This module must add no example run to the suite, and that is checked rather than assumed.

    Each scenario's request is keyed with the *production* key and compared against the request the
    owning module makes, rebuilt from **that module's own constants**. If either side moves -- a
    `creator_method`, a `needs_receipt`, a config folder -- the two stop sharing and the `e2e` job
    silently grows two more example runs, minutes per pipeline for no added coverage. Nothing else
    would notice: both modules would still pass.

    Built from the other modules' constants rather than from literals for the same reason
    `test_scenario_cache_key` is: a literal here would agree with itself after either side changed.

    Its reach stops at the arguments those modules expose. An owner's fixture that grows an
    `edits=` or `framework=` argument would split the run without failing here, because this
    rebuilds the call rather than observing it.
    """
    from tests.e2e import test_grid_examples, test_grid_restrictions
    from tests.scenario_cache import request_key

    by_name = {scenario.name: scenario for scenario in SCENARIOS}

    # `test_grid_examples` parametrises (example folder, scenario name, creator method) and asks
    # for a receipt; its first parameter is the `file`-built grid this module pins.
    example, name, creator_method = test_grid_examples.GRID_EXAMPLES[0].values[0]
    owners = {
        name: ((test_grid_examples.REPO_ROOT / 'examples' / example, name),
               {'creator_method': creator_method}, test_grid_examples.NEEDS_RECEIPT),
        test_grid_restrictions.SCENARIO: (
            (test_grid_restrictions.CONFIG_ROOT, test_grid_restrictions.SCENARIO),
            {'creator_method': test_grid_restrictions.CREATOR_METHOD}, False),
    }

    assert set(owners) == set(by_name), (
        f'this module pins {sorted(by_name)} but the modules it shares runs with request '
        f'{sorted(owners)}; one of them has been repointed at a different scenario')

    for scenario_name, (args, kwargs, needs_receipt) in owners.items():
        mine = by_name[scenario_name]
        assert (request_key(mine.config_dir, mine.name, creator_method=mine.creator_method),
                mine.needs_receipt) == (request_key(*args, **kwargs), needs_receipt), (
            f'{scenario_name}: this module no longer makes the same request as the module that '
            f'already runs it, so the e2e job would run it twice')
