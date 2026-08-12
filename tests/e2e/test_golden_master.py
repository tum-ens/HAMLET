"""Golden master — shipped scenarios, compared against committed reference numbers.

Every other test here pins a property someone thought to check. This one pins the numbers
themselves, so a change that moves results has to be acknowledged rather than noticed. Of the
defects found while assembling this branch, most were caught by review or by measurement rather
than by a test failing; this is the test that would have caught them.

    python -m pytest tests -m golden

**When it fails**, the diff tells you which table and column moved. Decide whether that is the
change you meant to make. If it is, regenerate the reference and commit it *with* the change,
so the review sees the numbers move:

    HAMLET_UPDATE_GOLDEN=1 python -m pytest tests -m golden               # every scenario
    HAMLET_UPDATE_GOLDEN=simple_scenario python -m pytest tests -m golden  # just this one

Name the scenario when more than one is pinned. `1` rewrites every reference, so a re-baseline
aimed at one change also commits any unrelated movement in the others.

**One legitimate cause of movement is not a defect: a different equally-optimal solution.** The
agent models are degenerate MILPs -- a battery or EV can very often shift charging between
timesteps at identical cost -- so changing the solver, or the *modelling backend* that presents
the same model to it, can break a tie the other way. The chosen state of charge then feeds the
next timestep, and a receding-horizon run amplifies one tie into visibly different trajectories.
Measured for `framework: linopy` vs `poi`: the two MPC models are mathematically identical
(verified by exporting both to LP and matching constraints by shape) and their first-timestep
objectives agree to ~1e-12, yet by step 23 the run-level numbers differ by tens of percent. An
agent owning neither a battery nor an EV stays at ~1e-13 throughout, having no state to carry a
divergence forward.

Telling the two apart, when the numbers move:

- *Degeneracy* leaves the objective unchanged at equal state. Structure holds (same tables, same
  row counts, no column added or dropped) and the divergence appears as a discrete jump at some
  timestep rather than from the first one.
- *A defect* shows a difference from the first timestep at identical state, or moves structure.

So a solver or backend change may be re-baselined once the movement is shown to be of the first
kind; a change that was meant to be numerically inert may not.

Reproducibility rests on seeding `random` and `numpy.random` and pinning `PYTHONHASHSEED`. The
Creator draws agent ids, plant ownership and sizings from all three. Verified: two seeded runs
produce byte-identical scenarios and identical results.

The reference records per-table row counts and, for every numeric column, the sum, minimum and
maximum. Agent ids are random-but-seeded, so tables are grouped by kind rather than by agent --
that keeps the reference readable and stable against an id-scheme change, while still moving the
moment the physics does.

**Adding a scenario.** Append a `GoldenScenario` to `SCENARIOS` and create its reference with
`HAMLET_UPDATE_GOLDEN=<name>`. Each scenario carries its own reference file, named after it, and
is run once for the whole module however many assertions read it. A scenario earns its place by
reaching code the others do not, and costs a full run in the `golden` CI job every time:

- `simple_scenario` -- the shipped example. Sets `electricity.active: False`, so it pins nothing
  the grid stage produces.
- `grid_golden` -- a deliberately weak feeder under §14a. Pins the power flow, the variable grid
  fees and direct power control, none of which the other scenario reaches. It lives under
  `tests/e2e/scenarios/` rather than `examples/` because it is tuned to overload rather than to
  be copied; `tests/e2e/test_grid_restrictions.py` asserts *that* the restriction fires, while
  this file pins the numbers it produces.
"""
import json
import os
import shutil
from pathlib import Path
from typing import NamedTuple

import pytest

from tests.scenario_run import REPO_ROOT, run_example


class GoldenScenario(NamedTuple):
    """One scenario, pinned against one committed reference.

    `name` is both the scenario's config folder and the reference's filename, so the mapping
    between a scenario and its numbers stays greppable in both directions. It is also what
    `test_solver_backend_smoke.py` reads when checking that the backend cell it defers to this
    module is still covered here.

    `container` is the repository-relative directory that *holds* the scenario folder -- not the
    folder itself, because that is what `run_example` takes. Shipped examples sit under
    `examples/<example>/` and are a user's entry point; a scenario built purely to pin behaviour
    -- deliberately undersized, tuned to reach a particular code path -- sits under
    `tests/e2e/scenarios/` instead, because putting it in `examples/` would advertise it as
    something to copy. `creator_method` follows from where the agent ids have to come from.
    """

    container: str
    name: str
    creator_method: str = 'new_scenario_from_configs'

    @property
    def config_dir(self):
        return REPO_ROOT.joinpath(*self.container.split('/'))

    @property
    def reference(self):
        return Path(__file__).parent / 'golden' / f'{self.name}.json'


#: Every scenario the golden master pins. See "Adding a scenario" in the module docstring.
SCENARIOS = [
    GoldenScenario(container='examples/create_simple_scenario', name='simple_scenario'),
    # The grid scenario. `simple_scenario` sets `electricity.active: False`, so until this one
    # nothing here pinned a single number produced by the grid stage, the §14a restrictions or the
    # power flow. `new_scenario_from_files` because its topology assigns agents to buses by id,
    # and only that entry point keeps the ids `agents.xlsx` declares -- creating from configs
    # redraws them and the assignment stops meaning anything.
    GoldenScenario(container='tests/e2e/scenarios', name='grid_golden',
                   creator_method='new_scenario_from_files'),
]

# Solver output is bit-stable on a fixed platform, but HiGHS and polars versions move; this is
# loose enough to survive that and far tighter than any real modelling change.
RELATIVE_TOLERANCE = 1e-6


@pytest.fixture(scope='module', params=SCENARIOS, ids=lambda pinned: pinned.name)
def scenario(request):
    """The scenario under test. Parametrised here so each one is run once for the whole module."""
    return request.param


def check_every_committed_reference_is_still_pinned():
    """`SCENARIOS` and `golden/*.json` must agree, and neither may be empty.

    Parametrising this module created a way for the whole file to stop asserting anything without
    a job going red: with `SCENARIOS = []`, `pytest -m golden` reports `4 skipped` and exits 0 in
    under a second, because an empty parameter set skips rather than fails. Before, dropping the
    pinning meant deleting the tests, which a reviewer sees.

    Comparing against the committed references rather than against a hardcoded name is what makes
    this generalise: any scenario that has been pinned stays pinned, not just the first one.
    Removing one on purpose means deleting its reference in the same commit, which is exactly the
    visible act that ought to be required.

    Run from two tests, deliberately. Markers decide jobs, and `pytest.ini` deselects `golden` by
    default, so a single test cannot sit in both the fast tier and the golden job — and this needs
    to be in both. The golden job is the one someone re-runs in isolation from the pipeline view,
    and in isolation it is the only thing standing between an empty `SCENARIOS` and a green tick.
    """
    pinned = {pinned_scenario.name for pinned_scenario in SCENARIOS}
    committed = {path.stem for path in (Path(__file__).parent / 'golden').glob('*.json')}

    assert pinned, (
        'SCENARIOS is empty, so the golden master pins nothing and `pytest -m golden` passes by '
        'skipping every test')
    assert committed <= pinned, (
        f'these scenarios have a committed reference but are no longer in SCENARIOS, so nothing '
        f'compares against them: {sorted(committed - pinned)}. Delete the reference in the same '
        f'commit if that is intended')


def test_every_committed_reference_is_still_pinned():
    """The fast-tier arm. See `check_every_committed_reference_is_still_pinned`."""
    check_every_committed_reference_is_still_pinned()


@pytest.mark.golden
def test_every_committed_reference_is_still_pinned_in_the_golden_job():
    """The golden-job arm, so that job is self-sufficient when run alone."""
    check_every_committed_reference_is_still_pinned()


@pytest.fixture(scope='module')
def actual(scenario, tmp_path_factory):
    """Run the example once, seeded, against a temp copy of the config.

    The run and the fingerprint live in `tests/scenario_run.py`, shared with the backend
    equivalence tests -- the poi arm of those compares against this reference, which only means
    anything if both reduce results the same way. (It was the linopy arm until `poi` became the
    default; the control has to be whichever backend the shipped config selects.)
    """
    base = tmp_path_factory.mktemp(f'golden_{scenario.name}')
    try:
        yield run_example(base, scenario.config_dir, scenario.name,
                          creator_method=scenario.creator_method)
    finally:
        shutil.rmtree(base, ignore_errors=True)


@pytest.fixture(scope='module')
def expected(scenario, actual):
    """The committed reference, regenerated in place when explicitly asked for.

    `HAMLET_UPDATE_GOLDEN` takes a scenario name, or `1`/`all` for every scenario. Naming one
    matters once more than one is pinned: a re-baseline aimed at a change in one scenario would
    otherwise also rewrite the others, so an unrelated movement in a scenario you were not
    thinking about gets committed as though it had been reviewed.
    """
    reference = scenario.reference
    update = os.environ.get('HAMLET_UPDATE_GOLDEN')
    if update and update in ('1', 'all', scenario.name):
        reference.parent.mkdir(parents=True, exist_ok=True)
        reference.write_text(json.dumps(actual, indent=2, sort_keys=True) + '\n',
                             encoding='utf-8')
        pytest.skip(f'reference regenerated at {reference.relative_to(REPO_ROOT)}; '
                    f'review the diff and commit it with the change that caused it')
    if update:
        pytest.skip(f'HAMLET_UPDATE_GOLDEN={update} does not name this scenario '
                    f'({scenario.name}), so its reference was left alone')

    assert reference.exists(), (
        f'no golden reference at {reference.relative_to(REPO_ROOT)}. Create one with '
        f'HAMLET_UPDATE_GOLDEN=1 python -m pytest tests -m golden')

    return json.loads(reference.read_text(encoding='utf-8'))


@pytest.mark.golden
def test_the_same_tables_are_produced(scenario, actual, expected):
    """A table appearing or disappearing is a result change like any other."""
    assert sorted(actual) == sorted(expected), scenario.name


@pytest.mark.golden
def test_row_counts_match(scenario, actual, expected):
    """Catches trades appearing or vanishing, which several defects here did."""
    differences = {kind: (entry['rows'], expected[kind]['rows'])
                   for kind, entry in actual.items()
                   if kind in expected and entry['rows'] != expected[kind]['rows']}

    assert not differences, (
        f'{scenario.name}: row counts moved (actual, expected): {differences}')


@pytest.mark.golden
def test_column_statistics_match(scenario, actual, expected):
    """The substance: every numeric column's total, minimum and maximum.

    Reported all at once rather than failing on the first, because when the model changes it is
    the shape of the difference across tables that tells you whether it was intended.
    """
    differences = []
    for kind, entry in actual.items():
        if kind not in expected:
            continue
        for column, stats in entry['columns'].items():
            reference = expected[kind]['columns'].get(column)
            if reference is None:
                differences.append(f'{kind}:{column} is new')
                continue
            for statistic, value in stats.items():
                other = reference.get(statistic)
                if value is None or other is None:
                    if value != other:
                        differences.append(f'{kind}:{column}.{statistic} {value} != {other}')
                    continue
                if abs(value - other) > RELATIVE_TOLERANCE * max(1.0, abs(other)):
                    differences.append(
                        f'{kind}:{column}.{statistic} {value:,.3f} != {other:,.3f} '
                        f'(delta {value - other:+,.3f})')

    assert not differences, (
        f'{scenario.name} now produces different numbers:\n  '
        + '\n  '.join(differences[:40])
        + (f'\n  ... and {len(differences) - 40} more' if len(differences) > 40 else '')
        + '\n\nIf this change was intended, regenerate the reference with '
          'HAMLET_UPDATE_GOLDEN=1 and commit it alongside the change.')


@pytest.mark.golden
def test_no_column_was_dropped(scenario, actual, expected):
    """A column disappearing is easy to miss when only the ones present are compared."""
    missing = [f'{kind}:{column}'
               for kind, entry in expected.items() if kind in actual
               for column in entry['columns']
               if column not in actual[kind]['columns']]

    assert not missing, f'{scenario.name}: columns no longer produced: {missing}'
