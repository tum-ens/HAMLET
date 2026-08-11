"""Golden master — the shipped example, compared against committed reference numbers.

Every other test here pins a property someone thought to check. This one pins the numbers
themselves, so a change that moves results has to be acknowledged rather than noticed. Of the
defects found while assembling this branch, most were caught by review or by measurement rather
than by a test failing; this is the test that would have caught them.

    python -m pytest tests -m golden

**When it fails**, the diff tells you which table and column moved. Decide whether that is the
change you meant to make. If it is, regenerate the reference and commit it *with* the change,
so the review sees the numbers move:

    HAMLET_UPDATE_GOLDEN=1 python -m pytest tests -m golden

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
`HAMLET_UPDATE_GOLDEN=1`. Each scenario carries its own reference file, named after it, and is run
once per session however many assertions read it. A scenario earns its place by reaching code the
others do not -- `simple_scenario` sets `electricity.active: False`, so nothing pinned here
executes the grid stage.
"""
import json
import os
import shutil
from pathlib import Path
from typing import NamedTuple

import pytest

from tests.scenario_run import REPO_ROOT, run_example


class GoldenScenario(NamedTuple):
    """One example, pinned against one committed reference.

    `name` is both the example's scenario folder and the reference's filename, so the mapping
    between a scenario and its numbers stays greppable in both directions. It is also what
    `test_solver_backend_smoke.py` reads when checking that the backend cell it defers to this
    module is still covered here.
    """

    example: str
    name: str

    @property
    def config_dir(self):
        return REPO_ROOT / 'examples' / self.example

    @property
    def reference(self):
        return Path(__file__).parent / 'golden' / f'{self.name}.json'


#: Every scenario the golden master pins. See "Adding a scenario" in the module docstring.
SCENARIOS = [
    GoldenScenario(example='create_simple_scenario', name='simple_scenario'),
]

# Solver output is bit-stable on a fixed platform, but HiGHS and polars versions move; this is
# loose enough to survive that and far tighter than any real modelling change.
RELATIVE_TOLERANCE = 1e-6


@pytest.fixture(scope='module', params=SCENARIOS, ids=lambda pinned: pinned.name)
def scenario(request):
    """The scenario under test. Parametrised here so each one is run once for the whole module."""
    return request.param


@pytest.fixture(scope='module')
def actual(scenario, tmp_path_factory):
    """Run the example once, seeded, against a temp copy of the config.

    The run and the fingerprint live in `tests/scenario_run.py`, shared with the backend
    equivalence tests -- those compare their linopy arm against this reference, which only means
    anything if both reduce results the same way.
    """
    base = tmp_path_factory.mktemp(f'golden_{scenario.name}')
    try:
        yield run_example(base, scenario.config_dir, scenario.name)
    finally:
        shutil.rmtree(base, ignore_errors=True)


@pytest.fixture(scope='module')
def expected(scenario, actual):
    """The committed reference, regenerated in place when explicitly asked for."""
    reference = scenario.reference
    if os.environ.get('HAMLET_UPDATE_GOLDEN'):
        reference.parent.mkdir(parents=True, exist_ok=True)
        reference.write_text(json.dumps(actual, indent=2, sort_keys=True) + '\n',
                             encoding='utf-8')
        pytest.skip(f'reference regenerated at {reference.relative_to(REPO_ROOT)}; '
                    f'review the diff and commit it with the change that caused it')

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
