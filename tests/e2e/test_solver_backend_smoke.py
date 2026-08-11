"""End to end — the shipped example runs under each solver x framework combination.

`tests/integration/executor/test_solver_backend_matrix.py` shows the four combinations agree on a
single model's optimum. This file asks the cheaper and equally necessary question: does each one
survive a real run at all?

That is not a formality. Before !209, `framework: poi` imported `pyoptinterface.gurobi`
unconditionally and accepted only `solver: gurobi`; it worked on a machine with a system Gurobi
installation and was structurally broken everywhere else, and nothing noticed because nothing ever
ran that cell end to end. This test is what would have caught it on the first push.

**By default it runs only the cells nothing else covers, which is both Gurobi cells.** An example
run costs minutes, and the two HiGHS cells are already run end to end elsewhere -- see
`COVERED_ELSEWHERE`. Running them here too tripled the `e2e` CI job (338 s to ~1090 s) and bought
nothing, and on a shared runner that inflation is not free: it widened the window in which the
`golden` job competes for the same cores. Set `HAMLET_SMOKE_ALL=1` to run all four anyway:

    HAMLET_SMOKE_ALL=1 python -m pytest tests/e2e/test_solver_backend_smoke.py -m e2e -rs

The deferred cells are still *parametrised and skipped*, never filtered out, so they appear in the
report with a reason naming what covers them instead of vanishing. And the claim that they are
covered is itself tested -- `test_the_deferred_cells_are_still_covered_elsewhere` fails if the
covering runs are removed or repointed, so this file cannot go on deferring to coverage that has
quietly gone away.

**Deliberately not a numeric comparison.** It asserts that the run completes and produces results,
not that the four combinations produce the *same* results. They do not, they are not expected to,
and that is closed as #198 -- degenerate MILPs break ties differently and `rtc_base.update_socs`
feeds the chosen vertex into the next timestep. `tests/e2e/test_backend_equivalence.py` holds that
comparison as a permanent strict xfail; do not reopen it here.

**Each run reports which backend actually solved it.** The config edit is a request, and this
asserts it was honoured -- see `scenario_run.BACKEND_PROBE` for why the request alone is not
enough. Budget a few minutes per combination that runs.
"""
import inspect
import json
import os

import pytest

from tests.backend_matrix import COMBINATION_IDS, COMBINATIONS, require
from tests.scenario_run import REPO_ROOT, run_example

EXAMPLE = REPO_ROOT / 'examples' / 'create_simple_scenario'
SCENARIO_NAME = 'simple_scenario'

#: Set to run every combination here, including the ones covered elsewhere.
RUN_ALL = 'HAMLET_SMOKE_ALL'

#: Cells whose end-to-end coverage already exists, and where. Both are on HiGHS, which is what
#: makes them the expensive ones to duplicate: they are the cells that always run.
COVERED_ELSEWHERE = {
    ('poi', 'highs'):
        'tests/e2e/test_golden_master.py, which runs the shipped config (poi + highs) end to end '
        'in its own CI job, and test_backend_equivalence.py, whose poi arm does the same',
    ('linopy', 'highs'):
        "tests/e2e/test_backend_equivalence.py, whose `linopy_results` fixture runs the example "
        "end to end with framework='linopy' on the shipped solver",
}


@pytest.mark.e2e
@pytest.mark.solver
@pytest.mark.parametrize(('framework', 'solver'), COMBINATIONS, ids=COMBINATION_IDS)
def test_the_example_runs_under_this_combination(framework, solver, tmp_path):
    """Creator and Executor complete, the requested backend is what solved, results are written."""
    require(framework, solver)
    if (framework, solver) in COVERED_ELSEWHERE and not os.environ.get(RUN_ALL):
        pytest.skip(f'{framework} + {solver} already runs end to end in '
                    f'{COVERED_ELSEWHERE[(framework, solver)]}; set {RUN_ALL}=1 to run it here too')

    record = tmp_path / 'backends.json'
    fingerprint = run_example(tmp_path, EXAMPLE, SCENARIO_NAME, framework=framework,
                              solver=solver, record_backends=record)

    # `run_example` already fails on a missing RUN_OK, so reaching here means the run finished.
    # What is left is whether it finished having done the work, and with the requested backend.
    assert record.exists(), (
        'the run completed but wrote no backend record, so what solved it is unknown')
    used = {tuple(pair) for pair in json.loads(record.read_text(encoding='utf-8'))}

    assert used == {(framework, solver)}, (
        f'asked for {framework} + {solver}, but the run actually used '
        f'{sorted(used) or "no modelling backend at all"}')

    assert fingerprint, 'the run produced no result tables'
    assert sum(entry['rows'] for entry in fingerprint.values()) > 0, (
        'the run wrote result tables but every one of them is empty')


def test_the_deferred_cells_are_still_covered_elsewhere():
    """The cells skipped above must still be run end to end by the tests named in the skip.

    Without this, `COVERED_ELSEWHERE` is an unchecked assertion about other files: delete the
    linopy arm of the equivalence test, or repoint the golden master at a different backend, and
    this file would go on politely skipping a cell that nothing runs any more. Deferring coverage
    is only safe if the deferral is itself tested.

    Checked against the covering modules' source rather than by importing their fixtures, because
    what matters is *that the example is run under that backend*, not what the fixture is called.
    The exception is which scenarios the golden master pins, which is read as data: now that it is
    multi-scenario, "it calls run_example somewhere" no longer implies "it still runs *this*
    example", and no substring can tell those two apart.
    """
    from tests.e2e import test_backend_equivalence as equivalence
    from tests.e2e import test_golden_master as golden

    equivalence_source = inspect.getsource(equivalence)
    golden_source = inspect.getsource(golden)

    assert "framework='linopy'" in equivalence_source, (
        "test_backend_equivalence no longer runs the example with framework='linopy', so "
        "linopy + highs is not covered there and must stop being skipped here")
    assert 'run_example(' in golden_source, (
        'test_golden_master no longer runs the example, so poi + highs is not covered there and '
        'must stop being skipped here')
    # The deferral names *this* scenario, not merely "the golden master". Dropping it from
    # `SCENARIOS` while pinning some other one would leave poi + highs uncovered with every other
    # check here still passing.
    pinned = {scenario.name for scenario in golden.SCENARIOS}
    assert SCENARIO_NAME in pinned, (
        f'test_golden_master no longer pins {SCENARIO_NAME} (it pins {sorted(pinned)}), so '
        f'poi + highs is not covered there and must stop being skipped here')
    # The golden master must still run the *shipped* configuration -- it is only a stand-in for
    # the poi + highs cell for as long as it does not override the backend itself.
    assert 'framework=' not in golden_source, (
        'test_golden_master now overrides the backend, so it no longer covers whatever the '
        'shipped config selects; re-check which cell it stands in for')
