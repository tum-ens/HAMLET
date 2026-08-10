"""End to end — the shipped example runs under every available solver x framework combination.

`tests/integration/executor/test_solver_backend_matrix.py` shows the four combinations agree on a
single model's optimum. This file asks the cheaper and equally necessary question: does each one
survive a real run at all?

That is not a formality. Before !209, `framework: poi` imported `pyoptinterface.gurobi`
unconditionally and accepted only `solver: gurobi`; it worked on a machine with a system Gurobi
installation and was structurally broken everywhere else, and nothing noticed because nothing ever
ran that cell end to end. This test is what would have caught it on the first push.

    python -m pytest tests/e2e/test_solver_backend_smoke.py -m e2e -rs

**Deliberately not a numeric comparison.** It asserts that the run completes and produces results,
not that the four combinations produce the *same* results. They do not, they are not expected to,
and that is closed as #198 -- degenerate MILPs break ties differently and `rtc_base.update_socs`
feeds the chosen vertex into the next timestep. `tests/e2e/test_backend_equivalence.py` holds that
comparison as a permanent strict xfail; do not reopen it here.

**Each run reports which backend actually solved it.** The config edit is a request, and this
asserts it was honoured -- see `scenario_run.BACKEND_PROBE` for why the request alone is not
enough. Budget a few minutes per available combination.
"""
import json

import pytest

from tests.backend_matrix import COMBINATION_IDS, COMBINATIONS, require
from tests.scenario_run import REPO_ROOT, run_example

EXAMPLE = REPO_ROOT / 'examples' / 'create_simple_scenario'
SCENARIO_NAME = 'simple_scenario'


@pytest.mark.e2e
@pytest.mark.solver
@pytest.mark.parametrize(('framework', 'solver'), COMBINATIONS, ids=COMBINATION_IDS)
def test_the_example_runs_under_this_combination(framework, solver, tmp_path):
    """Creator and Executor complete, the requested backend is what solved, results are written."""
    require(framework, solver)

    record = tmp_path / 'backends.json'
    fingerprint = run_example(tmp_path, EXAMPLE, SCENARIO_NAME, framework=framework,
                              solver=solver, record_backends=record)

    # `run_example` already fails on a missing RUN_OK, so reaching here means the run finished.
    # What is left is whether it finished having done the work, and having done it with the
    # requested backend.
    assert record.exists(), (
        'the run completed but wrote no backend record, so what solved it is unknown')
    used = {tuple(pair) for pair in json.loads(record.read_text(encoding='utf-8'))}

    assert used == {(framework, solver)}, (
        f'asked for {framework} + {solver}, but the run actually used '
        f'{sorted(used) or "no modelling backend at all"}')

    assert fingerprint, 'the run produced no result tables'
    assert sum(entry['rows'] for entry in fingerprint.values()) > 0, (
        'the run wrote result tables but every one of them is empty')
