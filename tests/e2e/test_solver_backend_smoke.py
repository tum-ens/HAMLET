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

**The #206 read-back used to be here and is now `tests/e2e/test_ctsp_industry.py`.** It asks a
scenario built from `agents.xlsx` for a backend it does not ship and reads back what solved.
Pointed at `grid_golden`, which ships `poi`, the only licence-free backend it could ask for was
`linopy`, and the run cost **232-272 s** to assert a config-plumbing property that needs neither a
grid nor linopy. The `ctsp_industry` fixture ships `linopy`, so the same assertion is made by
asking for `poi`: **26-36 s**, on the default backend, and it carries the first ctsp/industry
coverage with it. Same assertion, same class of coverage, still selected by `-m e2e` rather than
hidden behind a marker of its own.
"""
import ast
import inspect
import json
import os

import pytest

from tests.backend_matrix import COMBINATION_IDS, COMBINATIONS, require
from tests.scenario_run import REPO_ROOT

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
def test_the_example_runs_under_this_combination(framework, solver, scenario_runs):
    """Creator and Executor complete, the requested backend is what solved, results are written."""
    require(framework, solver)
    if (framework, solver) in COVERED_ELSEWHERE and not os.environ.get(RUN_ALL):
        pytest.skip(f'{framework} + {solver} already runs end to end in '
                    f'{COVERED_ELSEWHERE[(framework, solver)]}; set {RUN_ALL}=1 to run it here too')

    entry = scenario_runs.run(EXAMPLE, SCENARIO_NAME, framework=framework, solver=solver)
    record, fingerprint = entry.record, entry.fingerprint

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


#: The two ways a module starts an end-to-end run. `scenario_runs.run` is the shared run cache
#: and forwards to `run_example` (`tests/scenario_cache.py`), so either spelling means "this
#: module runs the example".
RUN_CALLS = ('run', 'run_example')


def run_calls(module):
    """Every end-to-end run the module *makes*, read from its syntax tree.

    Returns one dict of constant keyword arguments per call, so a caller can ask what backend a
    run was started with rather than whether a string appears somewhere in the file.

    **Parsed rather than grepped, and that is the whole point.** The substring version of this
    guard was satisfiable by prose in both directions: a comment mentioning `framework=` could
    fail the negative check, and a docstring quoting `framework='linopy'` could pass the positive
    one. Both happened -- the second is `tests/unit/test_warning_policy.py`'s lesson, and the
    first broke this test when a docstring elsewhere was reworded. An `ast.Call` node cannot be
    written in a comment.

    Non-constant arguments are reported as `None` rather than guessed at. That is deliberate and
    it is the limit of this check: a run whose backend is computed at runtime reads here as "no
    backend named". `test_the_example_runs_under_this_combination` above is exactly that shape,
    which is why this only inspects the two modules it defers to, both of which pass literals.
    """
    tree = ast.parse(inspect.getsource(module))
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = (node.func.attr if isinstance(node.func, ast.Attribute)
                else getattr(node.func, 'id', None))
        if name not in RUN_CALLS:
            continue
        calls.append({keyword.arg: (keyword.value.value
                                    if isinstance(keyword.value, ast.Constant) else None)
                      for keyword in node.keywords})
    return calls


def test_the_deferred_cells_are_still_covered_elsewhere():
    """The cells skipped above must still be run end to end by the tests named in the skip.

    Without this, `COVERED_ELSEWHERE` is an unchecked assertion about other files: delete the
    linopy arm of the equivalence test, or repoint the golden master at a different backend, and
    this file would go on politely skipping a cell that nothing runs any more. Deferring coverage
    is only safe if the deferral is itself tested.

    Checked against the covering modules' *calls* rather than by importing their fixtures, because
    what matters is that the example is run under that backend, not what the fixture is called.
    See `run_calls` for why this reads the syntax tree instead of the text. Which scenarios the
    golden master pins is read as data below, because "it starts a run somewhere" still does not
    imply "it still runs *this* example" -- that part no parse can tell you either.

    **The run cache is a registry of what actually ran, and it still cannot replace this.**
    `scenario_runs.log` records every (request, scenario) really executed, which is stronger
    evidence than any static read -- but it is populated by *running the examples*, and this test
    is deliberately in the fast tier, where none of them run. Asserting against it here would read
    an empty log and pass for the wrong reason. An e2e-marked companion could read it, but only by
    relying on this file being collected after `test_backend_equivalence`; that is alphabetical
    accident, not a guarantee, so it is deliberately not done.

    Note that `SCENARIOS` being non-empty is not checked here but in `test_golden_master.py`
    itself, against the committed references -- that guard generalises to every pinned scenario,
    where this one only defends the cell this file defers.
    """
    from tests.e2e import test_backend_equivalence as equivalence
    from tests.e2e import test_golden_master as golden

    equivalence_calls = run_calls(equivalence)
    golden_calls = run_calls(golden)

    assert any(call.get('framework') == 'linopy' for call in equivalence_calls), (
        "test_backend_equivalence no longer runs the example with framework='linopy', so "
        f"linopy + highs is not covered there and must stop being skipped here (its runs: "
        f"{equivalence_calls})")
    assert golden_calls, (
        'test_golden_master no longer runs the example, so poi + highs is not covered there and '
        'must stop being skipped here')
    # The deferral names *this* example and *this* scenario, not merely "the golden master".
    # Matching on the scenario name alone would accept a different example that happened to carry
    # a folder of the same name -- which is only unreachable today because no two shipped examples
    # share a scenario folder name, and that is a coincidence rather than a rule.
    #
    # Note what this does and does not buy: it protects the golden-master *reference* comparison
    # of this cell. It is not the only thing running poi + highs end to end -- `COVERED_ELSEWHERE`
    # names the equivalence test's poi arm as well, and that has its own constants and would keep
    # running if this module were repointed.
    pinned = {(scenario.config_dir, scenario.name) for scenario in golden.SCENARIOS}
    assert (EXAMPLE, SCENARIO_NAME) in pinned, (
        f'test_golden_master no longer pins {EXAMPLE.name}/{SCENARIO_NAME} (it pins '
        f'{sorted((str(d), n) for d, n in pinned)}), so the poi + highs cell has no golden '
        f'reference behind it and must stop being skipped here')
    # The golden master must still run the *shipped* configuration -- it is only a stand-in for
    # the poi + highs cell for as long as it does not override the backend itself.
    overridden = [call for call in golden_calls
                  if call.get('framework') is not None or call.get('solver') is not None]
    assert not overridden, (
        f'test_golden_master now overrides the backend ({overridden}), so it no longer covers '
        f'whatever the shipped config selects; re-check which cell it stands in for')
