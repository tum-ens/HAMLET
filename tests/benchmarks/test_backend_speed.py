"""Benchmark — every available framework x solver combination, on one MILP.

This is the measurement behind ROADMAP item #10, moving HAMLET's default modelling backend off
linopy, extended to the solver axis so that the same table answers both questions a user actually
asks: *do the four combinations agree*, and *what does each cost*.

Every cell builds the **same** agent-MPC-shaped MILP and solves it, so the only variables are the
modelling framework and the solver. The model lives in `tests/backend_models.py`, shared with
`tests/integration/executor/test_solver_backend_matrix.py`, so the thing timed here and the thing
compared there cannot drift apart. Its shape: battery (charge / discharge / SoC with a binary mode
flag), an inflexible load, and market import/export over a T-step horizon — what HAMLET solves
~911k times per scenario-year.

Deselected by default (`benchmark` marker); it takes ~30 s per available cell. Run it with:

    uv run python -m pytest -m benchmark -s

`-s` is what shows the table; without it pytest captures the output and only pass/fail is visible.
Both Gurobi cells skip without a licence — `linopy` also needs `uv sync --extra gurobi`, since it
reaches Gurobi through `gurobipy` while PyOptInterface links a system installation directly.

Four methodology points, each of which has produced a wrong number here before:

- **The model is perturbed between calls.** A re-solve of an unchanged model short-circuits in both
  solvers and reports ~0.01 ms, which looks like a spectacular result and measures nothing. The
  price vector changes on every repetition — that is what `mpc_prices(rep)` is for.
- **Cells are interleaved, not run in blocks.** This laptop drifts more than 2x thermally, so
  whichever cell ran last in a blocked layout is penalised for the others' heat.
- **Every cell is warmed before timing starts.** Loading a solver's shared library is a one-off
  cost of tens of milliseconds; charged to the first cell it runs in, it inverts the ranking
  outright.
- **Build and solve are reported separately.** The finding of item #10 is that linopy's *build*
  dominates while the solver sits idle, and a single total hides it — including the fact that
  linopy's build cost is identical whichever solver it is pointed at.

**Only the framework assertion gates.** The solver axis is reported, never asserted: HAMLET's
models are small enough that per-model overhead dominates, so which solver wins is a property of
this model size and this machine rather than of the solvers. Read that column, do not pin it.
"""
import statistics
import time
import warnings

import pytest

from tests.backend_matrix import COMBINATIONS, is_available
from tests.backend_models import (MPC_HORIZON, build_mpc_linopy, build_mpc_poi, mpc_prices,
                                  solve_linopy, solve_poi)

HORIZON, REPS = MPC_HORIZON, 60

# The framework comparison this file was written for, and the only assertion that gates.
BASELINE, CANDIDATE = ('linopy', 'highs'), ('poi', 'highs')


def build(framework, solver, prices):
    if framework == 'poi':
        return build_mpc_poi(solver, HORIZON, prices)
    return build_mpc_linopy(HORIZON, prices)


def solve(framework, solver, model):
    return solve_poi(model) if framework == 'poi' else solve_linopy(model, solver)


def measure(cells, reps):
    """Median build and solve times in ms per cell, plus each cell's objective.

    Cells are interleaved within each repetition and warmed beforehand; see the module docstring
    for why both matter.
    """
    builds = {cell: [] for cell in cells}
    solves = {cell: [] for cell in cells}
    objectives = {}

    for framework, solver in cells:
        objectives[(framework, solver)] = solve(
            framework, solver, build(framework, solver, mpc_prices(0)))

    for rep in range(reps):
        prices = mpc_prices(rep, HORIZON)
        for cell in cells:
            framework, solver = cell
            t0 = time.perf_counter()
            model = build(framework, solver, prices)
            t1 = time.perf_counter()
            solve(framework, solver, model)
            t2 = time.perf_counter()
            builds[cell].append((t1 - t0) * 1e3)
            solves[cell].append((t2 - t1) * 1e3)

    return ({cell: (statistics.median(builds[cell]), statistics.median(solves[cell]))
             for cell in cells}, objectives)


@pytest.mark.benchmark
@pytest.mark.solver
def test_every_available_combination_agrees_and_poi_is_faster_than_linopy():
    """One table: what each combination costs, and that they all reach the same optimum.

    The objective check is the load-bearing half. Timings from cells that disagree on the answer
    are not comparable at all, so this is an equivalence test that happens to also report speed.
    """
    cells = [cell for cell in COMBINATIONS if is_available(*cell)]
    assert cells, ('not one framework x solver combination can solve here, so this benchmark '
                   'would measure nothing. HiGHS ships with HAMLET -- run `uv sync`.')

    warnings.filterwarnings('ignore')  # linopy is loud about coordinate alignment

    timings, objectives = measure(cells, REPS)

    print(f'\n  horizon={HORIZON}  reps={REPS}  interleaved   (medians, ms)')
    print(f"  {'cell':<16}{'build':>9}{'solve':>9}{'total':>9}   objective")
    for cell in sorted(cells, key=lambda c: sum(timings[c])):
        build_ms, solve_ms = timings[cell]
        print(f"  {cell[0] + '+' + cell[1]:<16}{build_ms:>9.2f}{solve_ms:>9.2f}"
              f"{build_ms + solve_ms:>9.2f}   {objectives[cell]:.6f}")
    for cell in COMBINATIONS:
        if cell not in timings:
            print(f"  {cell[0] + '+' + cell[1]:<16}{'-- not available here --':>27}")

    reference = objectives[cells[0]]
    for cell, objective in objectives.items():
        assert objective == pytest.approx(reference, rel=1e-9), (
            f'{cell[0]} + {cell[1]} reached {objective}, but {cells[0][0]} + {cells[0][1]} '
            f'reached {reference}; the timings above are not comparable')

    # The framework axis is the only one asserted. A loose floor, not the measured figure: this
    # guards against PyOptInterface silently regressing to linopy-like cost, and asserting the
    # actual ~50x would make the suite fail on a slow runner.
    if BASELINE in timings and CANDIDATE in timings:
        baseline, candidate = sum(timings[BASELINE]), sum(timings[CANDIDATE])
        print(f'  framework speedup on highs: {baseline / candidate:.1f}x')
        assert candidate < baseline / 5, (
            f'expected PyOptInterface to be far faster than linopy on the same solver; '
            f'got {candidate:.2f} ms vs {baseline:.2f} ms')
