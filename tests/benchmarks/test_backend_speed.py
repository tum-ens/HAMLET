"""Benchmark — PyOptInterface against linopy on the same MILP and the same solver.

This is the measurement behind ROADMAP item #10, moving HAMLET's default modelling backend off
linopy. Both paths build the *same* agent-MPC-shaped MILP and hand it to the *same* HiGHS, so what
is measured is the modelling layer and nothing else.

**The model itself lives in `tests/backend_models.py`**, shared with the solver x framework
correctness matrix, so that the thing timed here and the thing compared there cannot drift apart.
Its shape: battery (charge / discharge / SoC with a binary mode flag), an inflexible load, and
market import/export over a T-step horizon -- what HAMLET solves ~911k times per scenario-year.

Deselected by default (`benchmark` marker); it takes ~30 s. Run it with:

    uv run python -m pytest -m benchmark -s

`-s` is what shows the table; without it pytest captures the output and only pass/fail is visible.

Two methodology points, both learned the hard way:

- **The model is perturbed between calls.** A re-solve of an unchanged model short-circuits in
  both solvers and reports ~0.01 ms, which looks like a spectacular result and measures nothing.
  The price vector changes on every repetition -- that is what `mpc_prices(rep)` is for.
- **Build and solve are reported separately.** The finding of item #10 is that linopy's *build*
  dominates while HiGHS sits idle, so a single total would hide it.

This benchmark is HiGHS-only on purpose: it exists to compare *frameworks*, so the solver is held
fixed. Timing across solvers is not a thing this repository measures, and the informational
per-cell figures printed by the correctness matrix are not a substitute.
"""
import statistics
import time
import warnings

import pytest

from tests.backend_models import (MPC_HORIZON, build_mpc_linopy, build_mpc_poi, mpc_prices,
                                  solve_linopy, solve_poi)
from tests.poi_support import available_backend

HORIZON, REPS = MPC_HORIZON, 100

# Held fixed; see the module docstring.
SOLVER = 'highs'


def measure(build, solve, horizon, reps):
    """Median build and solve times in ms, plus the objective, for one backend."""
    builds, solves, objective = [], [], None
    for rep in range(reps):
        prices = mpc_prices(rep, horizon)
        t0 = time.perf_counter()
        model = build(horizon, prices)
        t1 = time.perf_counter()
        objective = solve(model)
        t2 = time.perf_counter()
        builds.append((t1 - t0) * 1e3)
        solves.append((t2 - t1) * 1e3)
    return statistics.median(builds), statistics.median(solves), objective


@pytest.mark.benchmark
@pytest.mark.solver
def test_poi_is_faster_than_linopy_on_the_same_model():
    """PyOptInterface must reach the same optimum, much faster, on an identical MILP.

    The objective check is the load-bearing part. Two backends that disagree on the answer make
    the timing meaningless, so this is an equivalence test that happens to also report speed.
    """
    if available_backend() is None:
        pytest.skip('no PyOptInterface solver library is loadable')

    warnings.filterwarnings('ignore')  # linopy is loud about coordinate alignment

    poi_build, poi_solve, poi_obj = measure(
        lambda horizon, prices: build_mpc_poi(SOLVER, horizon, prices), solve_poi, HORIZON, REPS)
    lin_build, lin_solve, lin_obj = measure(
        build_mpc_linopy, lambda model: solve_linopy(model, SOLVER), HORIZON, REPS)

    poi_total, lin_total = poi_build + poi_solve, lin_build + lin_solve
    print(f'\n  horizon={HORIZON}  reps={REPS}  solver={SOLVER}   (medians, ms)')
    print(f"  {'backend':<24}{'build':>9}{'solve':>9}{'total':>9}")
    print(f"  {'PyOptInterface + HiGHS':<24}{poi_build:>9.2f}{poi_solve:>9.2f}{poi_total:>9.2f}")
    print(f"  {'linopy + HiGHS':<24}{lin_build:>9.2f}{lin_solve:>9.2f}{lin_total:>9.2f}")
    print(f'  speedup: {lin_total / poi_total:.1f}x   objective: {poi_obj:.6f}')

    assert poi_obj == pytest.approx(lin_obj, rel=1e-9), (
        f'backends disagree on the optimum: POI {poi_obj} vs linopy {lin_obj}')
    # A loose floor, not the measured figure. This guards against the backend silently regressing
    # to linopy-like cost; asserting the actual ~30-50x would make the suite fail on a slow runner.
    assert poi_total < lin_total / 5, (
        f'expected PyOptInterface to be far faster; got {poi_total:.2f} ms vs {lin_total:.2f} ms')
