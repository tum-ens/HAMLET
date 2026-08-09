"""Benchmark — PyOptInterface against linopy on the same MILP and the same solver.

This is the measurement behind ROADMAP item #10, moving HAMLET's default modelling backend off
linopy. Both paths build the *same* agent-MPC-shaped MILP and hand it to the *same* HiGHS, so what
is measured is the modelling layer and nothing else.

Model: battery (charge / discharge / SoC with a binary mode flag), an inflexible load, and market
import/export over a T-step horizon -- the shape HAMLET solves ~911k times per scenario-year.

Deselected by default (`benchmark` marker); it takes ~30 s. Run it with:

    uv run python -m pytest -m benchmark -s

`-s` is what shows the table; without it pytest captures the output and only pass/fail is visible.

Two methodology points, both learned the hard way:

- **The model is perturbed between calls.** A re-solve of an unchanged model short-circuits in
  both solvers and reports ~0.01 ms, which looks like a spectacular result and measures nothing.
  The price vector changes on every repetition.
- **Build and solve are reported separately.** The finding of item #10 is that linopy's *build*
  dominates while HiGHS sits idle, so a single total would hide it.
"""
import statistics
import time
import warnings

import pytest

from tests.poi_support import available_backend, skip_on_windows

CAP, PMAX, ETA = 10_000.0, 5_000.0, 0.95
HORIZON, REPS = 24, 100


def prices_for(rep, horizon):
    """A fresh price vector per repetition, so every call is a genuine re-solve."""
    return [0.30 + 0.01 * ((rep + t) % 7) for t in range(horizon)]


def load_for(t):
    """A deterministic inflexible-load profile, in W."""
    return 2_000.0 + 500.0 * (t % 5)


def build_poi(horizon, prices):
    import pyoptinterface as poi

    from hamlet.executor.utilities.controller.poi_solver import create_model

    m = create_model('highs')
    chg, dis, soc, mode, imp, exp = [], [], [], [], [], []
    for _ in range(horizon):
        chg.append(m.add_variable(lb=0, ub=PMAX))
        dis.append(m.add_variable(lb=0, ub=PMAX))
        soc.append(m.add_variable(lb=0, ub=CAP))
        mode.append(m.add_variable(domain=poi.VariableDomain.Binary))
        imp.append(m.add_variable(lb=0, ub=20_000.0))
        exp.append(m.add_variable(lb=0, ub=20_000.0))

    for t in range(horizon):
        m.add_linear_constraint(chg[t] - PMAX * mode[t], poi.ConstraintSense.LessEqual, 0.0)
        m.add_linear_constraint(dis[t] + PMAX * mode[t], poi.ConstraintSense.LessEqual, PMAX)
        recursion = soc[t] - ETA * chg[t] + dis[t] / ETA
        if t == 0:
            m.add_linear_constraint(recursion, poi.ConstraintSense.Equal, 0.5 * CAP)
        else:
            m.add_linear_constraint(recursion - soc[t - 1], poi.ConstraintSense.Equal, 0.0)
        m.add_linear_constraint(imp[t] - exp[t] + dis[t] - chg[t],
                                poi.ConstraintSense.Equal, load_for(t))

    m.set_objective(sum(prices[t] * imp[t] - 0.8 * prices[t] * exp[t] for t in range(horizon)),
                    poi.ObjectiveSense.Minimize)
    return m


def solve_poi(model):
    import pyoptinterface as poi

    model.optimize()
    status = model.get_model_attribute(poi.ModelAttribute.TerminationStatus)
    assert status == poi.TerminationStatusCode.OPTIMAL, f'POI returned {status}'
    return model.get_model_attribute(poi.ModelAttribute.ObjectiveValue)


def build_linopy(horizon, prices):
    import linopy
    import pandas as pd
    import xarray as xr

    steps = pd.RangeIndex(horizon, name='timesteps')
    m = linopy.Model()
    chg = m.add_variables(lower=0, upper=PMAX, coords=[steps], name='chg')
    dis = m.add_variables(lower=0, upper=PMAX, coords=[steps], name='dis')
    soc = m.add_variables(lower=0, upper=CAP, coords=[steps], name='soc')
    mode = m.add_variables(coords=[steps], name='mode', binary=True)
    imp = m.add_variables(lower=0, upper=20_000.0, coords=[steps], name='imp')
    exp = m.add_variables(lower=0, upper=20_000.0, coords=[steps], name='exp')

    m.add_constraints(chg - PMAX * mode <= 0, name='mode_chg')
    m.add_constraints(dis + PMAX * mode <= PMAX, name='mode_dis')
    m.add_constraints(soc.isel(timesteps=0) - ETA * chg.isel(timesteps=0)
                      + dis.isel(timesteps=0) / ETA == 0.5 * CAP, name='soc_init')
    m.add_constraints(soc.isel(timesteps=slice(1, None))
                      - ETA * chg.isel(timesteps=slice(1, None))
                      + dis.isel(timesteps=slice(1, None)) / ETA
                      - soc.isel(timesteps=slice(None, -1)).assign_coords(
                          timesteps=steps[1:]) == 0, name='soc_rec')
    m.add_constraints(imp - exp + dis - chg
                      == xr.DataArray([load_for(t) for t in range(horizon)], coords=[steps]),
                      name='balance')

    price = xr.DataArray(prices, coords=[steps])
    m.add_objective((imp * price - exp * (0.8 * price)).sum())
    return m


def solve_linopy(model):
    # HiGHS option names, not Gurobi's. HAMLET's own linopy path passes `OutputFlag`/`LogToConsole`
    # whatever the solver, and HiGHS answers `getOptionIndex: Option "OutputFlag" is unknown` and
    # solves anyway -- harmless there, but it would print between every repetition here.
    status = model.solve(solver_name='highs', output_flag=False, log_to_console=False)
    assert status[0] == 'ok', f'linopy returned {status}'
    return float(model.objective.value)


def measure(build, solve, horizon, reps):
    """Median build and solve times in ms, plus the objective, for one backend."""
    builds, solves, objective = [], [], None
    for rep in range(reps):
        prices = prices_for(rep, horizon)
        t0 = time.perf_counter()
        model = build(horizon, prices)
        t1 = time.perf_counter()
        objective = solve(model)
        t2 = time.perf_counter()
        builds.append((t1 - t0) * 1e3)
        solves.append((t2 - t1) * 1e3)
    return statistics.median(builds), statistics.median(solves), objective


@skip_on_windows
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

    poi_build, poi_solve, poi_obj = measure(build_poi, solve_poi, HORIZON, REPS)
    lin_build, lin_solve, lin_obj = measure(build_linopy, solve_linopy, HORIZON, REPS)

    poi_total, lin_total = poi_build + poi_solve, lin_build + lin_solve
    print(f'\n  horizon={HORIZON}  reps={REPS}   (medians, ms)')
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
