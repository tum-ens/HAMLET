"""One MPC-shaped and one RTC-shaped model, expressed in both frameworks and either solver.

These are the models the solver x framework matrix compares, and the models
`tests/benchmarks/test_backend_speed.py` times. They live here rather than in either test file so
that the correctness matrix and the benchmark provably solve the *same* problem -- a benchmark
measuring a different model from the one the correctness tests exercise would be worse than no
benchmark, which is the argument `tests/poi_support.py` already makes about the solver.

They are deliberately *representative*, not real controllers. Comparing backends on a whole run is
already settled and closed: the two agree on the models and diverge downstream because a
degenerate MILP's tie breaks differently and `rtc_base.update_socs` feeds that vertex into the next
timestep (#198, and `tests/e2e/test_backend_equivalence.py`'s permanent xfail). What is worth
testing across all four cells is therefore a single solve of a single model, where no state
feedback can confound the answer.

**MPC shape** -- a battery with a binary charge/discharge mode flag, an inflexible load, and market
import/export over a 24-step horizon, minimising energy cost. This is the shape HAMLET solves
~911k times per scenario-year.

**RTC shape** -- one timestep, minimising a weighted sum of deviations from setpoints subject to
the energy balance, with HAMLET's own component weights. Its targets are deliberately infeasible
together, so the optimum must spend the cheapest deviation first and then the next cheapest: all
three weights bind, and perturbing any one of them moves the objective. A model whose optimum
ignored a coefficient would pass this comparison while that coefficient was wrong.

Both carry a binary, so both are MILPs -- which is what makes the matrix's tolerance a statement
about the configured MIP gap rather than about floating point.
"""
from collections import namedtuple

# A solved model, with the framework and solver read back off the model rather than echoed from
# the request. See `identify`.
Solved = namedtuple('Solved', 'objective framework solver')

# Per-solver log silencing for the linopy path. Named per solver on purpose: HAMLET's own linopy
# controllers pass Gurobi's `OutputFlag`/`LogToConsole` whatever the solver, and HiGHS answers
# `getOptionIndex: Option "OutputFlag" is unknown` and solves anyway. Harmless there, noisy here.
# Correcting it in the production path is roadmap item #11.
LINOPY_SILENCE = {'highs': {'output_flag': False, 'log_to_console': False},
                  'gurobi': {'OutputFlag': 0, 'LogToConsole': 0}}

# ---------------------------------------------------------------------------------------------
# MPC-shaped model
# ---------------------------------------------------------------------------------------------
CAP, PMAX, ETA = 10_000.0, 5_000.0, 0.95
MPC_HORIZON = 24


def mpc_prices(rep, horizon=MPC_HORIZON):
    """A price vector, varying with `rep` so a benchmark's repetitions are genuine re-solves.

    A re-solve of an unchanged model short-circuits in both solvers and reports ~0.01 ms, which
    looks like a spectacular result and measures nothing.
    """
    return [0.30 + 0.01 * ((rep + t) % 7) for t in range(horizon)]


def mpc_load(t):
    """A deterministic inflexible-load profile, in W."""
    return 2_000.0 + 500.0 * (t % 5)


def build_mpc_poi(solver, horizon, prices):
    import pyoptinterface as poi

    from hamlet.executor.utilities.controller.poi_solver import create_model

    m = create_model(solver)
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
                                poi.ConstraintSense.Equal, mpc_load(t))

    m.set_objective(sum(prices[t] * imp[t] - 0.8 * prices[t] * exp[t] for t in range(horizon)),
                    poi.ObjectiveSense.Minimize)
    return m


def build_mpc_linopy(horizon, prices):
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
                      == xr.DataArray([mpc_load(t) for t in range(horizon)], coords=[steps]),
                      name='balance')

    price = xr.DataArray(prices, coords=[steps])
    m.add_objective((imp * price - exp * (0.8 * price)).sum())
    return m


# ---------------------------------------------------------------------------------------------
# RTC-shaped model
# ---------------------------------------------------------------------------------------------
# HAMLET's own RTC weights, from `optim_poi.define_objective` and its linopy twin: the higher the
# weight, the higher the penalty for deviating from the setpoint.
W_BATTERY, W_HP, W_MARKET = 1, 3, 4

# Powers in W, positive into the main meter, as everywhere in HAMLET.
RTC_PV, RTC_LOAD = 4_000.0, -9_000.0            # fixed generation and fixed consumption
RTC_HP_MIN, RTC_HP_TARGET = -5_000.0, -3_000.0
RTC_BATTERY_MAX, RTC_BATTERY_TARGET = 1_000.0, 0.0
RTC_MARKET_MAX, RTC_MARKET_TARGET = 20_000.0, 2_000.0

# The optimum, by hand, so that this model can be checked without trusting a solver. The balance
# needs hp + battery + market = 5,000 W while the targets supply -1,000 W, so 6,000 W of deviation
# has to be found. It is bought cheapest-first: 1,000 W of battery (weight 1, at its bound), then
# 3,000 W of heat pump (weight 3, at its bound), then the remaining 2,000 W from the market
# (weight 4).
#     1*1,000 + 3*3,000 + 4*2,000 = 18,000
# Every weight binds, so perturbing any of the three moves this number.
RTC_OPTIMUM = 18_000.0


def build_rtc_poi(solver):
    import pyoptinterface as poi

    from hamlet.executor.utilities.controller.poi_solver import create_model

    m = create_model(solver)
    pv = m.add_variable(lb=RTC_PV, ub=RTC_PV)
    load = m.add_variable(lb=RTC_LOAD, ub=RTC_LOAD)
    hp = m.add_variable(lb=RTC_HP_MIN, ub=0.0)
    market = m.add_variable(lb=-RTC_MARKET_MAX, ub=RTC_MARKET_MAX)

    # The battery is split into charge and discharge behind a binary mode flag, as HAMLET models
    # it, so this is a MILP rather than an LP and the MIP gap is the relevant tolerance.
    charge = m.add_variable(lb=0.0, ub=RTC_BATTERY_MAX)
    discharge = m.add_variable(lb=0.0, ub=RTC_BATTERY_MAX)
    mode = m.add_variable(domain=poi.VariableDomain.Binary)
    m.add_linear_constraint(charge - RTC_BATTERY_MAX * mode, poi.ConstraintSense.LessEqual, 0.0)
    m.add_linear_constraint(discharge + RTC_BATTERY_MAX * mode,
                            poi.ConstraintSense.LessEqual, RTC_BATTERY_MAX)
    battery = discharge - charge

    m.add_linear_constraint(pv + load + hp + battery + market, poi.ConstraintSense.Equal, 0.0)

    objective = []
    for expression, target, weight in ((hp, RTC_HP_TARGET, W_HP),
                                       (battery, RTC_BATTERY_TARGET, W_BATTERY),
                                       (market, RTC_MARKET_TARGET, W_MARKET)):
        # deviation >= |expression - target|, which the minimisation drives to equality.
        deviation = m.add_variable(lb=0.0)
        m.add_linear_constraint(deviation - expression, poi.ConstraintSense.GreaterEqual, -target)
        m.add_linear_constraint(deviation + expression, poi.ConstraintSense.GreaterEqual, target)
        objective.append(weight * deviation)

    m.set_objective(sum(objective), poi.ObjectiveSense.Minimize)
    return m


def build_rtc_linopy():
    import linopy

    m = linopy.Model()
    pv = m.add_variables(lower=RTC_PV, upper=RTC_PV, name='pv')
    load = m.add_variables(lower=RTC_LOAD, upper=RTC_LOAD, name='load')
    hp = m.add_variables(lower=RTC_HP_MIN, upper=0.0, name='hp')
    market = m.add_variables(lower=-RTC_MARKET_MAX, upper=RTC_MARKET_MAX, name='market')

    charge = m.add_variables(lower=0.0, upper=RTC_BATTERY_MAX, name='charge')
    discharge = m.add_variables(lower=0.0, upper=RTC_BATTERY_MAX, name='discharge')
    mode = m.add_variables(name='mode', binary=True)
    m.add_constraints(charge - RTC_BATTERY_MAX * mode <= 0.0, name='mode_charge')
    m.add_constraints(discharge + RTC_BATTERY_MAX * mode <= RTC_BATTERY_MAX, name='mode_discharge')
    battery = discharge - charge

    m.add_constraints(pv + load + hp + battery + market == 0.0, name='balance')

    objective = []
    for label, expression, target, weight in (('hp', hp, RTC_HP_TARGET, W_HP),
                                              ('battery', battery, RTC_BATTERY_TARGET, W_BATTERY),
                                              ('market', market, RTC_MARKET_TARGET, W_MARKET)):
        deviation = m.add_variables(lower=0.0, name=f'{label}_deviation')
        m.add_constraints(deviation - expression >= -target, name=f'{label}_deviation_up')
        m.add_constraints(deviation + expression >= target, name=f'{label}_deviation_down')
        objective.append(weight * deviation)

    m.add_objective(sum(objective))
    return m


# ---------------------------------------------------------------------------------------------
# Solving, and reading back what actually solved
# ---------------------------------------------------------------------------------------------
def solve_poi(model):
    import pyoptinterface as poi

    model.optimize()
    status = model.get_model_attribute(poi.ModelAttribute.TerminationStatus)
    assert status == poi.TerminationStatusCode.OPTIMAL, f'POI returned {status}'
    return model.get_model_attribute(poi.ModelAttribute.ObjectiveValue)


def solve_linopy(model, solver):
    status = model.solve(solver_name=solver, **LINOPY_SILENCE[solver])
    assert status[0] == 'ok', f'linopy returned {status}'
    return float(model.objective.value)


def identify(model):
    """The framework and solver a *solved* model actually used, read off the model itself.

    This exists because of the failure mode that makes a comparison test worthless. In !212 the
    `run_example` helper carried a literal `framework: linopy` switch that became a no-op the
    instant the default flipped to `poi`; both arms of a backend comparison would then have run
    `poi`, agreed with each other, and passed while asserting nothing. Arms that silently collapse
    into each other are the worst failure available here, so nothing in this module trusts the
    argument it was called with -- the identity is recovered from the object that did the work.

    PyOptInterface has a separate `Model` class per solver, so its module name *is* the solver.
    linopy has one `Model` and records the solver it last used in `solver_name`.
    """
    module = type(model).__module__
    if module.startswith('linopy'):
        return 'linopy', getattr(model, 'solver_name', None)
    if module.startswith('pyoptinterface'):
        return 'poi', module.rsplit('.', 1)[-1]
    raise AssertionError(f'unrecognised model class {type(model)!r} from module {module!r}')


def solve_mpc(framework, solver, rep=0):
    """Solve the MPC-shaped model, returning its optimum and what actually solved it."""
    if framework == 'poi':
        model = build_mpc_poi(solver, MPC_HORIZON, mpc_prices(rep))
        objective = solve_poi(model)
    elif framework == 'linopy':
        model = build_mpc_linopy(MPC_HORIZON, mpc_prices(rep))
        objective = solve_linopy(model, solver)
    else:
        raise ValueError(f'unknown framework {framework!r}')
    return Solved(objective, *identify(model))


def solve_rtc(framework, solver):
    """Solve the RTC-shaped model, returning its optimum and what actually solved it."""
    if framework == 'poi':
        model = build_rtc_poi(solver)
        objective = solve_poi(model)
    elif framework == 'linopy':
        model = build_rtc_linopy()
        objective = solve_linopy(model, solver)
    else:
        raise ValueError(f'unknown framework {framework!r}')
    return Solved(objective, *identify(model))
