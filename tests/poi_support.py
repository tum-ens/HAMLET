"""Shared PyOptInterface support for the test suite: which backend to use.

Both the slack integration tests and the backend speed tests need the same answer, and they must
not drift apart -- a benchmark that silently measured a different solver from the one the
correctness tests exercise would be worse than no benchmark.

**`skip_on_windows` is gone, deliberately.** Eight tests carried it because PyOptInterface plus
the `highsbox` HiGHS crashed the interpreter on Windows with an access violation. That was issue
#202, its cause is a C++ runtime the Windows loader picked by base name, and `hamlet/__init__.py`
now claims that name before `pandas` can. These tests run on Windows again; if the fix ever stops
working the suite fails with the `RuntimeError` from `poi_solver`, naming the offending DLL,
rather than taking the process down. Do not reintroduce a platform skip here without a
measurement -- reinstating one would hide exactly the regression the fix exists to prevent.
"""


def can_solve(module):
    """Whether this backend can actually solve, not merely link.

    `is_library_loaded()` answers a narrower question than it looks like it does. Gurobi's shared
    library loads perfectly well without a valid licence and raises `GurobiError: License expired`
    at `optimize()` -- so a loadable-only check turns "no licence" into test *failures* rather than
    skips, which is exactly what CI hits. Solving a two-variable LP is the honest probe.
    """
    import pyoptinterface as poi

    try:
        model = module.Model()
        x = model.add_variable(lb=0, ub=1)
        model.set_objective(1.0 * x, poi.ObjectiveSense.Maximize)
        model.optimize()
        return model.get_value(x) is not None
    except Exception:
        return False


def available_backend():
    """The first PyOptInterface solver that is present, loadable and able to solve.

    HiGHS is tried first because it is what HAMLET ships and what CI runs; preferring Gurobi would
    mean a developer with a licence and CI silently exercise different backends. This goes through
    the production loader rather than probing `is_library_loaded()` directly, so that HiGHS is
    found here for the same reason it is found at runtime -- via `highsbox`. Probing directly made
    this fall through to Gurobi on any machine with a system Gurobi installation.
    """
    from hamlet.executor.utilities.controller import poi_solver

    for name in poi_solver.SUPPORTED_SOLVERS:
        try:
            module = poi_solver.get_solver_module(name)
        except (ImportError, ValueError, RuntimeError):
            continue
        if can_solve(module):
            return module
    return None
