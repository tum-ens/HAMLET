"""Shared PyOptInterface support for the test suite: which backend to use, and when to skip.

Both the slack integration tests and the backend speed tests need the same two answers, and they
must not drift apart -- a benchmark that silently measured a different solver from the one the
correctness tests exercise would be worse than no benchmark.
"""
import sys

import pytest

# PyOptInterface + the `highsbox` HiGHS crash the *interpreter* on Windows -- an access violation
# (0xC0000005) at a moving, unrelated location (ast, pathlib, pytest internals), which is the
# signature of memory corruption rather than a fault where it is reported.
#
# What it takes to reproduce, measured: two or more HiGHS solves inside one pytest process. A
# single solve always passes, and 200 consecutive solves *outside* pytest are fine -- which is why
# this is a test-harness limitation and not a runtime one. Windows simulations run normally, and
# so does the POI backend itself.
#
# Ruled out: pytest's output capture (`-s` still crashes, and silencing the solver does not help),
# `np.inf` bounds, HAMLET's own code (it reproduces on an unmodified tree with only `highsbox`
# installed), and either dependency's version alone. It does not reproduce on Linux at any version
# tested, and CI is Linux, so these tests keep their coverage where it counts.
#
# There is an escape -- pyoptinterface >= 0.5.1 with highsbox >= 1.12.0 does not crash -- but that
# HiGHS is 3-5x slower on HAMLET-shaped models, which is most of the speedup this backend exists
# to deliver. Skipping on one platform is the cheaper trade. Revisit if the cause is found.
skip_on_windows = pytest.mark.skipif(
    sys.platform == 'win32',
    reason='PyOptInterface + highsbox crash the interpreter on Windows under pytest; '
           'the backend itself works there, and CI covers these tests on Linux')


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
