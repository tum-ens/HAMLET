"""The solver x framework matrix -- all four combinations reach the same optimum.

`framework` (`linopy` | `poi`) and `solver` (`highs` | `gurobi`) are independent per-agent options,
so there are four supported combinations. Two of them had no coverage at all before this file:
`poi` + `gurobi`, which is what the POI backend was hardcoded to before !209 and which hid four
real defects for as long as it could not be exercised, and `linopy` + `gurobi`, which is what the
published paper runs used.

    python -m pytest tests/integration/executor/test_solver_backend_matrix.py -rs

**What is asserted is the objective value, and deliberately nothing else.** Equally-optimal
vertices differ between solvers and between backends -- !201 measured 3 row counts and 76 column
statistics moving on a Gurobi -> HiGHS switch, and !209 and !212 measured the same character of
movement for a backend switch. An equality assertion on setpoints or result tables would be wrong,
would flake, and would teach the next person to loosen it. Run-level divergence is degeneracy, is
expected, and is settled and closed as #198; this file works at the level of a single model, where
no state feedback can confound the answer.

**The read-back is the point, not a formality.** In !212 the `run_example` helper carried a literal
`framework: linopy` switch that became a no-op the instant the default flipped to `poi`, so both
arms of a backend comparison would have run `poi`, agreed, and passed while asserting nothing.
Every test below therefore checks what actually solved before it looks at any number, and a passing
matrix without that check would prove nothing. See `backend_models.identify`.

**Skips are loud and countable.** All four combinations are always parametrised and an unavailable
one skips from inside the test, so it appears in the report rather than vanishing from it -- most
environments skip both Gurobi cells, and a silent skip is how eight Windows tests hid a real
interpreter crash for a release (#202). `test_at_least_one_combination_is_available` fails rather
than skips if the whole matrix is empty, so this file cannot go green by testing nothing.
"""
import time

import pytest

from tests.backend_matrix import (COMBINATION_IDS, COMBINATIONS, FRAMEWORKS, REFERENCE, SOLVERS,
                                  describe, is_available, require)
from tests.backend_models import MPC_HORIZON, RTC_OPTIMUM, solve_mpc, solve_rtc

# The optimum of the MPC-shaped model at `rep=0`, measured once and committed. Every cell is
# compared against this fixed number rather than against another cell, so no arm is trivially
# self-comparing and a perturbed coefficient cannot be hidden by both arms moving together.
MPC_OPTIMUM = 21_287.25

# **Tied to the configured MIP gap, not to machine epsilon.** Both solvers default to a 1e-4
# *relative* MIP gap -- verified by reading the parameter back: HiGHS `mip_rel_gap` and Gurobi
# `MIPGap` are both 1e-4. Both models below carry a binary, so each solver is only obliged to
# return an incumbent within that gap of the true optimum; two such incumbents can therefore differ
# from each other, and from the reference, by up to twice the gap.
#
# The four cells actually agree to ~1e-12 relative (RTC: exactly), so this band is around eight
# orders looser than what is observed. That is deliberate. The band states what the solver
# *guarantees*, not what it happened to deliver, so it survives a solver version changing its
# branching or its presolve without anyone having to loosen it in a hurry -- which is the failure
# mode that turns a tolerance into a rubber stamp.
RELATIVE_TOLERANCE = 2e-4


def check(solved, framework, solver, expected):
    """Assert the requested combination solved, and that it reached `expected`.

    Identity first, and the order is the point. A wrong objective from the wrong backend is a
    different bug from a wrong objective from the right one, and the collapse case is the one that
    would otherwise pass silently -- two arms that have become the same arm still agree.
    """
    assert (solved.framework, solved.solver) == (framework, solver), (
        f'asked for {framework} + {solver} but {solved.framework} + {solved.solver} actually '
        f'solved -- the arms of this comparison have collapsed into each other and the objective '
        f'below would prove nothing')
    assert solved.objective == pytest.approx(expected, rel=RELATIVE_TOLERANCE), (
        f'{framework} + {solver} reached {solved.objective!r}, expected {expected!r} '
        f'(relative tolerance {RELATIVE_TOLERANCE:g}, twice the 1e-4 MIP gap both solvers default '
        f'to)')


# --------------------------------------------------------------------------------------------
# The matrix must not be able to shrink silently
# --------------------------------------------------------------------------------------------
def test_the_matrix_covers_every_supported_solver():
    """A solver added to HAMLET has to be added here too, or its cells would never be tested.

    `SOLVERS` is written out rather than read from `SUPPORTED_SOLVERS` precisely so that this test
    exists: importing the production tuple would make the matrix silently follow it, including
    downwards, and a matrix that quietly stops covering a cell is the failure this file is against.
    """
    from hamlet.executor.utilities.controller.poi_solver import SUPPORTED_SOLVERS

    assert set(SOLVERS) == set(SUPPORTED_SOLVERS), (
        f'the matrix covers {SOLVERS} but HAMLET supports {SUPPORTED_SOLVERS}; add the missing '
        f'solver to tests/backend_matrix.py')


def test_the_matrix_covers_every_optimisation_framework():
    """Same, for the frameworks: both must still be reachable from both controllers.

    `framework` is only consulted when the method is `optim`, and the RTC and the FBC each keep
    their own dispatch table, so a framework can be dropped from one and not the other. Checking
    both is what makes this test worth more than reading the constants back to themselves.
    """
    import hamlet.constants as c
    from hamlet.executor.utilities.controller.fbc.fbc import Fbc
    from hamlet.executor.utilities.controller.rtc.rtc import Rtc

    assert set(FRAMEWORKS) == {c.C_LINOPY, c.C_POI}, (
        f'the matrix covers {FRAMEWORKS}, but the optimisation frameworks are '
        f'{c.C_LINOPY!r} and {c.C_POI!r}')

    for controller in (Rtc(method=c.C_OPTIM), Fbc(method=c.C_OPTIM)):
        for framework in FRAMEWORKS:
            assert framework in controller.class_mapping, (
                f'{type(controller).__name__} no longer offers {framework!r}, so its cells in this '
                f'matrix test something no scenario can select')


def test_at_least_one_combination_is_available():
    """A matrix that skipped every cell would report green while testing nothing.

    This is the guard against trap 4: parametrisation that produces no executed case is not a
    pass. It fails rather than skips, on purpose.
    """
    available = [f'{f}+{s}' for f, s in COMBINATIONS if is_available(f, s)]
    assert available, (
        'not one of the four solver x framework combinations can solve on this machine, so every '
        'test in this file would skip. That is an environment failure, not a licence-free machine '
        '-- HiGHS ships with HAMLET. Run `uv sync`.')


def test_the_reference_combination_is_available():
    """linopy + HiGHS needs no licence and no extra, so its absence is a broken environment.

    It is also the implementation the POI backend was validated against, which is why its
    unavailability is worth a distinct failure rather than being folded into the test above.
    """
    framework, solver = REFERENCE
    assert is_available(framework, solver), (
        f'the reference combination {framework} + {solver} cannot solve. `highspy` is a hard '
        f'dependency, so this is not a missing licence. Run `uv sync`.')


# --------------------------------------------------------------------------------------------
# The comparison itself
# --------------------------------------------------------------------------------------------
@pytest.mark.solver
@pytest.mark.parametrize(('framework', 'solver'), COMBINATIONS, ids=COMBINATION_IDS)
def test_the_mpc_model_reaches_the_same_optimum(framework, solver):
    """A 24-step battery/load/market MILP -- the shape HAMLET solves ~911k times per year."""
    require(framework, solver)

    started = time.perf_counter()
    solved = solve_mpc(framework, solver)
    elapsed = (time.perf_counter() - started) * 1e3

    check(solved, framework, solver, MPC_OPTIMUM)

    # Informational only, and it gates nothing. One build plus one solve is not a benchmark: it
    # includes import and licence-check costs and has a sample size of one. Speed lives in
    # `tests/benchmarks/test_backend_speed.py`, which perturbs the model between repetitions and
    # reports medians. Visible with `-s`.
    print(f'\n  mpc  {framework}+{solver:<7} horizon={MPC_HORIZON}  '
          f'objective={solved.objective:.6f}  {elapsed:7.1f} ms (informational)')


@pytest.mark.solver
@pytest.mark.parametrize(('framework', 'solver'), COMBINATIONS, ids=COMBINATION_IDS)
def test_the_rtc_model_reaches_the_same_optimum(framework, solver):
    """One timestep of weighted setpoint deviation -- and the cleaner of the two tests.

    The RTC runs first within a timestep, so it is the controller for which both backends are
    provably handed identical inputs; the MPC reads a state of charge the RTC has already written.
    Its expected optimum is derived by hand at `backend_models.RTC_OPTIMUM`, so a disagreement here
    can be attributed without re-running anything.
    """
    require(framework, solver)

    started = time.perf_counter()
    solved = solve_rtc(framework, solver)
    elapsed = (time.perf_counter() - started) * 1e3

    check(solved, framework, solver, RTC_OPTIMUM)

    print(f'\n  rtc  {framework}+{solver:<7} '
          f'objective={solved.objective:.6f}  {elapsed:7.1f} ms (informational)')


@pytest.mark.solver
@pytest.mark.parametrize(('framework', 'solver'), COMBINATIONS, ids=COMBINATION_IDS)
def test_the_combination_that_solved_is_the_one_that_was_requested(framework, solver):
    """The collapse guard, stated on its own so it cannot be lost inside another assertion.

    Both models are checked, because they select their backend by different means: the POI path
    picks a solver-specific `Model` class at construction, the linopy path names a solver at solve
    time. A switch could no-op in one and not the other.
    """
    require(framework, solver)

    for label, solved in (('mpc', solve_mpc(framework, solver)),
                          ('rtc', solve_rtc(framework, solver))):
        assert (solved.framework, solved.solver) == (framework, solver), (
            f'{label}: asked for {framework} + {solver}, got {solved.framework} + {solved.solver}')


def test_the_report_header_names_every_cell():
    """The header line is how a reader sees which cells ran without reading the skip list.

    Pinned because it is easy to let a cell drop out of a summary string while the tests still
    parametrise over it, which would make the matrix look smaller than it is.
    """
    line = describe()

    for framework, solver in COMBINATIONS:
        assert f'{framework}+{solver}=' in line, f'{framework}+{solver} missing from: {line}'
