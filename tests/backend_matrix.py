"""Which of the four (framework, solver) combinations this machine can actually exercise.

HAMLET ships two modelling frameworks and supports two solvers, so `framework` x `solver` has
four live cells. Only two of them had any coverage before this module existed:

    | framework      | highs                     | gurobi     |
    |----------------|---------------------------|------------|
    | poi (default)  | covered                   | uncovered  |
    | linopy         | covered (reference impl)  | uncovered  |

Both uncovered cells have history. `poi` + `gurobi` is what the POI backend was hardcoded to
before !209 -- it worked only on a machine with a system Gurobi installation and was structurally
broken everywhere else, and !209 found four real defects in that backend the moment it could
finally be exercised. `linopy` + `gurobi` is what the published paper runs used. Neither was
tested, and `solver: gurobi` is documented and supported.

**Availability is answered by solving, never by importing.** That is the lesson `can_solve` was
written for and it is reused here rather than restated: Gurobi's shared library loads perfectly
well without a valid licence and only fails at `optimize()`, so an import or `is_library_loaded()`
check turns "no licence" into test *failures* instead of skips. `linopy.available_solvers` is an
import check of exactly that kind, so it is deliberately not used either -- it would list `gurobi`
on a machine whose licence expired last week.

**The two frameworks do not reach Gurobi the same way, and this is not a detail.** PyOptInterface
links a *system* Gurobi installation through its C API with no Python package involved; linopy goes
through `gurobipy`, which is an optional extra (`uv sync --extra gurobi`) and carries its own copy
of the solver. So the two cells can be available independently, they can be running different
Gurobi *versions* on the same machine, and each needs its own probe.
"""
import contextlib
import os
import sys
import tempfile
from functools import lru_cache

import pytest

from tests.backend_models import LINOPY_SILENCE
from tests.poi_support import can_solve

# The frameworks selectable per agent via `controller.<rtc|fbc>.optimization.framework`.
FRAMEWORKS = ('linopy', 'poi')

# Stated here rather than read from `poi_solver.SUPPORTED_SOLVERS` so that adding a solver to
# HAMLET cannot silently enlarge -- or worse, silently shrink -- this matrix.
# `test_the_matrix_covers_every_supported_solver` holds the two in step.
SOLVERS = ('highs', 'gurobi')

COMBINATIONS = tuple((framework, solver) for framework in FRAMEWORKS for solver in SOLVERS)

# pytest ids, so a skipped cell is named in the report rather than being an anonymous number.
COMBINATION_IDS = tuple(f'{framework}+{solver}' for framework, solver in COMBINATIONS)

# The reference implementation. linopy on HiGHS needs no licence and no extra -- `highspy` is a
# hard dependency -- so its absence is a broken environment, not a machine without a licence.
REFERENCE = ('linopy', 'highs')


@contextlib.contextmanager
def _silenced():
    """Swallow output written to file descriptor 1, not merely to `sys.stdout`.

    Both solvers print from C -- Gurobi its licence banner at environment creation, HiGHS its
    version line -- so `contextlib.redirect_stdout`, which only rebinds a Python attribute, does
    not stop either. That matters for the probes below: `can_solve` deliberately creates a bare
    `Model()` with no silencing, and it is shared with the rest of the suite, so it is not forked
    to add a quiet mode. Without this, every `pytest` run would open with a Gurobi banner, because
    the report header probes.

    Restored in a `finally`, which is the part worth being deliberate about: the two linopy
    controllers still do `sys.stdout = open(os.devnull, 'w')` around their solve
    (`optim_linopy.py:241`, `mpc_linopy.py:234`), which leaks a file object per solve and never
    restores it if the solve raises. That is roadmap item #11; do not copy the pattern here.
    """
    sys.stdout.flush()
    with tempfile.TemporaryFile() as sink:
        saved = os.dup(1)
        try:
            os.dup2(sink.fileno(), 1)
            yield
        finally:
            sys.stdout.flush()
            os.dup2(saved, 1)
            os.close(saved)


@lru_cache(maxsize=None)
def is_available(framework, solver):
    """Whether this combination can solve here -- probed once per session, then cached.

    Cached because the probe is not free: starting a Gurobi environment and running a linopy solve
    both cost tens of milliseconds, and this is asked once per parametrised test plus once for the
    report header.
    """
    if framework == 'poi':
        return _poi_can_solve(solver)
    if framework == 'linopy':
        return _linopy_can_solve(solver)
    raise ValueError(f'unknown framework {framework!r}; expected one of {FRAMEWORKS}')


def _poi_can_solve(solver):
    """Reach the solver the way production does, then actually solve with it.

    Going through `get_solver_module` rather than probing `pyoptinterface.<solver>` directly is
    what makes HiGHS findable at all: `highspy` hides HiGHS inside its extension module, and it is
    `poi_solver` that locates the `highsbox` shared library instead.
    """
    from hamlet.executor.utilities.controller import poi_solver

    try:
        with _silenced():
            module = poi_solver.get_solver_module(solver)
            return can_solve(module)
    except (ImportError, ValueError, RuntimeError):
        return False


def _linopy_can_solve(solver):
    """The same question for linopy: solve a one-variable LP and see whether an answer comes back.

    Deliberately the same shape as `can_solve` rather than a call to it -- linopy has no
    `pyoptinterface` module to hand it -- so that both halves of the matrix answer "can it solve"
    and not "does it import".
    """
    try:
        import linopy

        model = linopy.Model()
        x = model.add_variables(lower=0, upper=1, name='x')
        model.add_objective(-1 * x.sum())
        with _silenced():
            status = model.solve(solver_name=solver, **LINOPY_SILENCE[solver])
        return status[0] == 'ok'
    except Exception:
        return False


def require(framework, solver):
    """Skip the calling test unless this combination can solve, naming why it could not.

    The skip has to be *loud and countable*: most environments skip both Gurobi cells, and a silent
    skip is how eight Windows tests hid a real interpreter crash for a release (#202). So every
    combination is always parametrised and skipped from inside the test -- never filtered out of
    the parametrisation, which would make an unavailable cell vanish from the report instead of
    appearing in it. `pytest -rs` prints every skip reason.
    """
    if not is_available(framework, solver):
        pytest.skip(f'{framework} + {solver} cannot solve here: {_why_not(framework, solver)}')


def _why_not(framework, solver):
    """The likeliest reason a cell is unavailable, for the skip message."""
    if solver == 'gurobi' and framework == 'linopy':
        return ('linopy reaches Gurobi through gurobipy, an optional extra '
                '(uv sync --extra gurobi), and needs a valid licence')
    if solver == 'gurobi':
        return 'PyOptInterface links a system Gurobi installation, which needs a valid licence'
    return 'no loadable solver library -- run `uv sync`'


def describe():
    """One line naming every cell and whether it will run, for the pytest report header."""
    cells = '  '.join(f'{framework}+{solver}={"ok" if is_available(framework, solver) else "SKIP"}'
                      for framework, solver in COMBINATIONS)
    return f'solver x framework matrix: {cells}'
