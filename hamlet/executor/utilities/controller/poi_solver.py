__author__ = "MarkusDoepfert"
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

"""Solver selection for the PyOptInterface backends.

PyOptInterface reaches a solver through its C API, so a backend works only once the solver's
*shared library* is loaded. That is the whole reason this module exists:

- **Gurobi** loads itself. `pyoptinterface.gurobi` finds a system Gurobi installation without
  `gurobipy` being installed, which is why the POI backends appeared to work while being
  unusable for anyone without that installation.
- **HiGHS** does not. `highspy` bundles HiGHS inside its `_core` extension module and exposes no
  shared library, so `pyoptinterface.highs` has nothing to find. `highsbox` ships the same HiGHS
  build as a plain `highs.dll` / `libhighs.so`; it is pinned to `highspy`'s version so that the
  linopy and POI backends solve with an identical solver.

Loading is idempotent and lazy -- importing HAMLET must not require any solver to be present.
"""

import glob
import logging
import os
import sys

from hamlet import msvc_runtime
from hamlet.executor.utilities.controller.solver_options import reproducibility_options

LOGGER = logging.getLogger(__name__)

# Solvers selectable via `controller.<rtc|fbc>.optimization.solver`.
SUPPORTED_SOLVERS = ('highs', 'gurobi')

# Set once `_load_highs_library` has run, so repeated model creation does not re-probe the disk.
_highs_loaded = None


def _highs_library_candidates():
    """Paths that may hold the HiGHS shared library shipped by `highsbox`.

    `highsbox` exposes its distribution directory but not the library itself, and the two
    platforms disagree about where it lives: Windows puts the loadable `highs.dll` in `bin/` and
    leaves only the `highs.lib` import stub in `lib/`, while Linux and macOS put the real object
    in `lib/`. Both are searched rather than branching on `os.name`, so a layout change in
    `highsbox` degrades to "not found" instead of a wrong path.
    """
    try:
        import highsbox
    except ImportError:
        return []

    dist = highsbox.highs_dist_dir()
    if sys.platform == 'win32':
        patterns = ('highs*.dll',)
    elif sys.platform == 'darwin':
        patterns = ('libhighs*.dylib',)
    else:
        patterns = ('libhighs*.so*',)

    candidates = []
    for sub in ('bin', 'lib'):
        for pattern in patterns:
            candidates.extend(sorted(glob.glob(os.path.join(dist, sub, pattern))))
    return candidates


def _load_highs_library():
    """Make `pyoptinterface.highs` usable, returning whether it is.

    Returns True if the library was already loaded (a system HiGHS on the loader path) or if one
    of the `highsbox` candidates loaded successfully.
    """
    global _highs_loaded
    if _highs_loaded is not None:
        return _highs_loaded

    # Before the import, not after: `pyoptinterface.highs` loads `highs.dll` at *import* time, so
    # this is the last moment at which we can refuse. A too-old C++ runtime does not make the
    # library fail to load -- it makes the first solve corrupt the process, at a location that
    # moves between runs. Raising here converts that into a message naming the culprit. See
    # `hamlet/msvc_runtime.py` and issue #202.
    unsupported = msvc_runtime.describe_unsupported_msvcp140()
    if unsupported is not None:
        raise RuntimeError(unsupported)

    from pyoptinterface import highs

    if highs.is_library_loaded():
        _highs_loaded = True
        return True

    for path in _highs_library_candidates():
        try:
            if highs.load_library(path):
                LOGGER.debug('Loaded the HiGHS shared library from %s', path)
                _highs_loaded = True
                return True
        except Exception:  # a wrong-architecture or truncated library must not abort the run
            LOGGER.debug('Could not load a HiGHS shared library from %s', path, exc_info=True)

    _highs_loaded = False
    return False


def get_solver_module(solver):
    """The `pyoptinterface` submodule for `solver`, with its shared library loaded.

    Raises ValueError for an unknown solver name and RuntimeError when the solver is known but
    its library cannot be loaded -- the two failures have different fixes, so they are not
    collapsed into one message.
    """
    if solver not in SUPPORTED_SOLVERS:
        raise ValueError(f"Unsupported solver: {solver}. "
                         f"Supported solvers are {', '.join(SUPPORTED_SOLVERS)}.")

    if solver == 'highs':
        if not _load_highs_library():
            raise RuntimeError(
                "The HiGHS shared library could not be loaded, so the 'poi' framework cannot "
                "use it. HAMLET installs `highsbox` for exactly this purpose -- `highspy` alone "
                "is not enough, as it bundles HiGHS inside its extension module and exposes no "
                "shared library. Run `uv sync`, or select `solver: gurobi`.")
        from pyoptinterface import highs
        return highs

    from pyoptinterface import gurobi
    if not gurobi.is_library_loaded():
        raise RuntimeError(
            "The Gurobi shared library could not be loaded. Install Gurobi and its licence, or "
            "select `solver: highs`, which ships with HAMLET and needs no licence.")
    return gurobi


def create_model(solver):
    """A silenced, empty model for `solver`.

    Both solvers are silenced, but not by the same means: Gurobi has to be quietened on the
    environment *before* the model exists, because it prints its licence banner at model
    creation, whereas HiGHS is configured on the model itself. `set_raw_parameter` is also
    type-strict for HiGHS -- it dispatches to `set_raw_option_bool`, which rejects `0`/`1`.
    """
    module = get_solver_module(solver)

    if solver == 'gurobi':
        env = module.Env(empty=True)
        env.set_raw_parameter("OutputFlag", 0)
        env.start()
        model = module.Model(env)
        model.set_raw_parameter("OutputFlag", 0)
        model.set_raw_parameter("LogToConsole", 0)
    else:
        model = module.Model()
        model.set_raw_parameter("output_flag", False)
        model.set_raw_parameter("log_to_console", False)

    import pyoptinterface as poi
    model.set_model_attribute(poi.ModelAttribute.Silent, True)
    return model


def apply_reproducibility_options(model, solver, time_limit):
    """Pin the thread count, and the wall-clock limit in seconds, on `model`.

    `solver_options.reproducibility_options` decides what is applied and under which names; this
    only puts it on a PyOptInterface model.
    """
    for name, value in reproducibility_options(solver, time_limit).items():
        model.set_raw_parameter(name, value)


def raise_unless_optimal(status, agent_id, time_limit):
    """Refuse any solve that did not reach a proven optimum.

    `TerminationStatusCode.TIME_LIMIT` used to be whitelisted alongside `OPTIMAL`, so a solve that
    ran out of time returned its incumbent and the simulation carried on with a suboptimal
    schedule and no signal at all -- which made results a function of machine load (#204). Every
    other bad status was equally ignored: the `raise` was present but commented out, so a failed
    solve only printed.

    Raising is not new behaviour for HAMLET as a whole. The linopy controllers have always raised
    on a non-`ok` status; this is the POI half catching up.
    """
    import pyoptinterface as poi

    if status == poi.TerminationStatusCode.OPTIMAL:
        return

    detail = ''
    if status == poi.TerminationStatusCode.TIME_LIMIT:
        detail = (f' The solve hit its {time_limit} s limit, so the schedule it returned is an '
                  f'incumbent rather than an optimum. Raise `time_limit` under `optimization`, or '
                  f'run on a less loaded machine.')

    raise ValueError(f'Optimization failed for agent {agent_id}: solver returned "{status}".'
                     f'{detail}')
