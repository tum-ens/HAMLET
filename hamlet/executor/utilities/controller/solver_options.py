__author__ = "MarkusDoepfert"
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

"""The solver options that decide whether a run is reproducible.

Both frameworks reach the same two solvers, so the options below are named once here rather than
spelled out per backend. This module deliberately covers *only* the options that affect whether
two runs of the same scenario produce the same numbers; output suppression is left where each
framework already does it, because the two do it at different moments (roadmap item #11).

Two options qualify, and both were unset before issue #204:

- **`threads = 1`.** Both solvers default to `0`, meaning "pick a thread count". A parallel MIP
  explores nodes in an order that depends on how those threads interleave, so which of several
  equally optimal solutions comes back depends on how busy the machine is -- and HAMLET's agent
  models are degenerate enough for that to be a real difference rather than a formality. Each
  model is a few hundred variables over a 24-step horizon; parallelism belongs across agents, not
  inside one solve.
- **The time limit, in seconds.** HAMLET divided the configured value by 60 for years, so the
  example's `time_limit: 120` reached the solver as 2 seconds. Under `framework: linopy` that was
  inert -- the value was sent under Gurobi's `TimeLimit` key, which HiGHS does not know and
  silently discards -- but PyOptInterface sends HiGHS' own `time_limit`, so making POI the default
  activated a limit that had never applied. Measured under artificial load, 1 of 192 solves in the
  shipped example hit it, and its suboptimal incumbent was accepted silently.

The second half of that is in `poi_solver.raise_unless_optimal`: a limit is only a safeguard if
reaching it is an error.
"""

#: Solvers whose option names are known here. Kept separate from `poi_solver.SUPPORTED_SOLVERS`,
#: which answers a different question (can this framework load the library).
KNOWN_SOLVERS = ('highs', 'gurobi')

#: HAMLET's name for each option, per solver. HiGHS uses lower-case snake case and Gurobi
#: capitalised camel case, and neither accepts the other's spelling.
_OPTION_NAMES = {
    'highs': {'threads': 'threads', 'time_limit': 'time_limit'},
    'gurobi': {'threads': 'Threads', 'time_limit': 'TimeLimit'},
}

#: One thread per solve. See the module docstring for why this is not a performance setting.
THREADS = 1


def reproducibility_options(solver, time_limit=None):
    """The options above under `solver`'s own names, as a plain dict.

    `time_limit` is in **seconds** and is omitted when None, which leaves the solver's own default
    (no limit) in place. The value is passed through unscaled -- that is the fix for #204, so a
    division reappearing here is a regression, not a unit conversion.

    Types matter to the callers: PyOptInterface dispatches `set_raw_parameter` on the Python type
    of the value and rejects an int where the solver declares a double, so the thread count stays
    an `int` and the limit is coerced to `float`.
    """
    if solver not in _OPTION_NAMES:
        raise ValueError(f"Unsupported solver: {solver}. "
                         f"Supported solvers are {', '.join(KNOWN_SOLVERS)}.")

    names = _OPTION_NAMES[solver]
    options = {names['threads']: THREADS}
    if time_limit is not None:
        options[names['time_limit']] = float(time_limit)
    return options
