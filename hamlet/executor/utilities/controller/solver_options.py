__author__ = "MarkusDoepfert"
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

"""The solver options a HAMLET run sets: reproducibility, and keeping the solver quiet.

Both frameworks reach the same two solvers, so the options below are named once here rather than
spelled out per backend. Output suppression joined this module with issue #199 (roadmap item
#11); before that, `mpc_linopy` and `optim_linopy` sent the literal dict
`{'OutputFlag': 0, 'LogToConsole': 0}` to whichever solver was configured. Those are **Gurobi's**
names. HiGHS discards them unrecognised, so under the default backend nothing was actually being
switched off -- the `sys.stdout = open(os.devnull, 'w')` around the call was doing all the work.
That is the same defect as the `TimeLimit` key in #204, and it is why every option here is now
looked up per solver rather than written out inline.

Three options qualify. The first two were unset before issue #204:

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

The third is **output suppression**, which affects nothing numeric and is here only so that its
names are looked up the same way. HiGHS calls the two flags `output_flag` and `log_to_console`
and wants booleans; Gurobi calls them `OutputFlag` and `LogToConsole` and wants 0/1.
"""

#: Solvers whose option names are known here. Kept separate from `poi_solver.SUPPORTED_SOLVERS`,
#: which answers a different question (can this framework load the library).
KNOWN_SOLVERS = ('highs', 'gurobi')

#: HAMLET's name for each option, per solver. HiGHS uses lower-case snake case and Gurobi
#: capitalised camel case, and neither accepts the other's spelling.
_OPTION_NAMES = {
    'highs': {'threads': 'threads', 'time_limit': 'time_limit',
              'output_flag': 'output_flag', 'log_to_console': 'log_to_console'},
    'gurobi': {'threads': 'Threads', 'time_limit': 'TimeLimit',
               'output_flag': 'OutputFlag', 'log_to_console': 'LogToConsole'},
}

#: The value each solver wants for "off". HiGHS' options are declared bool and Gurobi's int, and
#: as with the time limit in #204, the *type* is part of getting the name right.
_OFF = {'highs': False, 'gurobi': 0}

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


def quiet_options(solver):
    """`solver`'s own names for "do not print anything", as a plain dict.

    Separate from `reproducibility_options` because the two answer different questions and only
    one of them can move a number. Callers that want both merge them; keeping them apart means a
    reader can see at a glance that nothing in here affects the solution.
    """
    if solver not in _OPTION_NAMES:
        raise ValueError(f"Unsupported solver: {solver}. "
                         f"Supported solvers are {', '.join(KNOWN_SOLVERS)}.")

    names, off = _OPTION_NAMES[solver], _OFF[solver]
    return {names['output_flag']: off, names['log_to_console']: off}
