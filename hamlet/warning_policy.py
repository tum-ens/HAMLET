__author__ = "MarkusDoepfert"
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

"""Which warnings a HAMLET run quietens, named one by one.

Until issue #199 this was a single `warnings.filterwarnings("ignore")` at the top of
`hamlet/executor/setup.py`, which `hamlet/__init__.py` imports -- so `import hamlet` silenced
every warning in the process, including HAMLET's own and every dependency's. A second blanket
filter (`FutureWarning`, all modules) sat at the top of `hamlet/creator/agents/agents.py`.

**Two things were wrong with that, and they need separate fixes.**

*It was global, and it was an import side effect.* A library has no business editing the
process-wide warning filters of whatever imports it. That is fixed by scope: nothing here runs at
import, and `quiet_known_noise()` is entered around a Creator or Executor run and left when it
finishes.

*It was blanket.* That is fixed by enumeration: `SUPPRESSED` below lists every message this
policy hides, with the reason and the issue tracking its removal. Anything not on that list --
a new deprecation, a `UserWarning` from pandas, a `RuntimeWarning` from numpy, a
`warnings.warn` added to HAMLET tomorrow -- reaches the user.

**What was actually being hidden, measured rather than guessed.** Both shipped scenarios were run
end to end with the filters lifted after import and every warning recorded:

| scenario | warnings raised | distinct locations | from HAMLET's own code |
|---|---|---|---|
| `examples/create_simple_scenario` (24 steps) | 4,688 | 30 | 100 % |
| `tests/e2e/scenarios/grid_golden` (30 steps) | 7,805 | 36 | 100 % |

Not one came from a dependency warning about itself. The volume is HAMLET calling polars 0.20
APIs that polars 0.20 already deprecates -- the inventory ROADMAP item #12 (Polars 1.x migration)
exists to clear -- and that is what `SUPPRESSED` covers.

**Why the blanket filter could not simply be deleted.** Python already ignores
`DeprecationWarning` outside `__main__`, so most of that count is invisible by default and
overstates what a user would see. What a bare removal actually prints was measured separately:
**491 lines for a 30-timestep run**, of which 360 are polars' `MapWithoutReturnDtypeWarning`,
whose origin Python records as `sys:1` so it never deduplicates. That is ~16 lines per timestep,
or roughly 140,000 over a simulated year. The concern that a bare removal would simply be
reinstated by the next person is correct, and this module is why it does not have to be.

**What is deliberately left visible**, because it is signal rather than noise. Two live HAMLET
defects, both in Section 14a, both hidden for as long as the blanket filter existed:

- `enwg_14a.py:517` and `:522` -- 64 per run of pandas' `UserWarning: Boolean Series key will be
  reindexed to match DataFrame index`, from chained boolean masks. Issue #210.
- `enwg_14a.py:87` -- 30 per run of `FutureWarning: Non-integer 'periods' in pd.date_range ...`,
  which pandas will turn into an error. Issue #211.

These fire only where Section 14a actually runs, which today is `grid_golden` -- the shipped
examples configure the restriction but leave the grid inactive. That is a property of the shipped
configs, not a guarantee. Neither is suppressed here: hiding them again is how they stayed hidden.

The list is **not** an exhaustive inventory of what a run prints. A few dependency warnings
survive too (pandapower's `output_writer` and `run_control` each raise a `FutureWarning` on the
grid path, roughly once per run rather than per timestep), and that is the intended state -- they
are somebody else's code telling us something true.

The `'S'` frequency aliases that produced the Creator's own `FutureWarning`s were fixed at source
rather than listed here: `'S'` and `'s'` are the same offset, so the rename costs nothing.
"""
import warnings
from contextlib import contextmanager
from functools import wraps

from polars.exceptions import MapWithoutReturnDtypeWarning

#: Every message this policy hides, as (category, message regex, why).
#:
#: The regex is matched against the *start* of the warning's message, which is what
#: `warnings.filterwarnings` does. Matching the message rather than the module is deliberate:
#: polars reports these against the HAMLET file that called it, so a module-scoped filter would
#: have to name half the package and would swallow unrelated warnings raised from the same files.
#:
#: If polars rewords one of these, the pattern stops matching and the warning becomes visible
#: again. That is the safe direction to fail in, and `tests/unit/test_warning_policy.py` pins each
#: pattern against the message it was added for so the drift is caught before a user sees it.
SUPPRESSED = (
    (DeprecationWarning, r'`cumsum` is deprecated',
     'polars 0.20 renamed it to `cum_sum`; ROADMAP item #12'),
    (DeprecationWarning, r'`groupby` is deprecated',
     'polars 0.20 renamed it to `group_by`; ROADMAP item #12'),
    (DeprecationWarning, r'`apply` is deprecated',
     'polars 0.20 renamed it to `map_elements`; ROADMAP item #12'),
    (DeprecationWarning, r'The `axis` parameter for `DataFrame\.\w+` is deprecated',
     'polars 0.20 wants the explicit `*_horizontal` methods; ROADMAP item #12'),
    (DeprecationWarning, r"Use of `how='outer'` should be replaced with `how='full'`",
     'polars 0.20 renamed the join strategy; ROADMAP item #12'),
    (DeprecationWarning, r'The default coalesce behavior of left join will change',
     'polars 0.20 announcing a 1.x default change; ROADMAP item #12'),
    (MapWithoutReturnDtypeWarning, r'Calling `map_elements` without specifying `return_dtype`',
     'the 360-per-run one. Adding `return_dtype` can change a column dtype, so it belongs '
     'with ROADMAP item #12 and its golden-master re-check, not here'),
)


@contextmanager
def quiet_known_noise():
    """Hide exactly the warnings in `SUPPRESSED` for the duration of the block.

    Restores the previous filters on exit, including when the block raises -- which the
    `sys.stdout = open(os.devnull, 'w')` hack this replaces did not do.

    Not thread-safe, because `warnings.filters` is process-global and `catch_warnings` swaps it
    wholesale. HAMLET simulates in a single process and a single thread (the multiprocessing path
    was removed in !215), so there is nothing to race with today. When threading returns -- see
    ROADMAP section 7.3 -- this has to be entered once around the whole run rather than per
    worker, which is already how `Executor.run` and the `Creator` entry points use it.
    """
    with warnings.catch_warnings():
        for category, message, _ in SUPPRESSED:
            warnings.filterwarnings('ignore', message=message, category=category)
        yield


def quiet(method):
    """Decorator form of `quiet_known_noise`, for the Creator's three public entry points.

    They nest -- `new_scenario_from_configs` and `new_scenario_from_grids` both end by calling
    `new_scenario_from_files` -- and that is harmless: the inner block re-installs filters the
    outer one already holds and restores them on the way out. Each is decorated anyway, because
    which of them a user calls is their choice and none should behave differently from the others.
    """
    @wraps(method)
    def wrapper(*args, **kwargs):
        with quiet_known_noise():
            return method(*args, **kwargs)

    return wrapper
