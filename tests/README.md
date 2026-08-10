# HAMLET test suite

Split by scope, mirroring the package layout so a test sits next to the thing it covers.

```
tests/
  unit/          one class or function in isolation; no solver, no scenario
    creator/agents/            creator/markets/
    executor/agents/           executor/markets/
    executor/utilities/controller/{,fbc,rtc}/
    executor/utilities/grid_restrictions/
  integration/   several components wired together, or code that touches the filesystem
    analyzer/                  creator/            executor/
    creator/format/            committed scenario-format references, one per version
  e2e/           the shipped example, Creator -> Executor -> Analyzer
    golden/                    committed reference numbers
```

`unit/` and `integration/` mirror `hamlet/`, so the test for
`hamlet/executor/markets/electricity.py` lives at `tests/unit/executor/markets/`. Reading or
writing files puts a test in `integration/` regardless of how small it is.

## Running

```bash
uv sync                 # once: installs pytest and HAMLET itself, from uv.lock
uv run python -m pytest tests
```

That runs unit and integration — seconds, HiGHS only, no solver licence. The end-to-end layer
is deselected by default because it runs the whole example:

```bash
uv run python -m pytest tests -m e2e      # smoke: does the example still run
uv run python -m pytest tests -m golden   # golden master: does it still produce the same numbers
```

The `uv run` prefix is dropped in the rest of this file for readability; if you have activated
`.venv` yourself, `python -m pytest` is the same thing.

Markers: `solver` (builds and solves a real optimisation model), `e2e` (runs the example),
`golden` (compares against committed reference numbers). The last two take a couple of minutes
each and are deselected by default.

## The solver x framework matrix

`framework` (`linopy` | `poi`) and `solver` (`highs` | `gurobi`) are independent per-agent options,
so there are four supported combinations. Every run of the suite opens with a header line saying
which of them this machine can exercise:

```
solver x framework matrix: linopy+highs=ok  linopy+gurobi=SKIP  poi+highs=ok  poi+gurobi=SKIP
```

Two tests cover them, and `-rs` names every cell that skipped:

| Test | Asks |
|---|---|
| `integration/executor/test_solver_backend_matrix.py` | do all four reach the same optimum, on one MPC-shaped and one RTC-shaped model |
| `e2e/test_solver_backend_smoke.py` | does the shipped example run end to end under each cell nothing else already covers |

The smoke arm defaults to the **uncovered** cells, which is both Gurobi ones. An example run costs
minutes, and the two HiGHS cells are already run end to end by the golden master (the shipped
config, in its own CI job) and by the equivalence test's linopy arm. Running them here as well
tripled the `e2e` job — 338 s to ~1090 s — and bought nothing, which on a shared runner is not free:
it widened the window in which the `golden` job competes for the same cores. To run all four:

```bash
HAMLET_SMOKE_ALL=1 python -m pytest tests/e2e/test_solver_backend_smoke.py -m e2e -rs
```

The deferred cells are still parametrised and skipped, with a reason naming what covers them, so
they appear in the report rather than vanishing. And the deferral is itself tested:
`test_the_deferred_cells_are_still_covered_elsewhere` fails if the covering runs are deleted or
repointed at another backend — deferring coverage is only safe if the deferral is checked.

**Objective values are compared, and nothing else.** Equally-optimal vertices differ between
solvers and between backends — !201 measured 3 row counts and 76 column statistics moving on a
Gurobi → HiGHS switch. An equality assertion on setpoints or result tables would be wrong and would
flake. Whole-run divergence between the two frameworks is degeneracy, is expected, and is closed as
#198; `e2e/test_backend_equivalence.py` holds that comparison as a permanent strict xfail.

**The tolerance is the MIP gap, not machine epsilon.** Both models carry a binary and both solvers
default to a 1e-4 *relative* MIP gap, so each is only obliged to return an incumbent within that of
the true optimum, and two incumbents can differ by twice it. The band is therefore `2e-4`, about
eight orders looser than the ~1e-12 the cells actually agree to. That is deliberate: it states what
the solver guarantees rather than what it happened to deliver, so a solver version changing its
branching does not send someone looking for a number to loosen.

**Both Gurobi cells skip on most machines, and that is the normal case.** The two frameworks do not
reach Gurobi the same way, so they fail independently:

- **`poi` + `gurobi`** links a *system* Gurobi installation through PyOptInterface's C API. No
  Python package is involved — `gurobipy` need not be installed at all.
- **`linopy` + `gurobi`** goes through `gurobipy`, which is an optional extra. Without
  `uv sync --extra gurobi` this cell skips even on a machine with a valid licence — and when both
  are available they may be running different Gurobi *versions*, since `gurobipy` carries its own.

Availability is decided by **solving a small model, never by importing one**. Gurobi's shared
library loads perfectly well without a licence and only fails at `optimize()`, so an import check —
`linopy.available_solvers` included — turns "no licence" into test *failures* rather than skips.
That is what `poi_support.can_solve` exists for, and `backend_matrix` reuses it rather than
restating it.

**Neither test can pass by doing nothing.** All four combinations are always parametrised and an
unavailable one skips from inside the test, so it appears in the report rather than vanishing from
it; `test_at_least_one_combination_is_available` fails rather than skips if the whole matrix is
empty; and every assertion checks *what actually solved* before it looks at a number, because a
comparison whose two arms have silently collapsed into one is the failure mode that matters here
(`backend_models.identify` for the models, `scenario_run.BACKEND_PROBE` for the example run).

**Speed for the same four cells** is measured by `benchmarks/test_backend_speed.py`, which shares
its model with the matrix through `tests/backend_models.py` so the thing timed and the thing
compared cannot drift apart. It is deselected by default:

```bash
uv run python -m pytest -m benchmark -s
```

On the development laptop (24-step horizon, 60 interleaved repetitions, medians in ms):

| cell | build | solve | total |
|---|---|---|---|
| `poi` + `highs` | 0.95 | 3.29 | **4.24** |
| `poi` + `gurobi` | 2.10 | 3.46 | 5.56 |
| `linopy` + `gurobi` | 150.36 | 56.60 | 206.96 |
| `linopy` + `highs` | 148.38 | 64.52 | 212.90 |

Two readings, and only the first is asserted. **The framework axis dominates: ~50×**, and it is
almost entirely *build* — linopy's build cost is the same whichever solver it is pointed at, which
is what identifies the cost as Python model construction rather than solving. **The solver axis is
reported, never asserted:** at HAMLET's model sizes (144 columns here) per-model overhead dominates,
so Gurobi's edge on `solve` under linopy and its loss on `build` under POI are properties of this
model size and this machine. Read that column; do not pin it.

Three methodology points, each of which has produced a wrong number here: the price vector is
perturbed between repetitions (an unchanged re-solve short-circuits and reports ~0.01 ms), cells
are interleaved rather than run in blocks (this laptop drifts >2× thermally), and every cell is
warmed before timing (loading a solver library is a one-off cost of tens of ms that otherwise
inverts the ranking outright).

**Per-solve speed is not run speed.** A run is not only its solves: the modelling layer is ~50×
faster while the shipped example's Executor stage is 4.7× and the whole process 3.3×. Quote the
run-level figures.

## The golden master

Every other test pins a property someone thought to check. `tests/e2e/test_golden_master.py`
pins the numbers themselves, so a change that moves results has to be acknowledged rather than
noticed later. It runs the shipped example under a fixed seed and compares per-table row counts
and per-column sum/min/max against `tests/e2e/golden/simple_scenario.json`.

**When it fails**, the message names the tables and columns that moved and by how much. Decide
whether that is the change you meant. If it is, regenerate the reference and commit it *with*
the change, so the review sees the numbers move:

```bash
HAMLET_UPDATE_GOLDEN=1 python -m pytest tests -m golden
```

Reproducibility rests on seeding `random` and `numpy.random` and pinning `PYTHONHASHSEED`; the
Creator draws agent ids, plant ownership and sizings from all three. Two seeded runs were
verified to produce byte-identical scenarios and identical results. The column names in the
reference contain those seeded agent and plant ids, so changing how ids are generated will fail
this test — correctly, since agent identities would genuinely have changed.

## The scenario format version

`hamlet.constants.SCENARIO_FORMAT_VERSION` is the version of the **on-disk scenario folder**, not
of HAMLET itself. The Creator stamps it into `general/general.json`; the Executor and the Analyzer
refuse to read a folder carrying anything else, including no stamp at all. It exists because the
retailer in/out convention fix spanned the Creator and the Executor, so a scenario generated
before it and executed after it silently applied grid fees and levies to feed-in — plausible
numbers, no error, no warning.

A version nobody remembers to bump is worse than no version at all, so
`tests/integration/creator/test_scenario_format_shape.py` remembers. It generates the shipped
example (Creator only, a couple of seconds, so it runs in the default suite) and fingerprints its
*shape* — which files exist, which columns each table has, which keys each JSON carries, with the
seeded-random agent and plant ids normalised away — against
`tests/integration/creator/format/scenario_format_v<N>.json`.

**When it fails**, the message names what moved and lays out the three possibilities: revert,
regenerate the current reference, or bump. If you bump, create the new reference and commit both:

```bash
HAMLET_UPDATE_SCENARIO_FORMAT=1 python -m pytest tests/integration/creator/test_scenario_format_shape.py
```

Older reference files are not disposable — they record what those scenarios looked like. Two
further tests guard the bookkeeping: a reference must exist for the current version, and none may
exist for a version above it (which would mean a bump was left half-done).

**What this cannot catch.** A column that keeps its name and changes its *meaning* is invisible to
a structural fingerprint — and that is precisely the change that motivated the versioning. What
sees that is the golden master, by the numbers moving. So the bump rule has two halves, written
out in full at `c.SCENARIO_FORMAT_VERSION`, and only the first is automated. When the golden
master moves, the question to ask is whether the cause was a change to what the *Creator writes*
rather than to what the Executor computes; if it was, bump.

## Provenance

This suite was seeded by porting bug fixes that had been stranded on the paper branch. Tests
name the defect they pin in their docstring.

The check that matters is not "does it pass" but "what would make it fail". The useful way to
answer that is to revert one source file at a time and see which tests notice:

```bash
git show origin/develop:PATH > PATH && python -m pytest tests -q; git checkout HEAD -- PATH
```

A production change that no test notices is a coverage gap. Several tests were deleted rather
than kept after that check showed they exercised no HAMLET code at all — they demonstrated
polars or solver behaviour while reading as regression tests.

## The in/out convention

The single most error-prone thing in this codebase, so it is stated once, here.

**Retailer input files** (`input_data/retailers/**/*.csv`) name every column from the
**retailer's** point of view:

- `_out` is the retailer selling, i.e. **the agent buying**. This is the expensive direction,
  and it is where the grid fee and the levy sit, because both are charged on consumption.
- `_in` is the retailer buying, i.e. the agent feeding in.

**Transaction and forecast columns** inside the executor are from the **agent's** point of view:
`energy_in` is energy flowing into the agent, i.e. the agent buying.

Therefore any code mapping a retailer column onto a transaction or forecast column **must cross
in↔out**. That applies uniformly to `energy`, `balancing`, `grid` and `levies` — there is no
per-file exception.

Buying is always more expensive than selling. A test that appears to show otherwise is showing a
convention bug, not an arbitrage opportunity.
