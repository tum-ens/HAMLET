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
  e2e/           a shipped example, Creator -> Executor -> Analyzer
    golden/                    committed reference numbers
```

Three of the four shipped examples are exercised. `create_simple_scenario` carries the golden
master and the backend comparisons; the two grid-enabled ones are run by `e2e/test_grid_examples.py`,
which is the only place a power flow is solved end to end — `create_simple_scenario` sets
`electricity.active: False` and calculates no grid at all (#205).

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

### Scenario runs are shared when, and only when, the request is identical

An end-to-end run costs minutes, and every e2e fixture is module-scoped, so two files wanting the
same run each paid for one. `tests/scenario_cache.py` provides the session-scoped `scenario_runs`
fixture instead: ask it for a run and you get the existing one if an *identical* request has
already been made, and a fresh one otherwise. Prefer it over calling `run_example` directly.

Two things about it are worth knowing before you use it, and the module docstring has the rest.

**Identical means the whole request.** The key is derived by binding your arguments against
`run_example`'s signature, so it covers every parameter except the two naming where output goes.
That is deliberately strict: `test_golden_master` passes no `framework` — it must run whatever the
config ships — while `test_backend_equivalence`'s poi arm passes `framework='poi'`, and those stay
two separate runs even though `poi` is currently what ships. A cache that merged them would turn
an independent check into the reference agreeing with itself.

**Sharing a run is a way to stop testing what you think you are testing,** so each consumer's
request is checked against what the run *actually did* — the backend receipt and the scenario
directory it wrote — every time, cache hit included. Do not "optimise" that to once per run: the
key that decided two requests match cannot also be the evidence that they do.
`tests/unit/test_scenario_cache_key.py` breaks the key deliberately and pins that the consumer is
what rejects a mis-served entry.

The measurable saving today is one run — `grid_golden`, 70–125 s — and **only in a session that
selects both markers**, e.g. `pytest tests -m "e2e or golden"`. The `e2e` and `golden` CI jobs are
separate processes, so nothing is shared between them.

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
noticed later. It runs each scenario in its `SCENARIOS` list under a fixed seed and compares
per-table row counts and per-column sum/min/max against `tests/e2e/golden/<scenario>.json`.

Two scenarios are pinned. `simple_scenario` is the shipped example and calculates no grid;
`grid_golden` is a deliberately weak feeder under §14a and is the only thing pinning the power
flow, the variable grid fees and direct power control. It lives in `tests/e2e/scenarios/` rather
than `examples/` because it is tuned to overload rather than to be copied, and
`e2e/test_grid_restrictions.py` asserts *that* the restriction fires while the golden master pins
what it produces.

**To pin another scenario**, append a `GoldenScenario(container, name)` and create its reference
with `HAMLET_UPDATE_GOLDEN=<name>`. One earns its place by reaching code the others do not, and costs
a full example run in the `golden` CI job every time it runs. Note that `simple_scenario` sets
`electricity.active: False`, so pinning it does not pin the grid stage.

**When it fails**, the message names the tables and columns that moved and by how much. Decide
whether that is the change you meant. If it is, regenerate the reference and commit it *with*
the change, so the review sees the numbers move:

```bash
HAMLET_UPDATE_GOLDEN=1 python -m pytest tests -m golden               # every scenario
HAMLET_UPDATE_GOLDEN=simple_scenario python -m pytest tests -m golden  # just this one
```

Name the scenario when more than one is pinned: `1` regenerates every reference, so a re-baseline
aimed at one change silently commits any unrelated movement in the others.

Reproducibility rests on seeding `random` and `numpy.random` and pinning `PYTHONHASHSEED`; the
Creator draws agent ids, plant ownership and sizings from all three. Two seeded runs were
verified to produce byte-identical scenarios and identical results. The column names in the
reference contain those seeded agent and plant ids, so changing how ids are generated will fail
this test — correctly, since agent identities would genuinely have changed.

## The Analyzer's data processors

`tests/e2e/test_analyzer_processors.py` pins what the Analyzer computes, against committed
references at `tests/e2e/analyzer/<scenario>.json`. Its module docstring says what is pinned and
why; this section covers what a reader of the suite needs that the file cannot say about itself.

Before it, nothing asserted on the Analyzer's **output**. `test_simple_scenario` ran it and checked
the process printed `E2E_OK`; `integration/analyzer/test_results_format_check.py` constructs it six
times and asserts only on its refusal to read an incompatible scenario format. **Four of the six
processors had never executed at all** — `test_simple_scenario` calls two plotters, and that
example has no grid.

**It adds no scenario run**, because both its requests are byte-identical to ones the `e2e` job
already makes. `test_the_pinned_runs_are_shared_with_an_existing_module` rebuilds them from the
owning modules' constants and fails if either side drifts; nothing else would notice, since both
modules would still pass while the job quietly grew two more example runs. That guard is why
`test_grid_examples.NEEDS_RECEIPT` is a module constant rather than a literal in its fixture — it
is part of the cache key, and a literal is not derivable from outside.

Four properties of the reduction are load-bearing and are pinned by
`tests/unit/test_analyzer_outputs.py` in the fast tier rather than only inside a minutes-long job:
every recorded statistic is **reproducible across processes** (two processors group by Categorical
columns, so *row order* is not — **#229**); a column of numbers held as **`object` is still
numeric** (`process_total_balancing` builds one, and `select_dtypes('number')` would skip the
Analyzer output that most directly becomes a figure); an **empty return fails separately and
first**, checked against the committed reference as well as the live run; and a **value/index
misalignment is visible**. `tests/analyzer_outputs.py` gives the reasoning for each.

That last one is the subtlest and it was added after a review panel broke all six processors with
every assertion green. Sum, min, max and a distinct count are invariant under any permutation of
values against the index, so "right numbers, wrong row" — positional indexing written back onto a
sorted frame, an off-by-one interval convention, a price series sorted into a duration curve — was
unassertable. `ordered` is a position-weighted total taken in **stringified-index order**: it
catches the permutation without reintroducing the #229 flake (the weights come from the index's
string form, never a Categorical encoding) and is compared with the same relative tolerance as
every other float, rather than being a digest that would flake on a last-place difference between
platforms. Text columns take a digest, where exact comparison is safe.

`tests/integration/analyzer/test_market_data_processor_multi_market.py` covers the one branch the
two pinned scenarios cannot reach: `process_agent_balancing` accumulates across markets, and no
scenario in the repository has more than one. It builds two `market_transactions.ft` files
directly, so the branch costs milliseconds rather than a scenario run.

### What this needed before it could run at all

`process_electricity_grid_topology` could not run **anywhere**, for two independent reasons, and
both are worth keeping because of how they stayed hidden.

It loaded the saved network as the hardcoded `electricity.xlsx`, which only the `file`
grid-generation method produces — a `topology`-built scenario writes `topology.xlsx`. It now reads
the name from the same `grids.yaml` key the Executor saves under. Both conventions are pinned, and
that is why the two scenarios differ in generation method rather than being the two most convenient
runs.

It also calls `create_generic_coordinates(..., library='igraph')`, and **`igraph` was declared
nowhere in the repository** — one `grep` hit, in that call — so it raised `ImportError` in every
environment `uv sync` produces. `igraph` is now in the `test` dependency group, which is what lets
the suite pin it; it is still broken for anyone who *installs* HAMLET, and that half is **#227**.

Neither was visible because `PlotterBase.plot_all` catches every exception and prints it, so
`Analyzer.plot_all()` reports success whatever happens (**#228**). The lesson generalises past this
file: **coverage.py marks a line that raises as executed**, so "runs but unasserted" and "has never
worked" look identical when the caller swallows everything.

## The `ctsp_industry` fixture

`tests/e2e/scenarios/ctsp_industry/` is the only scenario anywhere in the repository that declares
a `ctsp` or an `industry` agent. Before it, neither type had a single line of coverage — ~1900
lines of Creator and both Executor agent classes were never executed (#213), and the two Creator
classes had drifted apart in four behavioural ways while nobody could tell.

**It answers the Executor's `# TODO: Not yet tested and implemented`: both types run.** `Ctsp` and
`Industry` are `AgentBase` subclasses with no behavioural overrides, and a run produces the same
result tables `sfh` does. The TODO was about testing, not about missing code.

**Two tests use this one config folder, and each reaches a half the other cannot.** The distinction
is not pedantry — it is easy to assume the expensive e2e run covers everything, and it does not:

| Test | Entry point | What it reaches |
|---|---|---|
| `e2e/test_ctsp_industry.py` | `new_scenario_from_files` | the **Executor** classes and the workbook → scenario → simulation path. `create_agents_from_file` never consults `Agents.types`, so **no Creator class runs at all** — traced, not assumed |
| `integration/creator/test_ctsp_industry_creator.py` | `new_scenario_from_configs` | the **Creator** classes, the ~1900 lines. Runs in ~4 s, so it is in the fast tier |

Four values in the fixture are load-bearing, and `e2e/test_ctsp_industry.py` fails if any of them
moves without the test moving with it. `solver: highs` is *not* one of them — nothing asserts it,
and `run_example` rewrites it anyway; it is there so the fixture needs no licence.

| Value | Why |
|---|---|
| `framework: linopy` | **Not the default, deliberately.** The scenario is built with `new_scenario_from_files`, so `agents.xlsx` is what the Creator reads and the #206 read-back has to ask it for something it does not ship. Shipping `linopy` makes that request `poi` — the fast backend — so the read-back costs **26–36 s** where the same assertion against `grid_golden` cost **232–272 s** (4 and 2 runs, same harness and machine; quote the band, this runner's spread is wide). |
| two agent **types** | `create_agents_file_from_config` writes one sheet per type, so declaring both gives `agents.xlsx` **two sheets**. `grid_golden` has one, and the per-sheet half of the backend switch had no real fixture until this one. (`number_of: 1` is a separate choice — it is what pins the 24-row assertions.) |
| `ev` share `1` | this is the **only** EV coverage either agent type has. `check_the_ev_premise` fails by name if it goes back to 0, and separately if the nested `charging_scheme` columns are present but entirely NaN — which is #219 exactly. |
| `charging_scheme.method` `["full", "min_soc"]` | `full` is the arm #220 broke, and this is the only place anything exercises it. Reverting that fix makes the whole e2e module fail with the original `TypeError` — verified by reverting, not assumed. `check_the_ev_premise` asserts some agent actually **draws** `full`, deliberately a property of the draw rather than of the distribution: if a reseed makes both agents draw `min_soc` the #220 coverage is genuinely gone and that is worth a red test. This row was unguarded until a review panel changed the fixture to `min_soc` and got a green run. |

### The EV path took three fixes to turn on

The fixture shipped `ev.share: 0` until #218, #219 and #220 were closed. Raising it turned the
fixture red **five** times over, each defect hiding the next, so the sequence is worth keeping:

1. `config_templates/agents.yaml`'s **ctsp** block asked for forecast method `ev_close`. No such
   model is registered — `sfh` and `industry` both say `arrival`. `KeyError` at forecaster init.
2. The same block stated `charging_scheme` in a **flat schema** (`min_soc_val`, …) that the
   Executor does not read; `sfh` and `industry` carry the nested one. `KeyError: 'min_soc'`.
3. `Ctsp._ev_config` and `Industry._ev_config` fill `charging_scheme` with `_add_info_indexed`,
   which **did not descend into nested config** the way `_add_info_simple` does (which is what
   `sfh` uses). Every nested charging-scheme parameter was written as `NaN`, silently, and the
   Creator reported success.
4. POI's `__constraint_cs_full` passed a whole `Series` where `add_linear_constraint` needs a
   scalar RHS (its LHS was a whole variable array too); the sibling `__constraint_cs_min_soc`
   already looped per timestep, and linopy's version takes arrays natively. So
   `charging_scheme.method: full` could not work for **any** agent type on the default backend.
5. Found by `test_ctsp_industry_creator.py` rather than by reading: the ctsp block's EV forecast
   sub-block was `random_forest_classifier:` where the registered model — and what `sfh` and
   `industry` write — is `rfr`.

**1, 2 and 5 were all the same shape as #212**: a change that landed in `sfh` and `industry` and not
in the `ctsp` copy. Those three were **#218**, config only. 3 was **#219**, code, both classes.
4 was **#220**, backend-wide and latent — no shipped scenario asked for `full` until this fixture
did, which is why it read as urgent only once the share went up.

Note #213's fourth divergence runs the *other* way — `ctsp` has the `share > 0` guard and
`industry` does not — so "ctsp is always the stale copy" is not the rule.

**On #219, the obvious fix and the `sfh`-shaped fix are both wrong**, and
`tests/unit/creator/agents/test_add_info_indexed.py` pins each. The block those two classes pass is
the only *mixed* one in the repository: `method` is a per-agent distribution list while the nested
leaves are scalars. So plain recursion reaches `len(0.5)` and raises, and copying `sfh`'s
`_add_info_simple` call fills the nested columns while silently stringifying `method` into
`"['full', 'min_soc']"` for every agent — the same class of defect one level down. The helper now
dispatches per leaf on the value's own type.

Changing that shared helper was safe because of a two-part audit, and **both parts are tests rather
than claims** — each would otherwise pass by omission as the other changed:

| part | test | what it re-derives |
|---|---|---|
| the 35 call sites read only `sizing`, `parameters`, `charging_scheme` | `test_the_call_sites_still_read_only_the_subkeys_this_audit_covers` | parses the call sites with `ast`, so a new site reading a fourth subkey fails |
| those subkeys are flat lists in every config | `test_no_other_call_site_passes_a_nested_or_scalar_config` | walks every `agents.yaml`: `sizing` 501 leaves, `parameters` 52, all lists, none nested |

The second takes its subkey set from the first rather than hardcoding it. That pairing is why
neither golden reference moved — verified on Linux/x86_64, and separately by building
`simple_scenario` under both the old and new helper and getting a byte-identical workbook.

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
