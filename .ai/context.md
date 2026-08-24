# HAMLET — AI runtime context

> The single orientation document for AI coding agents working in this repository.
> `CLAUDE.md`, `GEMINI.md` and `AGENTS.md` are pointers to this file, so every tool reads the
> same thing.

**Every claim here was checked against the tree before it was written.** This file exists because
unverified documentation caused real harm here: a sibling branch carried documents describing a
database migration that was never written, and both humans and agents built on them. A hundred
accurate lines beat four hundred aspirational ones. If you cannot check a claim, leave it out.

---

## Architecture

Three stages, each reading the previous one's output directory.

**Creator** reads a folder of YAML/XLSX configs and writes a scenario folder under `scenarios/`.
**Executor** copies that scenario into `results/<name>/`, then simulates over the copy timestep by
timestep and writes results into the same folder — it never reads the original scenario again
(`hamlet/executor/setup.py:212-220`). **Analyzer** takes a `{name: results_folder}` mapping and
plots. `examples/create_simple_scenario/run.ipynb` is that sequence in eight cells and is the
shortest useful read in the repository.

Live simulation state and the append-only results log both live in one in-memory object graph:
`Database` → `RegionDB` → `AgentDB` / `MarketDB` / `GridDB`, with a file-tree serializer attached.
Most of the executor's performance workarounds follow from those two responsibilities sharing an
object. Splitting them is planned, not done.

**The Executor runs in one process, and `num_workers` above 1 is refused rather than ignored.**
There was a multiprocessing path — `tasks_execution/`, six files — that handed each agent to a
worker by writing the database to disk and reading it back. It did not work: on every shipped
example each worker raised inside `agent_pool.task`, whose bare `except` returned `None`, which
the parent then unpacked (`TypeError: cannot unpack non-iterable NoneType`). Nobody noticed
because no run ever used it; every paper run passed `num_workers=1`. It is deleted rather than
repaired, because it is also what forced the whole disk-serialisation design that the state/results
split has to undo. When parallelism returns it will be **threads over agents** — all three solver
bindings release the GIL, measured — which needs no state transfer at all. ROADMAP §6.3, §7.3.

## Repository data

Of those three directories **only `input_data/` is tracked** — 152 files, ~159 MB. `scenarios/`
and `results/` are run output and are gitignored. Nothing under `hamlet/` writes into
`input_data/`; the package reads it through the `input:` path in a scenario's `setup.yaml`.

So `input_data/` is source and is **not** ignored: a file added there shows up in `git status` like
any other. Do not re-add a blanket rule over it — that hid a needed input once already, and
`tests/unit/test_gitignore_input_data.py` fails if you do. Bulk data belongs outside the
repository; archives (`*.zip`, `*.rar`, `*.7z`, `*.tar.gz`) are ignored everywhere.

## Environment

`pyproject.toml` declares the dependencies and `uv.lock` pins them, transitives included. Those two
files are the only dependency definition in the tree — `env.yml`, `requirements.txt`,
`docs/requirements.txt` and `ci/requirements_from_env.py` are all gone, and CI installs from the
same lock a contributor does. HAMLET is installed rather than reached through `sys.path`; nothing
tracked appends to it any more.

```bash
uv sync                     # the environment; also installs Python 3.11 if needed
uv run python -m pytest     # the fast tier
```

Three things about it are load-bearing:

- **`xarray==2024.6.0`** is not HAMLET's dependency; it arrives through linopy. linopy 0.3.11
  imports `xarray.core.rolling`, which xarray removed after 2024.6.0, so with xarray unconstrained
  a resolver takes the newest and `import hamlet` fails outright, with a traceback naming neither
  package. It is stated in `pyproject.toml` rather than left to the lock so that
  `uv lock --upgrade` cannot move it. **The resolver will not catch a linopy bump for you** —
  linopy declares only `xarray>=2024.2.0`, never a ceiling — so
  `tests/unit/test_dependency_constraints.py` enforces the pair instead. Do not relax either
  without replacing linopy.
- **Every version is exact**, matching the environment the committed golden master was measured
  in. That is a deliberate stage, not the end state: it keeps a change in *results* attributable.
  Relaxing the pins to ranges, and moving off Python 3.11 (`requires-python = ">=3.11,<3.12"`),
  are separate measured steps — see ROADMAP item #4.
- **`tensorflow` and `gurobipy` are extras**, so the default environment has neither. Nothing under
  `hamlet/` may import them at module scope; `fast` fails if that changes.

A `venv/` directory in your working tree is not this project's environment — `uv sync` creates
`.venv/`. Both are gitignored and nothing tracked refers to either.

## Solver

HiGHS is installed with HAMLET and is what `examples/create_simple_scenario` uses
(`simple_scenario/agents.yaml:904,930`). No licence is needed for the example, the test suite, or
CI. Gurobi is optional and fully supported — set `solver: gurobi` under `optimization`; the other
three examples and `config_templates/` still specify it.

**`framework` and `solver` are independent, so there are four supported combinations, and all four
are tested.** The matrix and its rules are documented once, in `tests/README.md`. The one thing
worth knowing before you read it: **the two frameworks do not reach Gurobi the same way.**
PyOptInterface links a *system* Gurobi installation through its C API with no Python package
involved, while linopy goes through `gurobipy`, which is an optional extra — so
`uv sync --extra gurobi` is what a licensed machine needs before the linopy half of the matrix runs
at all, and the two halves can be exercising different Gurobi versions.

**Two solver libraries ship, deliberately.** `highspy` is the one linopy talks to. It bundles
HiGHS inside its `_core` extension and exposes no shared library, which PyOptInterface needs — so
`highsbox`, the same HiGHS build packaged as a plain `highs.dll` / `libhighs.so`, is also a
dependency. Both are pinned to `1.10.0` so the two backends solve with an identical solver; keep
them equal, or a backend comparison stops being a comparison.
`hamlet/executor/utilities/controller/poi_solver.py` is the only place that selects or loads
either, and `framework: poi` works on HiGHS because of it — before that it imported Gurobi
unconditionally and silently required a system Gurobi installation.

**Windows needs a modern `MSVCP140.dll`, and `hamlet/__init__.py` makes sure it gets one.**
`highsbox` 1.10.0's `highs.dll` corrupts memory if it binds to a C++ runtime older than **14.38**
— measured on a version ladder, 14.36 crashes and 14.38 does not. The Windows loader resolves that
import by *base name against whatever loaded first*, and `pyarrow` (14.28) and `scikit-learn`
(14.32) each ship an unmangled private copy, so `import pandas` used to hand HiGHS a runtime five
toolsets too old. That was #202. `hamlet/msvc_runtime.py` claims the name for the system runtime on
the **first line** of `hamlet/__init__.py`; keep it there, because the imports below it reach
`pandas`. If something else wins the race anyway, `poi_solver` raises instead of solving.

**A caller who imports `pandas` and never imports HAMLET is not covered by that**, which was
acceptable while `poi` was opt-in and not once it became the default. So a second, independent
claim runs at interpreter startup: `packaging/hamlet_msvcp140_hook.py`, invoked by
`hamlet-msvcp140.pth`, both force-included into the wheel *and* the editable target by
`pyproject.toml`. Three things about it are deliberate and easy to undo by accident — it must
short-circuit on `sys.platform` before touching the disk, it must swallow every exception (`site`
degrades a raising `.pth` to a traceback printed on **every** interpreter start in the
environment), and it must not import `hamlet`, whose `__init__` would drag `pandas` into every
Python process in the environment. It costs ~40 ms of Windows startup, ~26 ms of which is
`ctypes` and irreducible; `HAMLET_NO_MSVCP140_HOOK=1` turns it off.
`tests/unit/test_msvcp140_hook.py` covers all of it.

Note what the hook did to the tests that predate it: with the name already claimed, the two
subprocess tests in `tests/unit/test_msvc_runtime.py` could no longer provoke a stale runtime and
**silently skipped**. They set the opt-out now. Any future test of this mechanism has to do the
same, or it will pass without asserting anything.

Why not simply bump `highsbox` past the crash, which #202 measured as sufficient? Because that
HiGHS is 4.8–6.2× slower on HAMLET's models, and 1.11.0 — the very next release after the pin — is
already slow, so there is no fast *and* crash-free version to move to. The 2×2 establishing that
the `pyoptinterface` version is not the variable is in `hamlet/msvc_runtime.py`.

**`framework: poi` is the default.** Both controllers' models have been shown equivalent to their
linopy counterparts (below); what is left is a consequence of degeneracy, not an unvalidated
backend. `linopy` remains fully supported and selectable per agent, and is the reference
implementation POI was validated against — keep it working.

Measured on the shipped example, medians of three interleaved runs on the development laptop:
the Executor stage goes **73.9 s → 15.7 s (4.7×)** and the whole process, scenario creation
included, **86.2 s → 26.5 s (3.3×)**. Quote those and not the per-solve figure — the modelling
layer is 43× faster per solve (`pytest -m benchmark`), but a run is not only its solves, and the
gap between the two numbers is the point of measuring both.

- *Results*: `poi` does not reproduce the linopy numbers on the shipped example — but
  the reason is **degeneracy amplified by state feedback, not a modelling difference**, and that
  distinction decides what to do about it. Both the MPC and the RTC models were compared directly
  and are equivalent; the run-level difference arises downstream of them. Three real defects were fixed first (heat pump built
  with no constraints at all, market power declared integer, dead slack reporting), taking the gap
  from 110 differing column statistics to 85. What remains was then measured rather than reasoned
  about, by exporting both MPC models to LP files and diffing them by constraint *shape* (sense,
  RHS, coefficient multiset — invariant to variable naming, which differs):

  - At the first timestep the two models are **mathematically identical**: same constraint count,
    senses, RHS, objective coefficients and binaries. Every unmatched constraint is explained by
    one extra `+1.0` term for linopy's balance-dummy variable, which is fixed at zero and so is
    inert. Checked on two agents; nothing else was unexplained.
  - Objectives at the first timestep agree to ~1e-12 for three of four agents.
  - The error then stays at machine precision for several steps, **jumps discretely**, and grows
    (to ~6e-1 by step 23). An agent owning neither a battery nor an EV stays at ~1e-13 for all 24
    steps — it is the only one with no inter-timestep state to carry a divergence forward.

  So the models agree; the MILP is degenerate, the two backends present it in a different order,
  a tie breaks differently, and the resulting state of charge feeds the next timestep. This is the
  legitimate exception the golden master's own notes describe.

  The one first-timestep difference (~1e-5 on a single agent) was chased to the end and is the
  same story a layer up, **not** a second defect. Dumping every MPC variable's bounds by name
  showed exactly one differing input: `<plant>_heat-storage_soc_init`, 8470 under linopy against
  8469 under poi — one Wh, in a value the MPC *reads* rather than computes. Controllers run RTC
  first and FBC second within a timestep (`agent_base.set_controllers` iterates the configured
  controllers in order), and the RTC writes the state of charge the MPC then starts from, through
  `rtc_base.update_socs`, which quantises to the socs column dtype. So the MPC model is not
  implicated: its input had already moved by one quantisation step. Note it is **heat storage**,
  not the EV — the agent merely happens to own an EV as well.

**The RTC was then compared the same way, and it is the cleaner test.** The RTC runs *first* in a
timestep, so at the first timestep both backends are handed provably identical inputs — no state
feedback to confound the result. On all four agents of the shipped example:

- **Objectives at the first timestep agree exactly** — `0.0e+00` for three agents and `1.3e-16`
  for the fourth.
- **Variable bounds are identical** for every shared variable. The only extras are linopy's
  `balance_electricity` and `balance_heat`, and those are fixed at `[0.0, 0.0]`, so they cannot
  affect a solution whatever coefficient they carry.
- **Constraint counts match** (6/6, 8/8, 7/7, 7/7), and every unmatched constraint shape is
  accounted for by those same zero-fixed dummies.

So both controllers are equivalent across the backends, and the 1 Wh above is settled: the RTC
reaches the *same optimum* at the first timestep and simply lands on a different vertex of it, so
the difference is degeneracy rather than an RTC defect. Over the run the RTC mismatches 14 of 96
solves, appearing and disappearing rather than growing monotonically — its objective is a
deviation from target that is re-anchored each timestep, unlike the MPC's.

  `tests/e2e/test_backend_equivalence.py` holds the end-to-end comparison as a **permanent**
  `xfail(strict=True)` — not one pending a fix. It compares whole-run outputs, so it keeps failing
  under degeneracy however correct both backends are. It still earns its place by failing loudly if
  the divergence ever *disappears*, which would mean the two arms had stopped being two arms (a
  renamed key, a switch silently no-oping). Do not delete the marker on the grounds that the models
  are equivalent; that was established and is not what the test measures. See #198.

**The golden master was re-baselined when the default flipped, and the evidence is the point.**
Movement was expected here — it is the one time in this project that is true — so it was checked
against the mechanism rather than accepted because it appeared. Structure held (18 tables, no
column added or dropped); 3 row counts and 85 statistics moved, the same character as the
Gurobi→HiGHS switch (!201: 3 and 76). The decisive check is *when* each agent first diverges, per
the criterion in `tests/e2e/test_golden_master.py`: **no agent differs at timestep 0**, where both
backends are handed provably identical inputs. The agent owning neither battery nor EV diverges
latest (step 10), on 1 of 48 columns, by 1 Wh on `hp_heat` — it still owns heat storage, so it is
the *least*-stateful agent rather than a stateless one; there is no stateless agent in this
scenario. Agents owning a battery or EV diverge from step 1–12 by hundreds of Wh. That is a
degenerate tie propagating through `update_socs`, not a component-set difference.

**§14a grid restrictions are covered end to end since the #205 work, and were not before.** Direct power
control reaches the RTC only (never the FBC), through `apply_grid_commands`; indirect control
(variable grid fees) is applied outside the solver in `agent_base.py`, so it is backend-agnostic
and both MPC backends pick it up. No *shipped example* enables the mechanism — the two that set
`restrictions.apply: ['enwg_14a']` have `electricity.active: False` — so the coverage comes from a
test fixture instead: `tests/e2e/scenarios/grid_golden/`, a deliberately weak 21-bus feeder that
overloads at 132 % and is pinned by the golden master.

Three things about that fixture are load-bearing and easy to undo by accident. **The transformer
is sized between the peak and the floor of the agents that are actually curtailable** — 15 kVA
against a 19.8 kW uncontrolled peak and an 8.4 kW floor for the two agents above the 4200 W
threshold at those hours. Raise it and nothing overloads, which `test_the_feeder_actually_overloads`
exists to catch. Note it is *below* the 16.8 kW all four agents would be guaranteed together: bring
a third EV onto the same hour and §14a can no longer resolve the overload. **Every agent has an EV starting at
SoC 0.2**: `create_scenario_with_topology`, which this fixture derives from, starts at 0.8 against a
target of 0.8, so nothing charges, no agent exceeds the 4200 W threshold, and direct control
computes a reduction of exactly zero. And **each
agent needs a bus in `agents.xlsx`** — `agent_base.__update_variable_grid_fees` looks the fee up by
it, so a blank one is `KeyError: nan` the moment variable grid fees are switched on. The shipped
topology example leaves it blank and gets away with it only because it runs no restriction.

**The assertion that matters is that a curtailment command is *respected*.** The grid stage cannot
check that itself: it re-simulates, gets the same answer, and converges on an uncapped grid. So a
backend whose `apply_grid_commands` silently does nothing — which is what `poi` inherited from the
base class before !209 — looks identical to one that obeys. Restoring that no-op fails exactly one
test and leaves the other three green.

**Direct power control has two methods and `grid_golden` selects `ems`; the other is covered by a
second run of the same fixture (#232).** `__individual_device_control` curtails named devices in a
priority order — battery, EV, heat pump — and no shipped config reaches it, while
`config_templates/grids.yaml` selects it, so it is what a copied scenario runs.
`test_grid_restrictions.py` reaches it with `config_edits`. **Selecting the method is not enough,
and this is the trap:** the order is only observable where the reduction budget runs out *part-way*
through it, and that is governed by the §14a floor, which sets each device's headroom — not by how
many device classes are present. At the shipped 4200 W this fixture's devices sit too close to the
floor, both go to it whatever the order, and reversing both documented orders leaves all 152 calls
byte-identical. The second fixture therefore also lowers the floor to 1000 W, and costs the `e2e`
job one extra run (~185 s). Its over-generation half is reachable by nothing (#233).

**The grid stage itself is covered since #205, and it was not before.** Both grid-enabled examples
run again and `tests/e2e/test_grid_examples.py` runs them, asserting that a power flow was solved
and that every load and sgen belongs to an agent. Nothing else here would notice the stage
breaking: the example the golden master uses calculates no grid at all.

**A grid file states its per-element metadata in one of two conventions, and `register_grid` reads
both.** Either as real `load_type` / `plant_type` / `owner` / `agent_type` / `file` columns, or
packed into `description` as `key:value,key:value`. Which one applies is a property of the file:
`examples/create_scenario_with_grid` ships the packed form, while a network imported from an
operator has real columns and free-form prose in `description` (#216 records what parsing that
does). Real columns win, `description` is read only as a fallback, and a file in neither convention
is rejected by name. Do not make either branch unconditional again — each one breaks the other's
files, and the repository has now done it in both directions.

**`agents.xlsx` in an example's config folder is an output as well as an input.** Both
`new_scenario_from_configs` and `new_scenario_from_grids` write it back, with freshly drawn agent
ids. Running an example from a checkout therefore dirties the working tree, and for
`create_scenario_with_topology` it *breaks* the example: that scenario's `topology.xlsx` assigns
agents to buses by id, so only `new_scenario_from_files` preserves the ids those assignments refer
to. Run examples against a copy of the config tree — `tests/scenario_run.py` does, and takes the
Creator entry point as an argument for this reason.

**The golden reference is solver-coupled.** Running under a different solver moves the numbers.
Measured when the example switched Gurobi → HiGHS (commit `f65edf0`): structure unchanged — same
18 tables, no column added or dropped — but 3 row counts and 76 per-column statistics moved, e.g.
`hp_heat.sum` by 0.04 % and `hp_electricity.sum` by 0.7 %. The same example under HiGHS produces
identical results on Windows and in a Linux container, so the coupling is to the solver, not the
platform.

## Determinism

**Never index `os.listdir()` with a seeded RNG.** The seed fixes the *index*; the filesystem fixes
the *order* — alphabetical on NTFS, hash order on ext4 and overlayfs. The Creator did exactly this
when picking each agent's load, PV, heat and EV profiles, so the same config with the same seed
produced **different scenarios on Windows and Linux** for years: identical agent ids, plant counts
and sizings, with different profiles behind them. It surfaced only when CI produced the first
Linux-native checkout. Fixed with `sorted()` at four sites in `hamlet/creator/agents/agents.py`
(`:1558`, `:1710`, `:1741`, `:1759`) and guarded by
`tests/unit/creator/agents/test_file_selection_is_order_independent.py`.

The method lesson is worth more than the bug: mounting a Windows-materialised checkout into a
Linux container tests the kernel and the libraries, not the checkout. To test a platform, let that
platform's git write the tree.

Three unsorted `os.listdir()` calls of the same shape remain. Whether any of them can affect output
is **not established** — nobody has checked:

- `hamlet/creator/setup.py:609`, `:692`
- `hamlet/executor/utilities/database/market_db.py:158`

A fourth, in `tasks_execution/agent_pool.py`, went with that directory.

**A solve must be reproducible, so two solver options are set for you and one status is fatal.**
`hamlet/executor/utilities/controller/solver_options.py` names them once for both frameworks:
`threads = 1`, because a parallel MIP's incumbent depends on how its threads interleave; and the
configured `time_limit` **in seconds**, which HAMLET divided by 60 for years, so the example's
`time_limit: 120` reached the solver as 2 s. Under linopy that was inert — the value went under
Gurobi's `TimeLimit`, which HiGHS discards unrecognised — so making `poi` the default activated a
limit that had never applied. Measured on the shipped example, 1 of 192 solves hit it under
artificial load and its incumbent was accepted silently, because `TIME_LIMIT` was whitelisted
alongside `OPTIMAL`. `poi_solver.raise_unless_optimal` now refuses anything but a proven optimum,
which is what linopy's controllers always did. See #204.

Pinning the threads moved no numbers: the golden reference was recorded at `threads = 0` and still
matches, so HiGHS was already solving these models serially on an idle machine. The thread count is
pinned because it *can* vary, not because it was shown to have.

**Every solver option is looked up per solver, and writing one out inline is the bug.** The same
module answers `quiet_options(solver)` — `output_flag`/`log_to_console` for HiGHS,
`OutputFlag`/`LogToConsole` for Gurobi, each in the type that solver's API demands. The linopy
controllers used to send the Gurobi pair to both, so under HiGHS the flags did nothing;
`poi_solver.create_model` reads the same table, so the two backends cannot drift apart again.

**And the stdout hijack was not covering for them.** HiGHS logs from C to file descriptor 1;
`sys.stdout = open(os.devnull, 'w')` rebinds a Python attribute and never touched it. Measured
per solve under `framework: linopy` + `solver: highs`: **52 stdout lines before, 1 after** — two
of the 52 being `ERROR: getOptionIndex: Option "OutputFlag" is unknown`, which linopy discards
the status of. So the solver was complaining, on every solve, into a stream nobody was reading.
Only an fd-level redirect catches C output; `tests/backend_matrix._silenced` is the one place
that does it. The spelling is pinned by a test rather than discovered by running (#199).

## Warnings

**Nothing in `hamlet` may install a warning filter at import.** `hamlet/executor/setup.py` used to
call `warnings.filterwarnings("ignore")` at module scope, so `import hamlet` silenced every
warning in the process — HAMLET's own included. That is how a slack-variable warning added in
!195 was dead on arrival, and why the scenario-format check was made a hard error rather than a
warning. Two more blanket filters existed, in `creator/agents/agents.py` and `pytest.ini`. All
three are gone (#199), and `tests/unit/test_warning_policy.py` fails if any comes back — checked
by raising warnings in a subprocess after importing the package, not by grepping for the call.

Suppression that is still wanted is **enumerated** in `hamlet/warning_policy.py`: each entry names
a category, a message pattern and the reason, and `quiet_known_noise()` installs exactly that list
around a Creator or Executor run. `tests/conftest.py` registers the same list with pytest, so the
suite and the runtime agree by construction rather than by two people remembering. That module
records what the blanket filter was hiding, measured, and why enumerating beat deleting.

**Do not add a category to `SUPPRESSED` to quieten a run.** If a warning is worth hiding it is
worth an issue: lifting the filter is what surfaced #210 and #211, two live §14a defects that had
been invisible for as long as the filter existed.

## Branches and propagation

- `master` and `develop` are protected on GitLab with push access **No one**. Changes land by
  merge request only. Branch off `origin/develop`; naming convention is in `CONTRIBUTING.md`.
- `github` (`github.com/tum-ens/HAMLET`) is a **push mirror**, configured with
  `keep_divergent_refs: false`. Anything pushed there by hand is overwritten at the next sync.
  Never push to it.
- `TUM-Doepfert` is the paper fork. **Leave it alone.** It is far behind, and it is the only
  remote carrying the tag `paper-elsevier-2026-complexity`, which is not on `origin`.
- Check merge-request and issue state with the authenticated `glab` CLI rather than inferring it
  from git topology.

## The paper lineage is a separate, incompatible world

The `paper/*` branches and `develop` have diverged into **mutually incompatible scenario formats**.
`develop` cannot reproduce the published paper's numbers, at any point in its history; only the tag
`paper-elsevier-2026-complexity` can. Do not represent otherwise, and do not tidy the paper
branches — they are a frozen, citable snapshot.

This is now detected rather than silent: `hamlet.functions.check_scenario_format` refuses any
scenario or results folder whose format stamp is missing or does not match
`c.SCENARIO_FORMAT_VERSION`, and paper-lineage folders carry no stamp. That is detection, not
interoperability; making the two lineages interoperate would be its own project.

## Working rules

- **No AI attribution trailers on commits.** Do not add `Co-Authored-By: Claude` or any
  equivalent. 28 commits reachable from `develop` carry one; they will not be rewritten, and no
  more are to be added.
- **Never commit** `scenarios/`, `results/`, zip archives, or generated figures.
- **Never start a full-year or other multi-hour simulation without asking first.**
- **Document only what the code cannot say, and only if it will still matter later.** A constraint,
  a measured number, a rejected alternative, a trap — yes. A restatement of what the function
  already does — no; it goes stale and then it lies. **One fact, one home:** if a paragraph exists
  in a tracked file, link to it rather than repeat it. Comments in config files get two lines and a
  pointer. Long-form reasoning belongs in the merge request, which is read once and never has to
  stay true.
- If something looks strange, ask rather than guessing. Most of what is written above was learned
  by not doing that.

## Skill cards

`.ai/skills/` holds one card per recurring task where doing the obvious thing produces a
plausible, wrong result. Each names the situation, points at the canonical procedure, and ends
with an exit criterion you can run. Read the one that matches before you start.

| Card | Use when |
|---|---|
| [`verify-a-claim`](skills/verify-a-claim.md) | writing a factual claim into a tracked file, a commit message or an MR |
| [`golden-master-failed`](skills/golden-master-failed.md) | `pytest -m golden` fails, or you expect it to |
| [`write-a-regression-test`](skills/write-a-regression-test.md) | fixing a defect, or adding coverage |
| [`change-a-dependency`](skills/change-a-dependency.md) | touching a pin in `pyproject.toml` |
| [`review-a-change`](skills/review-a-change.md) | a branch is ready — the reviewer panel, scaled to what the diff touches |
| [`open-a-merge-request`](skills/open-a-merge-request.md) | work is finished and needs to reach `develop` |

## Read next

This file deliberately does not restate what is documented elsewhere.

| For | Read |
|---|---|
| Running the suite, the golden master, the scenario format version, the retailer in/out convention | `tests/README.md` |
| What CI actually runs, and why it is informational rather than required | `CI_CD_Guide.md` |
| Branch naming, merge-request workflow, code of conduct | `CONTRIBUTING.md` |
| Installation and user-facing documentation | <https://hamlet-ens.readthedocs.io> (sources in `docs/`) |

`tests/README.md` also states the **retailer in/out convention**, which is the single most
error-prone thing in this codebase. Read it before touching any code that maps a retailer column
onto a transaction or forecast column.
