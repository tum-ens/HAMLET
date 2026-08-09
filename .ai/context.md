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

**Two solver libraries ship, deliberately.** `highspy` is the one linopy talks to. It bundles
HiGHS inside its `_core` extension and exposes no shared library, which PyOptInterface needs — so
`highsbox`, the same HiGHS build packaged as a plain `highs.dll` / `libhighs.so`, is also a
dependency. Both are pinned to `1.10.0` so the two backends solve with an identical solver; keep
them equal, or a backend comparison stops being a comparison.
`hamlet/executor/utilities/controller/poi_solver.py` is the only place that selects or loads
either, and `framework: poi` works on HiGHS because of it — before that it imported Gurobi
unconditionally and silently required a system Gurobi installation.

**`framework: poi` does not work on Windows, and is not numerically validated anywhere.** Two
separate problems, both measured:

- *Windows*: PyOptInterface + highsbox crash the interpreter with an access violation. The shipped
  example under `framework: poi` segfaults at the first timestep, reproducibly, outside pytest.
  The affected tests carry `skip_on_windows` from `tests/poi_support.py`, which records the
  evidence and the ruled-out causes. Linux is unaffected and CI is Linux.
- *Results*: on Linux, where it does run, `poi` does **not** reproduce the linopy numbers. Three
  backend defects have been fixed (heat pump built with no constraints at all, market power
  declared integer, dead slack reporting), taking the gap from 110 differing column statistics to
  85; the remainder is concentrated in the EV and is open. The same harness under `linopy`
  reproduces the committed golden master exactly, before and after, so the difference is the
  backend, not the harness. `tests/e2e/test_backend_equivalence.py` holds this as an
  `xfail(strict=True)` — when it starts failing, the backends agree and the marker should go. See
  #198; `poi` stays experimental and `linopy` stays the default until it is fixed.

**§14a grid restrictions are implemented but almost entirely untested, and no example runs them.**
Direct power control reaches the RTC only (never the FBC), through `apply_grid_commands`; indirect
control (variable grid fees) is applied outside the solver in `agent_base.py`, so it is
backend-agnostic and both MPC backends pick it up. Nothing in `tests/` executes `EnWG14a`, and no
shipped example enables the mechanism: the two scenarios that set `restrictions.apply:
['enwg_14a']` have `electricity.active: False`, and the two with a live grid have an empty
restriction list. An end-to-end test needs a new fixture — and note that
`examples/create_scenario_with_grid` currently **cannot run at all**, on `develop` as well: it
raises `TypeError: cannot unpack non-iterable NoneType` in `grid_db._create_grid_from_file`,
because `__assign_inflexible_load_for_agent` returns `None` when no load matches.

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

Four unsorted `os.listdir()` calls of the same shape remain. Whether any of them can affect output
is **not established** — nobody has checked:

- `hamlet/creator/setup.py:609`, `:692`
- `hamlet/executor/utilities/database/market_db.py:158`
- `hamlet/executor/utilities/tasks_execution/agent_pool.py:108`

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
