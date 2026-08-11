
# Changelog

All notable changes to this project will be documented in this file. 
See below for the format and guidelines for updating the changelog.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]
### Added
- **The PyOptInterface backend now runs on HiGHS, so it no longer needs a Gurobi licence.**
  `framework: poi` imported `pyoptinterface.gurobi` unconditionally and accepted only
  `solver: gurobi`. Because PyOptInterface links a *system* Gurobi installation directly, without
  `gurobipy` being installed, this looked like it worked on a developer machine that had one and
  was unusable everywhere else. Solver selection now lives in
  `hamlet/executor/utilities/controller/poi_solver.py` and honours `solver: highs` as well. HiGHS
  needs a shared library, which `highspy` does not expose — it bundles HiGHS inside its `_core`
  extension — so `highsbox` is a new dependency, pinned to `highspy`'s exact version so that both
  backends solve with an identical solver. Unblocks #198
- **A solver x framework equivalence matrix, so all four supported combinations are tested.**
  `framework` (`linopy` | `poi`) and `solver` (`highs` | `gurobi`) are independent options, and
  until now only the two HiGHS combinations were exercised. Both Gurobi cells now run wherever a
  licence is present and skip visibly, with a stated reason, where it is not. The tests compare the
  *objective value* of one MPC-shaped and one RTC-shaped model at the solvers' 1e-4 MIP gap — not
  setpoints or result tables, which legitimately differ between equally-optimal vertices — and a
  smoke arm runs `examples/create_simple_scenario` end to end under each combination that nothing
  else already covers, with `HAMLET_SMOKE_ALL=1` to run all four. See `tests/README.md`

  **`tests/benchmarks/test_backend_speed.py` now covers the same four cells**, reporting build and
  solve medians alongside each cell's objective, so correctness and cost are read from one table.
  On the development laptop the framework axis is worth ~50× and is almost entirely model *build*;
  the solver axis is worth a few per cent at HAMLET's model sizes and is deliberately reported
  rather than asserted

  **If you use Gurobi, note that the two frameworks reach it differently.** PyOptInterface links a
  *system* Gurobi installation directly and needs no Python package. linopy goes through
  `gurobipy`, which is an optional extra: run `uv sync --extra gurobi` before selecting
  `framework: linopy` with `solver: gurobi`, or linopy reports `Solver gurobi not installed` on a
  machine with a perfectly valid licence
- **A backend speed benchmark, `tests/benchmarks/test_backend_speed.py`.** It builds the same
  agent-MPC-shaped MILP through linopy and through PyOptInterface, hands both to the same HiGHS,
  and asserts they reach the same optimum before reporting the build/solve split. Deselected by
  default; run it with `pytest -m benchmark -s`. In a Linux container: **2.68 ms** per solve for
  PyOptInterface against **191.76 ms** for linopy, and the split shows why — linopy spends 138.89
  ms building a model that HiGHS then solves in 52.86 ms

### Removed
- **`psutil` is no longer a dependency.** Its only importer was the multiprocessing path removed
  above, and its own pin comment already recorded that the sole use — `TaskExecutioner.
  enough_memory`, guarding on `MIN_GB_AVAILABLE = 35` — had no callers. So it was installed for a
  function nothing invoked, inside a module nothing could run. The default environment is one
  package smaller; it still arrives under the `notebooks` extra, through jupyter, so notebook users
  see no change. The lock diff is two lines and no other version moved
- **The multiprocessing path, `hamlet/executor/utilities/tasks_execution/` — it did not work.**
  Six files and 435 lines that handed each agent to a worker process by writing the database to
  disk every timestep and reading it back. It was believed to be merely unused and unvalidated;
  measured against `develop` before deleting it, `num_workers=2` fails at the **first timestep of
  every shipped example**:

  | example | grid | `num_workers=1` | `num_workers=2` |
  |---|---|---|---|
  | `create_simple_scenario` | inactive | runs (this is the golden master) | **crashes at timestep 1** |
  | `create_scenario_with_market` | inactive | runs, 18 tables | **crashes at timestep 1** |
  | `create_scenario_with_topology` | active | crashes before the executor | crashes before the executor |
  | `create_scenario_with_grid` | active | crashes before the executor | crashes before the executor |

  `agent_pool.get_grid_restriction_commands` lists a `grids/` directory that a grid-less scenario
  never has, so every worker raised; `agent_pool.task`'s bare `except` turned that into a `None`,
  and the parent unpacked it — `TypeError: cannot unpack non-iterable NoneType object`. The two
  examples that would have had a `grids/` directory never reach the executor at all, so that
  branch has never run either. Nothing noticed, because every run in this project's history
  passed `num_workers=1`.

  **Migration.** `num_workers` is still accepted, and still means what it did for every existing
  caller: `1` or `None` run the simulation. Anything else now raises `ValueError` naming the
  number asked for, rather than silently running serial — a caller who asked for eight workers
  should find out. When parallelism returns it will be **threads over agents**, not processes; all
  three solver bindings release the GIL during solve, measured at ~6× on 8 cores. See ROADMAP
  §6.3 and §7.3.

  Deleted with it: the per-timestep `save_database(save_restriction_commands_only=True)`, the
  `save_all` branch of `AgentDB.save_agent`, and the `forecaster_train.pickle` that every agent
  save wrote for a worker to read back — nothing else has ever read it, so results folders no
  longer contain it. **No results move**: the golden master is unchanged, and the fingerprint it
  compares reads `.ft` tables only.

### Changed
- **PyOptInterface is now the default modelling backend, and this changes your results.**
  `framework: poi` replaces `framework: linopy` throughout `config_templates/` and all four
  shipped examples. On the shipped example the Executor stage goes **73.9 s → 15.7 s (4.7×)** and
  the whole run, scenario creation included, **86.2 s → 26.5 s (3.3×)**; the modelling layer
  itself is ~43× faster per solve, and the gap between those figures is the share of a run that
  was never the modelling layer.

  **Migration.** Existing scenario folders are unaffected — `framework` is baked into a scenario
  when it is created, so anything already under `scenarios/` keeps running on linopy and keeps
  producing the numbers it produced before. The change reaches you when you *create a new
  scenario* from the templates or examples. To stay on linopy, set `framework: linopy` under
  `ems.controller.rtc.optimization` and `ems.controller.fbc.optimization` in your `agents.yaml`;
  it remains fully supported and is the reference implementation PyOptInterface was validated
  against.

  **Expect different numbers, and not because either backend is wrong.** The two produce
  *mathematically identical models* — verified by exporting both controllers to LP and diffing by
  constraint shape — but the agent MILPs are degenerate: a battery or EV can shift charging
  between timesteps at identical cost, the backends break that tie differently, and
  `rtc_base.update_socs` quantises the chosen vertex into a state of charge the next timestep
  reads. On the shipped example that shows up as 3 row counts and 85 column statistics moving,
  with structure unchanged (same 18 tables, no column added or dropped). No agent differs at the
  first timestep, where both backends are handed identical inputs; the agent owning neither a
  battery nor an EV first differs at step 10, by 1 Wh. The committed golden master
  (`tests/e2e/golden/simple_scenario.json`) was re-baselined on that evidence. Closes #198
- **`framework: poi` is documented as experimental.** With a HiGHS library finally available, the
  shipped example was run under both frameworks and compared for the first time — the comparison
  #198 asks for. They do not agree: same 18 tables and no column added or dropped, but 3 row
  counts and 110 column statistics moved, by up to 100 %. Three backend defects were found and
  fixed (below), which brings that to 85. **What remains is degeneracy, not a modelling
  difference.** Both MPC models were exported to LP files and compared by constraint shape (sense,
  RHS and coefficient multiset, so variable naming does not matter): at the first timestep they
  are mathematically identical, every unmatched constraint being explained by one extra `+1.0`
  term for linopy's balance-dummy variable, which is fixed at zero and therefore inert. Objectives
  agree to ~1e-12 there. The error then holds at machine precision for several steps, jumps
  discretely, and grows — and the one agent owning neither a battery nor an EV stays at ~1e-13 for
  all 24 steps, being the only one with no state of charge to carry a divergence forward. So a tie
  in a degenerate MILP breaks differently and feeds the next timestep. One loose end: the
  EV-owning agent differs by ~1e-5 at the first timestep at identical state. `linopy` remains the
  default. See #198
- **`framework: poi` now runs on Windows** — see the entry under Fixed. The eight tests that
  carried `skip_on_windows` run there again, so the Windows suite goes from 221 passed / 8 skipped
  to 233 passed / 3 skipped, matching the other platforms

### Fixed
- **Fixed results depending on how busy the machine was (#204).** Three defects compounded into
  one: nothing pinned the solver's thread count, the configured `time_limit` was divided by 60 on
  its way to the solver, and `TerminationStatusCode.TIME_LIMIT` was accepted as success alongside
  `OPTIMAL` — so the shipped example asked for 120 s, got 2 s, and a solve that ran out of time had
  its suboptimal incumbent used with no signal at all. Under `framework: linopy` the limit had
  always been inert, because it was sent under Gurobi's `TimeLimit` key and HiGHS discards names it
  does not know; making `poi` the default therefore activated a limit that had never applied.

  **This was measured, not reasoned about.** Instrumenting all 192 solves of the shipped example:
  on an idle machine every one is `OPTIMAL` and the slowest takes 62 ms, and under artificial CPU
  load — 48 spinning processes on 16 threads — the median solve goes 11 ms → 414 ms, the slowest
  takes 3.33 s, and one returns `TIME_LIMIT` and is accepted. End to end, on the same commit and
  the same machine:

  | tree | machine | `pytest -m golden` |
  |---|---|---|
  | before | idle | 4 passed, 37 s |
  | before | loaded | **2 failed** (row counts *and* column statistics), 18m26s |
  | after | idle | 4 passed, 37 s |
  | after | loaded | 4 passed, 15m46s |

  So the golden-master failure recorded as flakiness on 2026-08-10 was not flakiness.
  `hamlet/executor/utilities/controller/solver_options.py` now names the two
  reproducibility-critical options once for both frameworks and both solvers — `threads = 1` and
  the limit **in seconds** — and `poi_solver.raise_unless_optimal` refuses anything short of a
  proven optimum, which is what the linopy controllers have always done. The commented-out `raise`
  next to it is gone with it: no solver failure was ever surfaced.

  **No results move.** The golden reference was recorded with `threads = 0` and a 2 s limit and
  still matches exactly, so on an idle machine HiGHS was already solving these models serially and
  the limit never bound. The thread count is pinned because it *can* vary with machine load, not
  because it was shown to have. Output suppression is still spelled per framework and still sends
  Gurobi's `OutputFlag`/`LogToConsole` to HiGHS, which ignores them; that is roadmap item #11
- **Fixed `framework: poi` crashing the interpreter on Windows with an access violation (#202).**
  The shipped example segfaulted at the first timestep and the suite died at a location that moved
  between runs. Neither dependency was at fault on its own, and the cause was not in HAMLET:
  `highsbox` 1.10.0's `highs.dll` imports `MSVCP140.dll` **by base name**, and the Windows loader
  satisfies such an import from whatever module of that name loaded *first*. `pyarrow` and
  `scikit-learn` each ship an unmangled private `msvcp140.dll` — 14.28 and 14.32 — and
  `import pandas` pulls `pyarrow` in, so HiGHS ran against a C++ runtime far older than the 14.43
  toolset it was built with, and corrupted memory. Measured on a ladder of runtimes, 20 runs each:
  14.28, 14.32 and 14.36 crash; **14.38 and newer never do**. Two controls make it causal — a copy
  of the *system* runtime placed at a foreign path never crashes, so the path is irrelevant and the
  version is the cause; and importing `pyarrow` *after* the first solve never crashes, so it is the
  load order. `hamlet/msvc_runtime.py` now claims the `MSVCP140.dll` name for the system runtime on
  the first line of `hamlet/__init__.py`, before `pandas` can, and `poi_solver` raises a
  `RuntimeError` naming the offending DLL if something wins the race anyway — a lost race can no
  longer corrupt the process silently. This imposes one rule on callers on Windows: **import
  `hamlet` before `pandas`**. `highspy` and PyOptInterface's own extensions are built with 14.29
  and were never exposed, which is why only the POI path broke
- **Fixed the PyOptInterface backend ignoring direct power control (§14a EnWG).** The real-time
  controller never overrode `OptimBase.apply_grid_commands`, which is a bare `pass`, and never
  stored `grid_commands` at all — so an agent on `framework: poi` accepted the grid operator's
  power caps and then discarded them. Nothing raised and nothing warned: the grid stage cannot
  observe that its commands had no effect, so it re-simulated the timestep, got the same answer
  and converged on an uncapped grid. Both control methods now match the linopy backend —
  `individual` tightens a plant's power variable and moves its target with it, `ems` constrains
  the sum of the agent's plant powers. `grid_commands` moved to `OptimBase`, so a backend cannot
  silently forget to store it again, and `tests/unit/.../rtc/test_direct_power_control.py` pins
  the behaviour across both backends
- **Fixed the POI real-time controller applying no heat-pump constraints at all.** `Hp` had its
  `define_constraints` and `__constraint_cop` written at module scope rather than inside the class
  (`rtc/optim/poi/components.py`), so the lookup fell through to the base class's `pass` and the
  heat pump lost both its COP coupling (`heat + electricity * cop == 0`) and its target-deviation
  constraint. `hp_electricity` was left free within its bounds and decoupled from `hp_heat`,
  making it a costless sink the solver used to absorb market deviations — which is why the heat
  pump was off by 3x and other components appeared not to deviate at all. Nothing raised, because
  the base `define_constraints` is a `@staticmethod` with a matching signature
- **Fixed the POI backends declaring market power as integer.** All four market variables in the
  real-time controller and both in the MPC were `integer=True` where linopy has `integer=False`,
  turning an LP into a mixed-integer problem — and, since `market_power` defaults to unbounded, an
  integer variable with infinite bounds. Any fractional optimum resolved differently
- **Fixed the POI real-time controller never reporting balance slack.** The slack warning used
  `np` without importing it, and the bare `except Exception` around it swallowed the resulting
  `NameError` on every call, so a POI run that shed or dumped energy to close its balance said
  nothing while the linopy run warned
- **Fixed new files under `input_data/` being silently ignored.** `.gitignore` carried
  `input_data/*` while the 152 files under it were tracked anyway, so adding an input left
  `git status` clean and needed `git add -f` — that is how the benchmark input `energy_da_raw.csv`
  was lost. `input_data/` is tracked source, so the rule is gone and a new file there now shows up
  like any other; no tracked file changed. Archives (`*.zip`, `*.rar`, `*.7z`, `*.tar.gz`,
  `*.tgz`) are ignored repository-wide instead, which is what the rule was really guarding against
- **Fixed the flexibility, heat and hydrogen market modules being unimportable.** All three did
  `from markets import Markets`, a top-level module that does not exist, so importing any of them
  raised `ModuleNotFoundError` — they could never have been used. They now import
  `hamlet.creator.markets.markets`, which works from anywhere now that the package is installed
  rather than reached through `sys.path`. An unused `import pandas as pd` went with it
- **Fixed `load_file` being unable to read an XLSX as a polars frame.** `pl.read_excel` needs
  `xlsx2csv`, which was in no environment this repository ever shipped, so
  `load_file(path, df='polars')` raised `ModuleNotFoundError: required package 'xlsx2csv' not
  found` instead of reading anything (`hamlet/functions.py:189`). Nothing in the suite touched the
  branch, so it stayed broken silently; `xlsx2csv` is now a declared dependency and
  `tests/integration/test_load_file_xlsx.py` covers both frame types
- **Fixed grid fees and levies being applied in the wrong power-flow direction.** The scenario
  configuration lists grid fees and levies as `[buying, selling]` but energy as
  `[selling, buying]`, and that inconsistency survived into the retailer table, so downstream
  code had to compensate per component. The Creator now normalises both spellings to one
  convention — `_out` is always the direction the agent pays for — and the executor crosses
  every retailer column uniformly. Existing scenario **configuration** files keep their current
  meaning and need no changes -- but see the migration note below, because generated scenarios
  and user-supplied retailer CSVs do not
- Fixed the MPC reading the market energy prices the wrong way round, so agents saw a lower
  price for buying than for selling and could trade against the retailer at a profit
- Fixed the MPC bounding the agent's purchase with the retailer's purchase quantity rather than
  its sale quantity (no effect while the two are configured equal)
- Fixed the PyOptInterface backends, which had drifted from the linopy ones: the MPC read the
  energy prices the wrong way round and bounded the market power with retailer columns that do
  not exist (so it raised `KeyError` on construction and the backend was unusable), the EV
  state of charge and `min_soc` scheme still had their pre-fix forms, the real-time controller
  ignored the configurable optimisation bounds, and neither controller had balance slacks
- Fixed grid fees and levies being charged on gross rather than net energy, which overcharged
  every agent that both bought and sold within a timestep
- Fixed the net energy behind grid fees and levies omitting the trades cleared in the timestep
  being settled
- Fixed the EV state of charge ignoring the battery capacity and being able to go negative
- Fixed the EV `min_soc` charging scheme, which let the car leave below its minimum state of
  charge. The state-of-charge recursion now includes the energy the car spends driving; without
  it the modelled state of charge only ever rose, so once the car had driven anywhere the
  minimum was satisfied on paper and the car was never actually recharged
- Fixed EV time series being averaged when resampled, which lost most of a trip's driving
  energy and could truncate the availability flag to zero so the car never charged at all.
  Resampling an EV series to a finer resolution also used to drop the end of the series
- Fixed controllers configured as off still being run when the setting came from `agents.xlsx`
  as an empty cell rather than as `None`
- Fixed retailer prices being broadcast as a column instead of read as a scalar, which gave
  each transaction in a timestep a different price when the retailer table had several rows
- Fixed the weather file name being hard-coded in the executor rather than read from the
  scenario's `setup.yaml`, so a scenario declaring `weather.csv` failed with a
  FileNotFoundError naming a file the user never asked for
- Fixed `load_file` being unable to read any CSV with default arguments: polars rejected the
  default `parse_dates=None` with a TypeError naming an argument the caller never passed
- Fixed the forecaster failing with `list.remove(x): x not in list` on scenarios whose agent
  time series name the time column `index` rather than `timestamp`
- **Fixed generated scenarios depending on the filesystem, so the same configuration and seed
  produced different scenarios on different machines.** The Creator picks each agent's load, PV,
  heat and EV profile by drawing an index from a seeded random number generator and using it to
  index `os.listdir()`. The seed fixed the index, but the *order* came from the filesystem —
  alphabetical on NTFS, hash order on ext4 — so Windows and Linux assigned different profiles to
  the same agent. Agent ids, plant counts and sizings matched, which made it easy to believe the
  scenarios were identical when they were not. The listings are now sorted. On Windows the results
  are unchanged, because NTFS already returned sorted order; on Linux they change to match
- **Fixed `env.yml` not producing a working installation.** It pins only HAMLET's direct
  dependencies, so `xarray` — which arrives transitively through linopy — was resolved to the
  newest release, and linopy 0.3.11 then failed at import with
  `ModuleNotFoundError: No module named 'xarray.core.rolling'`. A fresh environment could not
  `import hamlet` at all. `xarray` is now pinned to the version HAMLET is developed against.
  This is a stopgap; the durable fix is a lockfile, so that no transitive dependency is left for
  the resolver to pick
- Fixed the PyOptInterface tests reporting a missing Gurobi licence as two failures rather than
  as skips. They asked whether the solver's shared library was loadable, which Gurobi's is even
  without a licence — the licence is only checked at `optimize()`. They now probe by actually
  solving a two-variable problem, and prefer HiGHS, so that a developer with a Gurobi licence
  and a machine without one exercise the same backend
### Added
- **Added continuous integration** (`.gitlab-ci.yml`). Every push and merge request runs the unit
  and integration suite, the shipped example end to end, and the golden master against its
  committed reference — as three separate jobs, so a failure names which layer broke. No solver
  licence is needed. A lint job runs `ruff` restricted to genuine errors (undefined names, broken
  syntax and asserts) rather than style, because the repository has no agreed style yet
- Added `ci/requirements_from_env.py`, which derives CI's pip requirements from `env.yml` so that
  there is still only one dependency definition. It omits `tensorflow`, `psycopg2` and `jupyter`
  from the CI environment, and `--verify` re-checks the two omissions that are claims about the
  source rather than about download size
- Added `pytest` to `env.yml`. It was in no environment definition at all, so an environment built
  as the documentation describes could not run the test suite
- Added a test suite (`tests/`) split into `unit/`, `integration/` and `e2e/`, mirroring the
  package layout. `python -m pytest tests` runs unit and integration in seconds on HiGHS;
  `python -m pytest tests -m e2e` runs the shipped example end to end
- **Added a scenario format version.** Generated scenarios now carry
  `scenario_format_version` in `general/general.json`, and the Executor and the Analyzer refuse
  to read a folder whose version they do not match — including one with no stamp at all, which
  means it was generated before the retailer in/out convention was fixed. The error names the
  folder, what would otherwise have gone wrong silently, and the remedy. This is the on-disk
  scenario format, deliberately independent of `VERSION`: a release that changes no scenario file
  does not invalidate anybody's scenarios. Override with `allow_incompatible_scenario=True`
- Added a test that fails when the generated scenario's shape changes without the format version
  being bumped (`tests/integration/creator/test_scenario_format_shape.py`, in the default suite).
  It fingerprints which files, columns and JSON keys the Creator writes, against a reference
  committed per version, and its failure message spells out the choice between reverting,
  regenerating and bumping. The rule for when to bump — including the case a structural
  fingerprint cannot see — is documented at `hamlet.constants.SCENARIO_FORMAT_VERSION`
- Added a golden-master test (`python -m pytest tests -m golden`) that runs the shipped example
  under a fixed seed and compares per-table row counts and per-column statistics against a
  committed reference, so a change that moves results has to be acknowledged rather than noticed
  later. Regenerate the reference with `HAMLET_UPDATE_GOLDEN=1` and commit it with the change
- Added `.ai/context.md`, one orientation document for AI coding agents, with `CLAUDE.md`,
  `GEMINI.md` and `AGENTS.md` as pointers to it so that every tool reads the same text rather
  than three copies that drift apart. It covers the Creator → Executor → Analyzer flow, the
  load-bearing environment pins, the solver-coupling of the golden reference, the seeded-RNG
  ordering hazard, the branch and mirror rules, and why the paper lineage cannot be reproduced
  from `develop`. It links `tests/README.md` and `CONTRIBUTING.md` rather than restating them,
  and is deliberately short: a long document that drifts is worse than none
- Added `.ai/skills/`, one card per recurring task where doing the obvious thing produces a
  plausible but wrong result: writing a factual claim into a tracked file, a golden-master
  failure, a regression test, an `env.yml` change, and opening a merge request. Each names
  the situation, points at the canonical procedure rather than restating it, and ends with
  an exit criterion that can be run. Every rule in them is traceable to something that
  actually went wrong here
- Added a reviewer-panel card (`.ai/skills/review-a-change.md`): independent reviewers, one per
  lens, each blind to the others, followed by an adversarial pass that tries to refute what
  they found. The panel scales to what the diff *touches* rather than to how large it is,
  because a three-line change to an optimisation bound moves every scenario and a large
  plotter change cannot. It exists because the defects that mattered here were caught by
  review and by measurement, and not once by the suite going green
### Changed
- **HAMLET is an installable package, defined by `pyproject.toml` and a committed `uv.lock`.**
  `uv sync` builds the environment — including Python 3.11 itself — installs the exact versions
  in the lock, and installs HAMLET in editable mode. `import hamlet` therefore works from any
  directory: the `sys.path` lines in `run.py` and `tests/conftest.py`, and the two that were
  generated into the e2e subprocess scripts, are gone, and so is the `PYTHONPATH` a benchmark
  used to need. `run.py`'s first two lines also contained a mangled statement
  (`...outside an IDEimport sys`) in which a missing newline had swallowed an `import`
- **There is now exactly one dependency definition.** `env.yml`, `docs/requirements.txt` and
  `ci/requirements_from_env.py` are removed; CI, Read the Docs and contributors all install from
  the same lock. `env.yml` pinned only HAMLET's *direct* dependencies, which is why an
  unconstrained transitive `xarray` could break `import hamlet` on a fresh install; the lock pins
  all 195 packages, and `uv sync --locked` in CI fails if the lock and `pyproject.toml` ever
  disagree
- The linopy/xarray ceiling is a declared constraint rather than a coincidence of resolution:
  `xarray==2024.6.0` sits in `pyproject.toml`, so `uv lock --upgrade` can no longer move it. A
  resolver will never enforce it on its own, because linopy declares only a floor
  (`xarray>=2024.2.0` in every release from 0.3.13 to 0.9.0) — so
  `tests/unit/test_dependency_constraints.py` enforces it instead, failing on the commit that
  relaxes either pin rather than on whoever next builds a fresh environment
- TensorFlow, Gurobi and Jupyter are optional extras (`uv sync --extra tensorflow`, `--extra
  gurobi`, `--extra notebooks`); pytest, ruff and Sphinx are dependency groups. The default
  environment is ~600 MB smaller and needs no solver licence. `sktime` stays core: unlike
  TensorFlow it is imported at module scope
- `psutil` moved from 5.9.0 to 5.9.4, the one pin this change had to alter. 5.9.0 predates
  Python 3.11 and ships no cp311 wheel for any platform, so installing it from PyPI meant
  compiling it — the reason CI ran the 900 MB `python:3.11` image rather than `-slim`, and an
  outright failure on Windows without MSVC. It is numerically inert: its only use,
  `TaskExecutioner.enough_memory`, has no callers
- All other versions are unchanged from the environment the golden master was measured in, and
  every pin is exact. That is a stage rather than an end state: it keeps this change attributable,
  so relaxing the pins to ranges and moving off Python 3.11 stay separate, measured steps
- Out-of-horizon market records are dropped more cheaply: most calls have nothing to drop and
  now exit after a single pass, and when there is, the membership test is evaluated once and
  reused for both sides of the split instead of scanning the table twice. The output folder is
  only created when something is written. Measured on a 20,000-row table, the case that occurs
  in practice went from 0.097 ms to 0.010 ms
- pandapower's progress bar is no longer printed once per horizon on every timestep when the
  §14a grid restriction is active
- The real-time controller's optimisation bounds (balancing power, market power and the two
  heat-pump fallbacks) are now configurable via a `limits` block under the `rtc` controller
  instead of being hard-coded. The defaults reproduce the previous behaviour exactly
- **The balance equations now carry slack variables, on by default**, so a single infeasible
  agent no longer aborts a whole run: the controller sheds or dumps energy at a
  value-of-lost-load penalty instead. This changes what a previously-infeasible scenario does --
  it now completes instead of raising. Every use is logged at WARNING, because the shed energy
  is not written to the setpoints and the results for that timestep therefore do not balance.
  Disable per agent with `slack: false` under the controller to get the old behaviour back
- **The shipped example `examples/create_simple_scenario` now solves with HiGHS instead of
  Gurobi, and its results change as a result.** HAMLET's own example should not require a
  commercial licence to run: HiGHS is installed with HAMLET, Gurobi is not, and the example is
  what the README names as the installation test. Gurobi remains fully supported — set
  `solver: gurobi` under `optimization` in `agents.yaml`, as the other examples and the config
  templates still do.

  The two solvers do not return the same dispatch. The model is degenerate enough that they pick
  different, equally optimal solutions: 84 more bids and offers clear (499 → 583), 168 more
  market transactions (1,193 → 1,361), and 76 per-column statistics move — most of them small
  (heat-pump heat by 0.04 %, heat-pump electricity by 0.7 %), a few large where an EV or battery
  is charged on a different schedule for the same cost. No table and no column appears or
  disappears, and total traded energy is unchanged in character. The committed golden-master
  reference has been regenerated accordingly, so anyone comparing against previously published
  example numbers should expect this shift. Verified to be solver-only and not
  platform-dependent: the same example under HiGHS produces identical results on Windows and in
  a Linux container
- **Corrected the clone URL in the README**, which named `github.com/tum-ewk/hamlet` — the wrong
  organisation, so a newcomer following the installation guide failed at its first step. The
  README also said documentation was "currently being developed"; it has been published at
  <https://hamlet-ens.readthedocs.io> for some time and is now linked
- **Rewrote `CI_CD_Guide.md`.** It was generic template boilerplate describing install / lint /
  test / build / scan / deploy / notify stages, none of which this project has, sitting next to a
  real `.gitlab-ci.yml` it did not describe. It now documents the pipeline that exists: the five
  jobs, why lint is restricted to genuine errors, why dependencies are derived from `env.yml`
  rather than copied, why installs go through uv into a venv on `/cache`, and why a pipeline
  running on a single laptop runner is informational rather than a merge blocker

### Removed
- **Removed `input_data/general/weather/weather.csv` (91 MiB), which nothing referenced.** Every
  shipped `setup.yaml`, `config_templates/`, the documentation and the executor default all name
  `weather.ft` — the same data, verified identical (813,100 rows, 17 columns, all 15 numeric
  columns equal and both timestamps equal once parsed) at a 24th of the size. A fresh checkout's
  `input_data/` drops from ~160 MB to 69 MB. If you have a hand-edited `setup.yaml` naming
  `weather.csv`, change it to `weather.ft`; the repository history still carries the file
- Removed `env.yml`, `docs/requirements.txt` and `ci/requirements_from_env.py`, superseded by
  `pyproject.toml` and `uv.lock`. Contributors using conda are not stranded — see the README —
  but the conda path is no longer version-locked, and `uv sync` is
- Removed the `dependency-exclusions` CI job with the script it ran. It asserted that nothing
  under `hamlet/` imported `tensorflow` or `psycopg2` at module scope, because CI installed a
  hand-maintained subset of `env.yml`. The claim is now structural: `fast` imports `hamlet` in an
  environment with neither installed, so a module-scope import fails on the commit that adds it
- Removed `psycopg2` and `numba` from the dependencies. Neither is used: `psycopg2` has no
  occurrence anywhere in the tree, and `numba` appears only in three commented-out lines
  (`executor/setup.py:16`, `linopy/components.py:683`, `poi/components.py:649`)
- Removed `TEMPLATE_USAGE_GUIDE.md`, scaffolding inherited from the repository template this
  project was started from. It described a `/src` layout, a `requirements.txt`, MkDocs and
  `bump2version`, and a `main` branch — none of which exist here. What of it applies is in
  `CONTRIBUTING.md`

### Migration
- **Re-create your scenarios — HAMLET now tells you when you have to.** The power-flow direction
  fix spans the Creator and the Executor, so a scenario folder generated by an earlier version
  and executed by this one would have its grid fees and levies applied to feed-in instead of
  consumption. That is no longer silent. Scenario folders now carry a format version, and one
  that predates it stops the run with

  ```
  ScenarioFormatError: This scenario carries no format version, so it was generated before
  HAMLET started stamping one.
  ```

  followed by what would have gone wrong and how to fix it. The remedy is to re-run the Creator
  on the same configuration folder: `setup.yaml`, `agents.xlsx`, `markets.yaml` and `grids.yaml`
  do not need to change, only the generated scenario does. Existing **results** folders are
  affected the same way, because the Executor copies the scenario into them, so the Analyzer
  refuses them too. If you are deliberately re-reading an old scenario and accept that its
  numbers are not comparable, pass `allow_incompatible_scenario=True` to `Executor` or
  `Analyzer` — it proceeds, and prints what it suppressed
- **Relabel your own retailer CSVs.** Scenarios using `method: file` read retailer columns by
  name and no normalisation is applied to them. In `grid.csv` and `levies.csv` the charged value
  must now sit in the `_out` column rather than `_in`; the two files shipped under
  `input_data/retailers/lem/` have been relabelled accordingly. `energy.csv` and `balancing.csv`
  are unaffected
- `Agents._resample_timeseries` takes an additional `plant_dict` argument, and
  `ElectricityMarket._create_fixed_cols` became a classmethod. Both are internal, but anything
  calling them from outside the package needs updating

## [Version 1.2.0] - 2025-07-29
### Added
- Added extensive expert documentation in the "advanced topics" section (!189)

## [Version 1.1.0] - 2025-06-13
### Added
- Added the analyzer architecture and a few plots (!150)
- Added some plots to the example notebooks (!179)
### Fixed
- Fixed the region_db as a function returns empty list instead of None now (!180)
- Fixed that grid results were stored in the wrong folder (!181)
- Fixed that the storage options were not calculating the SoC properly (!182)
- Fixed that the agents were not trading properly due to false forecasts (!183)
### Changed
- Changed the README as the installation is now tested using the example notebooks (!179)

## [Version 1.0.1] - 2025-03-28
### Added
- Added the readthedocs file to generate the website on the GitHub repo (!172).
### Fixed
- Fixed a bug in the example `create_scenario_with_grid` (!170)

## [Version 1.0.0] - 2025-03-28
### Added
- Added a changelog (mea culpa, should have done this earlier). Will be used from now on.
- Added examples to work with the tool.
- Added calculation of grid fees based on whether it is local or retail trade (!123)
- Added grid model (!147)
- Added $14a EnWG grid control (!148)
- Added parallel processing using multiprocessing (!126).
- Added PyOptInterface (poi) as optimization problem framework (!129).
- Added github and gitlab templates (!137)
### Changed
- Major restructuring of the codebase (!164).
- Major change of the repo (!165).
- Changed market names to energy type and removed the `local` prefix (!160).
- Changed electricity energy type name from `power` to `electricity` (!159).
- Changed environment
### Fixed
- Fixed a bug in the energy demand forecast model.
- Fixed broken scenarios (!162).
- Fixed clearing when there are no trades (!159).
- Fixed that local price was not updated in forecaster (!148).

---

# Guidelines for Updating the Changelog
## [Version X.X.X] - YYYY-MM-DD
### Added
- Description of newly implemented features or functions, with a reference to the issue or MR number if applicable (e.g., `!42`).

### Changed
- Description of changes or improvements made to existing functionality, where relevant.

### Fixed
- Explanation of bugs or issues that have been resolved.
  
### Deprecated
- Note any features that are marked for future removal.

### Removed
- List of any deprecated features that have been fully removed.

---

## Example Entries

- **Added**: `Added feature to analyze time-series data from smart meters. Closes #10.`
- **Changed**: `Refined energy demand forecast model for better accuracy.`
- **Fixed**: `Resolved error in database connection handling in simulation module.`
- **Deprecated**: `Marked support for legacy data formats as deprecated.`
- **Removed**: `Removed deprecated API endpoints no longer in use.`

---

## Versioning Guidelines

This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html):
- **Major** (X): Significant changes, likely with breaking compatibility.
- **Minor** (Y): New features that are backward-compatible.
- **Patch** (Z): Bug fixes and minor improvements.

**Example Versions**:
- **[2.1.0]** for a backward-compatible new feature.
- **[2.0.1]** for a minor fix that doesn’t break existing functionality.

## Best Practices

1. **One Entry per Change**: Each update, bug fix, or new feature should have its own entry.
2. **Be Concise**: Keep descriptions brief and informative.
3. **Link Issues or MRs**: Where possible, reference related issues or merge requests for easy tracking.
4. **Date Each Release**: Add the release date in `YYYY-MM-DD` format for each version.
5. **Organize Unreleased Changes**: Document ongoing changes under the `[Unreleased]` section, which can be merged into the next release version.

