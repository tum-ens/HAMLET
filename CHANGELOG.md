
# Changelog

All notable changes to this project will be documented in this file. 
See below for the format and guidelines for updating the changelog.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]
### Added
- **The first coverage of the `ctsp` and `industry` agent types, which had none of any kind.** No
  `agents.yaml` outside `config_templates/` declared either type and no test imported either class,
  so ~1900 lines of Creator and both Executor agent classes were never executed — while the two
  Creator classes, 92 % identical, quietly drifted apart in four behavioural ways (#213).
  `tests/e2e/scenarios/ctsp_industry/` is one agent of each type with PV and a battery, one day at
  hourly resolution, no grid, and **two tests read it through different Creator entry points**
  because neither half covers the other: `e2e/test_ctsp_industry.py` builds it with
  `new_scenario_from_files` and simulates it, which reaches the Executor classes but — traced, not
  assumed — no Creator class at all, since `create_agents_from_file` never consults `Agents.types`;
  `integration/creator/test_ctsp_industry_creator.py` builds the same folder with
  `new_scenario_from_configs` in ~4 s, which is what reaches the ~1900 Creator lines and pins the
  shape of the workbook each class writes.

  **It settles the Executor's `# TODO: Not yet tested and implemented` on those two classes: they
  run.** `Ctsp` and `Industry` are `AgentBase` subclasses with no behavioural overrides, and a run
  produces the same result tables `sfh` does. The TODO was about testing, not about missing code.

  The Creator-side test immediately earned itself: it found a **fifth** ctsp/industry divergence
  nobody had spotted by reading — the ctsp block's EV forecast sub-block is
  `random_forest_classifier:` where the registered model, and what `sfh` and `industry` write, is
  `rfr`. The known column differences are now pinned as data, so a deduplication (#213) has to
  decide each one visibly rather than silently.

  Four values in the fixture are load-bearing and `tests/README.md` says why, the least obvious
  being that it ships **`framework: linopy`, not the default**. The scenario is built with
  `new_scenario_from_files`, so `agents.xlsx` is what the Creator reads and the #206 read-back has
  to ask it for a backend it does *not* ship; shipping `linopy` makes that request `poi`, the fast
  one. It declared **no EV** when it landed, because the EV path did not work for either type; the
  three fixes below changed that, and it now carries one on each sheet
- **The EV path works for `ctsp` and `industry`, and the fixture above now exercises it.** It
  shipped `ev.share: 0` because raising it failed in five ways, each hiding the next: two stale
  spellings and an unregistered forecast model in `config_templates`' ctsp block (#218), the
  Creator writing `NaN` into every nested `charging_scheme` parameter for *both* classes (#219),
  and PyOptInterface refusing to build `charging_scheme.method: full` at all (#220). All three
  issues are closed and `tests/README.md` keeps the sequence.

  The share is now `1` on both sheets, and it is guarded rather than merely set:
  `check_the_ev_premise` fails by name if it returns to 0, and separately if the nested columns are
  present but entirely NaN — which is #219's exact signature, since that defect wrote the columns
  and left them empty while the Creator reported success
- **A golden master for the grid stage and §14a, which had none.** The only scenario the golden
  master pinned sets `electricity.active: False` and calculates no grid, so no committed reference
  number had ever come from the power flow, the variable grid fees or direct power control — while
  the grid stage was being measured and optimised. `tests/e2e/scenarios/grid_golden/` is a 21-bus
  feeder with four single-family homes over one day, deliberately weak: at 15 kVA the transformer
  is below the uncontrolled peak of 19.8 kW, so it overloads at 132 % and the restriction fires,
  and above the 8.4 kW §14a floor of the two agents that are actually curtailable at those hours
  (the other two draw 892 W between them, well under the 4200 W threshold). It is *below* the
  16.8 kW floor all four agents together would be guaranteed, so a fixture change that brought a
  third agent's EV onto the same hour would leave an overload §14a cannot resolve.

  The topology is a radial slice of the real low-voltage feeder the §14a study used — real cable
  lengths, real impedances and ampacities — **rebuilt from explicitly chosen electrical parameters
  rather than by deleting columns**, so no street address, asset identifier or utility tag can leak
  by being overlooked. It lives under `tests/` rather than `examples/` because it is tuned to
  overload rather than to be copied.

  `tests/e2e/test_grid_restrictions.py` asserts the mechanism rather than the numbers, in four
  claims that fail separately: the feeder overloads, the fees vary, a command is issued, and **the
  command is respected in the resulting power flow**. That last one is what the grid stage cannot
  check for itself — it re-simulates, gets the same answer and converges on an uncapped grid — and
  restoring the pre-!209 no-op `apply_grid_commands` fails it alone, with the other three still
  green.

  `fingerprint` now covers the grid stage's CSV results as well as the Feather tables; a scenario
  with no grid writes no CSV, so the existing reference does not move
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

### Changed

- **Two test modules asking for the identical end-to-end run now share one, through
  `tests/scenario_cache.py`.** Every e2e fixture is module-scoped, so each file paid for its own
  run. Across the whole suite that is exactly **one** duplicated pair — `e2e/test_grid_restrictions`
  and `e2e/test_golden_master[grid_golden]` make byte-identical requests — worth 70–125 s on the
  development laptop. **It saves nothing in CI and is not meant to**: `e2e` and `golden` are
  separate jobs and separate pytest processes, and the pair straddles that boundary, so the saving
  is real only for a local `pytest tests -m "e2e or golden"`. What the cache is for beyond those
  that is that it is the mechanism, so the next duplicate shares automatically instead of being
  noticed by someone counting runs.

  The key is derived **mechanically**, by binding each request against `run_example`'s own
  signature, so a parameter added to `run_example` later is part of the key with nothing to
  remember; `NOT_PART_OF_THE_REQUEST` is an exclusion list whose failure direction is the safe one
  (naming too little costs a run, where an allowlist that named too little would merge two
  different requests — the shape `ROUNDING`, `KEYS` and `AGENT_TABLES` each failed in). Two
  requests that differ at all do not share: the golden master passes no `framework` so that it runs
  whatever the config ships, and the equivalence test's `poi` arm passes `framework='poi'`, so
  those two correctly remain separate runs.

  Every consumer re-reads the run's own artefacts — the `BACKEND_PROBE` receipt and the scenario
  directory actually written — on cache hits as much as misses, because the key that decided two
  requests were the same cannot also be the evidence that they were.
  `tests/unit/test_scenario_cache_key.py` breaks the key on purpose and pins that the consumer,
  not the key, is what rejects the mis-served entry.

- **`CONTRIBUTING.md`'s branch naming convention now matches what the repository actually does.**
  It documented `type-issue-nr-short-description` with types `feature`/`hotfix`/`release`, and
  gave `feature-42-add-new-ontology-class` as the example. Not one of the last 40 merged branches
  followed it: the convention in use is `type/short-description` with the issue number leading the
  description, over nine types (`fix`, `feat`, `chore`, `test`, `perf`, `refactor`, `docs`, `ci`,
  `release`). The document is corrected to the observed practice rather than the practice to the
  document, and the stale `main` branch reference is now `master`

- **The `open-a-merge-request` skill card now requires `Closes #<n>` in the MR description.**
  `CONTRIBUTING.md` §5 has always said so, but the card did not repeat it and the rule was skipped
  every time — #195 and #200 each sat open for a week after the work that fixed them had merged.
  The mechanism was never broken: `develop` is the default branch and autoclose is on. The card
  also gains the one check that settles it, `glab api .../merge_requests/<iid>/closes_issues`

- **`import hamlet` no longer silences every warning in the process.** `hamlet/executor/setup.py`
  called `warnings.filterwarnings("ignore")` at module scope, and `hamlet/__init__.py` imports
  that module, so importing HAMLET installed a process-wide blanket filter — hiding HAMLET's own
  warnings, every dependency's, and any raised by the importing program. A second blanket filter
  (`FutureWarning`, all modules) sat in `hamlet/creator/agents/agents.py`, and `pytest.ini`
  carried a third (`ignore::DeprecationWarning`) for the suite. All three are gone.

  Suppression that is still wanted is now *enumerated* rather than blanket:
  `hamlet/warning_policy.py` lists each hidden message with its category and reason, and
  `quiet_known_noise()` installs exactly that list around a Creator or Executor run and removes
  it again afterwards, including when the run raises. Nothing runs at import.
  `tests/conftest.py` registers the same list with pytest so the suite and the runtime cannot
  disagree. Everything on the list is HAMLET calling deprecated polars 0.20 APIs
  (ROADMAP item #12); what it hid, what a bare removal would have cost, and why it is enumerated
  rather than deleted are recorded in that module. Anything *not* on the list now reaches the
  user — including two live §14a defects that had been invisible for years, now #210 and #211.
  Closes #199 and the warnings half of ROADMAP item #11

- **Solver output options are looked up per solver instead of written out inline.**
  `mpc_linopy.py` and `optim_linopy.py` sent the literal `{'OutputFlag': 0, 'LogToConsole': 0}`
  to whichever solver was configured. Those are Gurobi's names; HiGHS discards them unrecognised,
  so under the default backend nothing was switched off and the `sys.stdout` redirect around the
  call was doing all the work. This is the same defect as #204's `TimeLimit`, differing only in
  consequence — a discarded log flag prints, a discarded time limit changes results.
  `solver_options.quiet_options` now answers with `output_flag`/`log_to_console` for HiGHS and
  `OutputFlag`/`LogToConsole` for Gurobi, in each solver's required type, and `poi_solver`
  reads the same table so the two backends cannot drift. No numbers move: these options affect
  logging only, and the golden references are unchanged

- **The solve no longer leaks a file object or loses stdout on failure.**
  `sys.stdout = open(os.devnull, 'w')` before each linopy solve and `sys.stdout = sys.__stdout__`
  after it leaked the handle on every solve, never restored on an exception — a raising solve
  left the process writing to devnull permanently — and restored to `sys.__stdout__` rather than
  to whatever the caller had installed, which broke pytest's capture. Replaced with
  `contextlib.redirect_stdout`. The remaining half of roadmap item #11

- **`freq='S'` is now `freq='s'`** in `creator/agents/agents.py` and `creator/markets/electricity.py`.
  pandas deprecated the capitalised alias and will remove it; the two spellings denote the same
  offset, so this fixes 37 warnings per run at no behavioural cost rather than suppressing them

- **The golden master can pin more than one scenario.** `tests/e2e/test_golden_master.py` held its
  example and scenario name as module-scope constants, so exactly one scenario could ever be
  pinned — and the one it pins sets `electricity.active: False`, which is why no reference numbers
  have ever covered the grid stage. Scenarios are now a list, parametrised so each is run once for
  the whole module, with its reference at `tests/e2e/golden/<scenario>.json`. Adding one is appending a
  `GoldenScenario` and running `HAMLET_UPDATE_GOLDEN=1`; see `tests/README.md`. No scenario was
  added in the same change, so that the reference could be shown byte-identical before and after
- **Agents no longer scan the whole market-transaction table to find what they already traded.**
  `strategies.py` filtered and grouped `market_transactions` on every agent on every timestep;
  `MarketDB.get_net_energy` answers the same question from a running per (timestep, agent) sum.
  Measured on a 104-agent, three-month scenario with a live grid, the agent stage grew from 3.7 s
  to 12.1 s per timestep over 120 steps while every other stage stayed flat — the deepcopy that
  ROADMAP §6.1a names as the main mechanism was 0.6 % of a timestep throughout.

  **The type filter is the reason this is not a two-line change.** `market_transactions` also
  carries `grid` and `levies` rows that clone the netted energy, so summing without filtering to
  `retail | market | balancing` roughly triple-counts traded energy for any agent paying fees. A
  version of this cache without that filter was refused during the paper-fix port for exactly this
  reason; `MarketDB.NET_TRANSACTION_TYPES` is now the single place the filter lives, used both to
  build the cache and to answer without one. `tests/unit/executor/utilities/database/`
  `test_net_energy_cache.py` compares the two paths on every case. No results move

  **The real-time controller did the same scan, and it was the larger of the two.**
  `RtcBase._get_market_results` is now served from the same machinery. Measured end to end on that
  scenario, 150 timesteps per arm:

  | steps 141–160 | timestep | agent stage | RTC+FBC | trading |
  |---|---|---|---|---|
  | before | ~19 s and rising | — | — | — |
  | after the trading cache only | 14.63 s | 9.62 s | 8.99 s | 0.25 s |
  | after both | **7.84 s** | 2.99 s | 2.39 s | 0.24 s |

  The baseline reached 21.8 s per timestep by step 218 and was still climbing; with both caches the
  timestep is roughly flat
- **The real-time controller no longer counts grid fees and levies as traded energy.** It summed
  `market_transactions` with no transaction-type filter where the trading strategy has always
  excluded them. **This changes no results, and that was measured rather than assumed**: filtered
  and unfiltered sums agree on 96 of 96 calls in the shipped example and 1040 of 1040 in a
  three-month, 104-agent scenario whose table does hold 890 `grid` and 890 `levies` rows. Fees are
  written when a delivery timestep is settled, and agents run before markets, so the timestep the
  controller asks about never has any. The filter makes that independent of ordering instead of
  silently dependent on it
- **§14a's variable grid fees are five times cheaper to compute, and produce identical numbers.**
  The grid stage is the largest single term in a timestep once the market rescans are gone, and
  `pp.runpp` is **3 ms** of it — it was never power flow. Two things were: the horizon's flows went
  through `pandapower.timeseries.run_timeseries` (2.065 s per timestep, of which ~0.07 s is the 24
  flows and the rest is per-controller stepping over ~1,460 `ConstControl`s), and the network
  shortest paths were recomputed every timestep over a topology that does not change (0.609 s).
  Both are fixed. A third followed from them: `_write_grid_parameters` built one
  `ConstControl` per element per variable, ~1,460 objects per timestep, **only for the horizon
  loop to read them straight back** — so it records the profile arrays directly instead.
  Measured per timestep on design 6:

  | part | before | after |
  |---|---|---|
  | §14a restrictions | 3.021 s | **0.528 s** |
  | `_write_grid_parameters` | 0.524 s | **0.234 s** |
  | grid stage | ~3.6 s | **~0.77 s** |

  **The golden master does not cover this** — the shipped example has `electricity.active: False`,
  so it runs no grid at all. The evidence is a direct comparison against `run_timeseries` on a
  network with random per-step profiles, where all four logged series match exactly, plus an
  end-to-end run of the paper's design 6 where §14a's outputs are byte-for-byte identical over 8
  timesteps.

  **`numba` was measured for this and does not help, so it is not being added.** pandapower prints
  *"install numba to gain a massive speedup"* on every run, and with 24 power flows per timestep
  that looks like an obvious win. With numba installed and pandapower confirming it usable,
  `run_timeseries` moved 2.065 s → 2.075 s. At 246 buses the numerics were never the cost

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

- **A guard inside a fixture consumed only by `xfail(strict=True)` tests was not a guard.**
  `pytest.mark.xfail(strict=True)` converts a fixture *setup error* into a silent `xfailed`
  (verified on pytest 8.3.5). In `e2e/test_backend_equivalence.py` the `linopy_results` fixture is
  consumed only by the two permanently-xfailed comparisons, so anything it raised was absorbed —
  including `run_example`'s own `assert_backend_honoured`, the #206 read-back that exists to catch
  the two arms silently collapsing into one. The quiet form of that failure was still caught (two
  identical arms make a strict xfail XPASS); the loud form, a guard naming the exact cause, was
  swallowed. Found by deliberately breaking the new run cache's key and watching the module stay
  green while the guard fired. `test_the_linopy_arm_actually_ran_linopy` now reads the same receipt
  from a test carrying no xfail marker. This was the only such fixture in the suite.

- **Two identical runs of the same scenario produced different results, because `energy_types` was
  a `set` (#216, partial).** `RtcBase` and `FbcBase` collected each agent's energy types into a
  set, and all four backends iterate that set in `add_balance_constraints` to add one balance
  constraint and one `{energy_type}_{direction}_slack` variable pair each — so the set's iteration
  order *was* the optimisation model's row and column order. Python randomises string hashing per
  process, so the same agent's model reached HiGHS with its rows and columns permuted from run to
  run, and `update_socs` carried the resulting difference into the next timestep, where it
  compounded. (The interpretation — that the permuted model is degenerate enough for the solver to
  return a different equally-optimal vertex — is the explanation, not the observation; what was
  measured is that sorting the set removes the divergence and that the first agent-side seam to
  move is `setpoints`.)

  Both bases now derive their energy types through one shared `derive_energy_types`, which returns
  a **sorted list**. Everything downstream tests membership or builds a `str.startswith` tuple, so
  no call site depends on the order — only on its stability.

  **Every scenario was affected, not only multi-carrier ones.** #216 reasons that a one-element set
  has one ordering and so electricity-only scenarios are immune; that does not apply, because
  `mapping` is not the agent's own plants. Both call sites pass the module-level constant
  (`rtc.py:51`, `fbc.py:49`: `controller_class(**kwargs, mapping=c.COMP_MAP)`), so
  `energy_types` is `['electricity', 'heat']` for **every agent in every scenario** whatever plants
  it owns — two balance rows and, with slack on by default, four slack columns, in one of two
  orders.

  **The golden master does not move** — verified on Linux/x86_64, `9 passed` against the committed
  reference, not argued. It is immune because it runs under `PYTHONHASHSEED=0`
  (`tests/scenario_run.py:334`, `.gitlab-ci.yml:50`), and under seed 0 `list({'electricity',
  'heat'})` already equals `sorted(...)`. Under seeds 2, 3 or 5 it is the reverse — so **the golden
  reference was silently coupled to the value of that CI variable**, and this change removes that
  coupling.

  **This does not make HAMLET reproducible, and #216 stays open.** On the paper's design 6
  (104 agents, POI + HiGHS, 150 steps, Mac mini, pairs run back to back), two runs of unmodified
  `develop` still disagree on **6 of 1246** compared files — two of 135 timesteps, in
  `bids_cleared`, `offers_cleared` and `market_transactions`, and they differ in **row count**
  (1314 vs 1312, 852 vs 857), i.e. a genuinely different number of cleared trades. The agent side
  is identical in every arm measured, **832/832**, on `develop` as well as on this branch and with
  the hash seed pinned or left random.

  So this change does not alter design 6's results at this commit: it removes a real ordering
  dependency and the golden reference's coupling to `PYTHONHASHSEED`, and what supports it is the
  regression test, not a results delta.

  *Measurement note, because it invalidates the numbers this entry previously carried and much of
  the work recorded in #216.* Comparing two results trees by reading the frames back and sorting
  them **does not work on this data**: several columns are `Categorical`, `sort` orders those by
  the column's *local* integer encoding, and that encoding depends on the order the values were
  first seen — so two frames holding identical rows sort into different orders and compare unequal.
  That method reports **375** differing files where an anti-join finds **zero** rows unique to
  either side and identical column totals; the true figure for the same pair is 6. #216's own
  figures (285, 375, 105) were produced the same way and should be treated as unverified. Cast
  categoricals to `Utf8` before sorting.

- **Creating a CTSP agent from a grid file crashed on ordinary demand values (#212).**
  `Ctsp._inflexible_load_grid` sized the load as `(df['demand'] * 1e6).astype('Int64')`, and
  `demand * 1e6` is not exactly representable in float64 for **349 of the 10 000** three-decimal MW
  values between 0 and 10 — `1.001` among them, which is a 1 MW load stated to the precision the
  shipped grid file uses. pandas refuses to cast a non-integral float to a nullable integer, so
  scenario creation aborted with `TypeError` rather than returning a wrong number.

  The same line exists three times. `industry` floors it, `sfh` rounds it, and `industry.py:392`
  keeps the old expression commented out beneath its replacement — so the change was deliberate and
  `ctsp` is the copy it was not applied to. **`ctsp` now rounds, matching `sfh` rather than
  `industry`, and that choice moves a watt**: `1.001 MW` is 1 001 000 W under `round` and
  1 000 999 W under `floor`. Rounding recovers the decimal the file states, where flooring is biased
  low on every value that is inexact and never high. `industry` is left as it is — changing it would
  move generated sizings for a second agent type, which belongs with the deduplication decision in
  #213 rather than in a crash fix. Values that were always exact are unchanged in all three classes,
  which is asserted rather than assumed.

  **It is one line per device group, not one line.** The same bare cast sizes the plant power in
  `_pv_grid`, `_wind_grid`, `_fixed_gen_grid` and `_battery_grid` — four more sites in each of
  `ctsp.py` and `industry.py`, all eight of which `sfh` already rounds. Fixing only the demand
  column would have left four identical crashes per class behind, which is this repository's
  recurring shape: the fix for a failure contains the same failure one level down. All nine sites
  are fixed and each is covered; reverting the eight fails 16 of the tests.

  A detail worth keeping, because the first attempt got it wrong and the new test caught it in one
  run: these methods write `self.df.index.map(...)`, and `round()` on a pandas `Index` raises
  `TypeError: type Index doesn't define __round__`. The Series has to be rounded *before* mapping,
  which is what `sfh` does.

  Latent until now: these methods are reached only from `new_scenario_from_grids`, and no config in
  the repository is built that way, so no generated scenario and neither golden reference moves
- **Two shipped examples stated their modelling backend twice and contradicted themselves (#214).**
  `examples/create_scenario_with_market` shipped an `agents.xlsx` saying `linopy` against an
  `agents.yaml` saying `poi`, and because it is built with `new_scenario_from_files` **the workbook
  wins** — a user who opened the YAML, read `framework: poi` and ran the notebook got linopy.
  `create_scenario_with_grid` had the same drift on both `framework` and `solver`, inert only
  because `new_scenario_from_grids` regenerates the workbook before reading it. No test ran either
  example's config pair.

  Both workbooks now say what their own YAML says — each taking its own YAML's solver, so
  `scenario_with_grid` lands on `poi`/`highs` and `scenario_with_market` on `poi`/`gurobi`. The
  workbook moved rather than the YAML because `poi` is the documented default *framework* and the
  YAML is the file a reader opens; neither example's solver choice is changed by this.
  `tests/integration/test_shipped_configs_agree_with_their_workbooks.py` keeps them that way, in
  the fast tier and without running anything: it compares the two files **per key and per sheet**,
  so one agreeing sheet cannot vouch for another, and it fails if a scenario folder appears in the
  tree without being listed — a guard that passes by finding nothing is how this class of defect
  survives
- **A test asking a scenario for a different solver backend could have it silently ignored
  (#206).** `tests/scenario_run.run_example`'s `framework=` / `solver=` switch edited
  `agents.yaml`. A scenario built with `new_scenario_from_files` gets its agents from
  **`agents.xlsx`**, which nothing regenerates and which the YAML has no part in. So the request
  was accepted, the switch's own `assert switched` was satisfied by the file it edited, and the
  workbook's own backend ran anyway. Three shipped scenarios were exposed — `grid_golden` and
  `scenario_with_topology` pin `poi`/`highs`, `scenario_with_market` pins `linopy`/`gurobi`, so
  asking that one for HiGHS would have run Gurobi. Nothing called it that way yet, so no test was
  vacuous; the trap was armed rather than sprung.

  Two changes, because they close different things. The switch now reaches **every config file
  that can carry the key**, workbook included, so no file can look authoritative while another one
  quietly disagrees with it. And the backend receipt — the probe that records what actually built
  and solved each model — is no longer opt-in: whenever a backend is asked for, `run_example`
  writes it and checks it. The first fixes the entry point that was missed; the second catches the
  next one, whatever file it reads. `tests/e2e/test_solver_backend_smoke.py` gains one test that
  asks `grid_golden` for a backend it does not ship and asserts on what solved, not on what was
  requested. `tests/integration/test_scenario_run_backend_switch.py` covers the same ground in the
  fast tier: it reads the workbook back the way the Creator reads it, and it covers the receipt
  check directly — that guard was reachable only through `run_example`, so a typo in the part
  meant to be durable would have failed open until someone ran a job costing minutes. No golden
  reference moves — a run that names no backend takes neither of the new paths, and the golden
  master names none
- **Grid registration could drop an agent from the network and report success (follow-up to
  #205).** Two agents the matcher cannot tell apart — same bus, same agent type, same profile file
  — both matched the *same* inflexible load, and the second overwrote the first's `id_agent`.
  Elements with no agent are then dropped downstream, so the power flow was solved for a feeder
  missing one of its participants and reported a loading that was too low, with nothing raised.
  The defect predates the description reader restored for #205; that reader made the path
  reachable again. Candidates already claimed by an earlier agent are now excluded, so each agent
  keeps its own element, and an agent with genuinely nothing left to match is named in an error
  rather than lost.

  Three narrower robustness gaps went with it, each reachable from an ordinary network: a grid
  file with **no sgen rows at all** (a feeder with no PV or battery) was rejected as carrying no
  `plant_type` information; `_create_grid_from_topology` raised `KeyError: 'id_agent'` when no
  element of a kind was created; and `__assign_plants_for_agent` raised a bare `KeyError: 'owner'`
  on any file that never declares ownership. And the unassigned-agent check added for #205 was
  stricter than the code it guards — it demanded a bus for agents that place no electrical element
  at all, such as a heat-only agent or the parent of a set of sub-agents

- **§14a direct power control crashed the first time a heat pump took part in a reduction.**
  `EnWG14a` reads each heat pump's minimum controllable power from a `hp_min_control` column on
  the grid's load table, and nothing in HAMLET ever wrote it, so the run died with
  `KeyError: 'hp_min_control'`. Nothing caught it because nothing had ever reached that code: no
  shipped example enables `direct_power_control`, and the study the implementation was written
  for ran with it switched off. Both control methods need the column, so direct power control had
  never executed successfully at all.

  The minimum is now computed at grid registration from the heat pump's rated power, per BNetzA
  BK6-22-300: `0.4 × P_rated` where the grid connection power exceeds 11 kW, and the configured
  `direct_power_control.threshold` below it — the flat guarantee that EVs and batteries get. The
  same floor enters the EMS variant with the simultaneity factor fixed at 1. It is computed at
  registration rather than in the control because the rated power lives in the agent's plant
  configuration and not in the grid file, so both grid-creation methods need the same answer.

  **Known deviation, unchanged and worth an issue of its own:** `enwg_14a` decides whether the
  11 kW rule applies by comparing the heat pump's *instantaneous* power against 11 kW, where the
  regulation means grid connection power. It cannot over-curtail — the minimum computed here caps
  the reduction — but it is not what the regulation says

- **Grid registration could drop an agent from the network and report success (follow-up to
  #205).** Two agents the matcher cannot tell apart — same bus, same agent type, same profile file
  — both matched the *same* inflexible load, and the second overwrote the first's `id_agent`.
  Elements with no agent are then dropped downstream, so the power flow was solved for a feeder
  missing one of its participants and reported a loading that was too low, with nothing raised.
  The defect predates the description reader restored for #205; that reader made the path
  reachable again. Candidates already claimed by an earlier agent are now excluded, so each agent
  keeps its own element, and an agent with genuinely nothing left to match is named in an error
  rather than lost.

  Three narrower robustness gaps went with it, each reachable from an ordinary network: a grid
  file with **no sgen rows at all** (a feeder with no PV or battery) was rejected as carrying no
  `plant_type` information; `_create_grid_from_topology` raised `KeyError: 'id_agent'` when no
  element of a kind was created; and `__assign_plants_for_agent` raised a bare `KeyError: 'owner'`
  on any file that never declares ownership. And the unassigned-agent check added for #205 was
  stricter than the code it guards — it demanded a bus for agents that place no electrical element
  at all, such as a heat-only agent or the parent of a set of sub-agents

- **Both grid-enabled examples run again (#205, #201).** `examples/create_scenario_with_grid` and
  `examples/create_scenario_with_topology` are the only shipped scenarios that calculate a grid at
  all — `create_simple_scenario` sets `electricity.active: False` — and neither could be executed.

  **A grid file may carry HAMLET's per-element metadata in either of two places, and only one was
  being read.** Some files declare `load_type`, `plant_type`, `owner`, `agent_type` and `file` as
  real columns; others pack them into `description` as `key:value,key:value`. The packed reader was
  removed outright earlier in this same release (see the entry below) because a network imported
  from an operator puts prose in `description` — `'2022: 17209 kWh'`, or nothing at all — and parsing
  it either raised or, worse, invented columns that were then written back into the network. That
  fixed real networks and broke
  `create_scenario_with_grid`, whose `electricity.xlsx` is written in the packed convention, with
  `KeyError: 'load_type'`. Both conventions are now supported: real columns win, `description` is
  the fallback, and a file in neither convention is rejected with a message naming both.

  **Two mismatches now say what is wrong instead of crashing on a lookup.** An agent with no bus in
  the topology file raised `KeyError: '<random agent id>'`, and an agent whose plants match no
  inflexible load in the grid file returned `None` into a tuple unpacking
  (`TypeError: cannot unpack non-iterable NoneType object`). Both now raise a `ValueError` naming
  the agent, the file to fix and, for the topology case, the fact that creating a scenario with
  `new_scenario_from_configs` redraws the agent ids that the topology file refers to by name.
  Skipping the unmatched element was rejected as the alternative: it would leave the agent out of
  the network and solve the power flow for a feeder missing one of its participants.

  **Both examples now specify `framework: poi` and `solver: highs`**, like `create_simple_scenario`
  since !201, so neither needs a Gurobi licence. `tests/e2e/test_grid_examples.py` runs both end to
  end and asserts that the power flow actually ran and that every load and sgen belongs to an
  agent; `tests/integration/executor/test_grid_registration.py` covers both grid-file conventions
  and both mismatches in about a second. Before this, `GridDB.register_grid` had no test at all
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
- **Fixed the Executor being unable to read any scenario carrying a real electricity network.**
  `GridDB.__get_grid_element_dataframe` unpacked pandapower's `description` column as
  `key:value,key:value` HAMLET metadata. That column is not HAMLET's to read: nothing in HAMLET
  writes it — `_create_grid_from_topology` passes `plant_id`, `agent_id`, `agent_type`, `zone` and
  `load_type`/`plant_type` as real pandapower columns — and in a network imported from an operator
  it holds free-form text. On the paper's design 6 grid, 96 of 469 loads and 134 of 263 sgens have
  no description at all (`AttributeError: 'NoneType' object has no attribute 'split'`), and most of
  the rest are prose containing colons — `'Anlagenart: Photovoltaik \n Energieart: Sonne \n Baujahr:
  2022'`, `'2022: 17209 kWh'` — giving `ValueError: too many values to unpack (expected 2)`.

  **A parse that had succeeded would have been worse than the crash**, because the invented columns
  are joined on and written straight back into `self.grid.load`. So the call is removed rather than
  guarded — which is what `paper/elsevier-2026-complexity` does at this exact line, commented out,
  so the published runs never parsed descriptions either. Covered by
  `tests/unit/executor/utilities/database/test_grid_element_dataframe.py`, which builds a real
  pandapower network carrying each of the description shapes above

  > **Corrected while fixing #205, in this same release.** Two statements above are too strong.
  > *Nothing in HAMLET writes `description`* is true of HAMLET's code but not of its data:
  > `examples/create_scenario_with_grid`'s `electricity.xlsx` is authored entirely in the packed
  > convention. And *nothing downstream consumes a description-derived column* — which this entry
  > originally asserted, and which is what made removing the call look free — is false for that
  > file, where `load_type`, `owner`, `agent_type` and `file` are all description-derived and all
  > consumed by `_create_grid_from_file`. Removing the call therefore made that example
  > unreadable. The reader is now a typed fallback rather than absent or unconditional; see the
  > #205 entry above
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

