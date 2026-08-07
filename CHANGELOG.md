
# Changelog

All notable changes to this project will be documented in this file. 
See below for the format and guidelines for updating the changelog.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]
### Fixed
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
### Changed
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

