
# Changelog

All notable changes to this project will be documented in this file. 
See below for the format and guidelines for updating the changelog.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]
### Fixed
- Fixed the MPC reading the market energy prices the wrong way round, so agents saw a lower
  price for buying than for selling and could trade against the retailer at a profit
- Fixed grid fees and levies being charged on gross rather than net energy, which overcharged
  every agent that both bought and sold within a timestep
- Fixed the net energy behind grid fees and levies omitting the trades cleared in the timestep
  being settled
- Fixed the EV state of charge ignoring the battery capacity and being able to go negative
- Fixed the EV `min_soc` charging scheme, which let the car leave below its minimum state of
  charge
- Fixed EV time series being averaged when resampled, which lost most of a trip's driving
  energy and could truncate the availability flag to zero so the car never charged at all
- Fixed controllers configured as off still being run when the setting came from `agents.xlsx`
  as an empty cell rather than as `None`
- Fixed retailer prices being broadcast as a column instead of read as a scalar, which gave
  each transaction in a timestep a different price when the retailer table had several rows
### Added
- Added a test suite (`tests/`) with a layered layout: component physics, accounting
  invariants and targeted regression tests, runnable with `python -m pytest tests`

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

