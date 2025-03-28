
# Changelog

All notable changes to this project will be documented in this file. 
See below for the format and guidelines for updating the changelog.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

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

