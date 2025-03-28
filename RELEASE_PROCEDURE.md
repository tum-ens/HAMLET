# Release Procedure

This release procedure outlines the steps for managing releases in the GitLab environment.<br>
These symbols help with orientation:
- 🐙 GitLab
- 💠 git (Bash)
- 📝 File
- 💻 Command Line (CMD)


## Version Numbers

This software follows the [Semantic Versioning (SemVer)](https://semver.org/).<br>
It always has the format `MAJOR.MINOR.PATCH`, e.g. `1.5.0`.

The data follows the [Calendar Versioning (CalVer)](https://calver.org/).<br>
It always has the format `YYYY-MM-DD`, e.g. `1992-11-07`.


## GitLab Release

### 1. Update the `CHANGELOG.md`
- 📝 **File**: Open the CHANGELOG.md file and add a new entry under the `[Unreleased]` section.
- 💠 **Commit**: Commit your changes to the changelog, noting all new features, changes, and fixes.
- 📝 **Version Entry**: Format the new version entry as follows:
    ```
    ## [0.1.0] - 2022-01-01
  
    ### Added
    - New feature
    - Another new feature
  
    ### Changed
    - Change to existing feature
  
    ### Fixed
    - Bug fix
    ```
  
### 2. Create a `Draft GitLab Release` Issue
- 🐙 **Template**: Use the `📝Release_Checklist` template for the issue.
- 🐙 **Issue**: Create a new issue in the repository with the title `Release - Minor Version - 0.1.0`.
- 🐙 **Description**: Fill in the details of the release, including the name, Git tag, release manager, and date.
- 🐙 **Workflow Checklist**: Check off the steps in the workflow checklist for the release.
  
### 3. Update Version in Code
- 📝 **File**: Locate the version variable in the code (in the template it can be found in [VERSION](VERSION)).
- 💻 **Update**: Change the version number to the new release version following SemVer.
- 💠 **Commit**: Commit this version change with a message like:
    ```
    git commit -m "Bump version to 1.5.0"
    ```

### 4. Create a Release Branch
- 💠 **Branching**: Create a release branch from develop:
    ```bash
    git checkout develop
    git pull
    git checkout -b release-1.5.0
    ```
- 💠 **Push**: Push the release branch to GitLab:
    ```bash
    git push --set-upstream origin release-1.5.0
    ```
  
### 5. Finalize and Merge
- 🐙 **Merge Request**: In GitLab, open a merge request (MR) from `release-1.5.0` into `main`.
- 🐙 **Review**: Assign reviewers to the MR and ensure all tests pass.
- 🐙 **Merge**: Once approved, merge the MR into main and delete the release branch.

### 6. Tag the Release
- 💠 **Checkout** main: Ensure you’re on the main branch.
    ```bash
    git checkout main
    git pull
    ```
- 💠 **Tag**: Tag the new release in GitLab:
    ```bash
    git tag -a v1.5.0 -m "Release 1.5.0"
    git push origin v1.5.0
    ```
  
### 7. Create a GitLab Release (Optional)
- 🐙 **GitLab Release Page**: Go to the GitLab project’s Releases section and create a new release linked to the v1.5.0 tag.
- 📝 **Release Notes**: Add release notes using information from the changelog.

### 8. Update the Documentation
- 📝 **Documentation**: Update the documentation to reflect the new release version.
- 💻 **Build**: Build the documentation to ensure it’s up to date.
- 💻 **Deploy**: Deploy the documentation to the appropriate location.
- 💻 **Update**: Update any version references in the documentation.
- 💻 **Commit**: Commit the documentation changes.
- 💠 **Push**: Push the documentation changes to the repository.
- 🐙 **Merge**: Merge the documentation changes into the main branch.
- 🐙 **Delete Branch**: Delete the release branch after merging.

### 9. Merge Back into `develop`
- 💠 **Branch**: Create an MR from `main` into `develop` to merge the release changes back into the development branch.
```bash
git checkout develop
git pull
git merge main
git push
```

## PyPi Release

### 0. 💻 Check release on Test-PyPI
- Check if the release is correctly displayed on [Test-PyPI](https://test.pypi.org/)
- **Automatic Deployment**: With each push to the `release-*` or `test-release` branch the package is released on [Test-PyPI](https://test.pypi.org/) by GitLab CI/CD (assuming a corresponding job like test-pypi-publish.yml is configured).
  - Note: Pre-releases on Test-PyPI are only shown under `Release history` in the navigation bar.
  - Note: Each unique branch state can only be released to a single version on TestPyPI. For testing multiple states on TestPyPI, increment the build version (e.g., using `bump2version build`) and push the changes.
- When testing is complete, finalize the release version with `bump2version release`
  - Note: The release on Test-PyPI might fail, but it will be the correct release version for the PyPI server.
- Push commits to the `release-*` branch

### 1. 💻 Create and publish package on PyPI
1. **Navigate to Project Directory**:
    ```bash
    cd path/to/gitlab/group/repo
    ```
2. **Build the Package**:
  - Use `setup.py` or `pyproject.toml` to create a package distribution:
    ```bash
    python setup.py sdist
    ```
  - Confirm that the `.tar.gz` file is generated in the `dist` folder.
3. **Activate Release Environment**:
  - Activate the virtual environment used for PyPI releases:
    ```bash
    source path/to/release_env/bin/activate
    ```
4. **Upload to PyPI**:
  - Use `twine` to upload the package to PyPI:
    ```bash
    twine upload dist/package_name-X.X.X.tar.gz
    ```
  - Enter your PyPI username and password when prompted.
5. **Verify Release**:
    - Check the [PyPI](https://pypi.org/) website to confirm that the release has been successfully uploaded.
6. **Celebrate**:
    - Take a moment to confirm everything looks good and enjoy a brief celebratory break before the grind continues.

### Important Notes
- **Versioning**: Always increment the version correctly using `bump2version` before creating the final release.
- **Publishing Reminder**: Ensure your PyPI credentials are correctly set up in GitLab CI/CD or local `.pypirc` configuration for seamless uploads.
- **Final Check**: If issues arise post-release, refer to the [GitLab CI/CD guide](https://docs.gitlab.com/ee/development/cicd/) and [PyPI documentation](https://packaging.python.org/en/latest/) for troubleshooting.

