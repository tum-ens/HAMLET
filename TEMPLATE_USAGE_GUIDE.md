# Template Usage Guide

This guide provides detailed instructions for setting up, using, and maintaining the ENS template repository. It is intended for contributors who will actively develop and maintain projects based on this template.

---

## Contents
1. [Getting Started](#getting-started)
2. [Template Structure and Conventions](#template-structure-and-conventions)
3. [Detailed Workflows](#detailed-workflows)
4. [Versioning and Release Management](#versioning-and-release-management)
5. [Documentation Standards](#documentation-standards)
6. [CI/CD Integration](#cicd-integration)
7. [Troubleshooting and FAQs](#troubleshooting-and-faqs)

---

## 1. Getting Started

### Installation Instructions
1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
    ```
2. ** Set up a Virtual Environment**:
    - For Windows:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
    - For MacOS/Linux:
   ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4. **Optional Setup**:
- **SSH Configuration**: If you frequently push/pull code, setting up SSH for GitLab simplifies authentication.
- **IDE Plugins**: Recommended plugins include GitLens (for VS Code) and git integration plugins that enhance workflow.

## 2. Template Structure and Conventions

### Folder Structure and Contents
- **`/src`**: Main project code. Start here with all core functionality, organized by logical subfolders.
- **`/docs`**: Markdown or RST documentation files. Structure and organize docs by major sections or modules.
- **`/tests`**: Contains tests structured by functionality. Use unit tests for specific functions, and add integration tests for interconnected features.
- **`/scripts`**: Contains utility scripts for setting up the project, running tests, and other tasks.
- **`/examples`**: Contains example scripts, notebooks, or applications that demonstrate project usage.
- **`/data`** (optional): Contains sample datasets, configuration files, and other data used by the project.
- **`/notebooks`** (optional): Contains Jupyter notebooks for data exploration, analysis, and visualization.

### Naming and Code Style Conventions
- **Branch Naming**: Branches should follow the naming convention `type-issue#-short-description`, e.g., `feature-42-add-new-ontology-class`.
   - **Types:** `feature`, `bugfix`, `hotfix`, `release`, `docs`, `style`, `refactor`, `test`, `task`.
- **Commit Messages**: Use the Conventional Commits format. Examples:
   - `feat: add new energy demand forecasting feature`
   - `fix: resolve issue with data loading`
   - `docs: update README with new installation instructions`
   - `test: add unit tests for data processing functions`
   - `style: format code using black and flake8`
   - `refactor: simplify data processing logic`
   - `task: update dependencies in requirements.txt`
- **File Naming**: Use descriptive names for files and folders. Follow a consistent naming convention (e.g., snake_case or camelCase).
- **Code Style**: Follow PEP 8 guidelines. Use black for automatic formatting and flake8 for linting.

## 3. Detailed Workflows

### Branching Workflow
#### 1. Create a Branch:
```bash
git checkout develop
git pull
git checkout -b feature-issue#-short-description
```
#### 2. Work on the Branch:
- Regularly commit changes and use descriptive messages.
- Sync with the `develop` branch frequently to avoid conflicts.
#### 3. Submit a Merge Request:
- Open a merge request from your branch to `develop`.
- Assign reviewers, add relevant labels and link any related issues.
- Complete all CI/CD checks before requesting a final review.

### Testing Workflow
- **Run Tests Locally**: Run `pytest` locally to ensure your code works as expected before committing.
- **Add New Tests**: Each new feature should come with a corresponding test. Place unit tests in `tests/unit/` and integration tests in `tests/integration/`.
- **Code Coverage**: Aim to maintain at least 90% coverage. Use `pytest --cov=<code_path>` to generate a coverage report.

### Documentation Workflow
- **README.md**: Update any usage changes.
- **CHANGELOG.md**: Log new features, changes, and fixes with each release.
- **Docstrings**: For any new functions or classes, follow the Google-style docstring format.
- **API Documentation**: Run `mkdocs serve` to preview changes locally if using MkDocs.

## 4. Versioning and Release Management
### Versioning with Semantic Versioning (SemVer)
- **Major**: Significant changes, likely with breaking compatibility.
- **Minor**: New features that are backward-compatible.
- **Patch**: Bug fixes and minor improvements.

### Release Workflow
1. **Create a Release Branch**: Create a branch named `release-vX.X.X` from `develop`.
2. **Update Version**: Run `bump2version [major|minor|patch]` to update the version in relevant files.
3. **Update Changelog**: Add the version entry for the new release with relevant details.
4. **Tag Release**: Create a Git tag for the release version.
```bash
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```
5. **Publish Release**: Draft a GitHub release with the tag and release notes.
6. **Merge Changes**: Merge the release branch back into `develop` and `main`.
7. **Notify Stakeholders**: Inform users, contributors, and maintainers about the new release.
8. **Update Documentation**: Ensure all documentation is up-to-date with the latest changes.
9. **Enjoy the Release!**

## 5. Documentation Standards
### Docstring Format (Google Style)
See the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for detailed docstring conventions.
You can find examples for this repo in the [CONTRIBUTING.md](CONTRIBUTING.md) file.

### Building Documentation
#### 1. Build Locally:
- **MkDocs**: Run `mkdocs serve` to preview documentation locally.
- **Sphinx**: Use `make html` to build HTML documentation from reStructuredText files.
#### 2. Preview: Ensure no build errors and that all docstrings render correctly.

## 6. CI/CD Integration

### Basic Pipeline Overview
The template includes three main stages: **install**, **lint**, and **test**.

### Extending the Pipeline
- **New Stages**: For additional steps like deployment, add a new stage and job in the `.gitlab-ci.yml` file (see [CI_CD_Guide.md](CI_CD_Guide.md) for more information).
- **Environment Variables**: Store sensitive data like API keys or tokens as environment variables in GitLab CI/CD settings.

### Running the Pipeline Locally
Use a GitLab runner or Docker to test pipeline scripts locally if desired.



## 7. Troubleshooting and FAQs

### Dependency Issues
- **Pip Conflicts**: Use `pipdeptree` to analyze and resolve dependency issues.

### Git FAQs
- **Merge Conflicts**: Resolve conflicts by identifying lines with both versions in your editor and selecting the appropriate changes.
- **Reverting Commits**: If you need to undo a commit, use `git revert <commit_hash>` to undo changes from a specific commit.

### Testing Issues
- **Failed Tests**: Check the error log and confirm dependencies are correctly installed.
- **Flaky Tests**: If a test fails intermittently, investigate potential causes such as network dependencies or asynchronous code.

---