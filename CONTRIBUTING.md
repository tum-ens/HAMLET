<!--SPDX-License-Identifier: MIT-->
<!--Version: v1.0.0-->

# Collaborative Development

## Prerequisites
- [GitLab](https://gitlab.com/) as a public repository. Please create an account.
- [Git](https://git-scm.com/) for version control. [Git How To](https://githowto.com/) and [Git Cheat Sheet](https://training.github.com/downloads/github-git-cheat-sheet.pdf) provide an introduction into working with Git.
- Development Environment Setup:
  - Clone the repository and set up a virtual environment as described in the `README.md`.
  - Install all necessary dependencies using `pip install -r requirements.txt`.

### Optional: Setting Up SSH for GitLab
To simplify authentication, consider setting up SSH keys for GitLab. Follow [GitLab’s SSH key setup guide](https://docs.gitlab.com/ee/user/ssh.html) for instructions.

## Types of interaction
This repository is following the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). <br>
Please be self-reflective and always maintain a good culture of discussion and active participation.

### A. Use
Since the open license allows free use, no notification is required. 
However, for the authors it is valuable information who uses the software for what purpose. 
Indicators are `Watch`, `Fork` and `Starred` of the repository. 
If you are a user, please add your name and details in [USERS.cff](USERS.cff).

### B. Comment
You can give ideas, hints or report bugs in issues, in MR, at meetings or other channels. 
This is no development but can be considered a notable contribution. 
If you wish, add your name and details to [CITATION.cff](CITATION.cff).

### C. Contribute and Review
You add code and become an author of the repository. 
You must follow the workflow!
Add your name and details to [CITATION.cff](CITATION.cff).

### D. Maintain and Release
You contribute and take care of the repository. 
You review and answer questions. 
You coordinate and carry out the release.
Add your name and details to [CITATION.cff](CITATION.cff).

## Workflow
The workflow for contributing to this project has been inspired by the workflow described by [Vincent Driessen](https://nvie.com/posts/a-successful-git-branching-model/).

##### Branches and their purpose
* main - holds the current stable version
* develop - holds all current developments
* see other types below

### 1. Describe the issue on GitLab
Create [an issue](https://docs.gitlab.com/ee/user/project/issues/#create-an-issue) in the GitLab repository. 
The `issue title` describes the problem you will address.  <br>
This is an important step as it forces one to think about the "issue".
Make a checklist for all needed steps if possible.
The issue templates guide you to ensure a consistent description.

### 2. Solve the issue locally
There are two ways to create the branch:
1. Create a new branch from the `develop` branch
2. Create a new branch directly in GitLab from the issue

#### Naming convention for branches
Naming convention for branches: `type`-`issue-nr`-`short-description`

##### `type`
* feature - used for new features
* hotfix - used for quick fixes, should be branched from the release branch
* release - used for preparing a new release, should be branched from the develop branch

Note: The majority of the development will be done in `feature` branches.

##### `issue-nr`
The `issueNumber` should be taken from Step 1. Do not use the "#". 

##### `short-description`
Describe shortly what the branch is about. Usually, the title of the issue.

##### Other hints
- Separate words with `-` (minus)
- Avoid using capital letters
- Do not put your name to the branch name, it's a collaborative project
- Branch names should be precise and informative

Examples of branch names: `feature-42-add-new-ontology-class`, `feature-911-branch-naming-convention`, `hotfix-404-update-api`, `release-v0.10.0`

#### 2.1. Create a new branch

##### Option 1: Get the latest version of the `develop` branch
1. Load the `develop branch`:
```bash
git checkout develop
```

2. Update with the latest version:
```bash
git pull
```

3. Create a new feature branch:
```bash
git checkout -b feature-1314-my-feature
```

##### Option 2: Create a new branch directly in GitLab from the issue
1. Click on the button `Create merge request` in the issue. Here you can either create a new branch or also create a merge request alongside with it (recommended).
2. Choose the appropriate source branch (most of the time develop).
3. Name the branch according to the naming convention.

#### 2.2. Start editing the files
- Divide your feature into small logical units
- Start to write the documentation or a docstring
- Don't rush, have the commit messages in mind
- Add your changes to the [CHANGELOG.md](CHANGELOG.md)

On first commit to the repo:
- Add your name and details to [CITATION.cff](CITATION.cff)

Check branch status:
```bash
git status
```

#### 2.3. Commit your changes 
If the file does not exist on the remote server yet, use:
```bash
git add filename.md
```

Then commit regularly with:
```bash
git commit filename.md
```

Write a good `commit message`:
- "If applied, this commit will ..."
- Follow [existing conventions for commit messages](https://chris.beams.io/posts/git-commit)
- Keep the subject line [shorter than 50 characters](https://chris.beams.io/posts/git-commit/#limit-50)
- Do not commit more than a few changes at the time: [atomic commits](https://en.wikipedia.org/wiki/Atomic_commit)
- Use [imperative mood](https://chris.beams.io/posts/git-commit/#imperative)
- Do not end the commit message with a [period](https://chris.beams.io/posts/git-commit/#end) ~~.~~ 
- Always end the commit message with the `issueNumber` including the "#"

Examples of commit message: `Added function with some method #42` or `Update documentation for commit messages #1`

#### 2.4 Fix your latest commit message
Do you want to improve your latest commit message? <br>
Is your latest commit not pushed yet? <br>
Edit the commit message of your latest commit:
```bash
git commit --amend
```

### 3. Run the tests
To maintain code quality and ensure reliability, please follow the guidelines below when contributing code to this project.

1. **Run Tests Locally**:
    - **Test Suite**: Before submitting code, run the full test suite to ensure your changes don’t introduce any issues.
    - **Test Command**: Use the following command to run the tests (adjust this based on your testing framework, e.g., `pytest`, `unittest`):
    ```bash
    pytest tests/
    ```
    - **Writing Tests**: New features should include relevant unit tests, ideally placed in the tests/ folder. Ensure each test covers a specific aspect of functionality and edge cases where applicable.

2. **Code Quality and Linting**:
To ensure consistency in code style and to prevent potential issues, it’s recommended to use linters and formatters. 
    - **Linting**: Use `flake8` to check for PEP 8 compliance and catch other common issues:
   ```bash
    pip install flake8
    flake8 path/to/your/code  # Run flake8 to check for style issues
    ```
    - **Code Formatting**: Use `black` oto automatically format your code according to PEP 8 standards.
    ```bash
    pip install black
    black path/to/your/code  # Run black to automatically format code
    ```
   - **Type Checking**: If applicable, use `mypy` to perform static type checking on your code:
    ```bash
    pip install mypy
    mypy path/to/your/code  # Run mypy to check for type errors
    ```
3. **Pre-Commit Hooks**:
To automate code quality checks before each commit, we use pre-commit hooks. These hooks ensure code adheres to the project’s standards, catching issues early in the development process.
    - **Pre-Commit**: To automate code quality checks, consider using pre-commit hooks. These hooks run checks before each commit to ensure code quality and style consistency.
    - **Installation**: Install pre-commit using pip:
    ```bash
    pip install pre-commit
    ```
   - **Setup**: If the repository includes a .pre-commit-config.yaml file, run the following command to install the hooks defined in the config file:
    ```bash
    pre-commit install
    ```
    - **Usage**: After installation, pre-commit will run automatically before each commit. If any issues are found, you will need to address them before the commit is accepted.
    - **Customization**: To customize the pre-commit hooks, modify the .pre-commit-config.yaml file in the repository. This file defines the hooks to run and their configurations.
    - **Manual Run**: To run the pre-commit checks manually, use the following command:
    ```bash
    pre-commit run --all-files
    ```
    **Note**: Pre-commit hooks may run linters, formatters, and other quality checks automatically on changed files when you commit, ensuring compliance without requiring manual intervention.

4. Ensuring Coverage:
    - **Code Coverage**: Aim to maintain high code coverage for new code by adding unit tests for all significant paths and edge cases.
    - **Coverage Tool**: Use `coverage.py` with `pytest` to generate a coverage report:
    ```bash
    pytest --cov=path/to/your/code tests/
    ```
   After running the tests, you can view the coverage report in the terminal or generate an HTML report for more detailed information.
    ```bash    
        coverage report
        coverage html
    ```
Following these steps will help ensure the project remains robust, maintainable, and consistent. Before pushing your changes, confirm that:
- All tests pass locally.
- Linting and formatting checks are successful.
- Pre-commit hooks have been applied without errors.

### 4. Push your commits
Push your `local` branch on the remote server `origin`. <br>
If your branch does not exist on the remote server yet, use:
```bash
git push --set-upstream origin feature-1314-my-feature
```

Then push regularly with:
```bash
git push
```

### 5. Submit a merge request (MR)
Follow the GitLab guide [Creating merge requests](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html). <br>
The MR should be directed: `base: develop` <- `compare: feature-1-collaboration`. <br>
Add the line `Close #<issue-number>` in the description of your MR.
When it is merged, it [automatically closes](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically) the issue. <br>
Assign a reviewer and get in contact.

#### 5.1. Let someone else review your MR
Follow the GitLab guide [Merge request views](https://docs.gitlab.com/ee/user/project/merge_requests/reviews/). <br>
Assign one reviewer or a user group and get into contact.

If you are the reviewer:
- Check the changes in all corresponding files.
- Checkout the branch and run code.
- Comment if you would like to change something (Use `Request changes`)
- If all tests pass and all changes are good, `Approve` the MR. 
- Leave a comment and some nice words!

#### 5.2. Merge the MR and delete the feature branch
Follow the GitLab guide [Merge methods](https://docs.gitlab.com/ee/user/project/merge_requests/methods/index.html).

### 6. Close the issue
Document the result in a few sentences and close the issue. <br>
Check that all steps have been documented:

- Issue title describes the problem you solved?
- All commit messages are linked in the issue?
- The branch was deleted?
- Entry in CHANGELOG.md?
- MR is closed?
- Issue is closed?

## Coding Standards

To maintain consistency and readability, please follow the coding standards outlined below when contributing to this project.

### Code Style

- **PEP 8**: All Python code should adhere to [PEP 8](https://www.python.org/dev/peps/pep-0008/) standards, which outline best practices for Python formatting, naming conventions, and structure. This includes:
  - Indentation (4 spaces per level).
  - Line length (maximum 79 characters, or 72 characters for docstrings).
  - Naming conventions for variables, functions, and classes.
  
- **Linting Tools**: To help ensure compliance with PEP 8, use a linter such as `flake8` or `black`:
  ```bash
  pip install flake8
  flake8 path/to/your/code  # Run flake8 to check for style issues
  
  # OR use black for auto-formatting
  pip install black
  black path/to/your/code  # Run black to automatically format code

### Documentation Style
To ensure consistent and clear documentation, please follow the Google-style docstring format. 
This style is straightforward and well-suited for research projects, as it provides structured sections for describing parameters, return values, examples, and more.

#### General Rules
1. **Module-Level Docstring**: Each file should start with a top-level docstring explaining its purpose and any important details.
2. **Class and Function Docstrings**: Each class and function should have a docstring summarizing its behavior. Class docstrings should also document class attributes if applicable.
3. **Attributes**: Document module-level and class attributes with type annotations and descriptions. This can be done inline or in the `Attributes` section.

#### Docstring Format
- **Summary**: Begin with a concise summary of the object’s purpose.
- **Args**: List and describe each parameter, including type annotations and default values.
- **Returns**: Describe the return value(s) of the function, including type annotations. Use `Yields` instead of `Returns` for generator functions. Use `None` if the function returns nothing.
- **Raises**: List any exceptions that the function may raise.
- **Examples**: Provide examples of how to use the function or class.
- **Notes**: Include any additional information that may be helpful.
- **References**: Link to any external resources or references.
- **TODOs**: List any future work or improvements that could be made.

#### Examples of Each Type of Docstring
1. **Module-Level Docstring**
   ```python
   """Brief module description.
    
    This module demonstrates the Google-style docstring format. It provides
    several examples of documenting different types of objects and methods.
    
    Attributes:
        module_level_variable1 (int): Module level variables can be documented here.
        module_level_variable2 (str): An example of an inline attribute docstring.
    """
   ```

2. **Function Docstring**
    ```python
    def example_function(param1: int, param2: str = 'default') -> bool:
         """Summary of the function.
         
         Args:
              param1 (int): The first parameter.
              param2 (str): The second parameter. Defaults to 'default'.
         
         Returns:
              bool: A boolean value indicating success.
         
         Raises:
              ValueError: If the parameter is invalid.
         """
    ```

3. **Generator Function Docstring**
    ```python
    def example_generator(param: int) -> Iterator[int]:
         """Summary of the generator function.
         
         Args:
              param (int): The parameter for the generator.
         
         Yields:
              int: The next value in the sequence.
         """
    ```
   
4. **Class Docstring with Attributes**
    ```python
    class ExampleClass:
    """A summary line for the class.
    
        Attributes:
            attr1 (str): Description of `attr1`.
            attr2 (int, optional): Description of `attr2`, which is optional.
        """
    
        def __init__(self, attr1: str, attr2: int = 0):
            """Initializes ExampleClass with specified attributes.
    
            Args:
                attr1 (str): Description of `attr1`.
                attr2 (int, optional): Description of `attr2`. Defaults to 0.
            """
            self.attr1 = attr1
            self.attr2 = attr2
    ```

5. **Exception Docstring**
    ```python
    class CustomError(Exception):
        """Exception raised for specific errors in ExampleClass.
    
        Args:
            msg (str): Explanation of the error.
            code (int, optional): Error code, if applicable.
    
        Attributes:
            msg (str): Explanation of the error.
            code (int): Error code.
        """
        def __init__(self, msg: str, code: int = None):
            self.msg = msg
            self.code = code
    ```

6. **Special Methods and Private Methods**
- Document special methods like `__init__`, `__str__`, and `__repr__` in the class docstring or their individual docstrings as you would any other method.
- Private methods (starting with `_`) generally do not require detailed docstrings unless they are complex.

#### Notes
- **Examples**: When adding examples use doctest format where possible. This allows you to test the examples as part of your test suite.
- **Formatting Tips**:
    - Use triple quotes for multi-line docstrings.
    - Use indentation consistently for multiline parameter descriptions.
    - Avoid documenting `self` in `Args` for instance methods, as it is implied.
    - Use `None` for default values instead of `NoneType`.
    - Use `Optional` for optional parameters with default values.
    - For properties, document only in the getter method.

Following this format will ensure all contributors create clear, structured documentation that can be easily read and maintained by others.

## Updating Documentation and Changelog
Keeping documentation up-to-date is essential for ensuring other contributors and users understand new features, changes, and fixes. Please follow these guidelines when updating documentation and the changelog.

### 1. Updating Documentation
- **Main Documentation**: If your changes introduce new features or modify existing functionality, update the relevant sections in the documentation located in the docs/ folder.
  - **Writing Style**: Keep the writing clear, concise, and objective.
  - **Code Examples**: Include code examples where applicable to demonstrate usage.
  - **Building Documentation**: If the project uses MkDocs or Sphinx, preview changes by building the documentation locally:
    ```bash
    mkdocs serve  # For MkDocs
    ```
    ```bash
    make html  # For Sphinx
    ```
- **Docstrings**: Update or add docstrings to any new or modified functions, classes, or modules, following the Google-style guidelines in the Documentation Style section above.

### 2. Updating Changelog
The `CHANGELOG.md` file keeps a record of significant changes to the project and helps users and contributors track updates. Please follow these conventions for maintaining the changelog:

- **Entry Format**:
  - Use the following structure for each new entry:
    ```markdown
    ## [Version] - [YYYY-MM-DD]
    
    ### Added
    - Brief description of the new feature, referencing the issue or merge request number if applicable.
    
    ### Changed
    - Description of any modifications to existing features.
    
    ### Fixed
    - Description of bug fixes or corrections.
    
    ### Removed
    - Deprecated features or removed functionality.
    ```
- **Examples**:
    - **New Feature**:
        ```markdown
        ## [1.0.0] - 2022-01-01
        
        ### Added
        - Implemented new feature X (#42).
        ```
    - **Bug Fix**:
        ```markdown
        ## [1.0.1] - 2022-01-02
        
        ### Fixed
        - Resolved issue with feature X not working correctly (#43).
        ```
- **Semantic Versioning**: Follow [Semantic Versioning](https://semver.org/) guidelines when updating the version number. Increment the version based on the impact of the changes:
  - **Major**: Breaking changes or significant new features.
  - **Minor**: New features or enhancements that are backward-compatible.
  - **Patch**: Bug fixes or minor improvements that do not affect compatibility.

### 3. Reviewing Documentation and Changelog Before Submission
- **Double-Check**: Ensure all new functionality is covered in the documentation, and verify that changelog entries are clear and concise.
- **Consistency**: Use consistent terminology and formatting across all documentation and changelog entries.
- **Preview Changes**: When possible, preview the documentation and changelog to ensure they are visually correct and well-organized.

By following these guidelines, you’ll help maintain high-quality, organized documentation that benefits all users and contributors.