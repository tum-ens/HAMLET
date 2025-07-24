Contributing to HAMLET
======================

We welcome contributions to HAMLET! Whether you are fixing a bug, improving documentation, or adding a new feature, your help is highly appreciated. This guide provides an overview of how you can contribute.

How to Contribute
-----------------
There are multiple ways to contribute to HAMLET:

1. **Report Issues**:
   - Found a bug? Have a feature request? Open an issue in our GitHub repository.
   - Clearly describe the problem and, if possible, provide a minimal example to reproduce it.

2. **Improve Documentation**:
   - If you find missing information or outdated documentation, submit a documentation update.
   - All documentation is written in reStructuredText (`.rst`) and located in the `docs/source/` directory.

3. **Submit Code Changes**:
   - Fork the repository and create a new branch for your changes.
   - Ensure that your code follows the project’s style guide.
   - Write tests for new functionality where applicable.
   - Open a pull request (PR) and describe your changes clearly.

4. **Enhance Testing & Validation**:
   - Help improve test coverage by adding unit tests.
   - Review and refine validation methods for simulation results.

Development Workflow
--------------------
To set up a development environment and contribute effectively, follow these steps:

1. **Clone the Repository**:

.. code-block:: bash

  git clone https://github.com/tum-ens/hamlet.git
  cd hamlet

2. **Set Up the Anaconda Environment**:
   HAMLET uses an Anaconda environment defined in `env.yml`. To set it up:

.. code-block:: bash

  conda env create -f env.yml
  conda activate hamlet

Submitting a Pull Request (PR)
-------------------------------
Once your changes are ready:

1. Push your branch:

.. code-block:: bash

  git push origin my-feature-branch

2. Open a Pull Request (PR) on GitHub.
3. Fill in the PR template, describing:
   - What problem this PR addresses.
   - What changes have been made.
   - How it was tested.

4. Request reviews from maintainers.

Code Style and Guidelines
-------------------------
- **Follow PEP 8**: Use `black` for formatting.
- **Use Type Hints**: Type annotations improve readability and maintainability.
- **Keep Functions Modular**: Avoid long, complex functions.
- **Write Docstrings**: Document all public functions and classes using NumPy-style docstrings.

Getting Help
------------
If you have questions about contributing, reach out through:

- **GitHub Issues**: Post questions or discussions.
- **GitHub Discussions**: Join discussions on development topics.

We appreciate your contributions to HAMLET!

Further Information
--------------------
For more details on contributing to HAMLET, please refer to the `contributing guide`_ and the `ci/cd guide`_ in the repository.

.. _contributing guide: https://github.com/tum-ens/hamlet/blob/master/CONTRIBUTING.md
.. _ci/cd guide: https://github.com/tum-ens/hamlet/blob/master/CI_CD_Guide.md