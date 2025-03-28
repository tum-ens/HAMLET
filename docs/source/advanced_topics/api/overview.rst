API Overview
============

HAMLET provides a well-structured API that allows users to interact with its core components programmatically. This section introduces the API, how it is structured, and how users can navigate the documentation.

API Structure
-------------

The HAMLET API is divided into the following categories:

- **Creator API**: Classes and functions related to setting up agents, markets, and grids.
- **Executor API**: Components responsible for executing simulations and managing data flows.
- **Analyzer API**: Tools for analyzing, visualizing, and extracting insights from simulation results.
- **Utilities**: Helper functions for forecasting, database access, trading strategies, and grid management.

How to Use the API
------------------

Users can interact with HAMLET by importing the relevant modules in Python scripts. Here are some common use cases:

- **Creating a new simulation scenario**:
  .. code-block:: python

      from hamlet.creator import Creator

      creator = Creator(path="./configs", name="example_scenario")
      creator.new_scenario_from_configs()

- **Running a simulation**:
  .. code-block:: python

      from hamlet.executor import Executor

      executor = Executor(path_scenario="./scenarios/example_scenario")
      executor.run()

- **Analyzing results**:
  .. code-block:: python

      from hamlet.analyzer import ScenarioAnalyzer

      analyzer = ScenarioAnalyzer(path_results="./results/example_scenario")
      analyzer.plot_virtual_feeder_flow()

Generating the API Documentation
--------------------------------

HAMLET's API documentation is automatically generated using **Sphinx** and the `autodoc` extension.

To update the API documentation, navigate to the `docs` folder and run:

.. code-block:: bash

    sphinx-apidoc -o source/api ../hamlet --force
    make html

The updated documentation can be found in `docs/build/html/index.html`.

Navigating the API Reference
----------------------------

The API reference provides detailed descriptions of each module, class, and function within HAMLET. Use the navigation menu to explore:

.. toctree::
   :maxdepth: 1

   creator
   executor
   analyzer
   utilities