Usage Guide
===========

HAMLET allows you to set up, execute, and analyze energy system simulations with ease. This guide provides an overview of common workflows and commands to help you get started.

Running a Basic Scenario
------------------------

Follow these steps to execute a pre-configured scenario:

1. **Navigate to the Examples Directory**:

   .. code-block:: bash

      cd examples

2. **Pick the `create_simple_scenario` folder**

   .. code-block:: bash

      cd create_simple_scenario

3. **Run the jupyter notebook**:

   .. code-block:: bash

      run.ipynb

Common Commands and Functions
-----------------------------

Creator Module
~~~~~~~~~~~~~~
- **Purpose**: Define agents, markets, and grids.
- **Example**:
.. code-block:: python

  from hamlet import Creator

  creator = Creator(path=\"./configs\", name=\"example_scenario\")
  creator.new_scenario_from_configs()

Executor Module
~~~~~~~~~~~~~~~
- **Purpose**: Execute the simulation scenarios.
- **Example**:
.. code-block:: python

  from hamlet import Executor

  executor = Executor(path_scenario=\"./scenarios/example_scenario\")
  executor.run()

Analyzer Module
~~~~~~~~~~~~~~~
- **Purpose**: Analyze and visualize the results.
- **Example**:
.. code-block:: python

  from hamlet import Analyzer

  analyzer = Analyzer(path_results=\"./results/example_scenario\")
  analyzer.plot_virtual_feeder_flow()

Advanced Options
----------------

1. **Custom Configurations**:
   Copy and configure the YAML files in the `configs` directory to customize your scenario.

2. **Parallel and Sequential Execution**:
   Parallel execution is turned on by default to decrease simulation time. However, in some circumstances (e.g. debugging), it might be better to run the simulation sequentially, which can be done by setting `num_workers`.
.. code-block:: python

  executor = Executor(path_scenario=\"./scenarios/example_scenario\", num_workers=1)

3. **Interactive Debugging**:
   Use IDEs like PyCharm or VS Code to debug specific steps in the simulation pipeline.

Next Steps
----------

Once you’re comfortable with the basics, explore more advanced use cases:

- **Custom Agents**: Define unique agent behaviors.
- **Complex Markets**: Simulate diverse market configurations.
- **Grid Variations**: Test different grid topologies and constraints.

Refer to the :doc:`examples` section for detailed workflows.
