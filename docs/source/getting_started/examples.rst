Examples for HAMLET
====================

Below are several examples on how to use HAMLET. Further examples can be found in the `examples` folder.

Basic Examples
--------------

**Example 1: Creating a Simple Scenario**

Purpose: Demonstrate how to set up a basic scenario with agents, grids, and markets.

Instructions:
1. Prepare YAML configuration files for agents, grids, and markets in the `configs` folder.
2. Use the Creator module to generate the scenario.

.. code-block:: python

  from hamlet import Creator

  sim = Creator(path='../scenarios/basic', name='basic_scenario')
  sim.new_scenario_from_configs()

3. Verify the created files in the `scenarios` folder.

---

**Example 2: Running a Simulation**

Purpose: Show how to execute a basic simulation for the created scenario.

Instructions:
1. Ensure the scenario folder contains the necessary files.
2. Run the Executor module to simulate.

.. code-block:: python

  from hamlet import Executor

  if __name__ == "__main__":  # required for multiprocessing
      executor = Executor(path_scenario="./scenarios/basic_scenario")
      executor.run()

3. Check results in the `results` folder.

---

**Example 3: Analyzing Results (not yet implemented)**

Purpose: Illustrate how to visualize and interpret simulation results.

Instructions:
1. Use the Analyzer module to load results.

.. code-block:: python

  from hamlet.analyzer import ScenarioAnalyzer

  analyzer = ScenarioAnalyzer(path_results="./results/basic_scenario")
  analyzer.plot_virtual_feeder_flow()

2. Generate additional plots or export data as needed.

Intermediate Examples
----------------------

**Example 4: Custom Agent Behaviors**

Purpose: Show how to define and integrate custom agent strategies.

Instructions:
1. Modify the agent configuration file to include custom behavior parameters.
2. Update the `Agent` class in the Executor module to include custom logic.
3. Run the scenario and analyze the impact of the custom behavior.

---

**Example 5: Market Dynamics**

Purpose: Simulate and analyze different market clearing mechanisms.

Instructions:
1. Define market configurations with different market designs.
2. Use the Creator module to generate the scenario.
3. Compare results for different market setups using the Analyzer module.

---

**Example 6: Grid Interaction**

Purpose: Highlight the interaction between agents and grids.

Instructions:
1. Define a distributed energy system with specific grid constraints.
2. Create the scenario either directly or use the topology setup to assign agents to buses.
3. Simulate the scenario using the Executor module.
4. Analyze grid flows and bottlenecks using the Analyzer module.

.. Advanced Examples
.. ------------------
..
.. **Example 7: Multi-Region Energy Systems**
..
.. Purpose: Showcase how to model interconnected regions.
..
.. Instructions:
.. 1. Create configurations for multiple regions with unique agents, markets, and grids.
.. 2. Simulate the interaction between regions.
.. 3. Visualize cross-region energy flows.
..
.. ---
..
.. **Example 8: Policy Impact Analysis**
..
.. Purpose: Demonstrate how policy changes affect market and agent behavior.
..
.. Instructions:
.. 1. Add policy parameters (e.g., carbon pricing) to the configuration files.
.. 2. Simulate the scenario with and without policy changes.
.. 3. Analyze the differences using the Analyzer module.
..
.. ---
..
.. **Example 9: Stress Testing and Scalability**
..
.. Purpose: Test HAMLET’s performance with large-scale simulations.
..
.. Instructions:
.. 1. Increase the number of agents and the duration of the simulation.
.. 2. Enable parallel execution in the Executor module.
..
.. .. code-block:: python
..
..   executor = Executor(path_scenario="./scenarios/example_scenario", num_workers=4)
..
.. 3. Monitor memory usage and runtime performance.
..
.. Specific Applications
.. ---------------------
..
.. **Example 10: Renewable Integration**
..
.. Purpose: Evaluate the impact of high renewable energy penetration.
..
.. Instructions:
.. 1. Add renewable energy sources (e.g., wind, solar) to the grid configuration.
.. 2. Simulate the scenario and analyze renewable energy contributions.
..
.. ---
..
.. **Example 11: Vehicle-to-Grid (V2G)**
..
.. Purpose: Model how electric vehicles interact with the grid.
..
.. Instructions:
.. 1. Define EV agents with charging and discharging behaviors.
.. 2. Simulate the impact on grid stability and energy prices.
..
.. ---
..
.. **Example 12: Local Energy Market Simulation**
..
.. Purpose: Simulate peer-to-peer energy trading among agents.
..
.. Instructions:
.. 1. Define a local energy market with different trading strategies.
.. 2. Simulate and analyze agent interactions and market outcomes.
..
.. Educational Use Cases
.. ---------------------
..
.. **Example 13: Step-by-Step Scenario Creation**
..
.. Purpose: Help students understand energy system modeling.
..
.. Instructions:
.. 1. Break down the scenario creation process into individual steps.
.. 2. Provide guided examples for each step.
..
.. ---
..
.. **Example 14: Market Mechanisms**
..
.. Purpose: Teach market concepts like clearing, pricing, and bidding strategies.
..
.. Instructions:
.. 1. Simulate markets with different clearing mechanisms.
.. 2. Visualize the impact of pricing and bidding strategies.
..
.. ---
..
.. **Example 15: Grid Planning Basics**
..
.. Purpose: Introduce students to grid design and optimization.
..
.. Instructions:
.. 1. Model a small grid with load balancing and generation constraints.
.. 2. Analyze grid performance and optimization results.
..
.. Suggestions for User Contributions
.. ----------------------------------
.. - Encourage users to submit their own scenarios or workflows.
.. - Provide templates for submitting examples to the HAMLET repository.
.. - Highlight contributed examples in the documentation for community engagement.

