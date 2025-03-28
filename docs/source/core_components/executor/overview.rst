Executor Module
===============

The **Executor** is the core simulation engine of HAMLET, responsible for processing interactions between **agents**, **markets**, and **grids** over time. It executes the predefined scenarios step by step, ensuring that agent decisions, market dynamics, and grid constraints are accurately modeled.

.. toctree::
   :maxdepth: 1
   :caption: Executor

   setup
   agents/agents
   markets/markets
   grids/grids

Key Responsibilities
--------------------
The **Executor** performs the following tasks:

1. **Time-Stepping**: Advances the simulation step by step based on the defined `timestep` in the configuration.
2. **Agent Decision-Making**: Executes the logic behind agent actions, including energy trading, load shifting, and self-consumption.
3. **Market Clearing**: Processes local market transactions and settles bids and offers.
4. **Grid Power Flow Calculation**: Evaluates energy flows to ensure grid constraints are met.
5. **Data Logging**: Records all market, grid, and agent actions for later analysis.

Simulation Workflow
-------------------
A typical HAMLET simulation follows this workflow:

1. **Initialize Simulation**
   - Load agents, markets, and grid configurations.
   - Set up initial conditions (e.g., weather, forecasts, battery state-of-charge).

2. **Execute Time Steps**
   - For each time step, the following actions occur in sequence:
     - Agents update their internal states (e.g., energy demand, generation, or forecasts).
     - Agents submit bids and offers to the market.
     - Markets clear trades based on their clearing method (e.g., **periodic double auction**).
     - The grid evaluates energy flows and constraints.
     - Unsettled trades may be carried forward to wholesale markets or retailers.
     - Agents receive updated price signals and adjust strategies accordingly.

3. **Store Results**
   - Logs energy consumption, trading activity, and grid states.
   - Data is used for post-simulation analysis.

Parallel Execution
------------------
HAMLET supports **parallel execution** to improve simulation speed by distributing tasks across multiple CPU cores.

.. code-block:: python

   from hamlet import Executor

   executor = Executor(path_scenario="./scenarios/example_scenario", num_workers=4)
   executor.run()

By default, HAMLET determines the optimal number of workers based on the available system resources. Users can **manually adjust** the number of workers depending on the **scenario complexity** and **hardware capabilities**.

