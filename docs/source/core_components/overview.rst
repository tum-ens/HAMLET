Overview
===============

HAMLET is structured into three main components, each serving a specific role in energy system simulation. This modular architecture ensures flexibility, scalability, and ease of customization.

Overview of Core Components
---------------------------

HAMLET consists of the following key components:

1. **Creator**
   - Responsible for setting up the simulation environment.
   - Generates agents, markets, and grids based on user-defined configurations.
   - Ensures all elements are initialized correctly before execution.

2. **Executor**
   - Runs the simulation by executing the interactions between agents, markets, and grids over time.
   - Manages time-stepping and parallel execution (if enabled) to ensure efficient simulations.
   - Processes decisions, transactions, and constraints dynamically.

3. **Analyzer**
   - Handles post-simulation analysis by processing and visualizing results.
   - Enables users to extract insights into agent behavior, market dynamics, and grid performance.
   - Supports visualization of energy flows, price trends, and grid constraints.

Each component operates independently but communicates through structured data flows, ensuring a seamless workflow from scenario creation to execution and analysis.
