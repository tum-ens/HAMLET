Setup
=====

The **Executor** is responsible for managing the entire simulation process, from setting up the environment to executing agent and market interactions and ensuring grid constraints are met.

What Does the Executor Do?
--------------------------
The **Executor** acts as the central controller of the simulation, ensuring that:

- All agents, markets, and grids operate in a synchronized manner.
- Simulations follow the predefined schedule from the **timetable**.
- Grid constraints are respected while maintaining a balance between supply and demand.
- Results are stored properly for analysis.

Execution Workflow
------------------
The simulation follows a structured workflow:

1. **Setup Phase**
   - Reads the scenario configuration and initializes relevant components.
   - Loads the timetable, scenario settings, and simulation parameters.
   - Establishes database connections and prepares storage locations for results.

2. **Execution Phase**
   - Iterates over the **timetable**, processing each timestep sequentially.
   - At each timestep:

     - Agents make decisions based on available information.
     - Markets execute clearing mechanisms to determine energy prices and trades.
     - Grids are evaluated to ensure power flow feasibility.

   - If grid constraints are not met, a recalculation is triggered, reverting to a previous database state if necessary.

3. **Cleanup Phase**
   - Saves the final simulation state and merges market and grid results.
   - Ensures all simulation data is properly stored for later analysis.

Key Components in Execution
---------------------------
1. **Agent Task Execution**
   - Manages agent decision-making.
   - Runs agent-related processes in parallel for performance optimization.

2. **Market Task Execution**
   - Handles bid/offer processing and market clearing.
   - Ensures smooth market operations across different clearing mechanisms.

3. **Grid Execution**
   - Simulates power flows based on market results and agent actions.
   - Ensures the grid remains within operational constraints.
   - Iteratively recalculates in case of violations.

4. **Database Management**
   - Stores and retrieves simulation data at each timestep.
   - Maintains backups to revert to previous states when needed.

By structuring execution this way, HAMLET ensures a **modular, scalable, and realistic** energy system simulation.
