Grid Overview
=============

The **Grid** in HAMLET represents the physical energy infrastructure where energy flows are simulated and constraints are enforced. The grid ensures that the energy transactions carried out in the markets remain technically feasible.

What Does the Grid Do?
----------------------
The grid is responsible for:
- **Simulating Power Flows**: Computes energy flow across nodes using power flow calculations.
- **Enforcing Grid Constraints**: Ensures that network limitations (e.g., voltage levels, line capacities) are respected.
- **Implementing Grid Control Measures**: Applies dynamic control mechanisms if grid stability is compromised.
- **Interfacing with Agents**: Retrieves energy setpoints from agents that influence grid behavior.

Grid Execution Workflow
-----------------------
Each simulation timestep involves a structured execution of grid operations:

1. **Retrieve Agent Setpoints**
   - Collects energy consumption and generation setpoints from agents.

2. **Update Grid Parameters**
   - Maps agent energy behavior onto the grid topology.
   - Assigns load and generation values to respective buses.

3. **Compute Power Flow**
   - Performs grid calculations based on the configured power flow method:
     - **DC Power Flow**: Simplified model assuming purely active power.
     - **AC Power Flow**: Full model considering active and reactive power.
     - **Optimal Power Flow (OPF)**: Optimization-based dispatch ensuring grid feasibility.

4. **Apply Grid Restrictions (If Configured)**
   - Implements regulatory or technical constraints such as:
     - **Dynamic Grid Fees**: Adjusts grid fees based on congestion levels.
     - **Direct Power Control**: Limits specific assets to avoid grid violations.

5. **Store Results**
   - Logs power flow outcomes, including node voltages, line loading, and constraint violations.

Grid Structure
--------------
Each grid setup consists of the following key components:

1. **Grid Type**
   - Defines the type of energy grid being simulated:
     - **Electricity**: Standard electricity network.
     - **Heat**: District heating network (future implementation).
     - **Hydrogen**: Hydrogen distribution grid (future implementation).

2. **Power Flow Calculation Method**
   - Determines how energy distribution is computed:
     - **DC Power Flow (`dc`)**: Assumes lossless, resistive transmission.
     - **AC Power Flow (`ac`)**: Considers voltage and reactive power.
     - **Optimal Power Flow (`acopf`, `dcopf`)**: Optimized dispatch to minimize cost while ensuring grid stability.

3. **Grid Restrictions**
   - Specifies regulatory measures and dynamic control mechanisms:
     - **Dynamic Grid Fees**: Implements variable pricing for network usage.
     - **Direct Power Control**: Adjusts power injection based on grid constraints.

4. **Grid Operator Role**
   - Unlike standard agents, the **grid operator** enforces technical constraints indirectly by:
     - Modifying grid fees dynamically.
     - Curtailing loads or generation if network limits are exceeded.
     - Influencing agent decisions through incentive-based mechanisms.

Extending Grid Models
---------------------
HAMLET allows users to extend grid functionalities:

- **Custom Power Flow Models**
  - Users can integrate alternative grid calculation methods (e.g., hybrid AC-DC models).

- **Advanced Constraint Enforcement**
  - Custom rules for congestion management and real-time network adjustments.

- **Multi-Energy Grids**
  - Future support for integrated electricity, heat, and hydrogen grids.

By modeling energy networks in a realistic yet flexible way, HAMLET enables the study of **power flow dynamics**, **market-grid interactions**, and **grid-aware agent behavior** in decentralized energy systems.
