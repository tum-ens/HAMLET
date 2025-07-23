Mathematical Formulations
========================

Introduction
-----------

This section presents the general mathematical framework used in HAMLET's optimization-based forecast-based controllers, including the problem structure, objective functions, and component models. The formulations are designed to be flexible, allowing for various energy carriers (electricity, heat, hydrogen), different forecast horizons, and diverse system configurations.

Mathematical Formulation Structure
---------------------------------

The mathematical formulation in HAMLET is organized into two main components:

1. **Objective Function**: Defines what the controller aims to optimize over the prediction horizon
2. **Component Models**: Defines the mathematical representation of system components and their temporal evolution

These components are combined to form the complete optimization problem, which is then solved using appropriate numerical solvers.

.. toctree::
   :maxdepth: 1

   objective_function
   component_models

General Problem Structure
------------------------

The optimization problem for forecast-based controllers can be generally formulated as:

.. math::

   \min_{\mathbf{x}} \quad & \sum_{t=0}^{T-1} f_t(\mathbf{x}_t) \\
   \text{subject to} \quad & g_{i,t}(\mathbf{x}_t, \mathbf{x}_{t+1}, \ldots, \mathbf{x}_{t+k}) \leq 0, \quad i = 1, \ldots, m, \quad t = 0, \ldots, T-1 \\
   & h_{j,t}(\mathbf{x}_t, \mathbf{x}_{t+1}, \ldots, \mathbf{x}_{t+k}) = 0, \quad j = 1, \ldots, p, \quad t = 0, \ldots, T-1

where:
- :math:`\mathbf{x}_t` is the vector of decision variables at timestep :math:`t`
- :math:`f_t(\mathbf{x}_t)` is the objective function at timestep :math:`t`
- :math:`g_{i,t}` are inequality constraints that may couple variables across timesteps
- :math:`h_{j,t}` are equality constraints that may couple variables across timesteps
- :math:`T` is the optimization horizon

The key difference between FBC and RTC optimization problems is that FBCs optimize over multiple timesteps simultaneously, with constraints that couple decisions across time.

Key Concepts
-----------

**Mathematical Programming**

Optimization-based controllers rely on mathematical programming techniques, primarily:

- **Linear Programming (LP)**: Problems with linear objective functions and constraints
- **Mixed-Integer Linear Programming (MILP)**: LP problems with some integer or binary variables
- **Quadratic Programming (QP)**: When the objective function includes quadratic terms (e.g., comfort maximization)
- **Mixed-Integer Quadratic Programming (MIQP)**: Combination of MILP and QP

In HAMLET, the focus is primarily on LP and MILP formulations due to their computational tractability and the availability of efficient solvers.

**Decision Variables**

Decision variables represent the quantities that the controller can adjust to optimize system performance. In forecast-based energy management systems, common decision variables include:

- Power consumption/production of each component at each timestep in the horizon
- Energy storage charging/discharging rates at each timestep in the horizon
- Energy storage state of charge at each timestep in the horizon
- Energy trading decisions (buy/sell quantities) at each timestep in the horizon
- Component operational states (on/off) at each timestep in the horizon

**Objective Function**

The objective function represents the goal of the optimization, typically expressed as a cost function to be minimized over the entire prediction horizon. Common objectives in forecast-based energy management include:

- Minimizing operational costs over the prediction horizon
- Minimizing energy consumption over the prediction horizon
- Minimizing carbon emissions over the prediction horizon
- Maximizing self-consumption of renewable energy over the prediction horizon
- Balancing multiple objectives through weighted combinations

For a detailed discussion of objective functions, see the :doc:`objective_function` section.

**Constraints**

Constraints define the feasible region of the optimization problem. In forecast-based energy systems, constraints typically include:

1. **Physical Constraints**: Ensure that the solution respects the physical limitations of the components at each timestep.
   - Power limits
   - Energy storage capacity limits
   - Ramp rate limits

2. **Balance Constraints**: Ensure energy balance in the system at each timestep.
   - Power balance: Total consumption equals total production plus grid exchange
   - Heat balance: Total heat demand equals total heat production

3. **Operational Constraints**: Ensure that the components operate within their specified ranges.
   - Minimum up/down times
   - Startup and shutdown constraints
   - Comfort range constraints

4. **Temporal Coupling Constraints**: Ensure proper coupling between timesteps.
   - Energy storage state of charge evolution
   - Thermal dynamics of buildings
   - Minimum runtime constraints

Component-specific constraints and system-level constraints are detailed in the :doc:`component_models` section.

**Receding Horizon Implementation**

In practice, FBCs are often implemented using a receding horizon approach, also known as Model Predictive Control (MPC):

1. At the current timestep :math:`t`, solve the optimization problem over the horizon :math:`[t, t+T-1]`.
2. Apply only the first decision :math:`\mathbf{x}_t` from the optimal solution.
3. Move to the next timestep :math:`t+1`, update forecasts, and repeat the process.

This approach allows the controller to adapt to changing conditions and forecast updates while still considering future impacts of current decisions.

FBC-specific aspects of the mathematical formulation are detailed in the :doc:`fbc_specifics` section.