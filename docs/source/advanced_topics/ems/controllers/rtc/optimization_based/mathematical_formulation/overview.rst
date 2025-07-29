Mathematical Formulations
========================

Introduction
-----------

Mathematical formulations provide the foundation for optimization-based controllers in energy management systems. These formulations translate physical systems, operational requirements, and control objectives into mathematical models that can be solved using numerical optimization techniques.

This section presents the general mathematical framework used in HAMLET's optimization-based controllers, including the problem structure, objective functions, and component models. The formulations are designed to be flexible, allowing for various energy carriers (electricity, heat, hydrogen), different time horizons, and diverse system configurations.

Mathematical Formulation Structure
---------------------------------

The mathematical formulation in HAMLET is organized into two main components:

1. **Objective Function**: Defines what the controller aims to optimize
2. **Component Models**: Defines the mathematical representation of system components

These components are combined to form the complete optimization problem, which is then solved using appropriate numerical solvers.

.. toctree::
   :maxdepth: 1

   objective_function
   component_models

General Problem Structure
------------------------

The optimization problem in HAMLET can be generally formulated as:

.. math::

   \min_{\mathbf{x}} \quad & f(\mathbf{x}) \\
   \text{subject to} \quad & g_i(\mathbf{x}) \leq 0, \quad i = 1, \ldots, m \\
   & h_j(\mathbf{x}) = 0, \quad j = 1, \ldots, p \\
   & x_k \in \mathbb{Z}, \quad k \in \mathcal{I}

where:

- :math:`\mathbf{x}` is the vector of decision variables (e.g., power flows, energy levels, binary states)
- :math:`f(\mathbf{x})` is the objective function to be minimized
- :math:`g_i(\mathbf{x})` are inequality constraints
- :math:`h_j(\mathbf{x})` are equality constraints
- :math:`x_k \in \mathbb{Z}` indicates that some variables are restricted to integer values
- :math:`\mathcal{I}` is the set of indices for integer variables

In the context of energy management, this general formulation is typically instantiated as a time-discretized problem over a finite horizon:

.. math::

   \min_{\mathbf{x}} \quad & \sum_{t=1}^{T} f_t(\mathbf{x}_t) \\
   \text{subject to} \quad & \mathbf{x}_{\min} \leq \mathbf{x}_t \leq \mathbf{x}_{\max}, \quad t = 1, \ldots, T \\
   & \mathbf{A}_t \mathbf{x}_t = \mathbf{b}_t, \quad t = 1, \ldots, T \\
   & \mathbf{C}_t \mathbf{x}_t + \mathbf{D}_{t-1} \mathbf{x}_{t-1} \leq \mathbf{e}_t, \quad t = 2, \ldots, T

where:

- :math:`T` is the optimization horizon
- :math:`\mathbf{x}_t` is the vector of decision variables at time step :math:`t`
- :math:`f_t(\mathbf{x}_t)` is the objective function at time step :math:`t`
- The second line represents variable bounds
- The third line represents energy balance constraints
- The fourth line represents time-coupling constraints (e.g., for storage components)

Key Concepts
-----------

**Mathematical Programming**

Optimization-based controllers rely on mathematical programming techniques, primarily:

- **Linear Programming (LP)**: Problems with linear objective functions and constraints
- **Mixed-Integer Linear Programming (MILP)**: LP problems with some integer or binary variables
- **Nonlinear Programming (NLP)**: Problems with nonlinear objectives or constraints
- **Mixed-Integer Nonlinear Programming (MINLP)**: NLP problems with some integer or binary variables

In HAMLET, the focus is primarily on LP and MILP formulations due to their computational tractability and the availability of efficient solvers.

**Decision Variables**

Decision variables represent the quantities that the controller can adjust to optimize system performance. In energy management systems, common decision variables include:

- Power flows between components
- Energy storage levels
- Binary variables for component states (on/off)
- Continuous variables for modulation levels
- Integer variables for discrete operational modes

**Objective Function**

The objective function represents the goal of the optimization, typically expressed as a cost function to be minimized. Common objectives in energy management include:

- Minimizing operational costs
- Minimizing energy consumption
- Minimizing carbon emissions
- Maximizing self-consumption of renewable energy
- Balancing multiple objectives through weighted combinations

For a detailed discussion of objective functions, see the :doc:`objective_function` section.

**Constraints**

Constraints define the feasible region of the optimization problem. In energy systems, constraints typically include:

- Physical limitations of components (power limits, energy capacity)
- Energy balance equations
- Operational requirements (comfort ranges, minimum run times)
- Temporal coupling between time steps (for storage components)
- Network constraints (voltage limits, power flow equations)

A particularly important class of constraints are the system-level energy balance constraints, which ensure that supply matches demand for each energy carrier (electricity, heat, hydrogen) at each time step. These constraints couple the operation of different components and are fundamental to the feasibility of the solution.

Component-specific constraints and system-level constraints are detailed in the :doc:`component_models` section.
