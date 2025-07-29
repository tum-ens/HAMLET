Optimization-based Methods
==========================

Introduction
-----------

In the context of forecast-based control (FBC), optimization-based methods determine the optimal operation of energy system components over multiple future timesteps by minimizing or maximizing an objective function while satisfying a set of constraints. These methods provide a structured and mathematically rigorous approach to decision-making in complex energy systems with forecasted inputs.

General Approach
--------------

The general approach to implementing optimization-based forecast-based controllers involves:

1. **System modeling**: Defining the mathematical representation of the energy system over multiple timesteps
2. **Problem formulation**: Specifying the objective function and constraints across the prediction horizon
3. **Forecast integration**: Incorporating forecasts of relevant parameters (e.g., weather, prices, loads)
4. **Solver selection**: Choosing an appropriate optimization solver for multi-period problems
5. **Solution implementation**: Applying the optimal control actions to the system (typically only the first timestep)
6. **Receding horizon implementation**: Re-solving the problem at each time step with updated forecasts and system states

Documentation Structure
---------------------

This section is organized as follows:

.. toctree::
   :maxdepth: 2

   mathematical_formulation/overview
   implementation/overview
   build_your_own/overview

The **Mathematical Formulation** section focuses on the general understanding of the objective function and component models over a prediction horizon, independent of specific implementation details.

The **Implementation** section provides concrete implementations using different frameworks (Linopy and PyOptInterface) for forecast-based control.

The **Build Your Own** section provides instructions on how to extend or customize the optimization-based forecast-based controllers for specific needs.