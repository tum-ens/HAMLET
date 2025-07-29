Controllers
===========

Introduction
------------

Controllers are the decision-making components of the Energy Management System (EMS). They use forecasts and current system state to determine optimal energy usage, storage, and trading strategies. Controllers are responsible for balancing multiple objectives such as cost minimization, comfort maximization, and environmental impact reduction.

HAMLET supports different types of controllers with varying approaches to decision-making.

Controller Types
----------------

.. toctree::
   :maxdepth: 2
   
   rtc/overview
   fbc/overview

Real-Time Controllers (RTC)
---------------------------

Real-Time Controllers make decisions for the current timestep, i.e. now.
They are reactive in nature and do not plan far into the future.

Forecast-Based Controllers (FBC)
--------------------------------

Forecast-Based Controllers use forecasts over a longer time horizon to plan ahead and make decisions that optimize performance over multiple timesteps.
FBCs can achieve better overall performance by anticipating future conditions but require more computational resources and accurate forecasts.

Controller Approaches
---------------------

Within each controller type, HAMLET supports different approaches to decision-making:

1. **Optimization-Based**: Formulate the control problem as a mathematical optimization problem and solve it using numerical solvers.

2. **Rule-Based**: Use predefined rules and heuristics to make decisions based on the current system state and forecasts. (currently none implemented)

3. **Reinforcement Learning**: Learn optimal control policies through interaction with the environment. (currently none implemented)

Implementation Options
----------------------

HAMLET provides multiple implementation options for optimization-based controllers:

1. **Linopy**: A high-level interface for linear and mixed-integer optimization problems, built on top of xarray and popular solvers.

2. **PyOptInterface**: A flexible interface to various optimization solvers, allowing for more customization and control over the optimization process.

The choice of implementation depends on the specific requirements of the simulation, such as problem complexity, solver availability, and performance considerations.

Configuration
-------------

Controllers can be configured in the agent configuration file. The specific parameters depend on the chosen controller type and approach, but typically include:

- Controller type (RTC or FBC)
- Optimization approach
- Objective function weights
- Constraints
- Solver parameters

Example configuration:

.. code-block:: yaml

   controller:
     type: fbc
     approach: optimization
     implementation: linopy
     objective:
       cost_weight: 1.0
       comfort_weight: 0.5
     horizon: 24
     solver:
       name: gurobi
       time_limit: 60