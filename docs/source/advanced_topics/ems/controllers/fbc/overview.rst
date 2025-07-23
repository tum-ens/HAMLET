Forecast-Based Controllers (FBC)
===========================

Introduction
-----------

Forecast-Based Controllers (FBCs) are a type of controller in HAMLET that use forecasts over a longer time horizon to plan ahead and make decisions that optimize performance over multiple timesteps. Unlike Real-Time Controllers (RTCs), which focus on immediate decisions, FBCs consider the future impact of current decisions.

.. toctree::
   :maxdepth: 1
   
   optimization_based/overview
   rule_based/overview
   reinforcement_learning/overview

Characteristics of FBCs
---------------------

FBCs have several key characteristics that distinguish them from other controller types:

1. **Predictive Decision-Making**: FBCs make decisions based on forecasts of future conditions, allowing them to anticipate and prepare for upcoming events.

2. **Temporal Coupling**: Decisions at different timesteps are coupled, meaning that the controller considers how current decisions affect future states and options.

3. **Optimization Horizon**: FBCs optimize over a finite time horizon, typically ranging from several hours to days, depending on the application.

4. **Receding Horizon Implementation**: In practice, FBCs are often implemented using a receding horizon approach, where only the first decision is applied, and the optimization is repeated at the next timestep with updated forecasts.

When to Use FBCs
--------------

FBCs are most appropriate in the following scenarios:

- When accurate forecasts are available for a reasonable time horizon
- When there are significant temporal dependencies in the system (e.g., energy storage)
- When current decisions have significant impact on future system states
- When global optimization over time is more important than computational efficiency

Limitations of FBCs
----------------

While FBCs offer advantages in terms of performance optimization, they also have limitations:

1. **Forecast Dependency**: The performance of FBCs heavily depends on the accuracy of forecasts. Poor forecasts can lead to suboptimal or even counterproductive decisions.

2. **Computational Complexity**: FBCs require solving larger optimization problems, which can be computationally intensive, especially for systems with many components or long optimization horizons.

3. **Model Complexity**: FBCs typically require more detailed models of system dynamics and constraints to accurately predict future states.

FBC Implementation in HAMLET
-------------------------

In HAMLET, FBCs can be implemented using different approaches:

1. **Optimization-Based**: Formulate the control problem as a mathematical optimization problem over a time horizon.

2. **Rule-Based**: Use predefined rules and heuristics that consider forecasted future conditions.

3. **Reinforcement Learning**: Learn optimal control policies that account for future rewards.

The specific implementation details, including the optimization problem formulation and component models, are described in the following sections.

Comparison with RTCs
-----------------

+----------------------------+----------------------+---------------------------+
| Feature                    | FBC                  | RTC                       |
+============================+======================+===========================+
| Decision Horizon           | Multiple timesteps   | Single timestep           |
+----------------------------+----------------------+---------------------------+
| Computational Complexity   | Higher               | Lower                     |
+----------------------------+----------------------+---------------------------+
| Forecast Dependency        | Higher               | Lower                     |
+----------------------------+----------------------+---------------------------+
| Performance Optimization   | Better long-term     | Better immediate response |
+----------------------------+----------------------+---------------------------+
| Implementation Complexity  | More complex         | Simpler                   |
+----------------------------+----------------------+---------------------------+


The choice between FBC and RTC depends on the specific requirements of the application, such as the availability of accurate forecasts, computational resources, and the importance of long-term optimization versus immediate response.