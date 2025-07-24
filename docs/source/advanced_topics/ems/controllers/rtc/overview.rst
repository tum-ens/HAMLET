Real-Time Controllers (RTC)
===========================

Introduction
-----------

Real-Time Controllers (RTCs) are a type of controller in HAMLET that make decisions at the current timestep based on the current system state.

.. toctree::
   :maxdepth: 1
   
   optimization_based/overview
   rule_based/overview
   reinforcement_learning/overview

Characteristics of RTCs
---------------------

RTCs have several key characteristics that distinguish them from other controller types:

1. **Reactive Decision-Making**: RTCs make decisions based primarily on the current system state and short-term forecasts.

2. **Computational Efficiency**: Since RTCs don't plan far into the future, they typically require less computational resources than controllers with longer planning horizons.

3. **Timestep Independence**: Each decision is made independently at each timestep, without considering the impact on future timesteps.

4. **Simplicity**: RTCs often use simpler models and objective functions compared to controllers with longer planning horizons.

When to Use RTCs
----------------

Please note, that an RTC always needs to be used in the models as they ensure the balance of supply and demand of each agent.

RTCs are most appropriate in the following scenarios:

- When computational resources are limited
- When accurate long-term forecasts are not available
- When the system dynamics change rapidly and unpredictably
- When immediate response to changing conditions is more important than long-term optimality

Limitations of RTCs
----------------

While RTCs offer advantages in terms of simplicity and computational efficiency, they also have limitations:

1. **Suboptimal Long-Term Performance**: By focusing on immediate decisions without considering future impacts, RTCs may not achieve globally optimal performance over longer time horizons.

2. **Limited Anticipation**: RTCs cannot effectively anticipate and prepare for future events or conditions.

3. **Myopic Decision-Making**: Decisions that are optimal in the short term may lead to suboptimal or even infeasible states in the future.

RTC Implementation in HAMLET
-------------------------

In HAMLET, RTCs can be implemented using different approaches:

1. **Optimization-Based**: Formulate the control problem as a mathematical optimization problem for the current timestep.

2. **Rule-Based**: Use predefined rules and heuristics to make decisions based on the current system state.

3. **Reinforcement Learning**: Learn optimal control policies through interaction with the environment.

The specific implementation details, including the optimization problem formulation and component models, are described in the following sections.