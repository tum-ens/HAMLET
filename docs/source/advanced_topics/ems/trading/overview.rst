Trading Strategies
================

Introduction
-----------

Trading strategies are a critical component of the Energy Management System (EMS) in HAMLET. They determine how agents interact with energy markets, including how they formulate bids and offers, respond to market signals, and make decisions about buying and selling energy. Trading strategies are executed after forecasting and control decisions have been made.

Available Strategies
--------------------

.. toctree::
   :maxdepth: 1
   
   strategies/zero_intelligence
   strategies/linear_bidding
   strategies/retailer_based

Role in the Simulation Flow
-------------------------

Trading strategies are the final step in the EMS workflow:

1. **Forecasting**: Agents generate forecasts for relevant parameters such as energy demand, generation potential, and market prices.
2. **Control**: Based on these forecasts, controllers make decisions about energy usage, storage, and trading.
3. **Trading**: Finally, agents execute their trading strategies to buy or sell energy in the market.

The trading strategies use the outputs from the controllers to determine the quantities and prices at which agents are willing to buy or sell energy.

Trading Strategy Types
-------------------

HAMLET supports various trading strategies with different levels of sophistication:

1. **Zero Intelligence**: Simple strategies that make randomized decisions without complex reasoning.

2. **Linear Bidding**: Strategies that adjust bids/offers linearly based on certain parameters.

3. **Retailer-Based**: Strategies that simulate traditional retailer-consumer relationships with fixed or time-of-use pricing.

More advanced strategies can be implemented by extending the existing framework, including:

- **Game-Theoretic Approaches**: Strategies based on game theory concepts such as Nash equilibrium.
- **Learning-Based Strategies**: Strategies that learn and adapt over time using techniques such as reinforcement learning.
- **Portfolio Optimization**: Strategies that optimize a portfolio of energy assets and market positions.

Configuration
------------

Trading strategies can be configured in the agent configuration file. The specific parameters depend on the chosen strategy, but typically include:

- Strategy type
- Risk preferences
- Price limits
- Adaptation parameters

Integration with Controllers
-------------------------

Trading strategies work closely with controllers in the EMS:

1. Controllers determine the available flexibility (e.g., how much energy can be bought or sold).
2. Trading strategies use this information to formulate bids and offers.
3. Market outcomes feed back into the controllers for the next decision cycle.

This integration ensures that trading decisions are consistent with the physical and operational constraints of the energy system.

The specific implementation details of each trading strategy are described in the following sections.