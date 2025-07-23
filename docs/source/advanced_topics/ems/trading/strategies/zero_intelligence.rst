Zero Intelligence (zi)
=====================

Introduction
------------

The Zero Intelligence (ZI) trading strategy is a simple approach where agents make randomized decisions without complex reasoning. This strategy is inspired by the concept of "zero intelligence traders" from economic literature, where agents place random bids and offers within certain constraints.

How It Works
------------

In HAMLET, the Zero Intelligence strategy generates random bid and offer prices for each timestep within the trading horizon. The prices are randomly selected between the forecasted buy and sell prices, with no consideration for strategic behavior or market conditions.

The randomization occurs independently for each timestep, meaning there is no correlation between prices across different timesteps. This approach simulates a market with agents who have no strategic intelligence or learning capabilities.

Mathematical Formulation
------------------------

For each timestep within the trading horizon, the bid price (price at which the agent is willing to buy energy) and offer price (price at which the agent is willing to sell energy) are calculated as:

.. math::

   P_{bid} = P_{sell} + random(0,1) \times (P_{buy} - P_{sell})

   P_{offer} = P_{sell} + random(0,1) \times (P_{buy} - P_{sell})

where:

- :math:`P_{bid}` is the bid price (price per unit for buying energy)
- :math:`P_{offer}` is the offer price (price per unit for selling energy)
- :math:`P_{buy}` is the forecasted buy price from the retailer
- :math:`P_{sell}` is the forecasted sell price to the retailer
- :math:`random(0,1)` is a random number between 0 and 1

Configuration
-------------

The Zero Intelligence trading strategy can be configured in the agent configuration file. Unlike other strategies, it doesn't require any additional parameters beyond specifying the strategy type.

.. code-block:: yaml

   ems:
     market:
       strategy: zi  # Zero Intelligence strategy
       horizon: 86400  # Trading horizon in seconds (e.g., 24 hours)

Notes
-----

- The Zero Intelligence strategy is useful as a baseline for comparison with more sophisticated strategies.
- It can be used to simulate markets with low information or high uncertainty.
- Despite its simplicity, ZI strategies can sometimes lead to efficient market outcomes in aggregate, as demonstrated in economic literature.
- This strategy does not adapt to market conditions or learn from past experiences.
- The random nature of this strategy means that results can vary significantly between simulation runs, even with identical initial conditions.