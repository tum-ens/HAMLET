Linear
======

Introduction
------------

The Linear Bidding trading strategy is a more sophisticated approach where agents adjust their bid and offer prices linearly over time. This strategy allows agents to gradually become more aggressive in their trading behavior as the delivery time approaches, helping to balance the trade-off between early execution and price optimization.

How It Works
------------

In HAMLET, the Linear Bidding strategy creates a linear progression of bid and offer prices for each timestep within the trading horizon. The progression is determined by two key parameters:

1. **steps_to_final**: The number of timesteps before reaching the maximum price adjustment
2. **steps_from_init**: The number of timesteps from the initial price adjustment

The strategy arranges the timesteps in order of proximity to delivery and applies a linear multiplier to adjust the prices. For timesteps far from delivery, the prices are closer to the forecasted retailer prices. As the delivery time approaches, the prices gradually shift to become more competitive.

Mathematical Formulation
------------------------

The Linear Bidding strategy calculates bid and offer prices using a linear interpolation between the forecasted buy and sell prices, with a multiplier that changes based on the timestep's position in the trading horizon:

.. math::

   P_{bid} = P_{sell} + \frac{(P_{buy} - P_{sell}) \times f}{N}

   P_{offer} = P_{buy} - \frac{(P_{buy} - P_{sell}) \times (N - f)}{N}

where:

- :math:`P_{bid}` is the bid price (price per unit for buying energy)
- :math:`P_{offer}` is the offer price (price per unit for selling energy)
- :math:`P_{buy}` is the forecasted buy price from the retailer
- :math:`P_{sell}` is the forecasted sell price to the retailer
- :math:`N` is the total number of timesteps in the trading horizon
- :math:`f` is the multiplication factor based on the timestep's position, ranging from 0 to N

The multiplication factor :math:`f` is determined by:
- For the first `steps_to_final` timesteps: :math:`f = 0`
- For the last `steps_from_init` timesteps: :math:`f = N`
- For timesteps in between: :math:`f` increases linearly from 0 to N

Configuration
-------------

The Linear Bidding trading strategy can be configured in the agent configuration file with the following parameters:

.. code-block:: yaml

   ems:
     market:
       strategy: linear  # Linear Bidding strategy
       horizon: 86400  # Trading horizon in seconds (e.g., 24 hours)
       linear:
         steps_to_final: 10  # Number of timesteps before reaching maximum price adjustment
         steps_from_init: 5  # Number of timesteps from initial price adjustment

Notes
-----

- The Linear Bidding strategy provides a balance between early execution and price optimization.
- By adjusting the `steps_to_final` and `steps_from_init` parameters, users can control how aggressively the agent's prices change over time.
- A larger `steps_to_final` value means the agent will maintain conservative prices for a longer period.
- A larger `steps_from_init` value means the agent will reach its most aggressive prices earlier.
- This strategy is particularly useful in markets where prices tend to follow predictable patterns over time.
- Unlike the Zero Intelligence strategy, Linear Bidding creates a deterministic price progression that can be tailored to specific market conditions.
- The strategy does not adapt to changing market conditions during execution; the price progression is determined at the beginning of the trading horizon.