Retailer-Based (retailer)
=========================

Introduction
------------

The Retailer-Based trading strategy is a straightforward approach that simulates traditional retailer-consumer relationships in energy markets. With this strategy, agents always use the forecasted retailer prices for their bids and offers, effectively accepting the retailer's price terms without attempting to optimize further.

How It Works
------------

In HAMLET, the Retailer-Based strategy sets bid and offer prices directly based on the forecasted retailer prices:

- For buying energy (bids), the agent uses the forecasted sell price from the retailer
- For selling energy (offers), the agent uses the forecasted buy price from the retailer

This approach represents a passive trading strategy where the agent accepts the retailer's price structure without negotiation or optimization. It's equivalent to having a fixed-price contract with a retailer, where the prices for buying and selling energy are predetermined.

Mathematical Formulation
------------------------

The Retailer-Based strategy uses a simple direct assignment of prices:

.. math::

   P_{bid} = P_{sell}

   P_{offer} = P_{buy}

where:

- :math:`P_{bid}` is the bid price (price per unit for buying energy)
- :math:`P_{offer}` is the offer price (price per unit for selling energy)
- :math:`P_{buy}` is the forecasted buy price from the retailer
- :math:`P_{sell}` is the forecasted sell price to the retailer

Configuration
-------------

The Retailer-Based trading strategy can be configured in the agent configuration file. As the default strategy in HAMLET, it doesn't require any additional parameters beyond specifying the strategy type.

.. code-block:: yaml

   ems:
     market:
       strategy: retailer  # Retailer-Based strategy (default)
       horizon: 86400  # Trading horizon in seconds (e.g., 24 hours)
       # No additional parameters needed for the Retailer-Based strategy

Notes
-----

- The Retailer-Based strategy is the default trading strategy in HAMLET.
- It's the simplest strategy to implement and understand, making it a good starting point for new users.
- This strategy is appropriate for simulating traditional energy markets where most consumers have fixed-price contracts with retailers.
- Unlike the other strategies, Retailer-Based doesn't attempt to optimize prices or respond to market conditions.
- The strategy assumes that the forecasted retailer prices are accurate and available for all timesteps in the trading horizon.
- In real-world applications, this strategy is equivalent to accepting the default tariff structure offered by an energy retailer.
- While simple, this strategy provides a useful baseline for comparing the performance of more sophisticated trading strategies.