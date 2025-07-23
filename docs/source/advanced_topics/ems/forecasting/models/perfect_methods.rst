Perfect Forecasting Methods
==========================

Introduction
-----------

Perfect forecasting methods represent the ideal case where the forecasting model has complete and accurate knowledge of future values. These methods are primarily used for benchmarking and theoretical analysis rather than practical applications, as they assume information that would not be available in real-world scenarios.

In the spectrum of forecasting complexity, perfect forecasting methods are conceptually the simplest but practically impossible to implement in real-world applications. They serve as an upper bound for forecasting performance and are useful for evaluating the maximum potential benefit of forecasting in energy management systems.

Perfect
-------

The perfect forecasting method provides exact future values without any error. It assumes complete knowledge of the future, which is only possible in simulation environments where future data is pre-generated or known in advance.

Available for
~~~~~~~~~~~~

Perfect forecasting is available for most components in HAMLET:

Inflexible Load | Heat Demand | DHW | PV | Wind | EV | Market Prices | Grid Fees | Levies

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \hat{y}_{t+k} = y_{t+k}

where:

   - :math:`\hat{y}_{t+k}` is the forecasted value for time :math:`t+k`
   - :math:`y_{t+k}` is the actual value at time :math:`t+k`

Configuration
~~~~~~~~~~~~

Perfect forecasting methods can be configured in the agent configuration file:

.. code-block:: yaml

   inflexible-load:
     fcast:
       method: perfect  # Options: perfect, naive, average, smoothed, sarma, rfr, cnn, rnn, arima

   # Or for market prices
   ems:
     market:
       fcast:
         wholesale:
           method: perfect  # Options: flat, naive, perfect

Notes
~~~~~

Perfect forecasting is primarily used for:

1. Benchmarking other forecasting methods
2. Theoretical analysis of control strategies
3. Understanding the upper bound of performance in energy management systems
4. Isolating the impact of forecast errors on system performance

In HAMLET, perfect forecasting is implemented by directly using the future values from the simulation data. This is only possible because HAMLET is a simulation environment where future data is pre-generated or known in advance.

While perfect forecasting is not realistic for real-world applications, it provides valuable insights into the maximum potential benefit of forecasting in energy management systems and serves as a reference point for evaluating other forecasting methods.