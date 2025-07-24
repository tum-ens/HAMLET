Naive Forecasting Methods
=========================

Introduction
------------

Naive forecasting methods are simple approaches that make minimal assumptions about the data. These methods serve as important baselines in forecasting and are often surprisingly effective for many time series, especially those with strong seasonal patterns or limited historical data. Despite their simplicity, naive methods can be computationally efficient and provide reasonable forecasts in many practical scenarios.

In the spectrum of forecasting complexity, naive methods represent the simpler end, offering computational efficiency and interpretability at the potential cost of accuracy for complex patterns.

Naive
-----

Description: The naive or persistence method assumes that future values will be the same as past values with a specified offset. This method is particularly useful for time series with strong daily or weekly patterns.

Available for
~~~~~~~~~~~~

The naive method is available for most components in HAMLET:

Inflexible Load | Heat Demand | DHW | Market Price

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \hat{y}_{t+h} = y_{t-k \cdot 24 + h}

where:

   - :math:`\hat{y}_{t+h}` is the forecasted value at time :math:`t+h`
   - :math:`y_{t-k \cdot 24 + h}` is the observed value at time :math:`t-k \cdot 24 + h`
   - :math:`k` is the offset in days
   - :math:`h` is the forecast horizon in hours

Configuration
~~~~~~~~~~~~

Naive forecasting methods can be configured in the agent configuration file:

.. code-block:: yaml

   component-name:
     fcast:
       method: naive
       naive:
         offset: 1  # offset in days to the current day

Average
------

Description: The average method assumes that future values will be the same as the average of several previous days. This method can smooth out day-to-day variations while still capturing daily patterns.

Available for
~~~~~~~~~~~~

The average method is available for most components in HAMLET:

Inflexible Load | Heat Demand | DHW

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \hat{y}_{t+h} = \frac{1}{n} \sum_{i=1}^{n} y_{t-i \cdot 24 + h - k \cdot 24}

where:

   - :math:`\hat{y}_{t+h}` is the forecasted value at time :math:`t+h`
   - :math:`y_{t-i \cdot 24 + h - k \cdot 24}` is the observed value at time :math:`t-i \cdot 24 + h - k \cdot 24`
   - :math:`n` is the number of days to average
   - :math:`k` is the offset in days
   - :math:`h` is the forecast horizon in hours

Configuration
~~~~~~~~~~~~

Average forecasting methods can be configured in the agent configuration file:

.. code-block:: yaml

   component-name:
     fcast:
       method: average
       average:
         offset: 1  # offset in days to the current day
         days: 2    # number of days to be used for averaging

Smoothed
-------

Description: The smoothed method applies a moving average to future values to reduce noise and short-term fluctuations. This method is useful for smoothing out irregular patterns while preserving the overall trend.

Available for
~~~~~~~~~~~~

The smoothed method is available for most components in HAMLET:

Inflexible Load | Heat Demand | DHW

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \hat{y}_{t+h} = \frac{1}{2s+1} \sum_{i=-s}^{s} y_{t+h+i}

where:

   - :math:`\hat{y}_{t+h}` is the forecasted value at time :math:`t+h`
   - :math:`y_{t+h+i}` is the observed or previously forecasted value at time :math:`t+h+i`
   - :math:`s` is the number of steps on each side of the central point
   - :math:`h` is the forecast horizon in hours

Configuration
~~~~~~~~~~~~

Smoothed forecasting methods can be configured in the agent configuration file:

.. code-block:: yaml

   component-name:
     fcast:
       method: smoothed
       smoothed:
         steps: 9  # number of future time steps to be used for smoothing

Flat
----

Description: The flat method assumes a constant value for all future time steps. This method is primarily used for market price forecasting when no better information is available.

Available for
~~~~~~~~~~~~

The flat method is available primarily for market components:

Market Price

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \hat{y}_{t+h} = c

where:

   - :math:`\hat{y}_{t+h}` is the forecasted value at time :math:`t+h`
   - :math:`c` is a constant value (typically based on the current or average price)
   - :math:`h` is the forecast horizon in hours

Configuration
~~~~~~~~~~~~

Flat forecasting methods can be configured in the agent configuration file:

.. code-block:: yaml

   market:
     fcast:
       method: flat

Notes
~~~~~

Naive forecasting methods in HAMLET have the following characteristics:

1. **Simplicity**:
   - Easy to understand and implement
   - Minimal computational requirements
   - No training phase required
   - Serve as useful baselines for more complex methods

2. **Data Requirements**:
   - Require limited historical data
   - Naive and average methods need data from previous days
   - Smoothed method requires future values (typically from other forecasting methods)
   - Flat method requires no historical data

3. **Performance Characteristics**:
   - Work well for time series with strong daily patterns
   - Less effective for time series with complex dependencies
   - Naive and average methods preserve the shape of daily profiles
   - Smoothed method reduces noise but may also smooth out important features

4. **Implementation Details**:
   - Implemented directly in HAMLET without external dependencies
   - Computationally efficient
   - Can be used as fallback methods when more complex methods fail
   - Often used for initialization of more complex forecasting methods

5. **Advantages and Limitations**:
   - **Advantages**:
     - Computational efficiency
     - Simplicity and interpretability
     - Minimal data requirements
     - Robustness to outliers (especially average and smoothed methods)

   - **Limitations**:
     - Limited ability to capture complex patterns
     - No adaptation to changing conditions
     - Cannot incorporate external factors
     - May perform poorly for time series with changing patterns

6. **Practical Considerations**:
   - Start with naive methods before moving to more complex approaches
   - Use naive methods as benchmarks to evaluate more complex methods
   - Consider the trade-off between simplicity and accuracy
   - For daily patterns, naive methods with appropriate offsets can be surprisingly effective