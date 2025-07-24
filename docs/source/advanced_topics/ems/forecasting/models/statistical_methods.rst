Statistical Forecasting Methods
===========================

Introduction
-----------

Statistical forecasting methods use historical data patterns and statistical techniques to predict future values. These methods range from simple averaging approaches to more sophisticated time series models. Statistical methods strike a balance between computational simplicity and forecasting accuracy, making them widely used in energy management systems.

In the spectrum of forecasting complexity, statistical methods occupy the middle ground between naive approaches and complex machine learning models. They can capture temporal patterns and seasonality while remaining computationally efficient and interpretable.

ARIMA
-----

The AutoRegressive Integrated Moving Average (ARIMA) model is a generalization of the ARMA model that incorporates differencing to handle non-stationary data. ARIMA models are versatile and can capture a wide range of time series patterns.

Available for
~~~~~~~~~~~~

The ARIMA method is available for most components in HAMLET:

Inflexible Load | Heat Demand | DHW

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \phi_p(B) (1 - B)^d y_t = \theta_q(B) \varepsilon_t

where:

   - :math:`\phi_p(B)` is the AR operator of order p
   - :math:`\theta_q(B)` is the MA operator of order q
   - :math:`(1 - B)^d` is the differencing operator of order d
   - :math:`\varepsilon_t` is white noise

Configuration
~~~~~~~~~~~~

ARIMA forecasting methods can be configured in the agent configuration file:

.. code-block:: yaml

   inflexible-load:
     fcast:
       method: arima  # Options: perfect, naive, average, smoothed, sarma, rfr, cnn, rnn, arima
       arima:
         order: [1, 0, 0]  # order (p,d,q) of the model
         days: 3  # past days that are used to train the arima model

SARMA
-----

The Seasonal AutoRegressive Moving Average (SARMA) model is a sophisticated time series forecasting method that captures both autoregressive patterns and moving average effects, along with seasonal components. This method is particularly effective for data with complex temporal patterns and multiple seasonality.

Available for
~~~~~~~~~~~~

The SARMA method is available for most components in HAMLET:

Inflexible Load | Heat Demand | DHW

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \phi_p(B) \Phi_P(B^s) (1 - B^s)^D (1 - B)^d y_t = \theta_q(B) \Theta_Q(B^s) \varepsilon_t

where:

   - :math:`\phi_p(B)` is the non-seasonal AR operator of order p
   - :math:`\Phi_P(B^s)` is the seasonal AR operator of order P
   - :math:`\theta_q(B)` is the non-seasonal MA operator of order q
   - :math:`\Theta_Q(B^s)` is the seasonal MA operator of order Q
   - :math:`(1 - B)^d` is the non-seasonal differencing operator of order d
   - :math:`(1 - B^s)^D` is the seasonal differencing operator of order D
   - :math:`s` is the seasonal period
   - :math:`\varepsilon_t` is white noise

Configuration
~~~~~~~~~~~~

SARMA forecasting methods can be configured in the agent configuration file:

.. code-block:: yaml

   inflexible-load:
     fcast:
       method: sarma  # Options: perfect, naive, average, smoothed, sarma, rfr, cnn, rnn, arima
       sarma:
         order: [2, 0, 2, 2, 0, 0, 96, 2, 0, 0, 672]  # order of double seasonal arma model

Notes
~~~~~

Statistical forecasting methods in HAMLET have the following characteristics:

1. **Data Requirements**: These methods require historical data, with the amount varying by method:
   - Average: Requires data from multiple previous days
   - Smoothed: Requires future values (only applicable in simulation)
   - SARMA/ARIMA: Require sufficient historical data to estimate model parameters

2. **Computational Considerations**:
   - Average and Smoothed methods are computationally efficient
   - SARMA and ARIMA models require parameter estimation, which can be more computationally intensive
   - Model fitting is typically done periodically rather than at every timestep

3. **Handling Seasonality**:
   - Average methods implicitly capture daily seasonality by using data from previous days
   - SARMA explicitly models multiple seasonal patterns (e.g., daily and weekly)
   - ARIMA can capture seasonality through seasonal differencing or seasonal terms

4. **Implementation Details**:
   - For SARMA and ARIMA, HAMLET uses the statsmodels Python package
   - Models are retrained periodically based on the retraining parameter in the configuration
   - Forecasts are updated at intervals specified by the update parameter

5. **Advantages and Limitations**:
   - Statistical methods provide a good balance between simplicity and accuracy
   - They can capture temporal patterns and seasonality
   - They may struggle with non-linear relationships or external factors (e.g., weather)
   - They typically outperform naive methods but may be less accurate than machine learning approaches for complex patterns