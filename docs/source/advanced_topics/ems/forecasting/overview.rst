Forecasting
===========

Introduction
-----------

Forecasting is the first step in the Energy Management System (EMS) workflow. Before making decisions about energy usage or trading, agents need to predict future conditions such as energy demand, generation potential, and market prices.

HAMLET has a two-level forecasting setup (defined in the `agents.yaml` configuration file):

1. **Global Forecasting Parameters**: Set at the EMS level, defining common parameters like prediction horizon, retraining frequency, and update intervals.
2. **Component-Specific Forecasting**: Each component (load, generation, storage, etc.) can use different forecasting methods tailored to its specific characteristics.

This flexible approach allows for both consistency in forecasting horizons while enabling specialized prediction methods for different types of components.

Forecasting Configuration
------------------------

**Global Forecasting Parameters**

Global forecasting parameters are configured at the EMS level in the agent configuration file:

.. code-block:: yaml

   fcasts:                                     # forecasting settings for all forecasts
       horizon: 86_400                         # forecasting horizon in seconds
       retraining: 86_400                      # period after which the forecast model is retrained
       update: 3_600                           # period after which the forecast model is updated

**Component-Specific Forecasting**

Each component can specify its own forecasting method and parameters:

.. code-block:: yaml

   inflexible-load:
     # other component parameters...
     fcast:
       method: average                         # forecasting method for this component
       average:                                # average forecasting method parameters
        offset: 1                              # offset in days to the current day
                                               # unit: days
       # other method-specific parameters...

Forecasting Methods
------------------

HAMLET supports various forecasting methods with different levels of complexity and accuracy. The choice of forecasting method can significantly impact the performance of the control and trading strategies.

.. toctree::
   :maxdepth: 1

   models/naive_methods
   models/statistical_models
   models/machine_learning

Forecasting in the Simulation Flow
---------------------------------

In the HAMLET simulation, forecasting typically occurs at the beginning of each decision cycle:

1. Agents collect historical data and current system state
2. Forecasting methods are applied to predict future conditions for each component
3. These forecasts are then passed to controllers for decision-making

The accuracy of forecasts directly impacts the quality of decisions made by controllers. More sophisticated forecasting methods generally provide better predictions but may require more computational resources and historical data.