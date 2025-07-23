Special Forecasting Methods
==========================

Introduction
-----------

Special forecasting methods are designed for specific components or scenarios where standard statistical or machine learning approaches may not be suitable. These methods leverage domain-specific knowledge and physical models to generate more accurate forecasts for particular types of components.

Weather
-------------

Weather-based forecasting uses physical models and weather data to predict the output of renewable energy sources.

Available for
~~~~~~~~~~~~~

PV | Wind | Heat Pump

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~~~~~

The mathematical formulation varies by component type:

For PV systems:

.. math::

   P_{PV} = f(GHI, DHI, DNI, T_{air}, \text{location}, \text{orientation})

where:
   - GHI: Global Horizontal Irradiance
   - DHI: Diffuse Horizontal Irradiance
   - DNI: Direct Normal Irradiance
   - T_{air}: Air temperature
   - location: Geographic coordinates and altitude
   - orientation: Tilt and azimuth of the PV panels

For Wind turbines:

.. math::

   P_{Wind} = f(v_{wind}, \rho_{air}, \text{power curve})

where:
   - v_{wind}: Wind speed
   - \rho_{air}: Air density (affected by temperature, pressure, humidity)
   - power curve: The turbine's characteristic power output at different wind speeds

Configuration
~~~~~~~~~~~~~

Weather-based forecasting methods can be configured in the agent configuration file:

.. code-block:: yaml

   pv:  # or wind:
     fcast:
       method: weather
       weather:
         # No additional parameters required as the method uses
         # the component specifications and weather data

Notes
~~~~~

The weather-based forecasting method relies on accurate weather forecasts and detailed component specifications. For PV systems, it uses the pvlib library to model the PV system's behavior based on solar position and weather conditions. For wind turbines, it uses the windpowerlib library to model the turbine's output based on wind conditions.

Arrival
-------------

Arrival provides predictions about electric vehicle availability and energy consumption based on the current availability status of the vehicle.

Available for
~~~~~~~~~~~~~

EV

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~~~~~

This method uses a simple logical approach:

If EV is currently available (availability = 1):
   - Use perfect forecast for the prediction period
   - Accurately predicts when the EV will depart

If EV is currently unavailable (availability = 0):
   - Forecast zero availability and energy consumption for the entire prediction period
   - Does not predict when the EV will arrive

Configuration
~~~~~~~~~~~~~

EV Arrival forecasting methods can be configured in the agent configuration file:

.. code-block:: yaml

   ev:
     fcast:
       method: arrival  # Options: perfect, arrival, ev_close, rfr

Notes
~~~~~

The Arrival method provides a simple but effective forecast based on the current availability status. It works well when the EV's arrival and departure times are unpredictable, but once the EV is available, its departure schedule is known.