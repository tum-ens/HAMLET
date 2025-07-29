Component Models
================

This page provides concise explanations of how component models work specifically in the Real-Time Control (RTC) context.

As examples, these are component models that are modeled within HAMLET:

1. **Battery Storage**

   In RTC, batteries are modeled with state-of-charge constraints and charging/discharging efficiency. The key decision variables are charging/discharging power at the current timestep.

   - **Simplified time-coupling**: Only considers a short horizon for state-of-charge evolution
   - **Current state focus**: Uses the current battery state as the initial condition
   - **Operational constraints**: Enforces power limits and prevents simultaneous charging/discharging

2. **PV Systems**

   RTC models PV systems with available power as an input parameter and actual power utilization as a decision variable.

   - **Real-time data integration**: Uses current measurements of solar irradiance
   - **Curtailment decisions**: Determines whether to use or curtail available PV power
   - **No forecasting complexity**: Simplifies PV modeling compared to forecast-based controllers

3. **Heat Pumps**

   Modeled with COP (Coefficient of Performance) as a parameter that converts electrical input to thermal output.

   - **Current COP**: Uses the current operating conditions to determine efficiency
   - **Binary operation**: Often modeled with on/off decisions for computational efficiency
   - **Simplified thermal dynamics**: Focuses on immediate heating/cooling capacity

4. **Grid Connection**

   Models import/export power flows with current price signals as parameters.

   - **Current prices**: Uses current electricity prices rather than complex price forecasts
   - **Power balance**: Ensures instantaneous balance between generation and consumption
   - **Grid constraints**: Enforces power limits for grid connection

The RTC implementation emphasizes:

- **Computational efficiency**: Formulated for real-time solving (seconds to minutes)
- **Current state focus**: Uses current system states rather than complex forecasts
- **Simplified constraints**: Focuses on immediate operational limits
- **Balance between optimality and solution speed**: Trades off some optimality for faster decisions

These simplified models enable RTC to make quick decisions based on current conditions, which is essential for real-time energy management.