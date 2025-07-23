Component Models
===============

Introduction
-----------

Forecast-Based Controllers (FBCs) in HAMLET incorporate various component models to represent the physical behavior and constraints of energy system elements over a time horizon. This page describes how different components are modeled within the FBC optimization problem.

Each component model defines:
- Decision variables related to the component for each timestep in the horizon
- Constraints that ensure physically realistic behavior, including temporal coupling
- Contribution to the objective function

Battery Storage
-------------

Battery storage is a key component in many energy systems, allowing for temporal shifting of energy use. In FBCs, battery models must capture the evolution of the state of charge over time.

**Decision Variables** (for each timestep :math:`t` in the horizon :math:`[0, T-1]`):
- :math:`P_{\text{charge}}(t)`: Charging power at timestep :math:`t`
- :math:`P_{\text{discharge}}(t)`: Discharging power at timestep :math:`t`
- :math:`E(t)`: Energy stored in the battery at timestep :math:`t`

**Constraints**:

1. Power limits (for each timestep :math:`t`):

   .. math::

      0 \leq P_{\text{charge}}(t) \leq P_{\text{charge}}^{\max}
      
      0 \leq P_{\text{discharge}}(t) \leq P_{\text{discharge}}^{\max}

2. Energy balance (temporal coupling):

   .. math::

      E(t) = E(t-1) + \eta_{\text{charge}} \cdot P_{\text{charge}}(t) \cdot \Delta t - \frac{P_{\text{discharge}}(t) \cdot \Delta t}{\eta_{\text{discharge}}} - E_{\text{loss}}(t)

   where:
   - :math:`\eta_{\text{charge}}` is the charging efficiency
   - :math:`\eta_{\text{discharge}}` is the discharging efficiency
   - :math:`\Delta t` is the timestep duration
   - :math:`E_{\text{loss}}(t)` represents self-discharge losses

3. Energy capacity limits:

   .. math::

      E^{\min} \leq E(t) \leq E^{\max}

4. No simultaneous charging and discharging (using binary variable :math:`b(t)`):

   .. math::

      P_{\text{charge}}(t) \leq P_{\text{charge}}^{\max} \cdot b(t)
      
      P_{\text{discharge}}(t) \leq P_{\text{discharge}}^{\max} \cdot (1 - b(t))

5. Terminal state constraint (optional):

   .. math::

      E(T-1) \geq E^{\text{terminal}}

Photovoltaic (PV) System
----------------------

PV systems convert solar radiation into electricity. In FBCs, PV models use forecasts of solar radiation over the optimization horizon.

**Decision Variables** (for each timestep :math:`t`):
- :math:`P_{\text{pv}}(t)`: PV power output at timestep :math:`t`
- :math:`P_{\text{curtailed}}(t)`: Curtailed PV power at timestep :math:`t`

**Constraints**:

1. Maximum power output based on forecasted solar radiation:

   .. math::

      P_{\text{pv}}(t) + P_{\text{curtailed}}(t) = P_{\text{available}}(t)

   where :math:`P_{\text{available}}(t)` is the forecasted available PV power at timestep :math:`t`

2. Non-negativity:

   .. math::

      P_{\text{pv}}(t) \geq 0
      
      P_{\text{curtailed}}(t) \geq 0

Heat Pump
--------

Heat pumps convert electricity into heat with a coefficient of performance (COP) that depends on operating conditions. In FBCs, heat pump models may include thermal dynamics of the building.

**Decision Variables** (for each timestep :math:`t`):
- :math:`P_{\text{hp}}(t)`: Electrical power consumed by the heat pump at timestep :math:`t`
- :math:`Q_{\text{hp}}(t)`: Heat output of the heat pump at timestep :math:`t`

**Constraints**:

1. Heat output relation:

   .. math::

      Q_{\text{hp}}(t) = \text{COP}(t) \cdot P_{\text{hp}}(t)

   where :math:`\text{COP}(t)` is the forecasted coefficient of performance at timestep :math:`t`

2. Power limits:

   .. math::

      0 \leq P_{\text{hp}}(t) \leq P_{\text{hp}}^{\max}

3. Ramp rate limits (temporal coupling):

   .. math::

      -R_{\text{down}} \leq P_{\text{hp}}(t) - P_{\text{hp}}(t-1) \leq R_{\text{up}}

   where :math:`R_{\text{down}}` and :math:`R_{\text{up}}` are the ramp-down and ramp-up limits

Electric Vehicle (EV)
------------------

Electric vehicles can be modeled as controllable loads with specific charging requirements and forecasted availability.

**Decision Variables** (for each timestep :math:`t`):
- :math:`P_{\text{ev}}(t)`: EV charging power at timestep :math:`t`
- :math:`E_{\text{ev}}(t)`: Energy stored in the EV battery at timestep :math:`t`

**Constraints**:

1. Power limits based on forecasted availability:

   .. math::

      0 \leq P_{\text{ev}}(t) \leq P_{\text{ev}}^{\max} \cdot A_{\text{ev}}(t)

   where :math:`A_{\text{ev}}(t)` is the forecasted availability of the EV (1 if connected, 0 otherwise)

2. Energy balance (temporal coupling):

   .. math::

      E_{\text{ev}}(t) = E_{\text{ev}}(t-1) + \eta_{\text{ev}} \cdot P_{\text{ev}}(t) \cdot \Delta t - E_{\text{trip}}(t)

   where:
   - :math:`\eta_{\text{ev}}` is the charging efficiency
   - :math:`E_{\text{trip}}(t)` is the forecasted energy consumed by trips during timestep :math:`t`

3. Energy capacity limits:

   .. math::

      E_{\text{ev}}^{\min} \leq E_{\text{ev}}(t) \leq E_{\text{ev}}^{\max}

4. Charging requirements at departure times:

   .. math::

      E_{\text{ev}}(t_{\text{departure}}) \geq E_{\text{ev}}^{\text{required}}

Building Thermal Model
-------------------

Building thermal dynamics are particularly important in FBCs to model the evolution of indoor temperature over time.

**Decision Variables** (for each timestep :math:`t`):
- :math:`T_{\text{indoor}}(t)`: Indoor temperature at timestep :math:`t`
- :math:`Q_{\text{heating}}(t)`: Heating power at timestep :math:`t`
- :math:`Q_{\text{cooling}}(t)`: Cooling power at timestep :math:`t`

**Constraints**:

1. Thermal dynamics (temporal coupling):

   .. math::

      T_{\text{indoor}}(t) = T_{\text{indoor}}(t-1) + \frac{\Delta t}{C} \left( Q_{\text{heating}}(t) - Q_{\text{cooling}}(t) + \sum_{i} U_i A_i (T_{\text{outdoor}}(t) - T_{\text{indoor}}(t-1)) + Q_{\text{internal}}(t) + Q_{\text{solar}}(t) \right)

   where:
   - :math:`C` is the thermal capacitance of the building
   - :math:`U_i` is the heat transfer coefficient of building element :math:`i`
   - :math:`A_i` is the area of building element :math:`i`
   - :math:`T_{\text{outdoor}}(t)` is the forecasted outdoor temperature
   - :math:`Q_{\text{internal}}(t)` is the forecasted internal heat gain
   - :math:`Q_{\text{solar}}(t)` is the forecasted solar heat gain

2. Comfort constraints:

   .. math::

      T_{\text{indoor}}^{\min}(t) \leq T_{\text{indoor}}(t) \leq T_{\text{indoor}}^{\max}(t)

   where the comfort bounds may vary over time (e.g., day/night setpoints)

3. Heating and cooling power limits:

   .. math::

      0 \leq Q_{\text{heating}}(t) \leq Q_{\text{heating}}^{\max}
      
      0 \leq Q_{\text{cooling}}(t) \leq Q_{\text{cooling}}^{\max}

Integration in the Optimization Problem
------------------------------------

These component models are integrated into the FBC optimization problem by:

1. Including all component-specific decision variables for all timesteps in the overall decision variable vector
2. Adding all component-specific constraints to the constraint set
3. Incorporating component-specific costs and benefits into the objective function

The temporal coupling constraints are particularly important in FBCs, as they ensure that the decisions made at one timestep properly affect the system state at future timesteps.

Differences from RTC Component Models
----------------------------------

The main differences between FBC and RTC component models are:

1. **Time Horizon**: FBC models include variables and constraints for multiple timesteps in the future.

2. **Temporal Coupling**: FBC models explicitly model how decisions at one timestep affect the system state at future timesteps.

3. **Forecast Integration**: FBC models incorporate forecasts of external conditions (e.g., weather, prices, availability) over the entire optimization horizon.

4. **Terminal Constraints**: FBC models may include constraints on the terminal state to ensure desirable conditions at the end of the horizon.

The specific implementation details depend on the chosen optimization solver and framework, which are described in the implementation section.