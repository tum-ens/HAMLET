
Agent Configuration
===================

Agents in HAMLET are configured through YAML files, allowing users to define their behavior, energy consumption, production capacities, forecasting methods, and market participation. This section explains the structure of the agent configuration file.

General Settings
----------------
Each agent type (e.g., SFH, MFH, C&I) is defined under a specific category. The general configuration includes:

- **Number of Agents** (`number_of`): Defines how many agents of this type will be created.
- **Basic Parameters** (`parameters`): Specifies agent attributes such as number of occupants, living area, and building efficiency.
- **Market Participation** (`market_participant_share`): Determines what fraction of the agents participate in the market.

Example:
.. code-block:: yaml

   sfh:
     general:
       number_of: 10
       parameters:
         occupants: [3, 4]
         area: [100, 130]
         efficiency: [55, 150]
       market_participant_share: 1

Energy Demand & Supply
----------------------
Agents can consume and produce energy through different means:

### **Inflexible Load**
Represents the baseline energy demand that cannot be shifted or controlled.

- **Annual Demand (`demand`)**: Specifies total energy consumption in Wh/a.
- **Deviation (`demand_deviation`)**: Allows random variation to simulate different households.

Example:
.. code-block:: yaml

   inflexible-load:
     share: 1
     num: [1]
     sizing:
       distribution: [0.2, 0.6, 0.2]
       demand: [2.5e6, 4e6, 5.5e6]
       demand_deviation: [5e5, 1e6, 5e5]

### **Flexible Load**
Represents shiftable demand, such as electric heating, EV charging, or industrial processes.

- **Time Offset (`time_offset`)**: Defines how long the demand can be postponed.
- **Forecasting Method (`fcast`)**: Specifies how the agent predicts its energy demand.

Example:
.. code-block:: yaml

   flexible-load:
     share: 0.5
     num: [1]
     sizing:
       distribution: [0.4, 0.4, 0.2]
       demand: [2.5e6, 4e6, 5.5e6]
       time_offset: [6, 12, 24]

### **Renewable Generation**
Agents can own **photovoltaic (PV) systems**, **wind turbines**, or **fixed generators**.

- **Installed Power (`power`)**: Defines the system's capacity.
- **Forecasting (`fcast`)**: Configures methods for predicting generation.

Example:
.. code-block:: yaml

   pv:
     share: 0.7
     num: [1]
     sizing:
       power: [1.5, 1.5]
       orientation: [0, -20]
       angle: [40, 50]
       controllable: [false, false]
     fcast:
       method: naive

Energy Storage
--------------
Agents may have energy storage systems such as **batteries** or **heat storage**.

- **Capacity (`capacity`)**: Defines storage size in Wh.
- **Efficiency (`efficiency`)**: Represents charge-discharge efficiency.
- **State of Charge (`soc`)**: Initial energy stored in the system.

Example:
.. code-block:: yaml

   battery:
     share: 0.8
     num: [1]
     sizing:
       power: [1]
       capacity: [1]
       efficiency: [0.95]
       soc: [0.1]

Electric Vehicles (EVs)
-----------------------
EVs are modeled as mobile storage units with specific charging strategies.

- **Charging Power (`charging_home`, `charging_AC`, `charging_DC`)**: Maximum power for each charging location.
- **Vehicle-to-Grid (`v2g`)**: Defines whether the EV can discharge back into the grid.

Example:
.. code-block:: yaml

   ev:
     share: 0.5
     num: [1]
     sizing:
       capacity: [5e4, 7.5e4, 1e5]
       charging_home: [7.2e3, 7.2e3, 1.1e4]
       v2g: [false, false, false]

Energy Management Systems (EMS)
-------------------------------
Each agent can be equipped with an **Energy Management System (EMS)** to optimize energy usage.

- **Real-Time Controller (`rtc`)**: Optimizes energy dispatch at each timestep.
- **Forecast-Based Controller (`fbc`)**: Plans energy use based on future predictions.

Example:
.. code-block:: yaml

   ems:
     controller:
       rtc:
         method: linopy
         linopy:
           solver: gurobi
           time_limit: 120
       fbc:
         method: linopy
         horizon: 86400

Market Participation
--------------------
Agents can participate in energy markets using different **trading strategies**.

- **Strategy (`strategy`)**: Defines how agents place bids and offers.
- **Trading Horizon (`horizon`)**: Specifies how far ahead agents plan trades.

Example:
.. code-block:: yaml

   market:
     strategy: linear
     horizon: [86400, 72000, 57600, 43200]
     linear:
       steps_to_final: 1
       steps_from_init: 0

Summary
-------
- Agents in HAMLET represent energy consumers, producers, and flexible assets.
- YAML configuration defines their behavior, forecasting, and market participation.
- Different forecasting models and control strategies allow detailed customization.
