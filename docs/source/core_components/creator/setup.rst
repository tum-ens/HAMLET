General Configuration
=====================

Before running a HAMLET simulation, a general configuration file (`config_setup.yaml`) is required to specify essential parameters. This file defines:

- **Time settings** (simulation duration and timestep)
- **Location parameters** (latitude, longitude, and weather file)
- **File paths** (input data, scenarios, and results storage)

Time Settings
-------------

The simulation requires a starting time, duration, and timestep resolution.

.. code-block:: yaml

    time:
      start: 2021-03-24T00:00:00+01:00  # Simulation start time (ISO 8601 format)
      duration: 1                       # Duration in days
      timestep: 900                      # Minimum timestep (seconds)

- `start`: The start of the simulation, provided in **ISO 8601 format** (YYYY-MM-DDTHH:MM:SS+HH:MM).
- `duration`: Defines how many **days** the simulation runs.
- `timestep`: The minimum time resolution in **seconds** (e.g., 900s = 15 minutes).

Location Settings
-----------------

Location parameters specify where the simulation takes place, affecting weather conditions and geographic constraints.

.. code-block:: yaml

    location:
      latitude: 48.137154   # Latitude (degrees)
      longitude: 11.576124  # Longitude (degrees)
      name: Munich          # Location name (used in result naming)
      altitude: 520         # Altitude (meters)
      weather: weather.ft   # Weather file to use (csv or ft format)

- `latitude` & `longitude`: The **coordinates** of the location.
- `name`: The name of the simulation location (used for labeling results).
- `altitude`: Height above sea level in meters.
- `weather`: Path to the **weather file** (must be in **CSV or Feather (`.ft`) format**).

Paths and Data Storage
----------------------

HAMLET organizes its data across different folders:

.. code-block:: yaml

    paths:
      input: ../03_input_data     # Directory for input data
      scenarios: ../04_scenarios  # Directory for scenario setup
      results: ../05_results      # Directory for storing simulation results

- `input`: Contains all necessary **input data** (e.g., weather, profiles, market parameters).
- `scenarios`: Stores the **scenario configuration files**.
- `results`: Holds all generated **output files** from a simulation run.

How It Works in HAMLET
----------------------

1. **Configuration Parsing**: The Creator module reads the `config_setup.yaml` file.
2. **Simulation Initialization**: The Executor module loads these settings when running a scenario.
3. **Results Naming**: The location and start date influence how results are labeled and stored.

Next Steps
----------

With the **general configuration** set up, the next sections will cover the individual components:

- **Agents**: How different energy system participants are modeled.
- **Markets**: How trading rules and pricing mechanisms are defined.
- **Grids**: How the physical network and constraints are structured.
