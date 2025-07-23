.. _configuration_guide:

Configuration Guide
===================

HAMLET's configuration system allows users to customize simulations through structured YAML files. This guide outlines the key configuration options available for defining scenarios, agents, markets, and grids.

General Setup
-------------
The general setup configuration file defines the **time settings**, **location**, and **paths** for input and output data.

Example:

.. code-block:: yaml

    time:
      start: 2021-03-24T00:00:00+01:00  # Start of the simulation
      duration: 1                        # Simulation duration in days
      timestep: 900                      # Simulation timestep in seconds

    location:
      latitude: 48.137154                # Latitude of the simulation
      longitude: 11.576124               # Longitude
      altitude: 520                      # Altitude in meters
      weather: weather.ft                 # Weather data file

    paths:
      input: ../03_input_data
      scenarios: ../04_scenarios
      results: ../05_results

Agents Configuration
--------------------
The **agents** configuration defines various participant types, their energy demand profiles, and available flexibility resources.

Example (Single Family Home - SFH):

.. code-block:: yaml

    sfh:
      general:
        number_of: 10                    # Number of SFHs to simulate
        occupants: [3, 4]                 # Number of occupants
        efficiency: [55, 150]             # Energy efficiency category

      inflexible-load:
        share: 1                          # Fraction of households with inflexible loads
        demand: [2.5e6, 4e6, 5.5e6]       # Yearly energy demand in Wh

      flexible-load:
        share: 0.5                        # Fraction of households with flexible loads
        demand: [3e6, 5e6, 7e6]           # Yearly demand in Wh
        time_offset: [6, 12, 24]          # Maximum shifting in hours

      pv:
        share: 0.7                        # Share of households with PV systems
        power: [1.5, 3.0]                 # Installed PV power per household

Markets Configuration
---------------------
The **markets** configuration file defines trading mechanisms, pricing schemes, and the interaction between participants.

Example (Local Electricity Market - LEM):

.. code-block:: yaml

    lem_continuous:
      active: True                        # Enable the market
      type: electricity                    # Type of traded energy
      clearing:
        type: ex-ante                     # Market clearing type
        method: pda                        # Periodic double auction
        pricing: uniform                   # Uniform pricing rule
      timing:
        opening: 900                       # Time in seconds before market opens
        duration: 900                      # Duration of one clearing period
        frequency: 900                      # Frequency of market clearings
      pricing:
        retailer:
          energy:
            method: fixed
            fixed:
              price: [0.05, 0.15]         # Sell/buy price in €/kWh

Grid Configuration
------------------
The **grid** configuration defines how power flow calculations are executed and which network constraints are applied.

Example (Electricity Grid):

.. code-block:: yaml

    electricity:
      active: True                        # Enable grid calculations
      powerflow: dc                        # Direct current (DC) power flow
      generation:
        method: file                      # Use predefined grid topology
        file: electricity.xlsx             # Grid definition file
      restrictions:
        apply: ['enwg_14a']               # Apply grid constraints
        max_iteration: 10                  # Max iterations for constraint resolution
      enwg_14a:
        variable_grid_fees:
          active: True
          grid_fee_base: 0.07             # €/kWh base grid fee

Customizing the Configuration
-----------------------------
Users can modify YAML files in the `configs` directory to fit their specific simulation requirements. Common modifications include:

- Adjusting the number and type of agents.
- Enabling/disabling market clearing mechanisms.
- Changing forecasting models for agents and markets.
- Defining new grid constraints.
