Grids Overview
==============

The grid component in HAMLET models the physical energy infrastructure that supports electricity, heat, and hydrogen systems. This section provides an overview of grid types, power flow models, and available constraints.

Grid Types
----------
HAMLET supports multiple energy grids, each representing a different energy carrier:

- **Electricity Grid**: Simulates power distribution and flow constraints.
- **Heat Grid**: Represents district heating networks (planned feature).
- **Hydrogen Grid**: Models hydrogen transportation and supply (planned feature).

The configuration file enables activating or deactivating each grid type:

.. code-block:: yaml

   electricity:
      active: True    # Enable (True) or disable (False) electricity grid

   heat:
      active: False   # Currently inactive (planned feature)

   hydrogen:
      active: False   # Currently inactive (planned feature)

Power Flow Models
-----------------
Each grid type uses a specific **power flow model** to compute energy flows and constraints.

For **electricity grids**, HAMLET supports:
- **DC Power Flow** (currently implemented): Approximates power flows using a linearized DC model.
- **AC Power Flow** (future implementation): Models full alternating current behavior, including voltage and reactive power effects.

.. code-block:: yaml

   electricity:
      powerflow: dc  # Options: dc (current implementation), ac (future implementation)

Grid Representation Methods
---------------------------
Users can define the **grid topology** using one of two methods:

1. **File-Based Method** (Recommended)
   - Reads an Excel file (`electricity.xlsx`) containing **bus, line, load, and generator data**.
   - Used for predefined grid configurations.

   .. code-block:: yaml

      generation:
         method: file
         file: electricity.xlsx  # File must contain grid topology

2. **Topology-Based Method** (Advanced)
   - The user first generates a scenario, then manually assigns agents to grid buses.
   - Requires a separate `topology.xlsx` file.

   .. code-block:: yaml

      generation:
         method: topology
         topology:
            file: topology.xlsx

Grid Restrictions and Regulations
---------------------------------
HAMLET allows users to apply **grid constraints** and **regulatory policies** that affect energy trading and power flow.

.. code-block:: yaml

   restrictions:
      apply: ['enwg_14a']  # List of applied restrictions (empty list means no restrictions)
      max_iteration: 10    # Maximum iterations for applying grid constraints

### **Energy Industry Act §14a (Germany)**
HAMLET includes **grid management rules** based on **§14a EnWG (Energy Industry Act, Germany)**:

#### **Variable Grid Fees** (Indirect Grid Control)
Dynamic grid fees are applied based on system load conditions.

.. code-block:: yaml

   variable_grid_fees:
      active: True
      horizon: 86400  # Forecast horizon (seconds)
      update: 3600    # Update frequency (seconds)
      grid_fee_base: 0.07  # Base grid fee in €/kWh

#### **Direct Power Control** (Active Load Management)
Allows grid operators to **control devices** to ensure system stability.

.. code-block:: yaml

   direct_power_control:
      active: True
      method: individual  # Options: individual (device-level), ems (EMS-based)
      threshold: 4200     # Guaranteed minimum power (W)

Next Steps
----------
- Learn how **agents interact with the grid** in :doc:`agent_configuration`.
- Explore **market-grid integration** in :doc:`market_configuration`.
