Grids Configuration
===================

This section explains how to configure energy grids in HAMLET, including **grid topology**, **power flow models**, and **regulatory constraints**.

Electricity Grid Configuration
------------------------------
The electricity grid is the **primary energy network** in HAMLET, supporting decentralized energy trading. The configuration allows users to define power flow models, topology methods, and grid restrictions.

### **Activating the Grid**
To enable or disable the electricity grid, set the `active` flag:

.. code-block:: yaml

   electricity:
      active: True  # Enables the electricity grid

If set to `False`, the grid is ignored in the simulation.

### **Power Flow Model**
HAMLET supports **DC power flow** (current implementation) and **AC power flow** (planned feature):

.. code-block:: yaml

   electricity:
      powerflow: dc  # Options: dc (current), ac (planned)

DC power flow is computationally efficient and suitable for most applications.

### **Grid Topology Methods**
Users can define the grid using either:
1. **File-Based Method** (Recommended)
2. **Topology-Based Method** (Advanced)

#### **File-Based Method**
Loads grid topology from an Excel file (`electricity.xlsx`) containing buses, lines, loads, and generators.

.. code-block:: yaml

   generation:
      method: file
      file: electricity.xlsx  # Must be located in the same folder as the configuration file

#### **Topology-Based Method**
Requires a **manual agent-to-bus assignment** in `topology.xlsx`:

.. code-block:: yaml

   generation:
      method: topology
      topology:
         file: topology.xlsx

Grid Constraints and Regulations
--------------------------------
HAMLET supports **grid constraints** and **regulatory mechanisms** such as **§14a EnWG (Energy Industry Act, Germany)**.

### **Applying Grid Restrictions**
Users can apply predefined **grid restrictions** by specifying them in the `apply` list.

.. code-block:: yaml

   restrictions:
      apply: ['enwg_14a']  # List of applied restrictions (empty list means no restrictions)
      max_iteration: 10    # Maximum iterations for applying constraints

If no restrictions are applied, the grid operates without regulatory limits.

### **Energy Industry Act §14a (Germany)**
HAMLET implements **grid management policies** based on **§14a EnWG**, including **variable grid fees** and **direct power control**.

#### **Variable Grid Fees** (Indirect Grid Control)
Adjusts grid fees dynamically to manage peak loads.

.. code-block:: yaml

   variable_grid_fees:
      active: True
      horizon: 86400  # Forecast horizon (seconds)
      update: 3600    # Update frequency (seconds)
      grid_fee_base: 0.07  # Base grid fee in €/kWh

#### **Direct Power Control** (Active Load Management)
Allows grid operators to **limit device power usage** during congestion.

.. code-block:: yaml

   direct_power_control:
      active: True
      method: individual  # Options: individual (device-level), ems (EMS-based)
      threshold: 4200     # Guaranteed minimum power (W)

Heat and Hydrogen Grids
-----------------------
Future versions of HAMLET will support heat and hydrogen grids. These grids can be toggled using:

.. code-block:: yaml

   heat:
      active: False  # Planned feature

   hydrogen:
      active: False  # Planned feature
