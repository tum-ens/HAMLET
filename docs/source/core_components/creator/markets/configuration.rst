Market Configuration
====================

The market configuration in HAMLET defines how energy trading occurs, including market mechanisms, clearing types, pricing structures, and grid interactions. This section explains how to configure different markets in the simulation.

General Market Settings
-----------------------
Each market is defined in the configuration file under a **unique market name** and contains the following key parameters:

.. code-block:: yaml

   lem_continuous:    # Market name (must be unique in the region)
      active: True    # Enable (True) or disable (False) the market
      type: electricity  # Type of energy market (e.g., electricity)

Market Clearing Options
-----------------------
The **clearing type** determines whether trades are settled before or after energy delivery.

.. code-block:: yaml

   clearing:
      type: ex-ante     # Options: ex-ante (before delivery), ex-post (after delivery)
      method: pda       # Options: None, pda (Periodic Double Auction), community (ex-post)
      pricing: uniform  # Options: uniform (same price for all), discriminatory (different prices)
      coupling: None    # Options: None, above_c, above_l, below_c, below_l

- **Ex-Ante Clearing**: Matches bids/offers before energy is delivered.
- **Ex-Post Clearing**: Settles transactions after energy usage (planned feature).
- **Market Coupling**: Determines whether **uncleared bids** are passed to other markets.

Market Timing
-------------
The **timing** section defines when and how often the market clears:

.. code-block:: yaml

   timing:
      start: 0           # Start time of the market (Unix timestamp or timedelta)
      opening: 900       # Time until the first clearing (seconds)
      horizon: [0, 86400] # Market clearing horizon (start and end in seconds)
      duration: 900      # Duration of each energy delivery period (seconds)
      frequency: 900     # Interval between market clearings (seconds)
      closing: 900       # Time before delivery when last auction occurs (seconds)
      settling: continuous  # Options: continuous (settle each step), period (settle for entire period)

Retailer and Pricing Configuration
----------------------------------
If no local trade is possible, agents **buy or sell energy through a retailer**. Retail pricing can be fixed or time-varying.

.. code-block:: yaml

   pricing:
      retailer:
         energy:
            method: fixed  # Options: fixed, file
            fixed:
               price: [0.05, 0.15]  # Prices for selling and buying (€/kWh)
               quantity: [1e5, 1e5] # Max trade quantity (Wh)
               quality: 0  # Energy quality label (market-defined)
            file:
               file: retailer_prices.csv  # External price file

Grid Fees and Levies
--------------------
Markets can **include grid fees, levies, and balancing costs**.

**Grid Fees**
.. code-block:: yaml

   grid:
      method: fixed  # Options: fixed, file
      fixed:
         market: [0.04, 0]  # Grid fees for local market trades (€/kWh)
         retail: [0.08, 0]  # Grid fees for retailer transactions (€/kWh)
      file:
         file: grid_fees.csv  # External grid fee file

.. note::
   Grid fees and levies are written ``[buying, selling]`` -- the opposite order from the energy
   and balancing blocks, which are written ``[selling, buying]``. Both spellings are accepted
   and normalised by the Creator, so existing configuration files keep their meaning. Whichever
   order a block uses, the charge belongs on the buying side: fees and levies are owed on the
   energy an agent draws from the grid, not on what it feeds in.

   Files supplied with ``method: file`` are **not** normalised -- they are read by column name.
   Their columns must already follow the convention the executor uses, in which ``_out`` is the
   direction the agent pays for, and their values must be integers in 0.01 ct/kWh rather than
   €/kWh.

**Levies**
.. code-block:: yaml

   levies:
      method: fixed  # Options: fixed, file
      fixed:
         price: [0.18, 0]  # Levies, buying first (€/kWh)
      file:
         file: levie
