Energy Management System
========================

Introduction
------------

The Energy Management System (EMS) is a critical component of HAMLET's simulation framework, responsible for decision-making processes within each agent. The EMS determines how agents interact with markets, grids, and their own energy assets.

Simulation Flow
---------------

The EMS follows a logical flow in the simulation process:

1. **Forecasting & historic data**: First, agents generate forecasts for relevant parameters such as energy demand, generation potential, and market prices as well as historic information such as purchased energy or previous grid commands by the grid operator
2. **Control**: Based on the forecasts and historic data, controllers make decisions about energy usage, storage, and trading.
3. **Trading**: Finally, agents execute their trading strategies to buy or sell energy in the market.

This documentation section is structured to follow this simulation flow, providing detailed information about each step in the process.

Contents
--------

.. toctree::
   :maxdepth: 1
   
   forecasting/overview
   controllers/overview
   trading/overview
