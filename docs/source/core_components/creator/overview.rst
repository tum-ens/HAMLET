Creator Module
==============

The **Creator** module in HAMLET is responsible for setting up the simulation environment. It defines agents, markets, and grids based on user configurations, ensuring that all components are initialized correctly before execution.

.. toctree::
   :maxdepth: 1
   :caption: Creator

   setup
   agents/agents
   agents/configuration
   markets/markets
   markets/configuration
   grids/grids
   grids/configuration

Role of the Creator
-------------------

The Creator module serves the following functions:

- **Scenario Setup**: Reads and processes configuration files to define the simulation environment.
- **Agent Initialization**: Defines different types of agents, including their behaviors, energy needs, and decision-making rules.
- **Market Setup**: Configures market structures, pricing mechanisms, and trading rules.
- **Grid Definition**: Establishes the physical infrastructure for energy transactions, including electricity, heat, and hydrogen grids.

Key Components
--------------

The Creator module consists of three submodules:

1. **Agents**: Represents participants in the simulation, such as households, industries, and grid operators.
2. **Markets**: Facilitates trading between agents, enabling various market mechanisms.
3. **Grids**: Defines the physical network where energy flows occur, ensuring that technical constraints are respected.

Workflow of the Creator
-----------------------

1. Users provide configuration files specifying agents, market rules, and grid parameters.
2. The Creator processes these configurations and generates the required data structures.
3. The initialized scenario is stored and made available for execution.

There are three ways to create scenarios:
1. From configs alone: Create a scenario from the config files alone. This will create the agents probabilistically. In these scenarios there is no grid.
2. From configs and topology: Create a scenario from the config files and topology file. This will create the agents probabilistically which you can then assign afterwards to buses in the grid topology.
3. From a grid file: The file contains all information about the grid and the agents at each bus.

In all cases you can fine tune the agents using the `agents.xlsx` file that is created before creating the final scenario.

Next Steps
----------

Each of the submodules—**Agents**, **Markets**, and **Grids**—will be covered in detail in the following sections.

