Creator API
===========

The Creator API provides classes and functions for setting up simulation scenarios, including agents, markets, and grids.

Module Overview
--------------

The Creator module is responsible for:

- Reading configuration files
- Creating agent instances with specified properties
- Setting up market structures
- Defining grid topologies
- Generating scenario files for the Executor

Key Classes
----------

Creator
~~~~~~~

.. code-block:: python

    from hamlet.creator import Creator

    creator = Creator(path="./configs", name="example_scenario")
    creator.new_scenario_from_configs()

The ``Creator`` class is the main entry point for creating simulation scenarios. It reads configuration files and generates the necessary files for the Executor.

**Key Methods**:

- ``new_scenario_from_configs()``: Creates a new scenario from configuration files
- ``new_scenario_from_template(template_name)``: Creates a new scenario from a template
- ``save_scenario()``: Saves the current scenario to disk

Agent Creation
~~~~~~~~~~~~~

The Creator module provides classes for creating different types of agents:

- ``ConsumerAgent``: For energy consumers
- ``ProducerAgent``: For energy producers
- ``ProsumerAgent``: For entities that both consume and produce energy
- ``StorageAgent``: For energy storage systems

Market Creation
~~~~~~~~~~~~~~

Markets can be configured with different clearing mechanisms:

- ``P2PMarket``: For peer-to-peer trading
- ``PoolMarket``: For centralized market clearing
- ``RetailerMarket``: For traditional retailer-based markets

Grid Creation
~~~~~~~~~~~~

Grid components include:

- ``Bus``: Represents a node in the grid
- ``Line``: Represents a connection between buses
- ``Transformer``: Represents a transformer between voltage levels

Example Usage
------------

Creating a Simple Scenario
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from hamlet.creator import Creator

    # Initialize the Creator
    creator = Creator(path="./configs", name="simple_scenario")
    
    # Create a new scenario from configuration files
    creator.new_scenario_from_configs()

Customizing Agent Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Excel file ``agents.xlsx`` contains all the information about the agents once they are created.
It is intended to be used to fine-tune the agents to your needs without changing anything else.
This is especially useful if you want to compare scenarios as the generatino from the config files
occurs probabilistically and therefore changes all agents.

Extending the Creator
-------------------

Users can extend the Creator functionality by:

1. Creating custom agent types
2. Implementing new market mechanisms
3. Defining specialized grid components

For more detailed information on specific classes and methods, refer to the API reference documentation.