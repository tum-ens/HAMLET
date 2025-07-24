Executor API
============

The Executor API provides classes and functions for running simulation scenarios created by the Creator module. It handles the time-stepping mechanism, agent interactions, market clearing, and grid operations.

Module Overview
--------------

The Executor module is responsible for:

- Loading scenario files created by the Creator
- Managing the simulation timeline
- Executing agent decision-making processes
- Facilitating market interactions and clearing
- Simulating grid operations and constraints
- Recording simulation results for analysis

Key Classes
----------

Executor
~~~~~~~~

.. code-block:: python

    from hamlet.executor import Executor

    executor = Executor(path_scenario="./scenarios/example_scenario")
    executor.run()

The ``Executor`` class is the main entry point for running simulations. It loads a scenario and executes it according to the specified parameters.

**Key Methods**:

- ``run()``: Executes the simulation
- ``load_scenario()``: Loads a scenario from disk
- ``save_results()``: Saves simulation results to disk
- ``set_num_workers(num)``: Sets the number of parallel workers for execution

Agent Execution
~~~~~~~~~~~~~~

The Executor module includes classes for executing different types of agent behaviors. For example:

- ``Sfh``: Single-family household
- ``Ctsp``: Commerce, trading, service & public
- ``Industry``: Industry
- ``Producer``: Producer (can also operate storage systems)
- ``Storage``: Storage operator (only provides storage flexibility)

All classes inherit from ``AgentBase`` which handles the standard execution tasks.

Each agent executor handles:
- Forecasting future energy needs or production
- Making control decisions for energy assets
- Formulating bids and offers for markets
- Responding to grid signals and constraints

Market Execution
~~~~~~~~~~~~~~~

Market execution components include:

- ``Electricity``: Energy-only electricity markets
- ``Heat``: Energy-only heat markets (currently only placeholder)
- ``Hydrogen``: Energy-only hydrogen markets (currently only placeholder)

All classes inherit from ``MarketBase`` which handles the standard execution tasks.

Each market executor handels:
- Clearing
- Settling

Grid Execution
~~~~~~~~~~~~~

Grid execution components include:

- ``Electricity``: Electricity grids
- ``Heat``: Heat grids
- ``Hydrogen``: Hydrogen grids

All classes inherit from ``GridBase`` which handles the standard execution tasks.

Each grid executor handels:
- Power flow calculations
- Threshold detections
- Direct and indirect grid control

Example Usage
------------

Running a Basic Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from hamlet.executor import Executor

    # Initialize the Executor with a scenario path
    executor = Executor(path_scenario="./scenarios/example_scenario")
    
    # Run the simulation
    executor.run()

Parallel Execution
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from hamlet.executor import Executor

    # Initialize the Executor with parallel processing
    executor = Executor(
        path_scenario="./scenarios/example_scenario",
        num_workers=4  # Use 4 parallel workers
    )
    
    # Run the simulation in parallel
    executor.run()

Utilities
--------

The Executor module includes various utility functions and classes that support the simulation process. These utilities are organized into the following categories based on their functionality:

Controller Utilities
~~~~~~~~~~~~~~~~~~~

Controller utilities help agents make decisions about energy usage and control:

- ``OptimalController``: Implements optimization-based control strategies
- ``RuleBasedController``: Implements rule-based control strategies
- ``PIDController``: Implements PID control for energy systems

Database Utilities
~~~~~~~~~~~~~~~~

Database utilities handle data storage and retrieval:

- ``DatabaseConnector``: Manages connections to databases
- ``DataReader``: Reads data from databases
- ``DataWriter``: Writes data to databases

Forecasting Utilities
~~~~~~~~~~~~~~~~~~~

Forecasting utilities help agents predict future values:

- ``DemandForecaster``: Predicts future energy demand
- ``GenerationForecaster``: Predicts future energy generation
- ``PriceForecaster``: Predicts future market prices
- ``WeatherForecaster``: Predicts future weather conditions

Grid Restriction Utilities
~~~~~~~~~~~~~~~~~~~~~~~~

Grid restriction utilities handle grid constraints:

- ``GridConstraintChecker``: Checks for grid constraint violations
- ``GridConstraintSolver``: Resolves grid constraint violations
- ``GridRestrictionManager``: Manages grid restrictions

Tasks Execution Utilities
~~~~~~~~~~~~~~~~~~~~~~~

Tasks execution utilities manage the execution of simulation tasks:

- ``TaskScheduler``: Schedules tasks for execution
- ``TaskExecutor``: Executes scheduled tasks
- ``ParallelTaskManager``: Manages parallel task execution

Trading Utilities
~~~~~~~~~~~~~~~

Trading utilities implement various trading strategies:

- ``ZeroIntelligence``: Implements a simple random trading strategy
- ``LinearBidding``: Implements a linear price adjustment strategy
- ``RetailerBased``: Implements a strategy based on retailer prices
- ``OrderBook``: Manages bids and offers in markets

.. code-block:: python

    from hamlet.executor.utilities.trading.strategies import LinearBidding

    # Create a linear bidding strategy
    strategy = LinearBidding(
        steps_to_final=10,
        steps_from_init=5
    )
    
    # Generate bids and offers
    bids_offers = strategy.generate_bids_offers(
        forecast_buy_prices,
        forecast_sell_prices,
        energy_demand,
        energy_generation
    )

Example Usage of Utilities
-------------------------

Using Forecasting Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from hamlet.executor.utilities.forecasts import DemandForecaster

    # Create a demand forecaster
    forecaster = DemandForecaster(
        method="arima",
        historical_data=historical_demand,
        forecast_horizon=24  # hours
    )
    
    # Generate a forecast
    forecast = forecaster.forecast()
    
    # Plot the forecast
    forecaster.plot_forecast()

Implementing Trading Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from hamlet.executor.utilities.trading.strategies import ZeroIntelligence

    # Create a zero intelligence trading strategy
    strategy = ZeroIntelligence(
        min_price=0.05,  # €/kWh
        max_price=0.20,  # €/kWh
        random_seed=42
    )
    
    # Generate random bids and offers
    bids_offers = strategy.generate_bids_offers(
        energy_demand=5.0,  # kWh
        energy_generation=3.0  # kWh
    )

Extending the Executor
--------------------

Users can extend the Executor functionality by:

1. Implementing custom agent behaviors
2. Creating new market clearing algorithms
3. Developing specialized grid operation methods
4. Adding custom data logging and processing
5. Creating custom utility functions for specific needs

For more detailed information on specific classes and methods, refer to the API reference documentation.