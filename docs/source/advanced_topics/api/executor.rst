Executor API
============

The Executor API provides classes and functions for running simulation scenarios created by the Creator module. It handles the time-stepping mechanism, agent interactions, market clearing, and grid operations.

Module Overview
---------------

The Executor module is responsible for:

- Loading scenario files created by the Creator
- Managing the simulation timeline
- Executing agent decision-making processes
- Facilitating market interactions and clearing
- Simulating grid operations and constraints
- Recording simulation results for analysis

Key Classes
-----------

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
~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~

Market execution components include:

- ``Electricity``: Energy-only electricity markets
- ``Heat``: Energy-only heat markets (currently only placeholder)
- ``Hydrogen``: Energy-only hydrogen markets (currently only placeholder)

All classes inherit from ``MarketBase`` which handles the standard execution tasks.

Each market executor handels:
- Clearing
- Settling

Grid Execution
~~~~~~~~~~~~~~~

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
-------------

Running a Basic Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from hamlet.executor import Executor

    # Initialize the Executor with a scenario path
    executor = Executor(path_scenario="./scenarios/example_scenario")
    
    # Run the simulation
    executor.run()

Parallel Execution
~~~~~~~~~~~~~~~~~~~

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
---------

The Executor module includes various utility functions and classes that support the simulation process. These utilities are organized into several categories, each providing specific functionality to support agent decision-making, market operations, and grid management.

Controller Utilities
~~~~~~~~~~~~~~~~~~~~~

Controller utilities help agents make decisions about energy usage and control:

- ``Controller``: Main entry point for controller operations
- ``Fbc`` (Forecast-Based Control): For planning ahead with implementations:

   - ``Linopy``: Model Predictive Control using the Linopy optimization framework
   - ``POI``: Model Predictive Control using the POI (Python Optimization Interface)
- ``Rtc`` (Real-Time Control): For immediate control with implementations:

   - ``Linopy``: Optimization using the Linopy framework
   - ``POI``: Optimization using the POI framework

Database Utilities
~~~~~~~~~~~~~~~~~~

Database utilities handle data storage and retrieval:

- ``Database``: Main entry point for database operations
- ``AgentDB``: Manages information related to agents
- ``MarketDB``: Manages information related to markets
- ``GridDB``: Base class for grid databases

   - ``ElectricityGridDB``: Manages information related to electricity grids
   - ``HeatGridDB``: Manages information related to heat grids
   - ``HydrogenGridDB``: Manages information related to hydrogen grids
- ``RegionDB``: Manages information related to regions

Forecasting Utilities
~~~~~~~~~~~~~~~~~~~~~

Forecasting utilities help agents predict future values:

- ``Forecaster``: Main entry point for forecasting operations
- Various forecast models that inherit from ``ModelBase``:

   - ``PerfectModel``: Provides perfect forecasts (for testing/benchmarking)
   - ``NaiveModel``: A simple forecasting model
   - ``AverageModel``: Uses averages for forecasting
   - ``SmoothedModel``: Uses smoothing techniques
   - ``SARMAModel``: Uses Seasonal AutoRegressive Moving Average
   - ``RandomForest``: Uses Random Forest algorithm
   - ``CNNModel``: Uses Convolutional Neural Networks
   - ``RNNModel``: Uses Recurrent Neural Networks
   - ``ARIMAModel``: Uses AutoRegressive Integrated Moving Average
   - ``WeatherModel``: Specialized for weather forecasting
   - ``ArrivalModel``: For forecasting arrivals/events

Grid Restriction Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~

Grid restriction utilities handle grid constraints:

- ``GridRestriction``: Main entry point for grid restriction operations
- ``GridRestrictionBase``: Base class for grid restrictions
- ``EnWG14a``: Implementation related to the German Energy Industry Act

Tasks Execution Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~

Tasks execution utilities manage the execution of simulation tasks:

- ``TaskExecutioner``: Base class for task execution

   - ``AgentTaskExecutioner``: Manages agent task execution
   - ``MarketTaskExecutioner``: Manages market task execution
- ``ProcessPool``: Base class for multiprocessing pools

   - ``AgentPool``: Manages agent multiprocessing
   - ``MarketPool``: Manages market multiprocessing

Trading Utilities
~~~~~~~~~~~~~~~~~

Trading utilities implement various trading strategies:

- ``Trading``: Main entry point for trading operations
- ``TradingBase``: Base class for trading strategies

   - ``Linear``: Implements a linear trading strategy
   - ``Zi``: Implements a zero intelligence trading strategy
   - ``Retailer``: Implements a retailer-based trading strategy

Example Usage of Utilities
--------------------------

Using Forecasting Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from hamlet.executor.utilities.forecasts.forecaster import Forecaster

    # Create a forecaster
    forecaster = Forecaster(
        agent_db=agent_db,
        method="arima"
    )
    
    # Generate a forecast
    forecast = forecaster.forecast_load()
    
    # Access forecast results
    load_forecast = forecaster.load_forecast

Using Trading Strategies
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from hamlet.executor.utilities.trading.strategies import Zi

    # Create a zero intelligence trading strategy
    strategy = Zi(
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
----------------------

Users can extend the Executor functionality by:

1. Implementing custom agent behaviors
2. Creating new market clearing algorithms
3. Developing specialized grid operation methods
4. Adding custom data logging and processing
5. Creating custom utility functions for specific needs

For more detailed information on specific classes and methods, refer to the API reference documentation.