poi
===

Introduction
-----------
The PyOptInterface implementation provides a flexible, lower-level approach to formulating and solving optimization problems in HAMLET's real-time controllers. It leverages the PyOptInterface package, which offers a unified interface to various optimization solvers, allowing for more direct control over the optimization process.

The implementation is organized around two main Python files:
- ``components.py``: Defines component models for various energy system elements
- ``optim_poi.py``: Implements the main optimization controller

This implementation is designed for users who need greater flexibility in problem formulation and solver configuration, providing a more direct mapping between the mathematical formulation and the code implementation.

Objective Function
----------------
The primary goal of the PyOptInterface implementation, like the Linopy implementation, is to minimize deviations from target values for different components, with weights assigned to prioritize certain components over others.

**Mathematical Formulation**

The objective function can be mathematically expressed as:

.. math::

   \min \sum_{c \in C} \sum_{v \in V_c} w_c \cdot d_v

where:

- :math:`C` is the set of component types (battery, heat storage, EV, heat pump, market)
- :math:`V_c` is the set of deviation variables for component type :math:`c`
- :math:`w_c` is the weight assigned to component type :math:`c`
- :math:`d_v` is the value of deviation variable :math:`v`

The weights :math:`w_c` are defined as:

.. math::

   w_c = 
   \begin{cases}
   1, & \text{if } c \in \{\text{battery}, \text{heat storage}\} \\
   2, & \text{if } c = \text{EV} \\
   3, & \text{if } c = \text{heat pump} \\
   4, & \text{if } c = \text{market}
   \end{cases}

This formulation captures the prioritization of different components in the system, with higher weights indicating higher priority for minimizing deviations from target values.

**Code Implementation**

.. code-block:: python

    def define_objective(self):
        # Weights to prioritize components (the higher the weight, the higher the penalty for deviation)
        weights = {
            c.P_BATTERY: 1,  # weight for battery
            c.P_HEAT_STORAGE: 1,  # weight for heat storage
            c.P_EV: 2,  # weight for electric vehicle
            c.P_HP: 3,  # weight for heat pump
            'market': 4  # weight for market energy
        }

        # Initialize the objective function as zero
        objective = []

        # Loop through the model's variables to identify the balancing variables that need to be minimized
        for variable_name, variable in self.variables.items():
            # Check if variable_name contains an underscore
            if "_deviation_" in variable_name:
                # Extract component type from variable name using the weights mapping
                component_type = next((key for key in weights.keys() if f'_{key}_' in variable_name), None)
                # If component type is None assign market weight
                component_type = 'market' if component_type is None else component_type

                # Get the weight for the component type
                weight = weights.get(component_type)

                # Add deviation to objective function
                objective.append(variable * weight)

        # Set the objective function to the model with the minimize direction
        self.model.set_objective(sum(objective), poi.ObjectiveSense.Minimize)

The weights in the objective function determine the priority of different components:

- **Battery and Heat Storage (weight=1)**: Lowest priority, allowing these storage components to deviate from their targets when necessary to accommodate higher-priority components.
- **Electric Vehicle (weight=2)**: Medium-low priority, balancing flexibility with user needs.
- **Heat Pump (weight=3)**: Medium-high priority, reflecting the importance of maintaining thermal comfort.
- **Market (weight=4)**: Highest priority, minimizing deviations from market commitments to avoid potential penalties or imbalance costs.

The higher the weight, the higher the penalty for deviation from the target value, which means the optimizer will try harder to keep that component close to its target value when conflicts arise.

Code Implementation
-----------------------
The main implementation of the PyOptInterface controller is in the ``optim_poi.py`` file, which defines the ``POI`` class:

.. code-block:: python

    class POI(OptimBase):
        def get_model(self, **kwargs):
            env = gurobi.Env(empty=True)
            env.set_raw_parameter("OutputFlag", 0)
            env.start()
            model = gurobi.Model(env)
            model.set_model_attribute(poi.ModelAttribute.Silent, True)
            model.set_raw_parameter("OutputFlag", 0)
            model.set_raw_parameter("LogToConsole", 0)
            return model

The class inherits from ``OptimBase``, which provides common functionality for optimization-based controllers.

**Model Initialization**

The model is initialized in the ``get_model`` method, which creates a Gurobi model with specific parameters:

.. code-block:: python

    def get_model(self, **kwargs):
        env = gurobi.Env(empty=True)
        env.set_raw_parameter("OutputFlag", 0)
        env.start()
        model = gurobi.Model(env)
        model.set_model_attribute(poi.ModelAttribute.Silent, True)
        model.set_raw_parameter("OutputFlag", 0)
        model.set_raw_parameter("LogToConsole", 0)
        return model

**Solving the Model**

The model is solved in the ``run`` method:

.. code-block:: python

    def run(self):
        # Solve the optimization problem
        solver = self.ems[c.C_OPTIM].get('solver')
        match solver:
            case 'gurobi':
                self.model.set_model_attribute(poi.ModelAttribute.Silent, True)
                self.model.set_raw_parameter("OutputFlag", 0)
                self.model.set_raw_parameter("LogToConsole", 0)
                if self.ems[c.C_OPTIM].get('time_limit') is not None:
                    self.model.set_raw_parameter('TimeLimit', self.ems[c.C_OPTIM]['time_limit'] / 60)
                self.model.optimize()
                status = self.model.get_model_attribute(poi.ModelAttribute.TerminationStatus)
            case _:
                raise ValueError(f"Unsupported solver: {solver}")

        # Check if the solution is optimal
        if status not in [poi.TerminationStatusCode.OPTIMAL, poi.TerminationStatusCode.TIME_LIMIT]:
            print(f'Exited with status "{status[0]}". \n')
            # raise ValueError(f"Optimization failed: {status}")

        # Process the solution into control commands and return
        self.agent = self.process_solution()

        return self.agent

**Getting the Solution**

The solution is retrieved in the ``get_solution`` method:

.. code-block:: python

    def get_solution(self):
        # Obtain the solution values
        return {var_name: int(self.model.get_value(var)) for var_name, var in self.variables.items()}

Mathematical Formulation
------------------------
The PyOptInterface implementation follows the general mathematical formulation described in the Mathematical Formulation section, with specific adaptations for the PyOptInterface framework.

**Problem Structure**

The optimization problem is formulated as a minimization problem with variables, constraints, and an objective function:

.. code-block:: python

    # Create a new model
    model = gurobi.Model(env)
    
    # Define variables
    self.define_variables()
    
    # Define constraints
    self.define_constraints()
    
    # Define objective
    self.define_objective()
    
    # Solve the model
    self.model.optimize()

**Decision Variables**

Variables are defined for each component using the ``define_variables`` method, which calls the component-specific ``define_variables`` methods. Unlike the Linopy implementation, variables are stored in a dictionary rather than as attributes of the model:

.. code-block:: python

    def define_variables(self):
        self.variables = {}
        # Define variables for each plant
        for plant_name, plant in self.plant_objects.items():
            plant.define_variables(self.model, self.variables, comp_type=self.plants[plant_name]['type'])

        # Define variables for each market
        for market_name, market in self.market_objects.items():
            # Balancing markets are not explicitly modeled and have the same comp_type as their original market
            if c.TT_BALANCING in market_name:
                energy_type = self.markets[market_name.rsplit('_', 1)[0]]
            else:
                energy_type = self.markets[market_name]

            market.define_variables(self.model, self.variables, energy_type=energy_type)

        return self.model

**Constraints**

Constraints are defined using the ``define_constraints`` method, which calls the component-specific ``define_constraints`` methods and adds system-level constraints:

.. code-block:: python

    def define_constraints(self):
        # Define constraints for each plant
        for plant_name, plant in self.plant_objects.items():
            plant.define_constraints(self.model, self.variables)

        # Define constraints for each market
        for market_name, market in self.market_objects.items():
            market.define_constraints(self.model, self.variables)

        # Additional constraints for energy balancing, etc.
        self.add_balance_constraints()

        return self.model

**Energy Balance Constraints**

The energy balance constraints are implemented in the ``add_balance_constraints`` method:

.. code-block:: python

    def add_balance_constraints(self):
        # Initialize the balance equations for each energy type by creating a zero variable for each energy type
        balance_equations = {energy_type: [] for energy_type in self.energy_types}

        # Loop through each energy type
        for energy_type in self.energy_types:
            # Loop through each variable and add it to the balance equation accordingly
            for variable_name, variable in self.variables.items():
                # Add the variable as generation if it is a market variable for the current energy type
                if (variable_name.startswith(tuple(self.market_objects))
                        and variable_name.endswith(f'_{energy_type}')):
                    balance_equations[energy_type].append(variable)
                # Add the variable if it is a plant variable for the current energy type
                elif (variable_name.startswith(tuple(self.plant_objects))
                      and variable_name.endswith(f'_{energy_type}')):
                    # Get the component name by splitting the variable name at the underscore
                    component_name = variable_name.split('_', 1)[0]

                    # Get the component type by comparing the ID with the plant names
                    component_type = [vals['type'] for plant, vals in self.plants.items()
                                      if plant == component_name][0]

                    # If the component type is in the mapping for the current energy type, add the variable to the
                    # balance equation
                    if energy_type in self.mapping[component_type].keys():
                        # Get the operation mode for the component and energy type
                        component_energy_mode = self.mapping[component_type][energy_type]

                        # Add the variable to the balance equation
                        balance_equations[energy_type].append(variable)
                    else:
                        # The component type is not in the mapping for the current energy type
                        pass
                else:
                    pass

        # Add the constraints for each energy type
        for energy_type, variables in balance_equations.items():
            self.model.add_linear_constraint(sum(variables), poi.ConstraintSense.Equal, 0,
                                             name=f"balance_{energy_type}")

Component Models
----------------
The PyOptInterface implementation includes models for various energy system components, defined in the ``components.py`` file. Each component is implemented as a class that inherits from the base ``POIComps`` class. Here we focus on three key component models: inflexible load, PV, and market.

**Base Component Class**

The ``POIComps`` class provides common functionality for all components:

.. code-block:: python

    class POIComps:
        def __init__(self, name, timeseries, **kwargs):
            # Get the data
            self.name = name
            self.ts = timeseries
            self.info = kwargs

            # Other attributes (to be defined in subclasses)
            self.comp_type = None
            self.target = None
            self.upper = None
            self.lower = None

        def define_variables(self, model, variables, **kwargs):
            raise NotImplementedError(f'{self.name} has not been implemented yet.')

        @staticmethod
        def define_constraints(model, variables):
            pass

        @staticmethod
        def add_variable_to_model(model, variables, name, **kwargs):
            var_name = name
            lb = kwargs.get("lower", -math.inf)
            ub = kwargs.get("upper", math.inf)
            kwargs_var = {
                'name': var_name,
                'lb': lb,
                'ub': ub,
                'domain': poi.VariableDomain.Integer if kwargs.get('integer', False)
                else poi.VariableDomain.Binary
                if kwargs.get('binary', False) else poi.VariableDomain.Continuous,
            }
            variables[name] = model.add_variable(**kwargs_var)

**Inflexible Load**

The ``InflexibleLoad`` class represents electrical loads that cannot be controlled or shifted. These loads must be satisfied exactly as specified:

.. code-block:: python

    class InflexibleLoad(POIComps):
        def __init__(self, name, **kwargs):
            # Call the parent class constructor
            super().__init__(name, **kwargs)

            # Get specific object attributes
            self.power = self.ts[f'{self.name}_{c.ET_ELECTRICITY}'][0]

        def define_variables(self, model, variables, **kwargs):
            comp_type = kwargs['comp_type']

            # Define the power variable
            self.define_electricity_variable(model, variables, comp_type=comp_type, lower=-self.power, upper=-self.power)

The power variable has fixed lower and upper bounds equal to the negative of the load power (indicating consumption), ensuring that the load must be satisfied exactly.

**PV Systems**

PV systems are implemented in the ``Pv`` class, which inherits from ``SimplePlant``:

.. code-block:: python

    class Pv(SimplePlant):
        def __init__(self, name, **kwargs):
            # Call the parent class constructor
            super().__init__(name, **kwargs)

The ``SimplePlant`` class defines the common functionality for generation components:

.. code-block:: python

    class SimplePlant(POIComps):
        def __init__(self, name, **kwargs):
            # Call the parent class constructor
            super().__init__(name, **kwargs)

            # Get specific object attributes
            try:
                self.power = self.ts[f'{self.name}_{c.ET_ELECTRICITY}'][0]
                self.target = kwargs['targets'][f'{self.name}'][0]
            except pl_e.ColumnNotFoundError:
                self.power = self.ts[f'{self.name}_power'][0]
                self.target = kwargs['targets'][f'{self.name}_{c.P_PLANT}_{c.ET_ELECTRICITY}'][0]

            self.lower = 0
            self.upper = self.power

        def define_variables(self, model, variables, **kwargs):
            comp_type = kwargs['comp_type']

            # Define the power variable
            self.define_electricity_variable(model, variables, comp_type=comp_type, lower=self.lower, upper=self.power)

PV systems have a power variable with a lower bound of 0 and an upper bound equal to the available power, allowing for curtailment when necessary.

**Market**

The market component represents the connection to external energy networks:

.. code-block:: python

    class Market(POIComps):
        def __init__(self, name, **kwargs):
            # Call the parent class constructor
            super().__init__(name, **kwargs)

            # Get specific object attributes
            self.dt = kwargs['delta'].total_seconds()  # time delta in seconds
            self.market_power = int(round(kwargs['market_result'] * c.HOURS_TO_SECONDS / self.dt))  # from Wh to W
            self.balancing_power = 10000000000  # Maximum available balancing power

            # Get the energy type
            self.energy_type = None

        def define_variables(self, model, variables, **kwargs):
            self.energy_type = kwargs['energy_type']

            # Define the market power variable
            self.add_variable_to_model(model, variables, name=f'{self.name}_{self.energy_type}', lower=-inf, upper=inf, integer=False)

            # Define the target variable (what was previously bought/sold on the market)
            self.add_variable_to_model(model, variables, name=f'{self.name}_{self.energy_type}_target',
                                    lower=self.market_power, upper=self.market_power, integer=False)

            # Define the deviation variable for positive and negative deviations
            # Deviation when more is bought/sold on the market than according to the market
            self.add_variable_to_model(model, variables, name=f'{self.name}_{self.energy_type}_deviation_pos',
                                    lower=0, upper=self.balancing_power, integer=False)
            # Deviation when less is needed from the grid than according to the market
            self.add_variable_to_model(model, variables, name=f'{self.name}_{self.energy_type}_deviation_neg',
                                    lower=0, upper=self.balancing_power, integer=False)

        def define_constraints(self, model, variables):
            # Define the deviation constraint
            cons_name = f'{self.name}_deviation'
            model.add_linear_constraint(
                variables[f'{self.name}_{self.energy_type}'] - variables[f'{self.name}_{self.energy_type}_target'],
                poi.ConstraintSense.Equal,
                variables[f'{self.name}_{self.energy_type}_deviation_pos'] - variables[f'{self.name}_{self.energy_type}_deviation_neg'],
                name=cons_name
            )

Configuration
-----------
The PyOptInterface implementation can be configured through the agent configuration file. The configuration is specified in the `ems.controller.rtc` section of the agent config file:

.. code-block:: yaml

    ems:
      controller:
        rtc:
          method: optimization
          optimization:
            framework: poi
            solver: gurobi
            time_limit: 120

**Configuration Parameters**

- **method**: The control method to use (set to "optimization" for the PyOptInterface implementation)
- **optimization.framework**: The optimization implementation to use (set to "poi" for this implementation)
- **optimization.solver**: The solver to use for the optimization problem (currently only "gurobi" is supported)
- **optimization.time_limit**: Maximum solving time in seconds (default: 120s)

The objective function weights are not configurable through the agent config file but are hardcoded in the implementation. To change them they need to be adjusted directly in the controller module.