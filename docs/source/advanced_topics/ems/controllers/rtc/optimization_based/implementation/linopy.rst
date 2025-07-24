linopy
======

Introduction
-----------
The Linopy implementation is HAMLET's high-level approach to formulating and solving optimization problems for real-time control. It leverages the Linopy package, which provides a convenient interface for creating and solving linear and mixed-integer linear programming problems using labeled arrays (via xarray).

The implementation is organized around two main Python files:
- ``components.py``: Defines component models for various energy system elements
- ``optim_linopy.py``: Implements the main optimization controller

This implementation is designed to be readable, maintainable, and accessible to users who may not have extensive experience with mathematical optimization.

Objective Function
----------------
The primary goal of the Linopy implementation is to minimize operational costs while satisfying energy balance constraints and component-specific constraints. The objective function is defined in the ``define_objective`` method of the ``Linopy`` class.

The objective function focuses on minimizing deviations from target values for different components, with weights assigned to prioritize certain components over others.

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
        for variable_name, variable in self.model.variables.items():
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
        self.model.add_objective(sum(objective), overwrite=True)

        return self.model

The weights in the objective function determine the priority of different components:

- **Battery and Heat Storage (weight=1)**: Lowest priority, allowing these storage components to deviate from their targets when necessary to accommodate higher-priority components.
- **Electric Vehicle (weight=2)**: Medium-low priority, balancing flexibility with user needs.
- **Heat Pump (weight=3)**: Medium-high priority, reflecting the importance of maintaining thermal comfort.
- **Market (weight=4)**: Highest priority, minimizing deviations from market commitments to avoid potential penalties or imbalance costs.

The higher the weight, the higher the penalty for deviation from the target value, which means the optimizer will try harder to keep that component close to its target value when conflicts arise.

Code Implementation
-----------------------
The main implementation of the Linopy controller is in the ``optim_linopy.py`` file, which defines the ``Linopy`` class:

.. code-block:: python

    class Linopy(OptimBase):
        def __init__(self, **kwargs):
            self.loaded_model = False
            self.model_path = f"{kwargs['agent'].agent_save}/linopy_rtc.nc"
            # grid commands
            self.grid_commands = kwargs['grid_commands']
            super().__init__(**kwargs)
            self.ems = self.ems[c.C_OPTIM]
            # Save first model to file to load later
            self.save_model()

The class inherits from ``OptimBase``, which provides common functionality for optimization-based controllers.

**Model Initialization**

The model is initialized in the ``get_model`` method:

.. code-block:: python

    def get_model(self, **kwargs):
        # Check for existing saved models
        if os.path.exists(self.model_path):
            # Load model
            model = read_netcdf(self.model_path)
            self.loaded_model = True
        else:
            # Create a new model
            model = Model()
        return model

**Solving the Model**

The model is solved in the ``run`` method:

.. code-block:: python

    def run(self):
        # Get the model
        self.model = self.get_model()

        # Define the variables
        self.define_variables()

        # Define the constraints
        self.define_constraints()

        # Define the objective
        self.define_objective()

        # Solve the model
        try:
            result = self.model.solve(solver=self.ems['solver'], sense="minimize")
            self.solution = result
            self.status = 'optimal'
        except Exception as e:
            print(f"Error solving model: {e}")
            self.status = 'error'
            return None

        # Apply the grid commands
        self.apply_grid_commands()

        return self.solution

Mathematical Formulation
------------------------
The Linopy implementation follows the general mathematical formulation described in the Mathematical Formulation section, with specific adaptations for the Linopy framework.

**Problem Structure**

The optimization problem is formulated as a minimization problem with variables, constraints, and an objective function:

.. code-block:: python

    # Create a new model
    model = Model()
    
    # Define variables
    self.define_variables()
    
    # Define constraints
    self.define_constraints()
    
    # Define objective
    self.define_objective()
    
    # Solve the model
    self.model.solve(solver=self.ems['solver'], sense="minimize")

**Decision Variables**

Variables are defined for each component using the ``define_variables`` method, which calls the component-specific ``define_variables`` methods:

.. code-block:: python

    def define_variables(self):
        # Define variables for each plant
        for plant_name, plant in self.plant_objects.items():
            self.model = plant.define_variables(self.model, comp_type=self.plants[plant_name]['type'])

        # Define variables for each market
        for market_name, market in self.market_objects.items():
            # Balancing markets are not explicitly modeled and have the same comp_type as their original market
            if c.TT_BALANCING in market_name:
                energy_type = self.markets[market_name.rsplit('_', 1)[0]]
            else:
                energy_type = self.markets[market_name]

            self.model = market.define_variables(self.model, energy_type=energy_type)

        return self.model

**Constraints**

Constraints are defined using the ``define_constraints`` method, which calls the component-specific ``define_constraints`` methods and adds system-level constraints:

.. code-block:: python

    def define_constraints(self):
        # Define constraints for each plant
        for plant_name, plant in self.plant_objects.items():
            plant.define_constraints(self.model)

        # Define constraints for each market
        for market_name, market in self.market_objects.items():
            market.define_constraints(self.model)

        # Additional constraints for energy balancing, etc.
        self.add_balance_constraints()

        return self.model

**Energy Balance Constraints**

The energy balance constraints are implemented in the ``add_balance_constraints`` method:

.. code-block:: python

    def add_balance_constraints(self):
        # If model was loaded, no changes required for these constraints
        if self.loaded_model:
            return
        # Initialize the balance equations for each energy type by creating a zero variable for each energy type
        balance_equations = {energy_type: self.model.add_variables(name=f'balance_{energy_type}',
                                                                lower=0, upper=0, integer=True)
                            for energy_type in self.energy_types}

        # Loop through each energy type
        for energy_type in self.energy_types:
            # Loop through each variable and add it to the balance equation accordingly
            for variable_name, variable in self.model.variables.items():
                # Add the variable as generation if it is a market variable for the current energy type
                if (variable_name.startswith(tuple(self.market_objects))
                        and variable_name.endswith('import')
                        and self.markets[variable_name.split('_')[0]] == energy_type):
                    balance_equations[energy_type] += variable

                # Add the variable as consumption if it is a market variable for the current energy type
                elif (variable_name.startswith(tuple(self.market_objects))
                        and variable_name.endswith('export')
                        and self.markets[variable_name.split('_')[0]] == energy_type):
                    balance_equations[energy_type] -= variable

                # Add the variable as generation or consumption if it is a plant variable for the current energy type
                elif variable_name.endswith(energy_type):
                    # Get the plant name and type
                    plant_name = '_'.join(variable_name.split('_')[:-2])
                    plant_type = variable_name.split('_')[-2]

                    # Check if the plant exists
                    if plant_name in self.plant_objects:
                        # Add the variable to the balance equation
                        balance_equations[energy_type] += variable

        # Add the balance equations as constraints
        for energy_type, equation in balance_equations.items():
            self.model.add_constraints(equation == 0, name=f'balance_{energy_type}')

        return self.model

**Grid Control**

Commands by the grid operator (e.g. reducing power) are applied in the ``apply_grid_commands`` method:

.. code-block:: python

    def apply_grid_commands(self):
        # Loop through each plant and apply the grid commands
        for plant_name, plant in self.plant_objects.items():
            # Get the plant type
            plant_type = self.plants[plant_name]['type']

            # Loop through each energy type
            for energy_type in self.energy_types:
                # Check if the variable exists
                variable_name = f'{plant_name}_{plant_type}_{energy_type}'
                if variable_name in self.model.variables:
                    # Get the variable value
                    value = float(self.solution[variable_name].values)

                    # Apply the grid command
                    self.grid_commands[plant_name][energy_type] = value

        # Loop through each market and apply the grid commands
        for market_name, market in self.market_objects.items():
            # Get the energy type
            energy_type = self.markets[market_name]

            # Check if the variable exists
            variable_name_import = f'{market_name}_import'
            variable_name_export = f'{market_name}_export'
            if variable_name_import in self.model.variables and variable_name_export in self.model.variables:
                # Get the variable values
                value_import = float(self.solution[variable_name_import].values)
                value_export = float(self.solution[variable_name_export].values)

                # Apply the grid command
                self.grid_commands[market_name][energy_type] = value_import - value_export

Component Models
----------------
The Linopy implementation includes models for various energy system components, defined in the ``components.py`` file. Each component is implemented as a class that inherits from the base ``LinopyComps`` class. Here we focus on three key component models as examples: inflexible load, PV, and market. For further information please see the code itself.

**Inflexible Load**

The ``InflexibleLoad`` class represents electrical loads that cannot be controlled or shifted. These loads must be satisfied exactly as specified:

.. code-block:: python

    class InflexibleLoad(LinopyComps):
        def __init__(self, name, **kwargs):
            # Call the parent class constructor
            super().__init__(name, **kwargs)

            # Get specific object attributes
            self.power = self.ts[f'{self.name}_{c.ET_ELECTRICITY}'][0]

        def define_variables(self, model, **kwargs) -> Model:
            comp_type = kwargs['comp_type']

            # Define the power variable
            model = self.define_electricity_variable(model, comp_type=comp_type, lower=-self.power, upper=-self.power)

            return model

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

    class SimplePlant(LinopyComps):
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

        def define_variables(self, model, **kwargs) -> Model:
            comp_type = kwargs['comp_type']

            # Define the power variable
            model = self.define_electricity_variable(model, comp_type=comp_type, lower=self.lower, upper=self.power)

            return model

PV systems have a power variable with a lower bound of 0 and an upper bound equal to the available power, allowing for curtailment when necessary.

**Market**

The market component represents the connection to external energy networks:

.. code-block:: python

    class Market(LinopyComps):
        def __init__(self, name, **kwargs):
            # Call the parent class constructor
            super().__init__(name, **kwargs)

            # Get specific object attributes
            self.dt = kwargs['delta'].total_seconds()  # time delta in seconds
            self.market_power = int(round(kwargs['market_result'] * c.HOURS_TO_SECONDS / self.dt))  # from Wh to W
            self.balancing_power = 10000000000  # Maximum available balancing power

            # Get the energy type
            self.energy_type = None

        def define_variables(self, model, **kwargs) -> Model:
            self.energy_type = kwargs['energy_type']

            # Define the market power variable
            self.add_variable_to_model(model, name=f'{self.name}_{self.energy_type}', lower=-inf, upper=inf, integer=False)

            # Define the target variable (what was previously bought/sold on the market)
            self.add_variable_to_model(model, name=f'{self.name}_{self.energy_type}_target',
                                    lower=self.market_power, upper=self.market_power, integer=False)

            # Define the deviation variable for positive and negative deviations
            # Deviation when more is bought/sold on the market than according to the market
            self.add_variable_to_model(model, name=f'{self.name}_{self.energy_type}_deviation_pos',
                                    lower=0, upper=self.balancing_power, integer=False)
            # Deviation when less is needed from the grid than according to the market
            self.add_variable_to_model(model, name=f'{self.name}_{self.energy_type}_deviation_neg',
                                    lower=0, upper=self.balancing_power, integer=False)

            return model

        def define_constraints(self, model) -> Model:
            # Define the deviation constraint
            cons_name = f'{self.name}_deviation'
            if cons_name not in model.constraints:
                equation = (model.variables[f'{self.name}_{self.energy_type}']
                            - model.variables[f'{self.name}_{self.energy_type}_target']
                            == model.variables[f'{self.name}_{self.energy_type}_deviation_pos']
                            - model.variables[f'{self.name}_{self.energy_type}_deviation_neg'])

                model.add_constraints(equation, name=cons_name)
            return model


Configuration
-----------
The Linopy implementation can be configured through the agent configuration file. The configuration is specified in the `ems.controller.rtc` section of the agent config file:

.. code-block:: yaml

    ems:
      controller:
        rtc:
          method: optimization
          optimization:
            framework: linopy
            solver: gurobi
            time_limit: 120

**Configuration Parameters**

- **method**: The control method to use (set to "optimization" for the Linopy implementation)
- **optimization.framework**: The optimization implementation to use (e.g. "linopy" for this implementation)
- **optimization.solver**: The solver to use for the optimization problem
- **optimization.time_limit**: Maximum solving time in seconds (default: 120s)

The objective function weights are not configurable through the agent config file but are hardcoded in the implementation.
To change them they need to be adjusted directly in the controller module.