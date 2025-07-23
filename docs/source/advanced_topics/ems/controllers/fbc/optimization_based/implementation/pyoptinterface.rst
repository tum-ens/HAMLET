poi
===

Introduction
------------

The PyOptInterface implementation for Forecast-Based Controllers (FBC) in HAMLET provides a flexible, lower-level approach to formulating and solving multi-period optimization problems. It leverages the PyOptInterface package, which offers a unified interface to various optimization solvers, allowing for more direct control over the optimization process.

The implementation is organized around two main Python files:
- ``components.py``: Defines component models for various energy system elements over multiple timesteps
- ``mpc_poi.py``: Implements the main optimization controller with forecast integration

This implementation is designed for users who need greater flexibility in problem formulation and solver configuration, providing a more direct mapping between the mathematical formulation and the code implementation. The key difference from the Real-Time Controller (RTC) implementation is the handling of multiple timesteps and forecasts, enabling the controller to make decisions that optimize performance over a prediction horizon rather than just for the current timestep.

Objective Function
------------------
The primary goal of the PyOptInterface implementation for FBC is to minimize operational costs over the entire prediction horizon. Unlike the RTC implementation, which focuses on minimizing deviations from target values, the FBC implementation directly minimizes costs and maximizes revenues from market interactions.

**Mathematical Formulation**

The objective function can be mathematically expressed as:

.. math::

   \min \sum_{t=0}^{T-1} \sum_{m \in M} \left( C_m(t) - R_m(t) \right)

where:

- :math:`T` is the optimization horizon
- :math:`M` is the set of markets
- :math:`C_m(t)` is the cost of buying energy from market :math:`m` at timestep :math:`t`
- :math:`R_m(t)` is the revenue from selling energy to market :math:`m` at timestep :math:`t`

The costs and revenues are defined as:

.. math::

   C_m(t) = P_{m,in}(t) \cdot \Delta t \cdot \left( p_{buy}(t) + g_{buy}(t) + l_{buy}(t) \right)

   R_m(t) = -P_{m,out}(t) \cdot \Delta t \cdot \left( p_{sell}(t) - g_{sell}(t) - l_{sell}(t) \right)

where:

- :math:`P_{m,in}(t)` is the power imported from market :math:`m` at timestep :math:`t` [W]
- :math:`P_{m,out}(t)` is the power exported to market :math:`m` at timestep :math:`t` [W]
- :math:`\Delta t` is the timestep duration [h]
- :math:`p_{buy}(t)`, :math:`p_{sell}(t)` are the energy buying and selling prices at timestep :math:`t` [€/Wh]
- :math:`g_{buy}(t)`, :math:`g_{sell}(t)` are the grid fees for buying and selling at timestep :math:`t` [€/Wh]
- :math:`l_{buy}(t)`, :math:`l_{sell}(t)` are the levies for buying and selling at timestep :math:`t` [€/Wh]

**Code Implementation**

The objective function is defined in the ``define_objective`` method of the ``POI`` class:

.. code-block:: python

    def define_objective(self):
        """Defines the objective function. The objective is to reduce the costs."""

        # Initialize the objective function as zero
        objective = []

        # Loop through the model's variables to identify the balancing variables
        for variable_name, variables in self.variables.items():
            # Only consider the cost and revenue components of the markets
            if variable_name.startswith(tuple(self.market_names)):
                if variable_name.endswith('_costs'):
                    # Add the variable to the objective function
                    objective.append(variables)
                elif variable_name.endswith('_revenue'):
                    # Subtract the variable from the objective function
                    objective.append(-1 * variables)
                else:
                    pass
            else:
                pass

        # Set the objective function to the model with the minimize direction
        self.model.set_objective(np.sum(objective), poi.ObjectiveSense.Minimize)

The objective function minimizes the total cost over the prediction horizon by:
1. Adding cost variables (representing expenses for buying energy)
2. Subtracting revenue variables (representing income from selling energy)

This approach allows the controller to make decisions that optimize economic performance over time, taking advantage of price variations and forecasted conditions to minimize overall costs.

Code Implementation
-----------------------
The main implementation of the PyOptInterface controller for FBC is in the ``mpc_poi.py`` file, which defines the ``POI`` class:

.. code-block:: python

    class POI(MpcBase):
        def get_model(self, **kwargs):
            env = gurobi.Env(empty=True)
            env.set_raw_parameter("OutputFlag", 0)
            env.start()
            model = gurobi.Model(env)
            model.set_model_attribute(poi.ModelAttribute.Silent, True)
            model.set_raw_parameter("OutputFlag", 0)
            model.set_raw_parameter("LogToConsole", 0)
            return model

The class inherits from ``MpcBase``, which provides common functionality for Model Predictive Control (MPC) based controllers. This is different from the RTC implementation, which inherits from ``OptimBase``.

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
                if self.ems[c.C_OPTIM].get('time_limit') is not None:
                    self.model.set_raw_parameter('TimeLimit', self.ems[c.C_OPTIM]['time_limit'] / 60)
                self.model.optimize()
                status = self.model.get_model_attribute(poi.ModelAttribute.TerminationStatus)
            case _:
                raise ValueError(f"Unsupported solver: {solver}")

        # Check if the solution is optimal
        if status not in [poi.TerminationStatusCode.OPTIMAL, poi.TerminationStatusCode.TIME_LIMIT]:
            print(f'Exited with status "{status}". \n ')
            # raise ValueError(f"Optimization failed: {status}")

        # Process the solution into control commands and return
        self.agent = self.process_solution()

        return self.agent

**Getting the Solution**

The solution is retrieved in the ``get_solution`` method, which returns a dictionary mapping variable names to arrays of values (one value per timestep):

.. code-block:: python

    def get_solution(self):
        # Obtain the solution values
        return {var_name: np.array([self.model.get_value(var) for var in vars]) for var_name, vars in
                self.variables.items()}

Mathematical Formulation
------------------------
The PyOptInterface implementation for FBC follows the general mathematical formulation described in the Mathematical Formulation section, with specific adaptations for multi-period optimization and forecast integration.

**Problem Structure**

The optimization problem is formulated as a minimization problem over multiple timesteps.
In the code, this is implemented using PyOptInterface's API with variables and constraints defined for each timestep:

.. code-block:: python

    # Define variables for each timestep
    for timestep in range(horizon):
        # Create variables for this timestep
        # ...

    # Define constraints for each timestep
    for timestep in range(horizon):
        # Create constraints for this timestep
        # ...

    # Define temporal coupling constraints
    for timestep in range(1, horizon):
        # Create constraints that link variables across timesteps
        # ...

**Decision Variables**

Variables are defined for each component and each timestep in the optimization horizon. Unlike the Linopy implementation, which uses labeled arrays, the PyOptInterface implementation uses a dictionary to store variables:

.. code-block:: python

    def define_variables(self):
        self.variables = {}
        # Define variables for each plant
        for plant_name, plant in self.plant_objects.items():
            plant.define_variables(self.model, self.variables, comp_type=self.plants[plant_name]['type'])

        # Define variables for each market
        for market_name, market in self.market_objects.items():
            market.define_variables(self.model, self.variables, comp_type=self.markets[market_name])

**Temporal Coupling Constraints**

A key feature of FBC is the inclusion of constraints that couple variables across timesteps. For example, the state-of-charge evolution for a battery:

.. code-block:: python

    def _constraint_soc(self, model, variables, energy_type: str = c.ET_ELECTRICITY):
        """Adds the constraint that the soc of the battery is that of the previous timestep plus dis-/charging power"""

        # Get the variables
        var_soc = variables[f'{self.name}_{self.comp_type}_soc']
        var_charge = variables[f'{self.name}_{self.comp_type}_{energy_type}_{c.PF_OUT}']
        var_discharge = variables[f'{self.name}_{self.comp_type}_{energy_type}_{c.PF_IN}']
        var_soc_init = variables[f'{self.name}_{self.comp_type}_soc_init']

        # Define the constraint for each timestep
        for timestep in range(len(self.timesteps)):
            if timestep == 0:
                # For the first timestep, use the initial SOC
                model.add_linear_constraint(
                    var_soc[timestep] - var_soc_init - var_charge[timestep] * self.efficiency * self.dt_hours - 
                    var_discharge[timestep] / self.efficiency * self.dt_hours,
                    poi.ConstraintSense.Equal,
                    0,
                    name=f'{self.name}_soc_{timestep}'
                )
            else:
                # For subsequent timesteps, use the SOC from the previous timestep
                model.add_linear_constraint(
                    var_soc[timestep] - var_soc[timestep-1] - var_charge[timestep] * self.efficiency * self.dt_hours - 
                    var_discharge[timestep] / self.efficiency * self.dt_hours,
                    poi.ConstraintSense.Equal,
                    0,
                    name=f'{self.name}_soc_{timestep}'
                )

**Energy Balance Constraints**

Energy balance constraints ensure that supply matches demand for each energy carrier at each timestep:

.. code-block:: python

    def add_balance_constraints(self):
        # Initialize the balance equations for each energy type
        balance_equations = {energy_type: [] for energy_type in self.energy_types}
        
        # Loop through each energy type
        for energy_type in self.energy_types:
            # Loop through each variable and add it to the balance equation accordingly
            for variable_name, variables in self.variables.items():
                # Add variables to balance equations based on their type and energy carrier
                # ...

        # Add the constraints for each energy type and timestep
        for energy_type, expressions in balance_equations.items():
            timestep_equations = np.sum(expressions, axis=0)
            for timestep, equation in enumerate(timestep_equations):
                self.model.add_linear_constraint(
                    equation, 
                    poi.ConstraintSense.Equal, 
                    0,
                    name=f"balance_{energy_type}_{timestep}"
                )

Component Models
----------------
The PyOptInterface implementation for FBC includes models for various energy system components, defined in the ``components.py`` file. Each component is implemented as a class that inherits from the base ``POIComps`` class. Here we focus on three key component models: inflexible load, PV, and market.

**Base Component Class**

The ``POIComps`` class provides common functionality for all components:

.. code-block:: python

    class POIComps:
        def __init__(self, name, forecasts, **kwargs):
            # Get the data
            self.name = name
            self.fcast = forecasts
            self.timesteps = kwargs['timesteps']
            self.info = kwargs

        def define_variables(self, model, variables, **kwargs):
            raise NotImplementedError()

        @staticmethod
        def define_constraints(model, variables):
            pass

        @staticmethod
        def add_variable_to_model(model, variables, name, **kwargs):
            coords = kwargs.get('coords', [[0]])
            if len(coords) > 1:
                print('Warning: 2d coords are not currently supported. Exiting..')
                return
            for coord in coords:
                variables[name] = np.empty((len(coord)), dtype=object)
                for ind in coord:
                    var_name = name
                    if 'coords' in kwargs:
                        var_name += f'_{ind}'
                    lb = kwargs.get("lower", -math.inf)
                    if isinstance(lb, (pd.Series, list, np.ndarray)):
                        lb = lb[ind]
                    ub = kwargs.get("upper", math.inf)
                    if isinstance(ub, (pd.Series, list, np.ndarray)):
                        ub = ub[ind]
                    kwargs_var = {
                        'name': var_name,
                        'lb': lb,
                        'ub': ub,
                        'domain': poi.VariableDomain.Integer if kwargs.get('integer', False)
                        else poi.VariableDomain.Binary
                        if kwargs.get('binary', False) else poi.VariableDomain.Continuous,
                    }
                    variables[name][ind] = model.add_variable(**kwargs_var)

**Inflexible Load**

The ``InflexibleLoad`` class represents electrical loads that cannot be controlled or shifted. These loads must be satisfied exactly as specified for each timestep in the horizon:

.. code-block:: python

    class InflexibleLoad(POIComps):
        def __init__(self, name, **kwargs):
            # Call the parent class constructor
            super().__init__(name, **kwargs)

            # Get specific object attributes
            self.power = pd.Series(self.fcast[f'{self.name}_{c.ET_ELECTRICITY}'], index=self.timesteps, dtype='int32')

        def define_variables(self, model, variables, **kwargs):
            comp_type = kwargs['comp_type']

            # Define the power variable
            self.define_electricity_variable(model, variables, comp_type=comp_type, lower=-self.power, upper=-self.power)

The power variable has fixed lower and upper bounds equal to the negative of the load power (indicating consumption) for each timestep, ensuring that the load must be satisfied exactly.

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
            self.power = list(self.fcast[f'{self.name}_{c.ET_ELECTRICITY}'])
            self.controllable = self.info['sizing']['controllable']
            self.lower = [0] * len(self.power) if self.controllable else self.power

            self.lower = pd.Series(self.lower, index=self.timesteps)
            self.power = pd.Series(self.power, index=self.timesteps)

        def define_variables(self, model, variables, **kwargs):
            comp_type = kwargs['comp_type']

            # Define the power variable
            self.define_electricity_variable(model, variables, comp_type=comp_type, lower=self.lower, upper=self.power)

PV systems have a power variable with a lower bound of 0 and an upper bound equal to the forecasted available power for each timestep, allowing for curtailment when necessary.

**Market**

The market component represents the connection to external energy networks, with buying and selling capabilities over the prediction horizon:

.. code-block:: python

    class Market(POIComps):
        def __init__(self, name, **kwargs):
            # Call the parent class constructor
            super().__init__(name, **kwargs)

            # Get specific object attributes
            self.comp_type = None
            self.dt_hours = kwargs['delta'].total_seconds() * c.SECONDS_TO_HOURS  # time delta in hours

            # Calculate the upper and lower bounds for the market power from the energy quantity
            self.upper = [int(round(x / self.dt_hours)) for x in self.fcast[f'{c.TC_ENERGY}_{c.TC_ENERGY}_{c.PF_IN}']]
            self.lower = [int(round(x / self.dt_hours * -1)) for x in self.fcast[f'{c.TC_ENERGY}_{c.TC_ENERGY}_{c.PF_OUT}']]

            self.lower = pd.Series(self.lower, index=self.timesteps)
            self.upper = pd.Series(self.upper, index=self.timesteps)

            # Get market price forecasts
            self.price_sell = pd.Series(self.fcast[f'{c.TC_ENERGY}_{c.TC_PRICE}_{c.PF_OUT}'], index=self.timesteps)
            self.price_buy = pd.Series(self.fcast[f'{c.TC_ENERGY}_{c.TC_PRICE}_{c.PF_IN}'], index=self.timesteps)
            self.grid_sell = pd.Series(self.fcast[f'{c.TT_GRID}_{c.TT_MARKET}_{c.PF_OUT}'], index=self.timesteps)
            self.grid_buy = pd.Series(self.fcast[f'{c.TT_GRID}_{c.TT_MARKET}_{c.PF_IN}'], index=self.timesteps)
            self.levies_sell = pd.Series(self.fcast[f'{c.TT_LEVIES}_{c.TC_PRICE}_{c.PF_OUT}'], index=self.timesteps)
            self.levies_buy = pd.Series(self.fcast[f'{c.TT_LEVIES}_{c.TC_PRICE}_{c.PF_IN}'], index=self.timesteps)

        def define_variables(self, model, variables, **kwargs):
            self.comp_type = kwargs['comp_type']
            # Define the market power variables (need to be positive and negative due to different pricing)
            self.add_variable_to_model(model, variables, name=f'{self.name}_{self.comp_type}_{c.PF_OUT}', lower=self.lower,
                                   upper=0, coords=[self.timesteps],
                                   integer=True)  # outflow from the building (selling)
            self.add_variable_to_model(model, variables, name=f'{self.name}_{self.comp_type}_{c.PF_IN}', lower=0,
                                   upper=self.upper, coords=[self.timesteps],
                                   integer=True)  # inflow into the building (buying)

            # Define mode flag that decides whether the market energy is bought or sold
            self.add_variable_to_model(model, variables, name=f'{self.name}_mode', coords=[self.timesteps], binary=True)

            # Define the market cost and revenue variables
            self.add_variable_to_model(model, variables, name=f'{self.name}_costs', lower=0, upper=np.inf,
                                   coords=[self.timesteps])
            self.add_variable_to_model(model, variables, name=f'{self.name}_revenue', lower=0, upper=np.inf,
                                   coords=[self.timesteps])

        def define_constraints(self, model, variables):
            # Add constraint that the market can either buy or sell but not both at the same time
            self.__constraint_operation_mode(model, variables)

            # Add constraint that the market cost and revenue are linked to the power
            self.__constraint_cost_revenue(model, variables)

The market component includes variables for buying and selling power, as well as for costs and revenues, which are used in the objective function. It also includes constraints to ensure that buying and selling don't occur simultaneously and to link costs and revenues to power flows.

Configuration
-----------
The PyOptInterface implementation for FBC can be configured through the agent configuration file. The configuration is specified in the `ems.controller.fbc` section of the agent config file:

.. code-block:: yaml

    ems:
      controller:
        fbc:
          method: optimization
          horizon: 86_400  # control horizon in seconds
          optimization:
            framework: poi
            solver: gurobi
            time_limit: 120

**Configuration Parameters**

- **method**: The control method to use (set to "optimization" for the PyOptInterface implementation)
- **horizon**: The control horizon in seconds (e.g., 86_400 for 24 hours)
  - Note: Cannot exceed forecast horizon
- **optimization.framework**: The optimization implementation to use (set to "poi" for this implementation)
- **optimization.solver**: The solver to use for the optimization problem (currently only "gurobi" is supported)
- **optimization.time_limit**: Maximum solving time in seconds (default: 120s)
