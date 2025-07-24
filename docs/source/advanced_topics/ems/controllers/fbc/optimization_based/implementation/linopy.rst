linopy
======

Introduction
------------
The Linopy implementation for Forecast-Based Controllers (FBC) in HAMLET provides a high-level approach to formulating and solving multi-period optimization problems. It leverages the Linopy package, which offers a convenient interface for creating and solving linear and mixed-integer linear programming problems using labeled arrays (via xarray).

The implementation is organized around two main Python files:
- ``components.py``: Defines component models for various energy system elements over multiple timesteps
- ``mpc_linopy.py``: Implements the main optimization controller with forecast integration

This implementation is designed to be readable, maintainable, and accessible to users who may not have extensive experience with mathematical optimization. The key difference from the Real-Time Controller (RTC) implementation is the handling of multiple timesteps and forecasts, enabling the controller to make decisions that optimize performance over a prediction horizon rather than just for the current timestep.

Objective Function
------------------
The primary goal of the Linopy implementation for FBC is to minimize operational costs over the entire prediction horizon. Unlike the RTC implementation, which focuses on minimizing deviations from target values, the FBC implementation directly minimizes costs and maximizes revenues from market interactions.

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

The objective function is defined in the ``define_objective`` method of the ``Linopy`` class:

.. code-block:: python

    def define_objective(self):
        """Defines the objective function. The objective is to reduce the costs."""

        # Initialize the objective function as zero
        objective = []

        # Loop through the model's variables to identify the balancing variables
        for variable_name, variable in self.model.variables.items():
            # Only consider the cost and revenue components of the markets
            if variable_name.startswith(tuple(self.market_names)):
                if variable_name.endswith('_costs'):
                    # Add the variable to the objective function
                    objective.append(variable)
                elif variable_name.endswith('_revenue'):
                    # Subtract the variable from the objective function
                    objective.append(-1 * variable)
                else:
                    pass
            else:
                pass

        # Set the objective function to the model with the minimize direction
        self.model.add_objective(sum(objective), overwrite=True)

        return self.model

The objective function minimizes the total cost over the prediction horizon by:
1. Adding cost variables (representing expenses for buying energy)
2. Subtracting revenue variables (representing income from selling energy)

This approach allows the controller to make decisions that optimize economic performance over time, taking advantage of price variations and forecasted conditions to minimize overall costs.

Code Implementation
-------------------
The main implementation of the Linopy controller for FBC is in the ``mpc_linopy.py`` file, which defines the ``Linopy`` class:

.. code-block:: python

    class Linopy(MpcBase):
        def __init__(self, **kwargs):
            self.loaded_model = False
            self.model_path = f"{kwargs['agent'].agent_save}/linopy_mpc.nc"
            super().__init__(**kwargs)
            self.ems = self.ems[c.C_OPTIM]
            # Save first model to file to load later
            self.save_model()

The class inherits from ``MpcBase``, which provides common functionality for Model Predictive Control (MPC) based controllers. This is different from the RTC implementation, which inherits from ``OptimBase``.

**Model Initialization**

The model is initialized in the ``get_model`` method, with a key difference from the RTC implementation being the use of ``force_dim_names=True``, which is important for handling the time dimension in multi-period optimization:

.. code-block:: python

    def get_model(self, **kwargs):
        # Check for existing saved models
        if os.path.exists(self.model_path):
            # Load model
            model = read_netcdf(self.model_path)
            self.loaded_model = True
        else:
            # Create a new model
            model = Model(force_dim_names=True)
        return model

**Solving the Model**

The model is solved in the ``run`` method:

.. code-block:: python

    def run(self):
        # Solve the optimization problem
        solver = self.ems.get('solver')
        match solver:
            case 'gurobi' | 'highs':
                sys.stdout = open(os.devnull, 'w')  # deactivate printing from linopy
                solver_options = {'OutputFlag': 0, 'LogToConsole': 0}
                if self.ems.get('time_limit') is not None:
                    solver_options.update({'TimeLimit': self.ems['time_limit'] / 60})
                status = self.model.solve(solver_name=solver, **solver_options)
                sys.stdout = sys.__stdout__  # re-activate printing
            case _:
                raise ValueError(f"Unsupported solver: {solver}")

        # Check if the solution is optimal
        if status[0] != 'ok':
            print(f'Exited with status "{status[0]}". \n '
                  f'Infeasibilities for agent {self.agent.agent_id}:')
            print(self.model.print_infeasibilities())

            print('Model:')
            for name, var in self.model.variables.items():
                print(var)
            for name, con in self.model.constraints.items():
                print(con)
            print(self.model.objective)

            raise ValueError(f"Optimization failed: {status}")

        # Process the solution into control commands and return
        self.agent = self.process_solution()

        return self.agent

Mathematical Formulation
------------------------
The Linopy implementation for FBC follows the general mathematical formulation described in the Mathematical Formulation section, with specific adaptations for multi-period optimization and forecast integration.

**Problem Structure**

The optimization problem is formulated as a minimization problem over multiple timesteps.
In the code, this is implemented using Linopy's labeled arrays with a time dimension:

.. code-block:: python

    # Create a new model with dimension names
    model = Model(force_dim_names=True)
    
    # Define variables with time dimension
    self.add_variable_to_model(model, name=name, lower=lower, upper=upper, coords=[self.timesteps])
    
    # Define constraints with time dimension
    model.add_constraints(equation, name=name, coords=[self.timesteps])
    
    # Define objective
    self.define_objective()
    
    # Solve the model
    self.model.solve(solver_name=solver, **solver_options)

**Decision Variables**

Variables are defined for each component and each timestep in the optimization horizon. For example, for a battery storage component:

.. code-block:: python

    # Define the power variables (need to be positive and negative due to the efficiency)
    model = self.define_electricity_variable(model, comp_type=self.comp_type, lower=self.lower, upper=0,
                                             direction=c.PF_OUT)  # flow out of the home (charging battery)
    model = self.define_electricity_variable(model, comp_type=self.comp_type, lower=0, upper=self.upper,
                                             direction=c.PF_IN)  # flow into the home (discharging battery)
    
    # Define mode flag that decides whether the battery is charging or discharging
    model = self.define_mode_flag(model, comp_type=self.comp_type)
    
    # Define the soc variable
    model = self.define_storage_variable(model, comp_type=self.comp_type, lower=0, upper=self.capacity)

**Temporal Coupling Constraints**

A key feature of FBC is the inclusion of constraints that couple variables across timesteps. For example, the state-of-charge evolution for a battery:

.. code-block:: python

    def _constraint_soc(self, model: Model, energy_type: str = c.ET_ELECTRICITY) -> Model:
        """Adds the constraint that the soc of the battery is that of the previous timestep plus dis-/charging power"""

        # Define the variables
        dt_hours = pd.Series([self.dt * c.SECONDS_TO_HOURS] * len(self.timesteps), index=self.timesteps)  # time in h
        efficiency = pd.Series([self.efficiency] * len(self.timesteps), index=self.timesteps)  # efficiency
        var_soc = model.variables[f'{self.name}_{self.comp_type}_soc']  # soc variable
        var_charge = model.variables[f'{self.name}_{self.comp_type}_{energy_type}_{c.PF_OUT}']  # charging power
        var_discharge = model.variables[f'{self.name}_{self.comp_type}_{energy_type}_{c.PF_IN}']  # discharging

        # Define the array that contains all previous socs
        var_soc_init = model.variables[f'{self.name}_{self.comp_type}_soc_init']  # current soc
        var_soc_prev = var_soc.roll(timesteps=1)  # previous soc
        # Update the first soc value with the initial soc
        var_soc_prev.lower[0] = var_soc_init.lower
        var_soc_prev.upper[0] = var_soc_init.upper
        var_soc_prev.labels[0] = var_soc_init.labels

        # Define the constraint for charging
        # Constraint: soc_new = soc_old + charge * efficiency * dt - discharge / efficiency * dt
        cons_name = f'{self.name}_soc'
        if cons_name not in model.constraints:
            eq = (var_soc
                  - var_soc_prev
                  + var_charge * efficiency * dt_hours
                  + var_discharge / efficiency * dt_hours
                  == 0)
            model.add_constraints(eq, name=cons_name)
        else:
            model.constraints[cons_name].coeffs[:, 2] = dt_hours * efficiency
            model.constraints[cons_name].coeffs[:, 3] = dt_hours / efficiency

        return model

**Energy Balance Constraints**

Energy balance constraints ensure that supply matches demand for each energy carrier at each timestep:

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
                if ((variable_name.startswith(tuple(self.market_objects)))
                        and (energy_type in variable_name)
                        and (variable_name.endswith(f'_{c.PF_IN}') or variable_name.endswith(f'_{c.PF_OUT}'))):
                    balance_equations[energy_type] += variable
                # Add the variable if it is a plant variable
                elif (variable_name.startswith(tuple(self.plant_objects))) \
                        and (variable_name.endswith(f'_{energy_type}')
                             or variable_name.endswith(f'_{energy_type}_{c.PF_IN}')
                             or variable_name.endswith(f'_{energy_type}_{c.PF_OUT}')):
                    # Get the component name by splitting the variable name at the underscore
                    component_name = variable_name.split('_', 1)[0]

                    # Get the component type by comparing the ID with the plant names
                    component_type = [vals['type'] for plant, vals in self.plants.items()
                                      if plant == component_name][0]

                    # If the component type is in the mapping for the current energy type and the variable is for
                    # the energy type, add the variable to the balance equation
                    if (energy_type in self.mapping[component_type].keys()) and (energy_type in variable_name):
                        # Get the operation mode for the component and energy type
                        component_energy_mode = self.mapping[component_type][energy_type]

                        # Add the variable to the balance equation
                        balance_equations[energy_type] += variable
                    else:
                        # The component type is not in the mapping for the current energy type
                        pass
                else:
                    # The variable is not a market or plant variable
                    pass

        # Add the constraints for each energy type
        for energy_type, equation in balance_equations.items():
            self.model.add_constraints(equation == 0, name="balance_" + energy_type)

Component Models
----------------
The Linopy implementation for FBC includes models for various energy system components, defined in the ``components.py`` file. Each component is implemented as a class that inherits from the base ``LinopyComps`` class. Here we focus on three key component models: inflexible load, PV, and market.

**Base Component Class**

The ``LinopyComps`` class provides common functionality for all components:

.. code-block:: python

    class LinopyComps:
        def __init__(self, name, forecasts, **kwargs):
            self.name = name
            self.fcast = forecasts
            self.timesteps = kwargs['timesteps']
            self.info = kwargs

        def define_variables(self, model, **kwargs):
            raise NotImplementedError()

        @staticmethod
        def define_constraints(model):
            return model

        @staticmethod
        def add_variable_to_model(model, name, **kwargs):
            """Wrapper for adding variables to models"""
            if name not in model.variables:
                model.add_variables(name=name, **kwargs)
            else:
                # adjust lower and upper bounds only
                model.variables[name].lower = kwargs.get("lower", -math.inf)
                model.variables[name].upper = kwargs.get("upper", math.inf)

Note that unlike the RTC implementation, the FBC implementation uses ``forecasts`` instead of ``timeseries`` and includes a ``timesteps`` parameter for handling the time dimension.

**Inflexible Load**

The ``InflexibleLoad`` class represents electrical loads that cannot be controlled or shifted. These loads must be satisfied exactly as specified for each timestep in the horizon:

.. code-block:: python

    class InflexibleLoad(LinopyComps):
        def __init__(self, name, **kwargs):
            # Call the parent class constructor
            super().__init__(name, **kwargs)

            # Get specific object attributes
            self.power = pd.Series(self.fcast[f'{self.name}_{c.ET_ELECTRICITY}'].cast(int), index=self.timesteps, dtype='int32')

        def define_variables(self, model, **kwargs):
            comp_type = kwargs['comp_type']

            # Define the power variable
            model = self.define_electricity_variable(model, comp_type=comp_type, lower=-self.power, upper=-self.power)

            return model

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

    class SimplePlant(LinopyComps):
        def __init__(self, name, **kwargs):
            # Call the parent class constructor
            super().__init__(name, **kwargs)

            # Get specific object attributes
            self.power = list(self.fcast[f'{self.name}_{c.ET_ELECTRICITY}'])
            self.controllable = self.info['sizing']['controllable']
            self.lower = [0] * len(self.power) if self.controllable else self.power

            self.lower = pd.Series(self.lower, index=self.timesteps)
            self.power = pd.Series(self.power, index=self.timesteps)

        def define_variables(self, model, **kwargs):
            comp_type = kwargs['comp_type']

            # Define the power variable
            model = self.define_electricity_variable(model, comp_type=comp_type, lower=self.lower, upper=self.power)

            return model

PV systems have a power variable with a lower bound of 0 and an upper bound equal to the forecasted available power for each timestep, allowing for curtailment when necessary.

**Market**

The market component represents the connection to external energy networks, with buying and selling capabilities over the prediction horizon:

.. code-block:: python

    class Market(LinopyComps):
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

        def define_variables(self, model, **kwargs):
            self.comp_type = kwargs['comp_type']
            # Define the market power variables (need to be positive and negative due to different pricing)
            self.add_variable_to_model(model, name=f'{self.name}_{self.comp_type}_{c.PF_OUT}', lower=self.lower, upper=0,
                                       coords=[self.timesteps], integer=False)  # outflow from the building (selling)
            self.add_variable_to_model(model, name=f'{self.name}_{self.comp_type}_{c.PF_IN}', lower=0, upper=self.upper,
                                       coords=[self.timesteps], integer=False)  # inflow into the building (buying)

            # Define mode flag that decides whether the market energy is bought or sold
            self.add_variable_to_model(model, name=f'{self.name}_mode', coords=[self.timesteps], binary=True)

            # Define the market cost and revenue variables
            self.add_variable_to_model(model, name=f'{self.name}_costs', lower=0, upper=np.inf, coords=[self.timesteps])
            self.add_variable_to_model(model, name=f'{self.name}_revenue', lower=0, upper=np.inf, coords=[self.timesteps])

            return model

        def define_constraints(self, model):
            # Add constraint that the market can either buy or sell but not both at the same time
            model = self.__constraint_operation_mode(model)

            # Add constraint that the market cost and revenue are linked to the power
            model = self.__constraint_cost_revenue(model)

            return model

The market component includes variables for buying and selling power, as well as for costs and revenues, which are used in the objective function. It also includes constraints to ensure that buying and selling don't occur simultaneously and to link costs and revenues to power flows.

Configuration
-----------
The Linopy implementation for FBC can be configured through the agent configuration file. The configuration is specified in the `ems.controller.fbc` section of the agent config file:

.. code-block:: yaml

    ems:
      controller:
        fbc:
          method: optimization
          horizon: 86_400  # control horizon in seconds
          optimization:
            framework: linopy
            solver: gurobi
            time_limit: 120

**Configuration Parameters**

- **method**: The control method to use (set to "optimization" for the Linopy implementation)
- **horizon**: The control horizon in seconds (e.g., 86_400 for 24 hours)
  - Note: Cannot exceed forecast horizon
- **optimization.framework**: The optimization implementation to use (set to "linopy" for this implementation)
- **optimization.solver**: The solver to use for the optimization problem
- **optimization.time_limit**: Maximum solving time in seconds (default: 120s)
