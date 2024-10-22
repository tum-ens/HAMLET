__author__ = "HodaHamdy"
__credits__ = "MarkusDoepfert"
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

from hamlet.executor.utilities.controller.mpc.mpc_base import MpcBase
from hamlet.executor.utilities.controller.mpc.poi.components import *

# Define all the available plants for this controller
AVAILABLE_PLANTS = {
            c.P_INFLEXIBLE_LOAD: InflexibleLoad,
            c.P_FLEXIBLE_LOAD: FlexibleLoad,
            c.P_HEAT: Heat,
            c.P_DHW: Dhw,
            c.P_PV: Pv,
            c.P_WIND: Wind,
            c.P_FIXED_GEN: FixedGen,
            c.P_HP: Hp,
            c.P_EV: Ev,
            c.P_BATTERY: Battery,
            c.P_PSH: Psh,
            c.P_HYDROGEN: Hydrogen,
            c.P_HEAT_STORAGE: HeatStorage,
        }


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

    def get_available_plants(self):
        return AVAILABLE_PLANTS

    def get_market_class(self):
        return Market

    def define_variables(self):
        self.variables = {}
        # Define variables for each plant
        for plant_name, plant in self.plant_objects.items():
            plant.define_variables(self.model, self.variables, comp_type=self.plants[plant_name]['type'])

        # Define variables for each market
        for market_name, market in self.market_objects.items():
            market.define_variables(self.model, self.variables, comp_type=self.markets[market_name])

    def define_constraints(self):
        # Define constraints for each plant
        for plant_name, plant in self.plant_objects.items():
            plant.define_constraints(self.model, self.variables)

        # Define constraints for each market
        for market_name, market in self.market_objects.items():
            market.define_constraints(self.model, self.variables)

        # Additional constraints for energy balancing, etc.
        self.add_balance_constraints()

    def add_balance_constraints(self):
        # Initialize the balance equations for each energy type by creating a zero variable for each energy type
        balance_equations = {energy_type: [] for energy_type in self.energy_types}
        # Loop through each energy type
        for energy_type in self.energy_types:
            # Loop through each variable and add it to the balance equation accordingly
            for variable_name, variables in self.variables.items():
                # Add the variable as generation if it is a market variable for the current energy type
                if ((variable_name.startswith(tuple(self.market_objects)))
                        and (energy_type in variable_name)
                        and (variable_name.endswith(f'_{c.PF_IN}') or variable_name.endswith(f'_{c.PF_OUT}'))):
                    balance_equations[energy_type].append(variables)
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
                        # Note: All components are modeled positively meaning that positive flows flow into the
                        #  main meter while negative flows flow out of the main meter. The components are modeled
                        #  accordingly
                        if component_energy_mode == c.OM_GENERATION:
                            balance_equations[energy_type].append(variables)
                        elif component_energy_mode == c.OM_LOAD:
                            balance_equations[energy_type].append(variables)
                        elif component_energy_mode == c.OM_STORAGE:
                            balance_equations[energy_type].append(variables)
                        else:
                            raise ValueError(f"Unsupported operation mode: {component_energy_mode}")
                    else:
                        # The component type is not in the mapping for the current energy type
                        pass
                else:
                    # The variable is not a market or plant variable
                    pass

        # Add the constraints for each energy type
        for energy_type, expressions in balance_equations.items():
            timestep_equations = np.sum(expressions, axis=0)
            for timestep, equation in enumerate(timestep_equations):
                self.model.add_linear_constraint(equation, poi.ConstraintSense.Equal, 0,
                                                 name=f"balance_{energy_type}_{timestep}")

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

    def run(self):

        # Solve the optimization problem
        solver = 'gurobi'  # Note: Currently overwriting other solvers as only gurobi is available
        match solver:
            case 'gurobi':
                self.model.set_raw_parameter("TimeLimit", 2)
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

    def get_solution(self):
        # Obtain the solution values
        return {var_name: np.array([self.model.get_value(var) for var in vars]) for var_name, vars in
                self.variables.items()}
