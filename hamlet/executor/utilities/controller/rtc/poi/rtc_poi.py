__author__ = "HodaHamdy"
__credits__ = "MarkusDoepfert"
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

from pyoptinterface import gurobi

from hamlet.executor.utilities.controller.rtc.poi.components import *
from hamlet.executor.utilities.controller.rtc.rtc_base import RtcBase

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


class POI(RtcBase):
    def get_model(self, **kwargs):
        model = gurobi.Model()
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
            # Balancing markets are not explicitly modeled and have the same comp_type as their original market
            if c.TT_BALANCING in market_name:
                energy_type = self.markets[market_name.rsplit('_', 1)[0]]
            else:
                energy_type = self.markets[market_name]

            market.define_variables(self.model, self.variables, energy_type=energy_type)

        return self.model

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
                        # Note: All components are modeled positively meaning that positive flows flow into the
                        #  main meter while negative flows flow out of the main meter. The components are modeled
                        #  accordingly
                        if component_energy_mode == c.OM_GENERATION:
                            balance_equations[energy_type].append(variable)
                        elif component_energy_mode == c.OM_LOAD:
                            balance_equations[energy_type].append(variable)
                        elif component_energy_mode == c.OM_STORAGE:
                            balance_equations[energy_type].append(variable)
                        else:
                            raise ValueError(f"Unsupported operation mode: {component_energy_mode}")
                    else:
                        # The component type is not in the mapping for the current energy type
                        pass
                else:
                    pass

        # Add the constraints for each energy type
        for energy_type, vars in balance_equations.items():
            self.model.add_linear_constraint(sum(vars), poi.ConstraintSense.Equal, 0,
                                                 name=f"balance_{energy_type}")

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

    def run(self):

        # Solve the optimization problem
        solver = 'gurobi'  # Note: Currently overwriting other solvers as only gurobi is available
        match solver:
            case 'gurobi':
                self.model.set_model_attribute(poi.ModelAttribute.Silent, True)
                self.model.set_raw_parameter("OutputFlag", 0)
                self.model.set_raw_parameter("LogToConsole", 0)
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

    def get_solution(self):
        # Obtain the solution values
        return {var_name: int(self.model.get_value(var)) for var_name, var in self.variables.items()}
