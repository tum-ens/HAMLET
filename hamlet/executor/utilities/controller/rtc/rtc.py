__author__ = "MarkusDoepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

from linopy import Model, Variable
import polars as pl
from pprint import pprint
from hamlet import constants as c
import hamlet.executor.utilities.controller.rtc.lincomps as lincomps
from hamlet.executor.utilities.controller.base import ControllerBase
from hamlet.executor.utilities.database.database import Database as db


class RtcBase:
    def run(self):
        raise NotImplementedError()


class Rtc(ControllerBase):

    def __init__(self, method='linopy', **kwargs):

        # Call the super class
        super().__init__()

        # Store the method and kwargs
        self.method = method
        self.kwargs = kwargs

        # Mapping from input string to class name
        self.class_mapping = {
            'linopy': self.Linopy,
            'rule-based': self.RuleBased
        }

    def run(self, **kwargs):
        # Return if no method is specified
        if self.method is None:
            return

        # Use the mapping to get the class
        controller_class = self.class_mapping.get(self.method.lower())

        if controller_class is None:
            raise ValueError(f"Unsupported method: {self.method}.\n"
                             f"The available methods are: {self.class_mapping.keys()}")

        return controller_class(**kwargs, mapping=self.component_mapping).run()

    class Linopy(RtcBase):
        def __init__(self, **kwargs):
            # Create the model
            self.model = Model()
            self.vars = {}
            self.cons = {}
            self.obj = None

            # Store the mapping of the components to the energy types and operation modes
            self.mapping = kwargs['mapping']
            # Identify all unique energy types
            self.energy_types = set()
            for mapping in self.mapping.values():
                self.energy_types.update(mapping.keys())

            # Get the timetable and filter it to only include the rows with the current timestep
            self.timetable = kwargs[c.TN_TIMETABLE]
            # Get the delta between timestamps
            self.dt = self.timetable.collect()[1, c.TC_TIMESTEP] - self.timetable.collect()[0, c.TC_TIMESTEP]
            # Filter the timetable to only include the rows with the current timestamp
            self.timetable = self.timetable.filter(
                pl.col(c.TC_TIMESTAMP) == pl.col(c.TC_TIMESTEP))

            # Get the agent and other data
            self.agent = kwargs['agent']
            self.account = self.agent.account
            self.plants = self.agent.plants  # Formerly known as components
            self.setpoints = self.agent.setpoints
            self.socs = self.agent.socs.collect()
            self.timeseries = self.agent.timeseries
            # Filter the timeseries to only include the rows with the current timestamp
            self.timeseries = self.timeseries.join(self.timetable, on=c.TC_TIMESTAMP, how='semi')

            # Raise warning if timeseries exceeds one row
            if len(self.timeseries.collect()) != 1:
                raise ValueError(f"Timeseries has {len(self.timeseries)} rows. It should only have 1 row for the rtc.")

            # Get the market data
            self.market = kwargs['market']
            # Filter the market data to only include the rows concerning the agent
            # TODO: self.market = db.filter_market_data(self.market, [seller, vendor columns], self.agent.id)

            # Obtain maximum balancing power
            # TODO: self.balancing = db.get_balancing_power(market name, energy type)

            # Available plants
            self.available_plants = {
                c.P_INFLEXIBLE_LOAD: lincomps.InflexibleLoad,
                c.P_FLEXIBLE_LOAD: lincomps.FlexibleLoad,
                c.P_HEAT: lincomps.Heat,
                c.P_DHW: lincomps.Dhw,
                c.P_PV: lincomps.Pv,
                c.P_WIND: lincomps.Wind,
                c.P_FIXED_GEN: lincomps.FixedGen,
                c.P_HP: lincomps.Hp,
                c.P_EV: lincomps.Ev,
                c.P_BATTERY: lincomps.Battery,
                c.P_PSH: lincomps.Psh,
                c.P_HYDROGEN: lincomps.Hydrogen,
                c.P_HEAT_STORAGE: lincomps.HeatStorage,
            }

            # Create the plant objects
            self.plant_objects = {}
            self.create_plants()

            # Create the market objects
            self.market_objects = {}
            self.create_markets()

            # Define the model
            self.define_variables()
            self.define_constraints()
            self.define_objective()

        def create_plants(self):
            for plant_name, plant_data in self.plants.items():

                # Get the plant type from the plant data
                plant_type = plant_data['type']

                # Retrieve the timeseries data for the plant
                cols = [col for col in self.timeseries.columns if col.startswith(plant_name)]
                timeseries = self.timeseries.select(cols)

                # Retrieve the soc data for the plant (if applicable)
                cols = [col for col in self.socs.columns if col.startswith(plant_name)]
                socs = self.socs.select(cols)

                # Get the plant class
                plant_class = self.available_plants.get(plant_type)
                if plant_class is None:
                    raise ValueError(f"Unsupported plant type: {plant_name}")

                # Create the plant object
                self.plant_objects[plant_name] = plant_class(name=plant_name, timeseries=timeseries,
                                                             **plant_data, delta=self.dt, socs=socs)

            return self.plant_objects

        def create_markets(self):
            """"""

            # Define variables from the market results and a balancing variable for each energy type
            for energy_type in self.energy_types:
                # Create market object
                self.market_objects[f'market_{energy_type}'] = lincomps.Market(name=f'market_{energy_type}',
                                                                               timeseries=self.market)

                # Create balancing object
                self.market_objects[f'balancing_{energy_type}'] = lincomps.Balancing(name=f'balancing_{energy_type}',
                                                                                     timeseries=self.market)

            return self.market_objects

        def define_variables(self):
            # Define variables for each plant
            for plant_name, plant in self.plant_objects.items():
                self.model = plant.define_variables(self.model, comp_type=self.plants[plant_name]['type'])

            # Define variables for each market
            for market_name, market in self.market_objects.items():
                self.model = market.define_variables(self.model)

            return self.model

        def define_constraints(self):
            # Define constraints for each plant
            for plant_name, plant in self.plant_objects.items():
                plant.define_constraints(self.model)

            # Additional constraints for energy balancing, etc.
            self.add_balance_constraints()

            return self.model

        def add_balance_constraints(self):
            # Initialize the balance equations for each energy type by creating a zero variable for each energy type
            balance_equations = {energy_type: self.model.add_variables(name=f'balance_{energy_type}',
                                                                       lower=0, upper=0, integer=True)
                                 for energy_type in self.energy_types}

            # Loop through each energy type
            for energy_type in self.energy_types:
                # Loop through each variable and add it to the balance equation accordingly
                for variable_name, variable in self.model.variables.items():
                    # Skip the balance variable by checking the naming convention
                    if variable_name.startswith('balance_'):
                        continue
                    # Add the variable as generation if it is a market variable for the current energy type
                    elif variable_name.startswith('market_') and variable_name.endswith(f'_{energy_type}'):
                        balance_equations[energy_type] += variable
                    # Skip the variable as generation if it is a market variable but not of the current energy type
                    elif variable_name.startswith('market_') and not variable_name.endswith(f'_{energy_type}'):
                        continue
                    # Add the variable as generation if it is a balancing variable for the current energy type
                    elif variable_name.startswith('balancing_') and variable_name.endswith(f'_{energy_type}'):
                        balance_equations[energy_type] += variable
                    # Skip the variable as generation if it is a balancing variable but not of the current energy type
                    elif variable_name.startswith('balancing_') and not variable_name.endswith(f'_{energy_type}'):
                        continue
                    else:
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
                            # Note: Generation is positive, load and storage are negative (this follows the convention
                            #       that inflows are positive and outflows are negative)
                            if component_energy_mode == c.OM_GENERATION:
                                balance_equations[energy_type] += variable
                            elif component_energy_mode == c.OM_LOAD or component_energy_mode == c.OM_STORAGE:
                                balance_equations[energy_type] -= variable
                            else:
                                raise ValueError(f"Unsupported operation mode: {component_energy_mode}")
                        else:
                            # The component type is not in the mapping for the current energy type
                            pass

            # Add the constraints for each energy type
            for energy_type, equation in balance_equations.items():
                self.model.add_constraints(equation == 0, name="balance_" + energy_type)

            return self.model

        def define_objective(self):
            print(self.model.variables)
            exit()
            # TODO: This will need to be adjusted in the future once the storage items are following their setpoints
            #  from the mpc
            # Initialize the objective function as zero
            objective = self.model.add_variables(name='objective', lower=0, upper=0, integer=True)

            # Loop through the model's variables to identify the balancing variables
            for variable_name, variable in self.model.variables.items():
                # If the variable name starts with 'balancing_', it's a balancing variable
                if variable_name.startswith('balancing_'):
                    # Add the variable to the objective function
                    objective += variable

            # Set the objective function to the model with the minimize direction
            self.model.add_objective(objective)

            return self.model

        def run(self):

            # Solve the optimization problem
            solver = 'gurobi'
            match solver:
                case 'gurobi':
                    solver_options = {'OutputFlag': 0, 'LogToConsole': 0}
                    solution = self.model.solve(solver_name='gurobi', **solver_options)
                case _:
                    raise ValueError(f"Unsupported solver: {solver}")


            # TODO: Make the model silent and not put out any response.

            # TODO: Solution seems as if one variable is turned upside down (charging or discharging flipped)

            # Check if the solution is optimal
            if solution[0] != 'ok':
                print(self.model.print_infeasibilities())
                raise ValueError(f"Optimization failed: {solution}")

            variable_values = {name: int(var.solution.values.item()) for name, var in self.model.variables.items()}
            pprint(variable_values)

            exit()

            # Process the solution into control commands and return
            control_commands = self.process_solution(solution)
            return control_commands

        def process_solution(self, solution):
            # Process the optimization solution into actionable control commands
            return control_commands

    class RuleBased(RtcBase):  # Note the change in class name

        def __init__(self, **kwargs):
            pass

        def run(self):
            print('Running Rule-Based')

