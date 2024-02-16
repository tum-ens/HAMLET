__author__ = "MarkusDoepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

import os
import time

import numpy as np
from linopy import Model, Variable
import polars as pl
import pandas as pd
from pprint import pprint
from hamlet import constants as c
from hamlet.executor.utilities.controller.mpc import lincomps
from hamlet.executor.utilities.controller.controller_base import ControllerBase
from hamlet.executor.utilities.database.database import Database as db
from hamlet import functions as f
import sys

AGENT_ID = '1q5Nid2Bwz2WzxG' # 'HnYxh1u9BEFpWyK'


class MpcBase:
    def run(self):
        raise NotImplementedError()


class Mpc(ControllerBase):

    def __init__(self, method='linopy', **kwargs):

        # Call the super class
        super().__init__()

        # Store the method and kwargs
        self.method = method
        self.kwargs = kwargs

        # Mapping from input string to class name
        self.class_mapping = {
            'linopy': self.Linopy,
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

        return controller_class(**kwargs, mapping=c.COMP_MAP).run()

    class Linopy(MpcBase):
        def __init__(self, **kwargs):
            # Create the model
            self.model = Model(force_dim_names=True)

            # Store the mapping of the components to the energy types and operation modes
            self.mapping = kwargs['mapping']
            # Identify all unique energy types
            self.energy_types = set()
            for mapping in self.mapping.values():
                self.energy_types.update(mapping.keys())

            # Get the agent and other data
            self.agent = kwargs['agent']
            self.account = self.agent.account
            self.ems = self.account['ems']
            self.plants = self.agent.plants  # Formerly known as components
            self.setpoints = self.agent.setpoints
            self.forecasts = self.agent.forecasts
            self.socs = self.agent.socs

            # Get the timetable
            self.timetable = kwargs[c.TN_TIMETABLE]
            # Get the delta between timestamps
            self.dt = self.timetable[1, c.TC_TIMESTEP] - self.timetable[0, c.TC_TIMESTEP]
            # Get the current timestamp
            self.timestamp = self.timetable[0, c.TC_TIMESTAMP]
            # Get the time horizon
            self.horizon = pd.Timedelta(seconds=self.ems['controller']['mpc']['horizon'])

            # Reduce the forecast to the horizon length
            self.forecasts = self.forecasts.filter((self.timestamp < self.forecasts[c.TC_TIMESTEP])
                                                   & (self.forecasts[c.TC_TIMESTEP] < self.timestamp + self.horizon))
            # Get the timesteps and the number of timesteps in the forecast
            self.timesteps = pd.Index(self.forecasts.select(c.TC_TIMESTEP).to_pandas(), name='timesteps')
            self.timesteps = pd.Index(range(len(self.timesteps)), name='timesteps')
            # self.timesteps = self.n_steps  # Use this line if the timesteps are not needed and the index is sufficient
            # Reduce the socs to the current timestamp
            self.socs = self.socs.filter(self.socs[c.TC_TIMESTAMP] == self.timestamp)

            # Get the market types
            # TODO: Still needs to be done and then adjusted in the market objects (right now the names are simply
            #  local and wholesale as this will suffice as long as there is only one market)
            # Get the market data
            self.market = kwargs[c.TC_MARKET]
            # Get the market names and types
            self.market_names = self.timetable.select(c.TC_NAME).unique().to_series().to_list()
            self.market_types = self.timetable.select(c.TC_MARKET).unique().to_series().to_list()
            # Assign each market name to an energy type
            self.markets = {name: c.TRADED_ENERGY[mtype] for name, mtype in zip(self.market_names, self.market_types)}

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

                # Retrieve the forecast data for the plant
                cols = [col for col in self.forecasts.columns if col.startswith(plant_name)]
                forecasts = self.forecasts.select(cols)

                # Retrieve the soc data for the plant (if applicable)
                cols = [col for col in self.socs.columns if col.startswith(plant_name)]
                socs = self.socs.select(cols)

                # Get the plant class
                plant_class = self.available_plants.get(plant_type)
                if plant_class is None:
                    raise ValueError(f"Unsupported plant type: {plant_name}")

                # Create the plant object
                self.plant_objects[plant_name] = plant_class(name=plant_name,
                                                             forecasts=forecasts,
                                                             **plant_data,
                                                             socs=socs,
                                                             delta=self.dt,
                                                             timesteps=self.timesteps)

            return self.plant_objects

        def create_markets(self):
            """"""

            # Define variables from the market results and a balancing variable for each energy type
            for market in self.markets:
                # Create market object
                self.market_objects[f'{market}'] = lincomps.Market(name=market,
                                                                   forecasts=self.forecasts,
                                                                   timesteps=self.timesteps,
                                                                   delta=self.dt)

            return self.market_objects

        def define_variables(self):
            # Define variables for each plant
            for plant_name, plant in self.plant_objects.items():
                self.model = plant.define_variables(self.model, comp_type=self.plants[plant_name]['type'])

            # Define variables for each market
            for market_name, market in self.market_objects.items():
                self.model = market.define_variables(self.model, comp_type=self.markets[market_name])

            return self.model

        def define_constraints(self):
            # Define constraints for each plant
            for plant_name, plant in self.plant_objects.items():
                self.model = plant.define_constraints(self.model)

            # Define constraints for each market
            for market_name, market in self.market_objects.items():
                self.model = market.define_constraints(self.model)

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
                    # Add the variable as generation if it is a market variable for the current energy type
                    if ((variable_name.startswith(tuple(self.market_objects)))
                            and (energy_type in variable_name)
                            and (variable_name.endswith(f'_{c.PF_IN}') or variable_name.endswith(f'_{c.PF_OUT}'))):
                        balance_equations[energy_type] += variable
                    # Add the variable if it is a plant variable
                    elif (variable_name.startswith(tuple(self.plant_objects)))\
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
                                balance_equations[energy_type] += variable
                            elif component_energy_mode == c.OM_LOAD:
                                balance_equations[energy_type] += variable
                            elif component_energy_mode == c.OM_STORAGE:
                                balance_equations[energy_type] += variable
                            else:
                                raise ValueError(f"Unsupported operation mode: {component_energy_mode}")
                        else:
                            # The component type is not in the mapping for the current energy type
                            pass
                    else:
                        # The variable is not a market or plant variable
                        pass

            # Add the constraints for each energy type
            for energy_type, equation in balance_equations.items():
                self.model.add_constraints(equation == 0, name="balance_" + energy_type)

            return self.model

        def define_objective(self):
            """Defines the objective function. The objective is to reduce the costs."""

            # Initialize the objective function as zero
            objective = self.model.add_variables(name='objective', lower=0, upper=0, integer=True)

            # Loop through the model's variables to identify the balancing variables
            for variable_name, variable in self.model.variables.items():
                # Only consider the cost and revenue components of the markets
                if variable_name.startswith(tuple(self.market_names)):
                    if variable_name.endswith('_costs'):
                        # Add the variable to the objective function
                        objective += variable
                    elif variable_name.endswith('_revenue'):
                        # Subtract the variable from the objective function
                        objective -= variable
                    else:
                        pass
                else:
                    pass

            # Set the objective function to the model with the minimize direction
            self.model.add_objective(objective.sum())

            return self.model

        def run(self):

            # Solve the optimization problem
            solver = 'gurobi'  # Note: Currently overwriting other solvers as only gurobi is available
            match solver:
                case 'gurobi':
                    sys.stdout = open(os.devnull, 'w')  # deactivate printing from linopy
                    solver_options = {'OutputFlag': 0, 'LogToConsole': 0, 'TimeLimit': 2}
                    status = self.model.solve(solver_name='gurobi', **solver_options)
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

        def process_solution(self):

            # Obtain the solution values
            solution = {name: sol for name, sol in self.model.solution.items()}
            # Update setpoints
            self.setpoints = self.update_setpoints(solution)

            # Update the agent
            self.agent.setpoints = self.setpoints

            return self.agent

        def update_setpoints(self, solution: dict):

            # Get relevant column names
            src_cols = self.__get_src_cols(solution)

            # Change the solution so that the in and out columns are computed as one
            adjusted_solution = {}
            for col in src_cols:
                # Check if the column is an in or out column
                if col.endswith(f'_{c.PF_IN}'):
                    key = col.rsplit('_', 1)[0]
                    val = (np.array(solution[col].round()) + np.array(solution[key + f'_{c.PF_OUT}'].round())).astype(
                        int)
                # Skip the out columns as they are already included in the in columns
                elif col.endswith(f'_{c.PF_OUT}'):
                    continue
                else:
                    key = col
                    val = np.array(solution[col].round()).astype(int)

                # Add value to the adjusted solution
                adjusted_solution[key] = val

            # Make adjusted_solution into DataFrame
            adjusted_solution = pl.DataFrame(adjusted_solution)
            # Add timestamp column from setpoints
            adjusted_solution = adjusted_solution.hstack(self.setpoints.select(c.TC_TIMESTAMP).slice(1))

            # Update setpoints
            self.setpoints = self.setpoints.update(adjusted_solution, on=c.TC_TIMESTAMP)

            # with pl.Config(set_tbl_width_chars=400, set_tbl_cols=25, set_tbl_rows=100):
            #     print(self.model.objective)
            #     print(self.model.solution.to_pandas().to_string())
            #     print(self.setpoints)

            return self.setpoints

        def __get_src_cols(self, solution):
            # Get the relevant columns based on market names, plant names, plants and energy types
            src_cols = []
            for idx, sol_var in enumerate(solution.keys()):
                sol = sol_var
                # Check if sol starts with a market name
                if sol.startswith(tuple(self.market_objects.keys())):
                    # Get the plant it starts with
                    market = next((market for market in self.market_objects if market in sol))
                    # Subtract the market from sol
                    sol = sol.replace(f'{market}_', '')
                    # Check if sol starts with an energy type
                    if sol.startswith(tuple(self.energy_types)):
                        # Set src_cols to one assuming it contains a relevant column
                        src_cols.append(sol_var)
                # Check if sol starts with a plant name
                if sol.startswith(tuple(self.plant_objects.keys())):
                    # Subtract the name from sol
                    sol = sol.split('_', 1)[1]
                    # Check if sol starts with an available plant type
                    if sol.startswith(tuple(self.available_plants)):
                        # Subtract the plant from sol
                        sol = sol.split('_', 1)[1]
                        # Check if sol starts with an energy type
                        if sol.startswith(tuple(self.energy_types)):
                            # Set src_cols to one assuming it contains a relevant column
                            src_cols.append(sol_var)

            return src_cols

        def __get_src_cols2(self, solution):
            # Note: Twice as fast but currently not deployed as it does not allow underscores in market or plant names
            # Pre-compute tuples
            market_keys = tuple(self.market_objects.keys())
            plant_keys = tuple(self.plant_objects.keys())
            energy_types = tuple(self.energy_types)
            available_plants = tuple(self.available_plants)

            src_cols = []
            for sol_var in solution.keys():
                sol = sol_var

                # Market name check
                if sol.startswith(market_keys):
                    market, _, remainder = sol.partition('_')
                    if remainder.startswith(energy_types):
                        src_cols.append(sol_var)
                        continue  # Skip to next iteration

                # Plant name check
                if sol.startswith(plant_keys):
                    plant, remainder = sol.split('_', 1)
                    if remainder.startswith(available_plants):
                        _, energy_type = remainder.split('_', 1)
                        if energy_type.startswith(energy_types):
                            src_cols.append(sol_var)

            return src_cols

        def __get_src_cols3(self, solution):
            # Note: Probably fastest but does also not allow underscores in market or plant names
            # Convert the keys to sets for faster operations
            solution_keys = set(solution.keys())
            market_keys = set(self.market_objects.keys())
            plant_keys = set(self.plant_objects.keys())
            energy_types = set(self.energy_types)
            available_plants = set(self.available_plants)

            # Filter the intersections based on energy types
            market_intersection = {key for key in solution_keys
                                   if (any(market in key for market in market_keys)
                                       and any(energy in key for energy in energy_types))}

            plant_intersection = {key for key in solution_keys
                                  if (any(plant in key for plant in plant_keys)
                                      and any(energy in key for energy in energy_types)
                                      and any(plant in key for plant in available_plants))}

            # Combine the intersections
            src_cols = list(market_intersection | plant_intersection)

            return src_cols
