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
from hamlet.executor.utilities.controller.base import ControllerBase
from hamlet.executor.utilities.database.database import Database as db
import math  # TODO: Take back out again once forecasts exist
from hamlet import functions as f


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
            self.model = Model()

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
            # self.forecasts = self.agent.timeseries.collect()  # TODO: Replace with forecasts once they exist
            self.forecasts = self.agent.forecasts.collect()
            with pl.Config() as cfg:
                cfg.set_tbl_width_chars(200)
                cfg.set_tbl_cols(30)
                print(self.forecasts)
            print('Forecast is now implemented. Next step is to replace the dummy data with the actual forecasts. \n'
                  'Also probably necessary to convert the retailer data as it is still using fractions')
            exit()
            self.socs = self.agent.socs.collect()

            # Get the timetable
            self.timetable = kwargs[c.TN_TIMETABLE]
            # Get the delta between timestamps
            self.dt = self.timetable.collect()[1, c.TC_TIMESTEP] - self.timetable.collect()[0, c.TC_TIMESTEP]
            # Get the current timestamp
            self.timestamp = self.timetable.collect()[0, c.TC_TIMESTAMP]
            # Get the next timestamp
            self.next_timestamp = self.timestamp + self.dt  # TODO: Might not be needed
            # Get the time horizon
            self.horizon = pd.Timedelta(seconds=self.ems['controller']['mpc']['horizon'])

            # Reduce the forecast to the horizon length
            self.forecasts = self.forecasts.filter((self.timestamp < self.forecasts[c.TC_TIMESTAMP])
                                                   & (self.forecasts[c.TC_TIMESTAMP] < self.timestamp + self.horizon))
            # Get the timesteps and the number of timesteps in the forecast
            self.timesteps = pd.Index(self.forecasts.select(c.TC_TIMESTAMP).to_pandas(), name='timesteps')
            self.n_steps = pd.Index(range(len(self.timesteps)), name='timesteps')
            self.timesteps = self.n_steps  # TODO: Take out again
            # Reduce the socs to the current timestamp
            self.socs = self.socs.filter(self.socs[c.TC_TIMESTAMP] == self.timestamp)

            # Get the market types
            # TODO: Still needs to be done and then adjusted in the market objects
            # Get the market data
            self.market = kwargs['market']
            # Get the market names and types
            self.market_names = self.timetable.collect().select(c.TC_NAME).unique().to_series().to_list()
            self.market_types = self.timetable.collect().select(c.TC_MARKET).unique().to_series().to_list()
            # Assign each market name to an energy type
            self.markets = {name: c.TRADED_ENERGY[mtype] for name, mtype in zip(self.market_names, self.market_types)}

            # TODO: Take out again once the forecasts are available
            # Create dummy forecasts for the market and append them to the forecasts
            dummy = {}
            dummy['energy_price_sell'] = [(math.cos(x)  + 4) / 100 for x in np.linspace(0, 2 * math.pi, len(self.timesteps))]
            dummy['energy_price_buy'] = [(2 * math.cos(x) + 6) / 100 for x in np.linspace(0, 2 * math.pi, len(self.timesteps))]
            dummy['energy_quantity_sell'] = [1e5] * len(self.timesteps)
            dummy['energy_quantity_buy'] = [1e5] * len(self.timesteps)
            dummy['grid_sell'] = [0] * len(self.timesteps)
            dummy['grid_buy'] = [0.04] * len(self.timesteps)
            dummy['grid_retail_sell'] = [0] * len(self.timesteps)
            dummy['grid_retail_buy'] = [0.08] * len(self.timesteps)
            dummy['levies_price_sell'] = [0] * len(self.timesteps)
            dummy['levies_price_buy'] = [0.18] * len(self.timesteps)

            # Combine the forecasts with the dummy forecasts
            for col, vals in dummy.items():
                self.forecasts = self.forecasts.with_columns(pl.Series(name=col, values=vals))
            # print(self.forecasts)
            # print(self.forecasts.columns)

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

                # "inflexible_load", "heat", "dhw", "pv", "ev", "battery"
                # TODO: Take out again once hp is implemented
                if plant_type in ["heat", "dhw"]:
                    continue

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
            for market in self.markets: # TODO: Change once it is clear how the markets are defined
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
                    # Note: Markets are modeled as generators.
                    if ((variable_name.startswith(tuple(self.market_objects.keys())))
                          and (energy_type in variable_name)
                          and (variable_name.endswith(f'_{c.PF_IN}') or variable_name.endswith(f'_{c.PF_OUT}'))):
                        balance_equations[energy_type] += variable
                    # Add the variable if it is a plant variable
                    elif variable_name.startswith(tuple(self.plant_objects.keys())):
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
                            # Note: Generation is positive, load and storage are negative (this follows the convention
                            #       that inflows are positive and outflows are negative)
                            # TODO: Maybe change this already to positive and negative in the plant objects to avoid confusion
                            if component_energy_mode == c.OM_GENERATION:
                                balance_equations[energy_type] += variable
                            elif component_energy_mode == c.OM_LOAD or component_energy_mode == c.OM_STORAGE:
                                balance_equations[energy_type] -= variable
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

            # exit()


            return self.model

        def define_objective(self):
            """Defines the objective function. The objective is to reduce the costs."""

            # Initialize the objective function as zero
            objective = self.model.add_variables(name='objective', lower=0, upper=0, integer=True)

            # TODO: This is currently not the correct format as the names of the markets are not correct yet.
            #  They will be replaced by the names of the market inputs

            # TODO: Another issue that in the future there needs to be a forecast for wholesale and local prices
            #  (ponder if regional also needs to be addressed here)

            # Loop through the model's variables to identify the balancing variables
            for variable_name, variable in self.model.variables.items():
                # If the variable name starts with 'balancing_', it's a balancing variable
                if variable_name.startswith('market_'):
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
            solver = 'gurobi'
            match solver:
                case 'gurobi':
                    # solver_options = {'OutputFlag': 0, 'LogToConsole': 0}
                    status = self.model.solve(solver_name='gurobi') #, **solver_options)
                case _:
                    raise ValueError(f"Unsupported solver: {solver}")

            # TODO: Make the model silent and not put out any response.

            # Check if the solution is optimal
            if status[0] != 'ok':
                # subset = self.model.compute_set_of_infeasible_constraints()
                # print(subset)
                # labels = self.model.compute_infeasibilities()
                # print(labels)
                # print(self.model.constraints.print_labels(labels))
                print(self.model.print_infeasibilities())
                raise ValueError(f"Optimization failed: {status}")

            # Process the solution into control commands and return
            self.agent = self.process_solution()

            return self.agent

        def process_solution(self):

            # Obtain the solution values
            solution = {name: sol for name, sol in self.model.solution.items()}

            # TODO: Solution is definitely not correct yet but it is a start: Focus on market results first

            # Update setpoints
            self.setpoints = self.update_setpoints(solution)

            # Update the agent
            self.agent.setpoints = self.setpoints

            return self.agent

        def update_setpoints(self, solution: dict):

            # Make LazyFrame into DataFrame
            self.setpoints = self.setpoints.collect()

            # Get relevant column name components (i.e. the plant names and market and balancing)
            ids = set([col.split('_', 1)[0] for col in solution.keys()
                              if not col.startswith('objective') and not col.startswith('balance')])

            # with pl.Config(tbl_cols=20, fmt_str_lengths=50):
            #     print(self.setpoints)
            # Get the relevant columns
            # Filter for columns that start with one of the ids
            src_cols = [col for col in solution.keys() for i in ids if i in col]
            # Filter for columns that contain the energy type
            src_cols = [col for col in src_cols for e in self.energy_types if e in col]

            # print(np.array(solution[f'continuous_power_{c.PF_IN}']).astype(int))
            # print(np.array(solution[f'continuous_power_{c.PF_OUT}']).astype(int))

            # Change the solution so that the in and out columns are computed as one
            # TODO: Take out round() once values are surely correct
            adjusted_solution = {}
            for col in src_cols:
                # Check if the column is an in or out column
                if col.endswith(f'_{c.PF_IN}'):
                    key = col.rsplit('_', 1)[0]
                    val = (np.array(solution[col].round()) + np.array(solution[key + f'_{c.PF_OUT}'].round())).astype(int)
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
            # print(adjusted_solution)

            # Update setpoints
            self.setpoints = self.setpoints.update(adjusted_solution, on=c.TC_TIMESTAMP)

            # with pl.Config(tbl_cols=20, fmt_str_lengths=50):
            #     print(self.setpoints)
            # exit()

            # Make LazyFrame again
            self.setpoints = self.setpoints.lazy()

            return self.setpoints

