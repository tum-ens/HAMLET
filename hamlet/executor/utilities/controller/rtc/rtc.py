__author__ = "MarkusDoepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

import time
from linopy import Model, Variable
import polars as pl
from pprint import pprint
from hamlet import constants as c
from hamlet.executor.utilities.controller.rtc import lincomps
from hamlet.executor.utilities.controller.base import ControllerBase
from hamlet.executor.utilities.database.database import Database as db
from hamlet import functions as f
import warnings
import logging

# warnings.filterwarnings("ignore")
logging.getLogger('linopy').setLevel(logging.CRITICAL)


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

        return controller_class(**kwargs, mapping=c.COMP_MAP).run()

    class Linopy(RtcBase):
        def __init__(self, **kwargs):
            # Create the model
            self.model = Model()

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
            # Get the current timestamp
            self.timestamp = self.timetable.collect()[0, c.TC_TIMESTAMP]

            # Get the agent and other data
            self.agent = kwargs['agent']
            self.account = self.agent.account
            self.plants = self.agent.plants  # Formerly known as components
            self.setpoints = self.agent.setpoints
            self.timeseries = self.agent.timeseries
            self.socs = self.agent.socs
            self.meters = self.agent.meters
            # Filter the timeseries to only include the rows with the current timestamp
            self.timeseries = self.timeseries.join(self.timetable, on=c.TC_TIMESTAMP, how='semi')
            # Get the targets by filtering the setpoints to only include the rows with the current timestamp
            self.targets = self.setpoints.join(self.timetable, on=c.TC_TIMESTAMP, how='semi')

            # Raise warning if timeseries exceeds one row
            if len(self.timeseries.collect()) != 1:
                raise ValueError(f"Timeseries has {len(self.timeseries)} rows. It should only have 1 row for the rtc.")

            # Get the market data
            self.market = kwargs['market']
            # Get market name
            # Get the market names and types
            self.market_names = self.timetable.collect().select(c.TC_NAME).unique().to_series().to_list()
            self.market_types = self.timetable.collect().select(c.TC_MARKET).unique().to_series().to_list()
            # Assign each market name to an energy type
            self.markets = {name: c.TRADED_ENERGY[mtype] for name, mtype in zip(self.market_names, self.market_types)}
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

            # Note: This can probably be only done once and then stored in the agent. Afterwards, it only needs to be
            #  updated every timestep (will need considerable adjustment though and might not be worth the effort).

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

                # TODO: Take out once hp is implemented
                if plant_type in ['dhw', 'heat']:
                    continue

                # Retrieve the timeseries data for the plant
                cols = [col for col in self.timeseries.columns if col.startswith(plant_name)]
                timeseries = self.timeseries.select(cols).collect()

                # Retrieve the target setpoints for the plant
                cols = [col for col in self.targets.columns if col.startswith(plant_name)]
                targets = self.targets.select(cols).collect()

                # Retrieve the soc data for the plant (if applicable)
                cols = [col for col in self.socs.columns if col.startswith(plant_name)]
                socs = self.socs.select(cols).collect()

                # Get the plant class
                plant_class = self.available_plants.get(plant_type)
                if plant_class is None:
                    raise ValueError(f"Unsupported plant type: {plant_name}")

                # Create the plant object
                self.plant_objects[plant_name] = plant_class(name=plant_name, timeseries=timeseries,  **plant_data,
                                                             targets=targets, socs=socs, delta=self.dt)

            return self.plant_objects

        def create_markets(self):
            """"""

            # Define variables from the market results and a balancing variable for each energy type
            for market in self.markets:
                # Create market object
                self.market_objects[market] = lincomps.Market(name=market,
                                                              timeseries=self.market)

                # Create balancing market object
                self.market_objects[f'{market}_{c.MT_BALANCING}'] = lincomps.Balancing(
                    name=f'{market}_{c.MT_BALANCING}',
                    timeseries=self.market)

            return self.market_objects

        def define_variables(self):
            # Define variables for each plant
            for plant_name, plant in self.plant_objects.items():
                self.model = plant.define_variables(self.model, comp_type=self.plants[plant_name]['type'])

            # Define variables for each market
            for market_name, market in self.market_objects.items():
                # Balancing markets are not explicitly modeled and have the same comp_type as their original market
                if c.MT_BALANCING in market_name:
                    energy_type = self.markets[market_name.rsplit('_', 1)[0]]
                else:
                    energy_type = self.markets[market_name]

                self.model = market.define_variables(self.model, energy_type=energy_type)

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
                    # Add the variable as generation if it is a market variable for the current energy type
                    if (variable_name.startswith(tuple(self.market_objects))
                        and variable_name.endswith(f'_{energy_type}')):
                        balance_equations[energy_type] += variable
                    # Add the variable if it is a plant variable for the current energy type
                    elif (variable_name.startswith(tuple(self.plant_objects))
                          and variable_name.endswith(f'_{energy_type}')):  # TODO: Could be done so it checks for _power since that would save looking at the deviation and target specifically (can be done as else continue)
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
                            # TODO: Maybe change this already to positive and negative in the plant objects to avoid confusion
                            # TODO: Probably change convention of storage to positive as well (load becomes negative)
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
                        pass

            # Add the constraints for each energy type
            for energy_type, equation in balance_equations.items():
                self.model.add_constraints(equation == 0, name="balance_" + energy_type)

            return self.model

        def define_objective(self):
            # TODO: Something is still wrong here as it should discharge the battery and not use balancing energy
            #  Check as well if balancing constraint is correct and not tainted by the new variables
            #  Alternative would be introducing a milp in the balancing constraints of battery and ev but they should be correct already

            # TODO: Check balancing constraint and objective function for correctness

            # Weights to prioritize components
            w_bat = 1  # weight for battery
            w_ev = 2  # weight for electric vehicle
            w_hp = 3  # weight for heat pump
            w_bal = 4  # weight for balancing energy
            # Initialize the objective function as zero
            objective = self.model.add_variables(name='objective', lower=0, upper=0, integer=True)

            # Loop through the model's variables to identify the balancing variables
            for variable_name, variable in self.model.variables.items():
                # If the variable name starts with 'balancing_', it's a balancing variable
                if variable_name.startswith('balancing_'):
                    # Add the variable to the objective function
                    objective += variable * w_bal
                elif c.P_EV in variable_name and '_deviation_' in variable_name:
                    if variable_name.endswith('_pos'):
                        objective += variable * w_ev / 2
                    elif variable_name.endswith('_neg'):
                        objective += variable * w_ev / 2
                elif c.P_BATTERY in variable_name and '_deviation_' in variable_name:
                    if variable_name.endswith('_pos'):
                        objective += variable * w_bat / 2
                    elif variable_name.endswith('_neg'):
                        objective += variable * w_bat / 2
                else:
                    pass

            # Set the objective function to the model with the minimize direction
            self.model.add_objective(objective)

            return self.model

        def define_objective_gpt(self):
            # TODO: This is the answer it gave to me to the following request:
            #  "I want it to also include that preferably excess power generation is first put into electric vehicles,
            #  then batteries and then the heat pump. The electric vehicle should also not be charged by the battery,
            #  only if otherwise balancing energy would be needed.
            #  Can you rewrite the objective function to reflect that?"  --> Check if any good and debug if worth it
            # Weights to prioritize components
            w1 = 1  # weight for electric vehicle
            w2 = 2  # weight for battery
            w3 = 3  # weight for heat pump
            w4 = 4  # weight for balancing energy

            # Initialize the objective function as zero
            objective = 0

            # Loop through the model's variables to identify different components and balancing variables
            for variable_name, variable in self.model.variables.items():
                if variable_name.startswith('ev_charge'):
                    # Give priority to EV charging from excess generation, not from battery
                    if 'from_battery' not in variable_name:
                        objective += w1 * variable
                    else:
                        objective += w4 * variable
                elif variable_name.startswith('battery_charge'):
                    objective += w2 * variable
                elif variable_name.startswith('heat_pump_use'):
                    objective += w3 * variable
                elif variable_name.startswith('balancing_'):
                    objective += w4 * variable

            # Set the objective function to the model with the minimize direction
            self.model.add_objective(objective, sense="min")

            return self.model

        def run(self):

            # Solve the optimization problem
            solver = 'gurobi'
            match solver:
                case 'gurobi':
                    solver_options = {'OutputFlag': 0, 'LogToConsole': 0}
                    status = self.model.solve(solver_name='gurobi', **solver_options)
                case _:
                    raise ValueError(f"Unsupported solver: {solver}")

            # TODO: Make the model silent and not put out any response.

            # Check if the solution is optimal
            if status[0] != 'ok':
                print(self.model.print_infeasibilities())
                raise ValueError(f"Optimization failed: {status}")

            # Process the solution into control commands and return
            self.agent = self.process_solution()

            return self.agent

        def process_solution(self):

            # Obtain the solution values
            solution = {name: int(sol) for name, sol in self.model.solution.items()}

            # Update setpoints
            self.setpoints = self.update_setpoints(solution)

            # Update socs
            self.socs = self.update_socs(solution)

            # TODO: Update meters
            self.meters = self.update_meters(solution)

            # Update the agent
            self.agent.setpoints = self.setpoints
            self.agent.socs = self.socs
            self.agent.meters = self.meters

            return self.agent

        def update_setpoints(self, solution: dict):

            # Make LazyFrames into DataFrames
            self.setpoints = self.setpoints.collect()

            # Get relevant column name beginnings (i.e. the plant names and market and balancing)
            beginnings = set([col.split('_', 1)[0] for col in solution.keys()
                              if not col.startswith('objective') and not col.startswith('balance')])
            # Get relevant column name endings (i.e. the energy types)
            endings = self.energy_types
            # Get the relevant columns
            src_cols = [col for col in solution.keys()
                        if col.split('_', 1)[0] in beginnings and col.rsplit('_', 1)[-1] in endings]

            # Shift index according to timetable time
            self.setpoints.index = [self.timetable.collect()[0, c.TC_TIMESTAMP] + self.dt * t
                           for t in range(len(self.setpoints))]

            # Update setpoints
            # TODO: Do this similar to the one in mpc (columns are added if missing, otherwise replaced)
            for src_col in src_cols:
                # Check if the column is already in the setpoints
                if src_col not in self.setpoints.columns:
                    # Add column to dataframe with 0 values
                    self.setpoints = self.setpoints.with_columns(pl.lit(0).alias(src_col))
                # Assign setpoint value to first row
                self.setpoints[0, src_col] = solution[src_col]

            # Drop all setpoint columns that are not part of src_cols (plus keep timestamp and timestep column)
            sel_cols = [self.setpoints.columns[0]] + src_cols
            self.setpoints = self.setpoints.select(sel_cols)

            # Make LazyFrame again
            self.setpoints = self.setpoints.lazy()

            return self.setpoints

        def update_socs(self, solution: dict):
            # Make LazyFrame into DataFrame
            self.socs = self.socs.collect()

            # Find row that is to be updated (i.e. the row with the next timestamp)
            row_soc = self.socs.filter(self.socs[c.TC_TIMESTAMP] == self.timestamp + self.dt)

            # Update socs
            for col in self.socs.columns[1:]:
                # Extract power from variable values
                key = next((key for key in solution
                            if key.startswith(col) and (key.endswith('_power') or key.endswith('_heat'))),
                           None)

                if key:  # Check for matching key
                    # Get power from solution
                    power = solution[key]

                    # Get the column dtype
                    dtype = self.socs[col].dtype

                    # Get soc from plant object
                    soc = self.plant_objects[col].soc

                    # Calculate delta for the soc based on power and time step
                    delta_soc = power * self.dt.total_seconds() * c.SECONDS_TO_HOURS

                    # Adjust delta_soc by efficiency based on power being positive or negative
                    if power > 0:
                        delta_soc *= self.plant_objects[col].efficiency
                    elif power < 0:
                        delta_soc /= self.plant_objects[col].efficiency

                    # Update soc
                    soc += delta_soc

                    # Round soc to integer
                    soc = round(soc)

                    # Update the soc value in the DataFrame for the corresponding column
                    row_soc = row_soc.with_columns(pl.lit(soc).cast(dtype).alias(col))

                # Ensure that soc is within bounds (not implemented as of now to ensure that the model is working)
                # soc = max(0, min(self.plant_objects[col].capacity, soc))

            # Update socs dataframe
            self.socs = self.socs.filter(self.socs[c.TC_TIMESTAMP] != self.timestamp + self.dt)
            self.socs = self.socs.merge_sorted(row_soc, key=c.TC_TIMESTAMP)

            # Make LazyFrame again
            self.socs = self.socs.lazy()

            return self.socs

        def update_meters(self, solution: dict):
            # Make LazyFrame into DataFrame
            self.meters = self.meters.collect()

            # Find row that is to be updated as well as the previous row (i.e. current and next timestamp)
            row_now = self.meters.filter(self.meters[c.TC_TIMESTAMP] == self.timestamp)
            row_new = self.meters.filter(self.meters[c.TC_TIMESTAMP] == self.timestamp + self.dt)

            # Update meters
            for col in self.meters.columns[1:]:
                # Extract power from variable values
                key = next((key for key in solution
                            if key.startswith(col) and (key.endswith('_power') or key.endswith('_heat'))),
                           None)

                if key:  # Check for matching key
                    # Calculate energy from power
                    delta_energy = solution[key] * self.dt.total_seconds() * c.SECONDS_TO_HOURS

                    # Get the column dtype
                    dtype = self.meters[col].dtype

                    # Get old meter value from row now
                    meter_now = row_now[col]

                    # Update meter value in row new
                    row_new = row_new.with_columns(pl.lit(meter_now + round(delta_energy)).cast(dtype).alias(col))

            # Update meters dataframe
            self.meters = self.meters.filter(self.meters[c.TC_TIMESTAMP] != self.timestamp + self.dt)
            self.meters = self.meters.merge_sorted(row_new, key=c.TC_TIMESTAMP)

            # Make LazyFrame again
            self.meters = self.meters.lazy()

            return self.meters


    class RuleBased(RtcBase):  # Note the change in class name

        def __init__(self, **kwargs):
            pass

        def run(self):
            print('Running Rule-Based')

