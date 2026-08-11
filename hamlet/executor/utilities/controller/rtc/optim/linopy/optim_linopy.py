__author__ = "MarkusDoepfert"
__credits__ = "HodaHamdy"
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

import os
import sys
import logging

import numpy as np

from linopy.io import read_netcdf

from hamlet.executor.utilities.controller.rtc.optim.linopy.components import *
from hamlet.executor.utilities.controller.rtc.optim.optim_base import OptimBase
from hamlet.executor.utilities.controller.solver_options import reproducibility_options

LOGGER = logging.getLogger(__name__)

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


class Linopy(OptimBase):
    def __init__(self, **kwargs):
        self.loaded_model = False
        self.model_path = f"{kwargs['agent'].agent_save}/linopy_rtc.nc"
        # `grid_commands` is stored by OptimBase, for both backends alike.
        super().__init__(**kwargs)
        self.ems = self.ems[c.C_OPTIM]
        # Save first model to file to load later
        self.save_model()

    def save_model(self):
        # Save first model to file to load later
        if not os.path.exists(self.model_path):
            self.model.to_netcdf(self.model_path)

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

    def get_available_plants(self):
        return AVAILABLE_PLANTS

    def get_market_class(self):
        return Market

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
                        and variable_name.endswith(f'_{energy_type}')):
                    balance_equations[energy_type] += variable
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
                    pass

        # Give each balance equation a slack variable pair so a single infeasible agent cannot
        # abort the whole run. This matters most for the non-electricity carriers: only the
        # electricity balance has a market term to absorb a mismatch, so a heat balance that
        # cannot be met has no other way out.
        # See c.DEFAULT_SLACK_ENABLED; disable per agent with `slack: false`.
        for energy_type in (balance_equations if self.slack_enabled else {}):
            balance_equations[energy_type] += self.model.add_variables(
                name=f'{energy_type}_{c.OM_GENERATION}_slack', lower=0, integer=False)
            balance_equations[energy_type] -= self.model.add_variables(
                name=f'{energy_type}_{c.OM_LOAD}_slack', lower=0, integer=False)

        # Add the constraints for each energy type
        for energy_type, equation in balance_equations.items():
            self.model.add_constraints(equation == 0, name="balance_" + energy_type)

        return self.model

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
            elif variable_name.endswith('_slack'):
                # This objective is a weighted sum of set-point deviations rather than a cost, so
                # the slack penalty is a priority weight. It sits above every deviation weight
                # above, making slack the least preferred way to satisfy the balance.
                objective.append(variable * self.slack_penalty)

        # Set the objective function to the model with the minimize direction
        self.model.add_objective(sum(objective), overwrite=True)

        return self.model

    # Slack below this many W is solver tolerance rather than a real imbalance
    SLACK_REPORTING_THRESHOLD = 1e-3

    def _warn_on_slack(self):
        """Report when a balance was only closed by shedding or dumping energy.

        The slack variables keep an infeasible agent from aborting the run, but they are not
        written to the setpoints, so a run that used them produces results whose flows do not
        balance. That has to be visible: an agent that shed 3 kW must not look identical to one
        that served it.

        Note: this reports through `logging` rather than `warnings`. The executor installs a
        blanket `warnings.filterwarnings("ignore")` at import time, so a warning here would
        never reach the user and the shed energy would be silent.
        """

        for name, variable in self.model.variables.items():
            if not name.endswith('_slack'):
                continue
            try:
                peak = float(np.max(np.abs(variable.solution.values)))
            except (AttributeError, ValueError):
                continue
            if peak > self.SLACK_REPORTING_THRESHOLD:
                LOGGER.warning(
                    'Agent %s: energy balance closed with %.1f W of "%s". The setpoints for '
                    'this timestep do not balance.', self.agent.agent_id, peak, name)

    def run(self):

        # Solve the optimization problem
        solver = self.ems.get('solver')
        match solver:
            case 'gurobi' | 'highs':
                sys.stdout = open(os.devnull, 'w')  # deactivate printing from linopy
                # `OutputFlag` and `LogToConsole` are Gurobi's names, and HiGHS discards them
                # unrecognised -- the redirect above is what actually silences it. Tidying that is
                # roadmap item #11; the options added below are the ones that decide whether the
                # run is reproducible, so they are named per solver (#204).
                solver_options = {'OutputFlag': 0, 'LogToConsole': 0}
                solver_options.update(reproducibility_options(solver, self.ems.get('time_limit')))
                status = self.model.solve(solver_name=solver, **solver_options)
                sys.stdout = sys.__stdout__  # re-activate printing
            case _:
                raise ValueError(f"Unsupported solver: {solver}.")

        # Check if the solution is optimal
        if status[0] != 'ok':
            print(f'Exited with status "{status[0]}". \n'
                  f'Infeasibilities for agent {self.agent.agent_id}: \n'
                  f'{self.model.print_infeasibilities()}')

            # Print the model
            print('Model:')
            for name, var in self.model.variables.items():
                print(var)
            for name, con in self.model.constraints.items():
                print(con)
            print(self.model.objective)

            raise ValueError(f"Optimization failed: {status}")

        # Surface any energy that was shed or dumped to close the balance
        self._warn_on_slack()

        # Process the solution into control commands and return
        self.agent = self.process_solution()

        return self.agent

    def get_solution(self):

        # Obtain the solution values
        return {name: int(sol) for name, sol in self.model.solution.items()}

    def apply_grid_commands(self):
        """Adjust model variables according to grid control commands if necessary."""
        # apply direct power control (§14a EnWG regulation)
        if (c.G_ELECTRICITY in self.grid_commands and
                'current_direct_power_control' in self.grid_commands[c.G_ELECTRICITY] and
                self.agent.agent_id in self.grid_commands[c.G_ELECTRICITY]['current_direct_power_control']):

            control_target = self.grid_commands[c.G_ELECTRICITY]['current_direct_power_control'][self.agent.agent_id]

            # iterate through all relevant plants in grid commands
            for plant_id in control_target.keys():

                plant_power = control_target[plant_id]

                # get plant type
                if plant_id == 'ems' and 'direct_power_control_ems' not in self.model.variables.labels:  # EMS control
                    # add a new constraint to limit total load / generation
                    if plant_power > 0:
                        balance_equations = self.model.add_variables(name='direct_power_control_ems', lower=-inf,
                                                                     upper=plant_power, integer=False)
                    else:
                        balance_equations = self.model.add_variables(name='direct_power_control_ems',
                                                                     lower=plant_power,
                                                                     upper=inf, integer=False)

                    # Loop through each variable and add it to the balance equation accordingly
                    for variable_name, variable in self.model.variables.items():
                        # Add the variable if it is a plant variable for the current energy type
                        if variable_name.startswith(tuple(self.plant_objects)) and variable_name.endswith(
                                c.ET_ELECTRICITY):
                            # Get the component name by splitting the variable name at the underscore
                            component_name = variable_name.split('_', 1)[0]

                            # Get the component type by comparing the ID with the plant names
                            component_type = [vals['type'] for plant, vals in self.plants.items()
                                              if plant == component_name][0]

                            # If the component type is in the mapping for the current energy type, add the variable to
                            # the balance equation
                            if c.ET_ELECTRICITY in self.mapping[component_type].keys():
                                # Get the operation mode for the component and energy type
                                component_energy_mode = self.mapping[component_type][c.ET_ELECTRICITY]

                                # Add the variable to the balance equation
                                if component_energy_mode == c.OM_GENERATION:
                                    balance_equations += variable
                                elif component_energy_mode == c.OM_LOAD:
                                    balance_equations += variable
                                elif component_energy_mode == c.OM_STORAGE:
                                    balance_equations += variable
                                else:
                                    raise ValueError(f"Unsupported operation mode: {component_energy_mode}")
                            else:
                                # The component type is not in the mapping for the current energy type
                                pass

                    self.model.add_constraints(balance_equations == 0, name='direct_power_control_ems')

                elif plant_id == 'ems' and 'direct_power_control_ems' in self.model.variables.labels:
                    # check if ems power is positive (load) or negative (generation) and set boundary
                    if plant_power > 0:
                        self.model.variables['direct_power_control_ems'].upper = plant_power
                    else:
                        self.model.variables['direct_power_control_ems'].lower = plant_power

                else:  # individual device control
                    plant_type = self.plants[plant_id]['type']

                    power_variable_name = '_'.join([plant_id, plant_type, c.ET_ELECTRICITY])
                    target_variable_name = '_'.join([plant_id, plant_type, 'target'])

                    # check if plant power is positive (load) or negative (generation) and set boundary
                    if plant_power > 0:
                        self.model.variables[power_variable_name].upper = plant_power
                        self.model.variables[power_variable_name].lower = min(
                            self.model.variables[power_variable_name]
                            .lower, plant_power)
                    else:
                        self.model.variables[power_variable_name].lower = plant_power
                        self.model.variables[power_variable_name].upper = max(
                            self.model.variables[power_variable_name]
                            .upper, plant_power)

                    # set target value also to control target
                    if target_variable_name in self.model.variables.labels:
                        self.model.variables[target_variable_name].upper = plant_power
                        self.model.variables[target_variable_name].lower = plant_power
