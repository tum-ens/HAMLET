__author__ = "MarkusDoepfert"
__credits__ = "HodaHamdy"
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

import io
import logging
import os
from contextlib import redirect_stdout

import numpy as np

from linopy.io import read_netcdf

from hamlet.executor.utilities.controller.fbc.mpc.linopy.components import *
from hamlet.executor.utilities.controller.fbc.mpc.mpc_base import MpcBase
from hamlet.executor.utilities.controller.solver_options import (quiet_options,
                                                                 reproducibility_options)

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


class Linopy(MpcBase):
    def __init__(self, **kwargs):
        self.loaded_model = False
        self.model_path = f"{kwargs['agent'].agent_save}/linopy_mpc.nc"
        super().__init__(**kwargs)
        self.ems = self.ems[c.C_OPTIM]
        # Save first model to file to load later
        self.save_model()

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

    def save_model(self):
        if not os.path.exists(self.model_path):
            self.model.to_netcdf(self.model_path)

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

        # Give each balance equation a slack variable pair, carrying a value-of-lost-load
        # penalty, so a single infeasible agent cannot abort the whole run.
        # See c.DEFAULT_SLACK_ENABLED; disable per agent with `slack: false`.
        for energy_type in (balance_equations if self.slack_enabled else {}):
            balance_equations[energy_type] += self.model.add_variables(
                name=f'{energy_type}_{c.OM_GENERATION}_slack', lower=0, upper=np.inf,
                coords=[self.timesteps], integer=False)
            balance_equations[energy_type] -= self.model.add_variables(
                name=f'{energy_type}_{c.OM_LOAD}_slack', lower=0, upper=np.inf,
                coords=[self.timesteps], integer=False)

        # Add the constraints for each energy type
        for energy_type, equation in balance_equations.items():
            self.model.add_constraints(equation == 0, name="balance_" + energy_type)

        return self.model

    def define_objective(self):
        """Defines the objective function. The objective is to reduce the costs."""

        # Initialize the objective function as zero
        objective = []

        # Loop through the model's variables to identify the balancing variables
        for variable_name, variable in self.model.variables.items():
            # Note: the slack test must come first. Market names are user-defined, so a market
            # called `electricity` would otherwise capture `electricity_gen_slack` in the branch
            # below and the penalty would silently never reach the objective, making slack free.
            if variable_name.endswith('_slack'):
                # This objective is monetary, so the slack penalty is a value of lost load in the
                # same units as the price forecasts (0.01 ct/kWh, i.e. 0.1 EUR/MWh per unit),
                # scaled by the timestep so that it stays a price per unit of energy.
                # Shedding load must always be dearer than serving it at any market price.
                dt_hours = self.dt.total_seconds() * c.SECONDS_TO_HOURS
                objective.append(self.slack_penalty * dt_hours * variable)
            # Only consider the cost and revenue components of the markets
            elif variable_name.startswith(tuple(self.market_names)):
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

    # Slack below this many W is solver tolerance rather than a real imbalance
    SLACK_REPORTING_THRESHOLD = 1e-3

    def _warn_on_slack(self):
        """Report when a balance was only closed by shedding or dumping energy.

        The slack variables keep an infeasible agent from aborting the run, but they are not
        written to the setpoints, so a run that used them produces results whose flows do not
        balance. That has to be visible: an agent that shed 3 kW must not look identical to one
        that served it.

        Reported through `logging`, not `warnings`, which deduplicates by source line and would
        report only the first agent that shed. Pinned by `tests/.../test_slack_wiring.py`.
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
                # The flags are what silence the solver; the redirect is a cheap guard, not the
                # mechanism. Measured per solve on a real MPC model, stdout lines under
                # highs: 52 with the old Gurobi-named options (two of them
                # `ERROR: getOptionIndex: Option "OutputFlag" is unknown`), 1 with these -- the
                # HiGHS banner, emitted from C before options apply. **The old
                # `sys.stdout = open(os.devnull, 'w')` never suppressed any of it**, because
                # HiGHS writes to file descriptor 1 and that only rebinds a Python attribute;
                # only an fd-level redirect can, as `tests/backend_matrix._silenced` does.
                # `redirect_stdout` is kept because it is exception-safe where the assignment was
                # not: that leaked a handle per solve, never restored on a raise, and restored to
                # `sys.__stdout__` rather than the caller's stdout. It captures 0 bytes today.
                solver_options = quiet_options(solver)
                solver_options.update(reproducibility_options(solver, self.ems.get('time_limit')))
                with redirect_stdout(io.StringIO()):
                    status = self.model.solve(solver_name=solver, **solver_options)
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

        # Surface any energy that was shed or dumped to close the balance
        self._warn_on_slack()

        # Process the solution into control commands and return
        self.agent = self.process_solution()

        return self.agent

    def get_solution(self):
        # Obtain the solution values
        return {name: sol for name, sol in self.model.solution.items()}
