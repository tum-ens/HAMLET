__author__ = "HodaHamdy"
__credits__ = "MarkusDoepfert"
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

import logging

from hamlet.executor.utilities.controller.fbc.mpc.mpc_base import MpcBase
from hamlet.executor.utilities.controller.fbc.mpc.poi.components import *
from hamlet.executor.utilities.controller.poi_solver import (apply_reproducibility_options,
                                                             create_model, raise_unless_optimal)

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


class POI(MpcBase):
    def get_model(self, **kwargs):
        # `self.ems` is still the whole ems block here; the base narrows it to the fbc controller
        # only after the model exists.
        return create_model(self.ems[c.C_CONTROLLER][c.C_FBC][c.C_OPTIM].get('solver'))

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

        # Give each balance equation a slack variable pair, carrying a value-of-lost-load
        # penalty, so a single infeasible agent cannot abort the whole run.
        # See c.DEFAULT_SLACK_ENABLED; disable per agent with `slack: false`.
        for energy_type in (balance_equations if self.slack_enabled else {}):
            for direction, sign in ((c.OM_GENERATION, 1), (c.OM_LOAD, -1)):
                name = f'{energy_type}_{direction}_slack'
                self.variables[name] = np.array(
                    [self.model.add_variable(name=f'{name}_{t}', lb=0, ub=np.inf)
                     for t in range(len(self.timesteps))])
                balance_equations[energy_type].append(sign * self.variables[name])

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
        dt_hours = self.dt.total_seconds() * c.SECONDS_TO_HOURS
        for variable_name, variables in self.variables.items():
            # Note: the slack test must come first. Market names are user-defined, so a market
            # called `electricity` would otherwise capture `electricity_gen_slack` below and the
            # penalty would silently never reach the objective, making slack free.
            if variable_name.endswith('_slack'):
                # The penalty is a value of lost load in the price unit, scaled by the timestep
                # so that it stays a price per unit of energy.
                # Note: no reduction here. Every branch appends a per-timestep array and
                # the outer np.sum reduces across all of them; collapsing this one first
                # makes the list ragged and the objective un-buildable.
                objective.append(self.slack_penalty * dt_hours * variables)
            # Only consider the cost and revenue components of the markets
            elif variable_name.startswith(tuple(self.market_names)):
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

    # Slack below this many W is solver tolerance rather than a real imbalance
    SLACK_REPORTING_THRESHOLD = 1e-3

    def _warn_on_slack(self):
        """Report when a balance was only closed by shedding or dumping energy.

        The shed energy is never written to the setpoints, so without this an agent that shed
        3 kW is indistinguishable in the results from one that served it.

        Reported through `logging`. That was originally forced -- the executor installed a
        blanket `warnings.filterwarnings("ignore")` at import, so anything raised through the
        `warnings` machinery here reached nobody (#199). The blanket filter is gone, and
        `logging` is kept on its own merits: the `warnings` machinery deduplicates by source
        line, so it would report the first agent that shed and stay silent for every one after
        it, which is precisely the wrong shape for a per-agent, per-timestep report.
        """

        for name, variables in self.variables.items():
            if not name.endswith('_slack'):
                continue
            try:
                peak = float(np.max(np.abs([self.model.get_value(v)
                                            for v in np.atleast_1d(variables)])))
            except Exception:
                continue
            if peak > self.SLACK_REPORTING_THRESHOLD:
                LOGGER.warning(
                    'Agent %s: energy balance closed with %.1f W of "%s". The setpoints for '
                    'this timestep do not balance.', self.agent.agent_id, peak, name)

    def run(self):

        # Solve the optimization problem. The model was already created and silenced for this
        # solver in `get_model`, so only the reproducibility options are left to apply.
        solver = self.ems[c.C_OPTIM].get('solver')
        time_limit = self.ems[c.C_OPTIM].get('time_limit')
        apply_reproducibility_options(self.model, solver, time_limit)
        self.model.optimize()
        status = self.model.get_model_attribute(poi.ModelAttribute.TerminationStatus)

        # Anything short of a proven optimum is an error, the time limit included
        raise_unless_optimal(status, self.agent.agent_id, time_limit)

        # Surface any energy that was shed or dumped to close the balance
        self._warn_on_slack()

        # Process the solution into control commands and return
        self.agent = self.process_solution()

        return self.agent

    def get_solution(self):
        # Obtain the solution values
        return {var_name: np.array([self.model.get_value(var) for var in vars]) for var_name, vars in
                self.variables.items()}
