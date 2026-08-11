__author__ = "HodaHamdy"
__credits__ = "MarkusDoepfert"
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

import logging

# Used by the slack reporting in `run`. It was missing, and the bare `except Exception` around
# that block turned every call into a silently swallowed NameError -- so a POI run that closed its
# balance with slack reported nothing while the linopy run warned.
import numpy as np

from hamlet.executor.utilities.controller.poi_solver import (apply_reproducibility_options,
                                                             create_model, raise_unless_optimal)
from hamlet.executor.utilities.controller.rtc.optim.poi.components import *
from hamlet.executor.utilities.controller.rtc.optim.optim_base import OptimBase

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


class POI(OptimBase):
    # The variable the EMS-level cap is imposed through, shared with the linopy backend so a
    # scenario reads the same whichever framework produced it.
    EMS_CONTROL_VARIABLE = 'direct_power_control_ems'

    def get_model(self, **kwargs):
        # `self.ems` is still the whole ems block here; the base narrows it to the rtc controller
        # only after the model exists.
        return create_model(self.ems[c.C_CONTROLLER][c.C_RTC][c.C_OPTIM].get('solver'))

    def apply_grid_commands(self):
        """Impose the grid operator's direct power control (§14a EnWG) on the model.

        Mirrors the linopy backend method by method. Without it this inherited the base class's
        no-op, so an agent on `framework: poi` accepted every cap and then ignored it -- silently,
        because the grid stage has no way to tell that its commands were discarded.

        Two control methods, matching the `direct_power_control.method` configuration:
        `individual` tightens the bounds of one plant's power variable, `ems` caps the agent's
        whole electrical connection by constraining the sum of its plant powers.
        """
        commands = (self.grid_commands.get(c.G_ELECTRICITY, {})
                    .get('current_direct_power_control', {})
                    .get(self.agent.agent_id))
        if not commands:
            return

        for plant_id, plant_power in commands.items():
            if plant_id == 'ems':
                self.__apply_ems_control(plant_power)
            else:
                self.__apply_individual_control(plant_id, plant_power)

    def __apply_individual_control(self, plant_id, plant_power):
        """Cap a single plant, and move its target with it.

        The target has to follow the cap: it is the reference the deviation term is priced
        against, so leaving it at a now-unreachable setpoint would charge the agent for obeying
        the grid operator.
        """
        plant_type = self.plants[plant_id]['type']
        power_name = '_'.join([plant_id, plant_type, c.ET_ELECTRICITY])
        target_name = '_'.join([plant_id, plant_type, 'target'])

        lower, upper = self.__bounds(power_name)
        # A positive command is a load limit and a negative one a generation limit, so which
        # bound binds depends on the sign. The other is widened rather than left alone, so a cap
        # can never produce an empty interval.
        if plant_power > 0:
            self.__set_bounds(power_name, min(lower, plant_power), plant_power)
        else:
            self.__set_bounds(power_name, plant_power, max(upper, plant_power))

        if target_name in self.variables:
            self.__set_bounds(target_name, plant_power, plant_power)

    def __apply_ems_control(self, plant_power):
        """Cap the agent's whole electrical connection rather than one device."""
        if self.EMS_CONTROL_VARIABLE in self.variables:
            # Already built this timestep; only the bound moves.
            if plant_power > 0:
                self.__set_bounds(self.EMS_CONTROL_VARIABLE,
                                  self.__bounds(self.EMS_CONTROL_VARIABLE)[0], plant_power)
            else:
                self.__set_bounds(self.EMS_CONTROL_VARIABLE, plant_power,
                                  self.__bounds(self.EMS_CONTROL_VARIABLE)[1])
            return

        if plant_power > 0:
            slack = self.model.add_variable(lb=-inf, ub=plant_power, name=self.EMS_CONTROL_VARIABLE)
        else:
            slack = self.model.add_variable(lb=plant_power, ub=inf, name=self.EMS_CONTROL_VARIABLE)
        self.variables[self.EMS_CONTROL_VARIABLE] = slack

        # Same selection as the balance equation: every plant variable carrying electricity, for a
        # component whose mapping declares an electricity mode.
        terms = [slack]
        for variable_name, variable in self.variables.items():
            if not (variable_name.startswith(tuple(self.plant_objects))
                    and variable_name.endswith(c.ET_ELECTRICITY)):
                continue
            component_name = variable_name.split('_', 1)[0]
            component_type = [vals['type'] for plant, vals in self.plants.items()
                              if plant == component_name][0]
            if c.ET_ELECTRICITY not in self.mapping[component_type]:
                continue
            mode = self.mapping[component_type][c.ET_ELECTRICITY]
            if mode not in (c.OM_GENERATION, c.OM_LOAD, c.OM_STORAGE):
                raise ValueError(f"Unsupported operation mode: {mode}")
            terms.append(variable)

        self.model.add_linear_constraint(sum(terms), poi.ConstraintSense.Equal, 0,
                                         name=self.EMS_CONTROL_VARIABLE)

    def __bounds(self, name):
        variable = self.variables[name]
        return (self.model.get_variable_attribute(variable, poi.VariableAttribute.LowerBound),
                self.model.get_variable_attribute(variable, poi.VariableAttribute.UpperBound))

    def __set_bounds(self, name, lower, upper):
        variable = self.variables[name]
        self.model.set_variable_attribute(variable, poi.VariableAttribute.LowerBound, float(lower))
        self.model.set_variable_attribute(variable, poi.VariableAttribute.UpperBound, float(upper))

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

        # Give each balance equation a slack variable pair so a single infeasible agent cannot
        # abort the whole run. This matters most for the non-electricity carriers: only the
        # electricity balance has a market term to absorb a mismatch.
        # See c.DEFAULT_SLACK_ENABLED; disable per agent with `slack: false`.
        for energy_type in (balance_equations if self.slack_enabled else {}):
            for direction, sign in ((c.OM_GENERATION, 1), (c.OM_LOAD, -1)):
                name = f'{energy_type}_{direction}_slack'
                self.variables[name] = self.model.add_variable(name=name, lb=0, ub=inf)
                balance_equations[energy_type].append(sign * self.variables[name])

        # Add the constraints for each energy type
        for energy_type, variables in balance_equations.items():
            self.model.add_linear_constraint(sum(variables), poi.ConstraintSense.Equal, 0,
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
            elif variable_name.endswith('_slack'):
                # This objective is a weighted sum of set-point deviations rather than a cost, so
                # the slack penalty is a priority weight, above every deviation weight above.
                objective.append(variable * self.slack_penalty)

        # Set the objective function to the model with the minimize direction
        self.model.set_objective(sum(objective), poi.ObjectiveSense.Minimize)

    # Slack below this many W is solver tolerance rather than a real imbalance
    SLACK_REPORTING_THRESHOLD = 1e-3

    def _warn_on_slack(self):
        """Report when a balance was only closed by shedding or dumping energy.

        The shed energy is never written to the setpoints, so without this an agent that shed
        3 kW is indistinguishable in the results from one that served it.

        Note: reported through `logging`, not `warnings` -- the executor installs a blanket
        `warnings.filterwarnings("ignore")` at import time.
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
        return {var_name: int(self.model.get_value(var)) for var_name, var in self.variables.items()}
