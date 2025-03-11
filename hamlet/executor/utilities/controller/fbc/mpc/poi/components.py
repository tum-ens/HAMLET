__author__ = "HodaHamdy"
__credits__ = "MarkusDoepfert"
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

import math

import numpy as np
import pandas as pd
import polars.exceptions as pl_e
import pyoptinterface as poi
from pyoptinterface import gurobi

import hamlet.constants as c


class POIComps:
    def __init__(self, name, forecasts, **kwargs):
        self.name = name
        self.fcast = forecasts
        self.timesteps = kwargs['timesteps']
        self.info = kwargs

    def define_variables(self, model, variables, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def define_constraints(model, variables):
        pass

    @staticmethod
    def add_variable_to_model(model, variables, name, **kwargs):
        coords = kwargs.get('coords', [[0]])
        if len(coords) > 1:
            print('Warning: 2d coords are not currently supported. Exiting..')
            return
        for coord in coords:
            variables[name] = np.empty((len(coord)), dtype=object)
            for ind in coord:
                var_name = name
                if 'coords' in kwargs:
                    var_name += f'_{ind}'
                lb = kwargs.get("lower", -math.inf)
                if isinstance(lb, (pd.Series, list, np.ndarray)):
                    lb = lb[ind]
                ub = kwargs.get("upper", math.inf)
                if isinstance(ub, (pd.Series, list, np.ndarray)):
                    ub = ub[ind]
                kwargs_var = {
                    'name': var_name,
                    'lb': lb,
                    'ub': ub,
                    'domain': poi.VariableDomain.Integer if kwargs.get('integer', False)
                    else poi.VariableDomain.Binary
                    if kwargs.get('binary', False) else poi.VariableDomain.Continuous,
                }
                variables[name][ind] = model.add_variable(**kwargs_var)

    @staticmethod
    def _create_variable_name(name: str, component_type: str, energy_type: str, direction: str = None):
        if direction:
            return f'{name}_{component_type}_{energy_type}_{direction}'
        return f'{name}_{component_type}_{energy_type}'

    def define_electricity_variable(self, model: gurobi.Model, variables, comp_type: str, lower, upper,
                                    direction: str = None, integer=False):
        """Creates the electricity variable for the component. The direction is either in or out.

        Parameters
        ----------
        variables
        """

        # Set the name of the variable
        name = self._create_variable_name(self.name, comp_type, c.ET_ELECTRICITY, direction)

        # Define the variable
        self.add_variable_to_model(model, variables, name=name, lower=lower, upper=upper, coords=[self.timesteps],
                                   integer=integer)

    def define_heat_variable(self, model: gurobi.Model, variables, comp_type: str, lower, upper, direction: str = None,
                             integer=False):
        """Creates the heat variable for the component. The direction is either in or out.

        Parameters
        ----------
        variables
        """

        # Set the name of the variable
        name = self._create_variable_name(self.name, comp_type, c.ET_HEAT, direction)

        # Define the variable
        self.add_variable_to_model(model, variables, name=name, lower=lower, upper=upper,
                                   coords=[self.timesteps],
                                   integer=integer)

    def define_cool_variable(self, model: gurobi.Model, variables, comp_type: str, lower, upper, direction: str = None,
                             integer=False):
        """Creates the cooling variable for the component. The direction is either in or out.

        Parameters
        ----------
        variables
        """

        # Set the name of the variable
        name = self._create_variable_name(self.name, comp_type, c.ET_COOLING, direction)

        # Define the variable
        self.add_variable_to_model(model, variables, name=name, lower=lower, upper=upper,
                                   coords=[self.timesteps],
                                   integer=integer)

    def define_h2_variable(self, model: gurobi.Model, variables, comp_type: str, lower, upper, direction: str = None,
                           integer=False):
        """Creates the hydrogen variable for the component. The direction is either in or out.

        Parameters
        ----------
        variables
        """

        # Set the name of the variable
        name = self._create_variable_name(self.name, comp_type, c.ET_H2, direction)

        # Define the variable
        self.add_variable_to_model(model, variables, name=name, lower=lower, upper=upper,
                                   coords=[self.timesteps],
                                   integer=integer)

    def define_storage_variable(self, model: gurobi.Model, variables, comp_type: str, lower, upper, integer=False):
        """Creates the state-of-charge variable for the component.

        Parameters
        ----------
        variables
        """

        # Set the name
        name = f'{self.name}_{comp_type}_soc'

        # Define the variable
        self.add_variable_to_model(model, variables, name=name, lower=lower, upper=upper,
                                   coords=[self.timesteps],
                                   integer=integer)

    def define_mode_flag(self, model: gurobi.Model, variables, comp_type: str):
        """Creates the mode flag variable for the component. This is used to decide whether the component is charging
        or discharging.

        Parameters
        ----------
        variables"""

        # Set the name
        name = f'{self.name}_{comp_type}_mode'

        # Define the variable
        self.add_variable_to_model(model, variables, name=name, coords=[self.timesteps], binary=True)


class Market(POIComps):
    def __init__(self, name, **kwargs):
        # Call the parent class constructor
        super().__init__(name, **kwargs)

        # Get specific object attributes
        self.comp_type = None
        self.dt_hours = kwargs['delta'].total_seconds() * c.SECONDS_TO_HOURS  # time delta in hours

        # Calculate the upper and lower bounds for the market power from the energy quantity
        self.upper = [int(round(x / self.dt_hours)) for x in self.fcast[f'energy_quantity_sell']]
        self.lower = [int(round(x / self.dt_hours * -1)) for x in self.fcast[f'energy_quantity_buy']]

        # Get market price forecasts
        self.price_sell = pd.Series(self.fcast[f'{c.TC_ENERGY}_{c.TC_PRICE}_{c.PF_OUT}'], index=self.timesteps)
        self.price_buy = pd.Series(self.fcast[f'{c.TC_ENERGY}_{c.TC_PRICE}_{c.PF_IN}'], index=self.timesteps)
        self.grid_sell = pd.Series(self.fcast[f'{c.TT_GRID}_{c.TT_MARKET}_{c.PF_OUT}'], index=self.timesteps)
        self.grid_buy = pd.Series(self.fcast[f'{c.TT_GRID}_{c.TT_MARKET}_{c.PF_IN}'], index=self.timesteps)
        self.levies_sell = pd.Series(self.fcast[f'{c.TT_LEVIES}_{c.TC_PRICE}_{c.PF_OUT}'], index=self.timesteps)
        self.levies_buy = pd.Series(self.fcast[f'{c.TT_LEVIES}_{c.TC_PRICE}_{c.PF_IN}'], index=self.timesteps)

        # TODO: Add constraint that market value becomes zero if there is no market for this energy:
        #  One way to do this is check if market forecasts can be obtained. If that is not the case, it is assumed
        #  that there is no market.

        # TODO: Ponder how the interplay between markets should happen. In the future there will be a wholesale market
        #  regardless if there are other markets if there are forecast values

    def define_variables(self, model, variables, **kwargs):
        self.comp_type = kwargs['comp_type']
        # Define the market power variables (need to be positive and negative due to different pricing)
        self.add_variable_to_model(model, variables, name=f'{self.name}_{self.comp_type}_{c.PF_OUT}', lower=self.lower,
                                   upper=0, coords=[self.timesteps],
                                   integer=True)  # outflow from the building (selling)
        self.add_variable_to_model(model, variables, name=f'{self.name}_{self.comp_type}_{c.PF_IN}', lower=0,
                                   upper=self.upper, coords=[self.timesteps],
                                   integer=True)  # inflow into the building (buying)

        # Define mode flag that decides whether the market energy is bought or sold
        self.add_variable_to_model(model, variables, name=f'{self.name}_mode', coords=[self.timesteps], binary=True)

        # Define the market cost and revenue variables
        self.add_variable_to_model(model, variables, name=f'{self.name}_costs', lower=0, upper=np.inf,
                                   coords=[self.timesteps])
        self.add_variable_to_model(model, variables, name=f'{self.name}_revenue', lower=0, upper=np.inf,
                                   coords=[self.timesteps])

    def define_constraints(self, model, variables):
        # Add constraint that the market can either buy or sell but not both at the same time
        self.__constraint_operation_mode(model, variables)

        # Add constraint that the market cost and revenue are linked to the power
        self.__constraint_cost_revenue(model, variables)

    def __constraint_operation_mode(self, model, variables):
        """Adds the constraint that energy can either be bought or sold but not both at the same time.

        Parameters
        ----------
        variables
        """

        for timestep in self.timesteps:
            # Define the variables
            mode_var = variables[f'{self.name}_mode'][timestep]  # mode variable
            var_in = variables[f'{self.name}_{self.comp_type}_{c.PF_IN}'][timestep]  # inflow (buying)
            var_out = variables[f'{self.name}_{self.comp_type}_{c.PF_OUT}'][timestep]  # outflow (selling)
            # Define the constraint for outflow
            model.add_linear_constraint(var_out + self.lower[timestep] * mode_var - self.lower[timestep],
                                        poi.ConstraintSense.GreaterEqual,
                                        0, name=f'{self.name}_outflowflag_{timestep}')

            # Define the constraint for inflow
            # Note: The constraints should look something like: var_charge <= max_power * (1 - mode_var)
            #       However, linopy does not support it. Thus, the somewhat more complicated version below is used.
            model.add_linear_constraint(var_in - mode_var * self.upper[timestep], poi.ConstraintSense.LessEqual, 0,
                                        name=f'{self.name}_inflowflag_{timestep}')

    def __constraint_cost_revenue(self, model, variables):
        """Adds the constraint that the market cost and revenue are linked to the power."""
        for timestep in self.timesteps:
            # Define the variables
            var_in = variables[f'{self.name}_{self.comp_type}_{c.PF_IN}'][timestep]  # inflow (buying)
            var_out = variables[f'{self.name}_{self.comp_type}_{c.PF_OUT}'][timestep]  # outflow (selling)
            var_cost = variables[f'{self.name}_costs'][timestep]  # costs
            var_revenue = variables[f'{self.name}_revenue'][timestep]  # revenue

            # Define the constraint for costs
            model.add_linear_constraint(var_cost - var_in * self.dt_hours *
                                        (self.price_buy[timestep] + self.grid_buy[timestep] + self.levies_buy[
                                            timestep]), poi.ConstraintSense.Equal,
                                        0, name=f'{self.name}_costs_{timestep}')

            # Define the constraint for revenue
            model.add_linear_constraint(var_revenue + var_out * self.dt_hours * (
                    self.price_sell[timestep] - self.grid_sell[timestep] - self.levies_sell[timestep]),
                                        poi.ConstraintSense.Equal,
                                        0, name=f'{self.name}_revenue_{timestep}')


class InflexibleLoad(POIComps):

    def __init__(self, name, **kwargs):
        # Call the parent class constructor
        super().__init__(name, **kwargs)

        # Get specific object attributes
        self.power = pd.Series(self.fcast[f'{self.name}_power'], index=self.timesteps, dtype='int32')

    def define_variables(self, model, variables, **kwargs):
        comp_type = kwargs['comp_type']

        # Define the power variable
        self.define_electricity_variable(model, variables, comp_type=comp_type, lower=-self.power, upper=-self.power)


class FlexibleLoad(POIComps):

    def __init__(self, name, **kwargs):
        # Call the parent class constructor
        super().__init__(name, **kwargs)


class Heat(POIComps):

    def __init__(self, name, **kwargs):
        # Call the parent class constructor
        super().__init__(name, **kwargs)

        # Get specific object attributes
        self.heat = pd.Series(self.fcast[f'{self.name}_heat'], index=self.timesteps, dtype='int32')

    def define_variables(self, model, variables, **kwargs):
        comp_type = kwargs['comp_type']

        # Define the power variable
        self.define_heat_variable(model, variables, comp_type=comp_type, lower=-self.heat, upper=-self.heat)


class Dhw(POIComps):

    def __init__(self, name, **kwargs):
        # Call the parent class constructor
        super().__init__(name, **kwargs)

        # Get specific object attributes
        self.heat = pd.Series(self.fcast[f'{self.name}_dhw'], index=self.timesteps, dtype='int32')

    def define_variables(self, model, variables, **kwargs):
        comp_type = kwargs['comp_type']

        # Define the power variable
        self.define_heat_variable(model, variables, comp_type=comp_type, lower=-self.heat, upper=-self.heat)


class SimplePlant(POIComps):

    def __init__(self, name, **kwargs):
        # Call the parent class constructor
        super().__init__(name, **kwargs)

        # Get specific object attributes
        self.power = list(self.fcast[f'{self.name}_power'])
        self.controllable = self.info['sizing']['controllable']
        self.lower = [0] * len(self.power) if self.controllable else self.power

        self.lower = pd.Series(self.lower, index=self.timesteps)
        self.power = pd.Series(self.power, index=self.timesteps)

    def define_variables(self, model, variables, **kwargs):
        comp_type = kwargs['comp_type']

        # Define the power variable
        self.define_electricity_variable(model, variables, comp_type=comp_type, lower=self.lower,
                                         upper=self.power)


class Pv(SimplePlant):

    def __init__(self, name, **kwargs):
        # Call the parent class constructor
        super().__init__(name, **kwargs)


class Wind(SimplePlant):

    def __init__(self, name, **kwargs):
        # Call the parent class constructor
        super().__init__(name, **kwargs)


class FixedGen(SimplePlant):

    def __init__(self, name, **kwargs):
        # Call the parent class constructor
        super().__init__(name, **kwargs)


class Hp(POIComps):

    def __init__(self, name, **kwargs):
        # Call the parent class constructor
        super().__init__(name, **kwargs)

        # Get specific object attributes
        # Note: Currently all heat demand is considered to be covered using the heat data. Dhw is not separately
        #  modelled as this would be a different energy type. For simplification purposes this is not done yet but
        #  will be done in the future.
        self.comp_type = None
        self.cop_heat = np.array(self.fcast[f'{self.name}_{c.S_COP}_{c.P_HEAT}'] * c.COP100_TO_COP)
        # self.cop_dhw = list(self.fcast[f'{self.name}_{c.S_COP}_{c.P_DHW}'] * c.COP100_TO_COP)

        # Calculate the available power for heating and dhw
        # Note: The sizing power is a fallback method to ensure that there is always enough power so that the
        #  model does not fail.
        try:
            self.power_heat = np.array(self.fcast[f'{self.name}_{c.S_POWER}_{c.ET_HEAT}_{c.P_HEAT}'])
            self.power_heat = np.maximum(self.info['sizing']['power'], self.power_heat)
        except pl_e.ColumnNotFoundError:
            self.power_heat = [np.inf] * len(self.timesteps)

        # Calculate the power for electricity
        try:
            self.power_electricity = np.array(self.fcast[f'{self.name}_{c.S_POWER}_{c.ET_ELECTRICITY}_{c.P_HEAT}'])
            self.power_electricity = np.maximum(self.info['sizing']['power'] / self.cop_heat, self.power_electricity)
            self.power_electricity = np.rint(-self.power_electricity).astype(int)
        except (KeyError, pl_e.ColumnNotFoundError):
            self.power_electricity = [-np.inf] * len(self.timesteps)

        self.power_heat = pd.Series(self.power_heat, index=self.timesteps)
        self.power_electricity = pd.Series(self.power_electricity, index=self.timesteps)
        self.upper, self.lower = self.power_heat, 0

    def define_variables(self, model, variables, **kwargs):
        self.comp_type = kwargs['comp_type']
        # Define the heat power variable (positive as it generates heat)
        self.define_heat_variable(model, variables, comp_type=self.comp_type, lower=self.lower, upper=self.upper)

        # Define the electricity power variable (negative as it consumes electricity)
        self.define_electricity_variable(model, variables, comp_type=self.comp_type, lower=self.power_electricity,
                                         upper=0)

    def define_constraints(self, model, variables):
        # Add constraint that the heat power is the electricity power times the cop
        self.__constraint_cop(model, variables)

    def __constraint_cop(self, model, variables):
        """Adds the constraint that the heat power is the electricity power times the cop."""

        # Define the variables

        # Define the constraint
        for timestep in self.timesteps:
            var_heat = variables[f'{self.name}_{self.comp_type}_{c.ET_HEAT}'][timestep]
            var_electricity = variables[f'{self.name}_{self.comp_type}_{c.ET_ELECTRICITY}'][timestep]
            model.add_linear_constraint(var_heat + var_electricity * self.cop_heat[timestep], poi.ConstraintSense.Equal,
                                        0, name=f'{self.name}_cop_{timestep}')


class Ev(POIComps):

    def __init__(self, name, **kwargs):

        # Call the parent class constructor
        super().__init__(name, **kwargs)

        # Get specific object attributes
        self.comp_type = None
        self.capacity = self.info['sizing']['capacity']
        self.availability = list(self.fcast[f'{self.name}_availability'])
        self.energy = np.cumsum(list(self.fcast[f'{self.name}_energy_consumed']))  # cumulative energy consumption
        # For now this is always charging at home. In the future this can depend on the availability column if it shows
        # the location instead of availability at home. Most of it is already prepared.
        self.charging_power_home = self.info['sizing']['charging_home']
        self.charging_power_AC = self.info['sizing']['charging_AC']
        self.charging_power_DC = self.info['sizing']['charging_DC']
        self.charging_power = self.charging_power_home  # To be changed once more sophisticated EV modelling available
        self.efficiency = self.info['sizing']['charging_efficiency']
        self.v2g = self.info['sizing']['v2g']

        # Charging scheme
        self.dt = kwargs['delta'].total_seconds()  # time delta in seconds
        self.soc = min(self.capacity, kwargs['socs'][f'{self.name}'][0])  # soc at current timestamp (energy)
        self.scheme = self.info['charging_scheme']

        # Kwargs variables
        self.soc = [kwargs['socs'][f'{self.name}'][0]] * len(self.energy) - self.energy  # state of charge at timestep

        # Define the lower and upper bounds for the charging power based on the charging scheme
        self.lower, self.upper = [0] * len(self.availability), [0] * len(self.availability)
        self.lower, self.upper = self.__define_power_limits_based_on_availability()

    def define_variables(self, model, variables, **kwargs):
        self.comp_type = kwargs['comp_type']
        # Define the power variables (need to be positive and negative due to the efficiency)
        self.define_electricity_variable(model, variables, comp_type=self.comp_type, lower=self.lower, upper=0,
                                         direction=c.PF_OUT)  # flow out of the home (charging battery)
        self.define_electricity_variable(model, variables, comp_type=self.comp_type, lower=0, upper=self.upper,
                                         direction=c.PF_IN)  # flow into the home (discharging battery)
        # Define mode flag that decides whether the battery is charging or discharging
        self.define_mode_flag(model, variables, comp_type=self.comp_type)

        # Define the soc variable
        self.define_storage_variable(model, variables, comp_type=self.comp_type, lower=0, upper=self.capacity)

        # Define the soc variable for the previous timestep (thus the value of self.soc[0]) as needed for constraints
        self.add_variable_to_model(model, variables, name=f'{self.name}_{self.comp_type}_soc_init', lower=self.soc[0],
                                   upper=self.soc[0])

    def define_constraints(self, model, variables):

        # Add constraint that the EV can either charge or discharge but not both at the same time
        self.__constraint_operation_mode(model, variables)

        # Add constraint that adheres to the chosen charging scheme
        self.__constraint_charging_scheme(model, variables)

        # Add constraint that the soc of the battery is that of the previous timestep plus dis-/charging power
        #   times the efficiency and the time delta
        self.__constraint_soc(model, variables)

    def __define_power_limits_based_on_availability(self):
        """Defines the lower and upper bounds for the charging power based on the availability and v2g."""

        # Define the charging power variables (depends on availability and v2g)
        self.upper = self.charging_power * self.v2g * np.array(self.availability)
        self.lower = -self.charging_power * np.array(self.availability)

        return self.lower, self.upper

    def __constraint_operation_mode(self, model, variables):
        """Adds the constraint that the battery can either charge or discharge but not both at the same time."""

        for timestep in self.timesteps:
            # Define the variables
            mode_var = variables[f'{self.name}_{self.comp_type}_mode'][timestep]  # mode variable
            var_charge = variables[
                f'{self.name}_{self.comp_type}_{c.ET_ELECTRICITY}_{c.PF_OUT}'][timestep]  # charging power
            var_discharge = variables[
                f'{self.name}_{self.comp_type}_{c.ET_ELECTRICITY}_{c.PF_IN}'][timestep]  # discharging

            # Define the constraint for charging
            # Note: The constraints should look something like: var_charge <= max_power * (1 - mode_var)
            #       However, linopy does not support it. Thus, the somewhat more complicated version below is used.
            model.add_linear_constraint(var_charge + mode_var * self.lower[timestep], poi.ConstraintSense.GreaterEqual,
                                        self.lower[timestep], name=f'{self.name}_chargingflag_{timestep}')

            # Define the constraint for discharging
            model.add_linear_constraint(var_discharge - mode_var * self.upper[timestep], poi.ConstraintSense.LessEqual,
                                        0, name=f'{self.name}_dischargingflag_{timestep}')

    def __constraint_charging_scheme(self, model, variables):
        """Adds the constraint that the soc is always above the minimum to reach the target soc."""

        scheme = self.scheme['method']

        match scheme:
            case 'full':
                self.__constraint_cs_full(model, variables)
            case 'price_sensitive':
                raise NotImplementedError(f'Charging scheme {scheme} not implemented for this mpc model.')
            case 'min_soc':
                self.__constraint_cs_min_soc(model, variables)
            case 'min_soc_time':
                raise NotImplementedError(f'Charging scheme {scheme} not implemented for this mpc model.')
            case _:
                raise ValueError(f'Charging scheme {scheme} not available.')

    def __constraint_cs_full(self, model, variables):
        """Define constraints to ensure battery is fully charged.

        Args:
            model (Object): The optimization model.

        Returns:
            Object: The updated model with new constraints.
        """
        # Define the variable for state of charge
        var_soc = variables[f'{self.name}_{self.comp_type}_soc']
        target = 1  # Full state of charge

        # Calculate the energy needed to fill the battery at each timestep (if no (dis-)charging occurs)
        # Note: Here the energy is calculated that needs to be added to the battery without losses
        energy_to_target = self.__energy_to_reach_target(target)

        # Calculate the max net energy that can be charged at each timestep (if no (dis-)charging occurs)
        # Note: Net means that this is the energy that can be stored in the battery thus efficiency losses apply
        max_energy = self.__max_energy_at_timesteps()

        # Cumulate the values to get the energy that can be charged from the current timestep onwards but start over
        #   whenever there is a zero (thus the car has left)
        max_energy_cumulative = self.__cumulative_max_energy(max_energy)

        # Calculate the max capacity at each timestep depending on the availability
        max_capacity = self.__max_capacity_at_timesteps(energy_to_target)

        # Calculate target state of charge at each timestep
        target_soc = pd.Series(np.minimum(max_capacity, self.soc + max_energy_cumulative),
                               index=self.timesteps).astype(int)

        # Define the constraint
        model.add_linear_constraint(var_soc, poi.ConstraintSense.GreaterEqual, target_soc,
                                    name=f'{self.name}_soc_scheme')

    def __constraint_cs_min_soc(self, model, variables):
        """Define constraints to ensure a minimum state of charge is met.

        Args:
            model (Object): The optimization model.

        Returns:
            Object: The updated model with new constraints.
        """
        # Define the variable for state of charge
        target = self.scheme['min_soc']['val']  # target soc [0, 1]

        # Calculate the energy needed to fill the battery at each timestep (if no (dis-)charging occurs)
        # Note: Here the energy is calculated that needs to be added to the battery without losses
        energy_to_target = self.__energy_to_reach_target(target)

        # Calculate the max net energy that can be charged at each timestep (if no (dis-)charging occurs)
        # Note: Net means that this is the energy that can be stored in the battery thus efficiency losses apply
        max_energy = self.__max_energy_at_timesteps()

        # Cumulate the values to get the energy that can be charged from the current timestep onwards but start over
        #   whenever there is a zero (thus the car has left)
        max_energy_cumulative = self.__cumulative_max_energy(max_energy)

        # Calculate the max capacity at each timestep depending on the availability
        max_capacity = self.__max_capacity_at_timesteps(energy_to_target)

        # Calculate maximum and minimum state of charge at each timestep
        # Reverse the max_energy array to get the energy that can still be charged until the car leaves
        max_soc = np.minimum(max_capacity, self.soc + max_energy_cumulative)
        # Reverse the max_energy array to get the energy that can still be charged until the car leaves
        # self.reverse_non_zero_sequences(max_energy_cumulative)
        max_energy_reversed = self.__reverse_non_zero_sequences(max_energy_cumulative)
        # Target is the minimum soc necessary for the car to leave with the target soc
        target_soc = pd.Series(np.maximum(max_soc - max_energy_reversed, 0),
                               index=self.timesteps).astype(int)

        for timestep in self.timesteps:
            # Define the constraint
            model.add_linear_constraint(variables[f'{self.name}_{self.comp_type}_soc'][timestep],
                                        poi.ConstraintSense.GreaterEqual, target_soc[timestep],
                                        name=f'{self.name}_soc_scheme_{timestep}')

    def __constraint_soc(self, model, variables):
        """Adds the constraint that the soc of the battery is that of the previous timestep plus dis-/charging power"""
        dt_hours = self.dt * c.SECONDS_TO_HOURS  # time in h
        for timestep in self.timesteps:
            # Define the variables
            var_soc = variables[f'{self.name}_{self.comp_type}_soc'][timestep]  # soc variable
            if timestep == 0:
                var_soc_prev = variables[f'{self.name}_{self.comp_type}_soc_init'][0]  # current soc
            else:
                var_soc_prev = variables[f'{self.name}_{self.comp_type}_soc'][timestep - 1]  # previous soc
            var_charge = variables[
                f'{self.name}_{self.comp_type}_{c.ET_ELECTRICITY}_{c.PF_OUT}'][timestep]  # charging power
            var_discharge = variables[
                f'{self.name}_{self.comp_type}_{c.ET_ELECTRICITY}_{c.PF_IN}'][timestep]  # discharging

            # Define the constraint for charging
            # Constraint: soc_new = soc_old + charge * efficiency * dt - discharge / efficiency * dt
            # Note: Everything is moved to the lhs as linopy cannot handle it otherwise
            model.add_linear_constraint(var_soc
                                        - var_soc_prev
                                        + var_charge * self.efficiency * dt_hours
                                        + var_discharge / self.efficiency * dt_hours,
                                        # negative as discharging is negative
                                        poi.ConstraintSense.Equal, 0, name=f'{self.name}_soc_{timestep}')

    @staticmethod
    # @numba.jit(nopython=True) # TODO: Check if faster with numba
    def __cumsum_reset_at_zero(arr):
        """
        Compute the cumulative sum of an array, resetting the sum to zero whenever a zero value is encountered.

        Parameters:
        - arr (np.array): Input array to compute the cumulative sum of.

        Returns:
        - np.array: Array containing the cumulative sums, with resets at zero values.
        """

        # Initialize an array of the same shape as `arr` filled with zeros
        result = np.zeros_like(arr)

        # Variable to store the running sum
        running_sum = 0

        # Iterate over each value in the input array
        for i, val in enumerate(arr):
            # Reset running sum if current value is zero
            if val == 0:
                running_sum = 0
            # Otherwise, add the current value to the running sum
            else:
                running_sum += val
            # Assign the current running sum to the result array
            result[i] = running_sum

        return result

    @staticmethod
    # @numba.jit(nopython=True) # TODO: Check if faster with numba
    def __reverse_non_zero_sequences(arr):
        """Reverse sequences of non-zero values in the given array.

        This function identifies sequences of non-zero values in the array and
        reverses them in place. Sequences are separated by zero values.

        Args:
            arr (list or numpy.ndarray): The input array containing sequences to be reversed.

        Returns:
            list or numpy.ndarray: The modified array with reversed non-zero sequences.
        """

        # Starting position for a sequence
        start = None

        for i, val in enumerate(arr):
            if val != 0:
                # If starting position is not set, set it
                if start is None:
                    start = i
            else:
                # If there was a sequence before this 0, reverse it
                if start is not None:
                    arr[start:i] = arr[start:i][::-1]
                    start = None

        # Handle the case where the array ends with a non-zero sequence
        if start is not None:
            arr[start:] = arr[start:][::-1]

        return arr

    def __energy_to_reach_target(self, target):
        """
        Calculate the energy needed to reach a target state of charge (SoC).

        Parameters:
        - target (float): Desired target state of charge (SoC).

        Returns:
        - float: Energy needed to reach the target SoC.
        """
        return (self.capacity * target - self.soc).round()

    def __max_energy_at_timesteps(self):
        """
        Compute the maximum energy that can be added at each timestep.

        Returns:
        - numpy.array: Array representing the max energy at each timestep.
        """
        dt_hours = self.dt * c.SECONDS_TO_HOURS  # Convert delta time to hours
        return int(self.charging_power * self.efficiency * dt_hours) * np.array(self.availability)

    def __cumulative_max_energy(self, max_energy):
        """
        Calculate cumulative energy over timesteps with reset at zero values.

        Parameters:
        - max_energy (numpy.array): Array representing the max energy at each timestep.

        Returns:
        - numpy.array: Cumulative max energy with reset at zero values.
        """
        return self.__cumsum_reset_at_zero(max_energy)

    def __max_capacity_at_timesteps(self, energy_to_target):
        """
        Calculate the maximum capacity at each timestep based on availability.

        Parameters:
        - energy_to_target (float): Energy needed to reach the target SoC.

        Returns:
        - numpy.array: Max capacity at each timestep.
        """
        return (self.soc + energy_to_target) * np.array(self.availability)


class SimpleStorage(POIComps):

    def __init__(self, name, **kwargs):

        # Call the parent class constructor
        super().__init__(name, **kwargs)

        # Get specific object attributes
        self.comp_type = None
        self.capacity = self.info['sizing']['capacity']
        self.charging_power = self.info['sizing']['power']
        self.efficiency = self.info['sizing']['efficiency']

        # Kwargs variables
        self.dt = self.info['delta'].total_seconds()  # time delta in seconds
        self.soc = min(self.capacity, self.info['socs'][f'{self.name}'][0])  # soc at current timestamp (energy)
        self.energy_to_full = self.capacity - self.soc  # energy needed to charge to full

        # Define the charging and discharging power limits
        # Charging is treated as negative (load), discharging as positive (generation)
        self.lower, self.upper = -self.charging_power, self.charging_power

    def define_variables(self, model, variables, **kwargs):
        self.comp_type = kwargs['comp_type']
        # Define the power variables (need to be positive and negative due to the efficiency)
        self._define_power_variables(model, variables)

        # Define mode flag that decides whether the battery is charging or discharging
        self.define_mode_flag(model, variables, comp_type=self.comp_type)

        # Define the soc variable
        self.define_storage_variable(model, variables, comp_type=self.comp_type, lower=0, upper=self.capacity)

        # Define the soc variable for the previous timestep (thus the value of self.soc[0]) as needed for constraints
        self.add_variable_to_model(model, variables, name=f'{self.name}_{self.comp_type}_soc_init', lower=self.soc,
                                   upper=self.soc)

    def _define_power_variables(self, model: gurobi.Model, variables,
                                energy_type: str = c.ET_ELECTRICITY):

        # Define the power variables depending on the energy type
        match energy_type:
            case c.ET_ELECTRICITY:
                self.define_electricity_variable(model, variables, comp_type=self.comp_type, lower=self.lower, upper=0,
                                                 direction=c.PF_OUT)  # flow out of the home (charging battery)
                self.define_electricity_variable(model, variables, comp_type=self.comp_type, lower=0, upper=self.upper,
                                                 direction=c.PF_IN)  # flow into the home (discharging battery)
            case c.ET_HEAT:
                self.define_heat_variable(model, variables, comp_type=self.comp_type, lower=self.lower, upper=0,
                                          direction=c.PF_OUT)  # flow out of the home (charging battery)
                self.define_heat_variable(model, variables, comp_type=self.comp_type, lower=0, upper=self.upper,
                                          direction=c.PF_IN)  # flow into the home (discharging battery)
            case c.ET_COOLING:
                self.define_cool_variable(model, variables, comp_type=self.comp_type, lower=self.lower, upper=0,
                                          direction=c.PF_OUT)  # flow out of the home (charging battery)
                self.define_cool_variable(model, variables, comp_type=self.comp_type, lower=0, upper=self.upper,
                                          direction=c.PF_IN)  # flow into the home (discharging battery)
            case c.ET_H2:
                self.define_h2_variable(model, variables, comp_type=self.comp_type, lower=self.lower, upper=0,
                                        direction=c.PF_OUT)
                self.define_h2_variable(model, variables, comp_type=self.comp_type, lower=0, upper=self.upper,
                                        direction=c.PF_IN)
            case _:
                raise NotImplementedError(f'{energy_type} has not been implemented yet for the simple storage system.')

    def define_constraints(self, model, variables):

        # Add constraint that the battery can either charge or discharge but not both at the same time
        self._constraint_operation_mode(model, variables)

        # TODO: Limit the charging power to the own generation if the battery must not charge from grid
        self._constraint_power_limits(model, variables)

        # Add constraint that the soc of the battery is that of the previous timestep plus dis-/charging power
        #   times the efficiency and the time delta
        self._constraint_soc(model, variables)

    def _constraint_operation_mode(self, model: gurobi.Model, variables,
                                   energy_type: str = c.ET_ELECTRICITY):
        """Adds the constraint that the battery can either charge or discharge but not both at the same time."""

        for timestep in self.timesteps:
            # Define the variables
            mode_var = variables[f'{self.name}_{self.comp_type}_mode'][timestep]  # mode variable
            var_charge = variables[
                f'{self.name}_{self.comp_type}_{energy_type}_{c.PF_OUT}'][timestep]  # charging power
            var_discharge = variables[f'{self.name}_{self.comp_type}_{energy_type}_{c.PF_IN}'][timestep]  # discharging

            # Define the constraint for charging
            # Note: The constraints should look something like: var_discharge >= -max_power * (1 - mode_var)
            #       However, linopy does not support it. Thus, the somewhat more complicated version below is used.
            model.add_linear_constraint(var_charge + mode_var * self.lower, poi.ConstraintSense.GreaterEqual,
                                        self.lower, name=f'{self.name}_chargingflag_{timestep}')

            # Define the constraint for discharging
            model.add_linear_constraint(var_discharge - mode_var * self.upper, poi.ConstraintSense.LessEqual, 0,
                                        name=f'{self.name}_dischargingflag_{timestep}')

    def _constraint_power_limits(self, model: gurobi.Model, variables):
        pass

    def _constraint_soc(self, model: gurobi.Model, variables, energy_type: str = c.ET_ELECTRICITY):
        """Adds the constraint that the soc of the battery is that of the previous timestep plus dis-/charging power"""

        dt_hours = self.dt * c.SECONDS_TO_HOURS  # time in h
        for timestep in self.timesteps:
            # Define the variables
            var_soc = variables[f'{self.name}_{self.comp_type}_soc'][timestep]  # soc variable
            var_charge = variables[
                f'{self.name}_{self.comp_type}_{energy_type}_{c.PF_OUT}'][timestep]  # charging power
            var_discharge = variables[f'{self.name}_{self.comp_type}_{energy_type}_{c.PF_IN}'][timestep]  # discharging
            if timestep == 0:
                var_soc_prev = variables[f'{self.name}_{self.comp_type}_soc_init'][0]  # current soc
            else:
                var_soc_prev = variables[f'{self.name}_{self.comp_type}_soc'][timestep - 1]  # previous soc

            # Define the constraint for charging
            # Constraint: soc_new = soc_old + charge * efficiency * dt - discharge / efficiency * dt
            # Note: Everything is moved to the lhs as linopy cannot handle it otherwise
            model.add_linear_constraint(var_soc
                                        - var_soc_prev
                                        + var_charge * self.efficiency * dt_hours
                                        + var_discharge / self.efficiency * dt_hours, poi.ConstraintSense.Equal, 0,
                                        name=f'{self.name}_soc_{timestep}')


class Battery(SimpleStorage):

    def __init__(self, name, **kwargs):
        # Call the parent class constructor
        super().__init__(name, **kwargs)
        self.b2g = self.info['sizing']['b2g']
        self.g2b = self.info['sizing']['g2b']

    def define_constraints(self, model, variables):
        super().define_constraints(model, variables)

        # Add constraint that the battery can only charge from the grid if the b2g flag is set to true
        self._constraint_b2g(model, variables)

    def _constraint_b2g(self, model, variables):
        """Adds the constraint that the battery can only charge from the grid if the b2g flag is set to true."""

        # Define the variables
        markets = self.info['markets']

        # Define the constraint if b2g is disabled
        if not self.b2g:
            for market, energy in markets.items():
                if energy == c.ET_ELECTRICITY:  # Only electricity markets are considered
                    for timestep in self.timesteps:
                        model.add_linear_constraint(variables[f'{self.name}_{self.comp_type}_mode'][timestep] -
                                                    variables[f'{market}_mode'][timestep],
                                                    poi.ConstraintSense.LessEqual, 0,
                                                    name=f'{self.name}_b2g_{market}_{timestep}')


class Psh(SimpleStorage):

    def __init__(self, name, **kwargs):
        # Call the parent class constructor
        super().__init__(name, **kwargs)


class Hydrogen(SimpleStorage):

    def __init__(self, name, **kwargs):
        # Call the parent class constructor
        super().__init__(name, **kwargs)


class HeatStorage(SimpleStorage):

    def __init__(self, name, **kwargs):
        # Call the parent class constructor
        super().__init__(name, **kwargs)

    def define_variables(self, model, variables, **kwargs):
        self.comp_type = kwargs['comp_type']
        # Define the power variables (need to be positive and negative due to the efficiency)
        self._define_power_variables(model, variables, energy_type=c.ET_HEAT)

        # Define mode flag that decides whether the storage is charging or discharging
        self.define_mode_flag(model, variables, comp_type=self.comp_type)

        # Define the soc variable
        self.define_storage_variable(model, variables, comp_type=self.comp_type, lower=0, upper=self.capacity)

        # Define the soc variable for the previous timestep (thus the value of self.soc[0]) as needed for constraints
        self.add_variable_to_model(model, variables, name=f'{self.name}_{self.comp_type}_soc_init', lower=self.soc,
                                   upper=self.soc)

    def define_constraints(self, model, variables):
        # Add constraint that the battery can either charge or discharge but not both at the same time
        self._constraint_operation_mode(model, variables, energy_type=c.ET_HEAT)

        # TODO: Limit the charging power to the own generation if the battery must not charge from grid
        self._constraint_power_limits(model, variables)

        # Add constraint that the soc of the battery is that of the previous timestep plus dis-/charging power
        #   times the efficiency and the time delta
        self._constraint_soc(model, variables, energy_type=c.ET_HEAT)
