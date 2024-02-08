__author__ = "MarkusDoepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

import time

import hamlet.constants as c
from pprint import pprint
import numpy as np
import pandas as pd
import linopy
import polars.exceptions as pl_e
from linopy import Model


# import numba


class LinopyComps:

    def __init__(self, name, forecasts, **kwargs):

        # Get the data
        self.name = name
        self.fcast = forecasts
        self.timesteps = kwargs['timesteps']
        self.info = kwargs

    def define_variables(self, model, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def define_constraints(model):
        return model

    def define_electricity_variable(self, model: linopy.Model, comp_type: str, lower, upper, direction: str = None,
                                    integer=False):
        """Creates the electricity variable for the component. The direction is either in or out."""

        # Set the name of the variable
        if direction:
            name = f'{self.name}_{comp_type}_{c.ET_ELECTRICITY}_{direction}'
        else:
            name = f'{self.name}_{comp_type}_{c.ET_ELECTRICITY}'

        # Define the variable
        model.add_variables(name=name, lower=lower, upper=upper, coords=[self.timesteps], integer=integer)

        return model

    def define_heat_variable(self, model: linopy.Model, comp_type: str, lower, upper, direction: str = None,
                             integer=False):
        """Creates the heat variable for the component. The direction is either in or out."""

        # Set the name of the variable
        if direction:
            name = f'{self.name}_{comp_type}_{c.ET_HEAT}_{direction}'
        else:
            name = f'{self.name}_{comp_type}_{c.ET_HEAT}'

        # Define the variable
        model.add_variables(name=name, lower=lower, upper=upper, coords=[self.timesteps], integer=integer)

        return model

    def define_cool_variable(self, model: linopy.Model, comp_type: str, lower, upper, direction: str = None,
                             integer=False):
        """Creates the heat variable for the component. The direction is either in or out."""

        # Set the name of the variable
        if direction:
            name = f'{self.name}_{comp_type}_{c.ET_COOLING}_{direction}'
        else:
            name = f'{self.name}_{comp_type}_{c.ET_COOLING}'

        # Define the variable
        model.add_variables(name=name, lower=lower, upper=upper, coords=[self.timesteps], integer=integer)

        return model

    def define_h2_variable(self, model: linopy.Model, comp_type: str, lower, upper, direction: str = None,
                             integer=False):
        """Creates the heat variable for the component. The direction is either in or out."""

        # Set the name of the variable
        if direction:
            name = f'{self.name}_{comp_type}_{c.ET_H2}_{direction}'
        else:
            name = f'{self.name}_{comp_type}_{c.ET_H2}'

        # Define the variable
        model.add_variables(name=name, lower=lower, upper=upper, coords=[self.timesteps], integer=integer)

        return model

    def define_storage_variable(self, model: linopy.Model, comp_type: str, lower, upper,
                                integer=False):
        """Creates the state-of-charge variable for the component."""

        # Set the name
        name = f'{self.name}_{comp_type}_soc'

        # Define the variable
        model.add_variables(name=name, lower=lower, upper=upper, coords=[self.timesteps], integer=integer)

        return model

    def define_mode_flag(self, model: linopy.Model, comp_type: str):
        """Creates the mode flag variable for the component. This is used to decide whether the component is charging
        or discharging."""

        # Set the name
        name = f'{self.name}_{comp_type}_mode'

        # Define the variable
        model.add_variables(name=name, coords=[self.timesteps], binary=True)

        return model


class Market(LinopyComps):

    def __init__(self, name, **kwargs):
        # Call the parent class constructor
        super().__init__(name, **kwargs)

        # Get specific object attributes
        self.dt_hours = kwargs['delta'].total_seconds() * c.SECONDS_TO_HOURS  # time delta in hours
        self.comp_type = None

        # Calculate the upper and lower bounds for the market power from the energy quantity
        self.upper = [int(round(x / self.dt_hours)) for x in self.fcast[f'energy_quantity_sell']]
        self.lower = [int(round(x / self.dt_hours * -1)) for x in self.fcast[f'energy_quantity_buy']]

        # Get market price forecasts
        self.price_sell = pd.Series(self.fcast[f'energy_price_sell'], index=self.timesteps)
        self.price_buy = pd.Series(self.fcast[f'energy_price_buy'], index=self.timesteps)
        self.grid_sell = pd.Series(self.fcast[f'grid_local_sell'], index=self.timesteps)
        self.grid_buy = pd.Series(self.fcast[f'grid_local_buy'], index=self.timesteps)
        self.levies_sell = pd.Series(self.fcast[f'levies_price_sell'], index=self.timesteps)
        self.levies_buy = pd.Series(self.fcast[f'levies_price_buy'], index=self.timesteps)

        # TODO: Add constraint that market value becomes zero if there is no market for this energy:
        #  One way to do this is check if market forecasts can be obtained. If that is not the case, it is assumed
        #  that there is no market.

        # TODO: Ponder how the interplay between markets should happen. In the future there will be a wholesale market
        #  regardless if there are other markets if there are forecast values

    def define_variables(self, model, **kwargs):
        self.comp_type = kwargs['comp_type']

        # Define the market power variables (need to be positive and negative due to different pricing)
        model.add_variables(name=f'{self.name}_{self.comp_type}_{c.PF_OUT}', lower=self.lower, upper=0,
                            coords=[self.timesteps], integer=True)  # outflow from the building (selling)
        model.add_variables(name=f'{self.name}_{self.comp_type}_{c.PF_IN}', lower=0, upper=self.upper,
                            coords=[self.timesteps], integer=True)  # inflow into the building (buying)
        # Define mode flag that decides whether the market energy is bought or sold
        model.add_variables(name=f'{self.name}_mode', coords=[self.timesteps], binary=True)

        # Define the market cost and revenue variables
        model.add_variables(name=f'{self.name}_costs', lower=0, upper=np.inf, coords=[self.timesteps])
        model.add_variables(name=f'{self.name}_revenue', lower=0, upper=np.inf, coords=[self.timesteps])

        return model

    def define_constraints(self, model):
        # Add constraint that the market can either buy or sell but not both at the same time
        model = self.__constraint_operation_mode(model)

        # Add constraint that the market cost and revenue are linked to the power
        model = self.__constraint_cost_revenue(model)

        return model

    def __constraint_operation_mode(self, model):
        """Adds the constraint that energy can either be bought or sold but not both at the same time."""

        # Define the variables
        mode_var = model.variables[f'{self.name}_mode']  # mode variable
        var_in = model.variables[f'{self.name}_{self.comp_type}_{c.PF_IN}']  # inflow (buying)
        var_out = model.variables[f'{self.name}_{self.comp_type}_{c.PF_OUT}']  # outflow (selling)

        # Define the constraint for outflow
        eq_out = (var_out >= -(mode_var - 1) * self.lower)
        model.add_constraints(eq_out, name=f'{self.name}_outflowflag', coords=[self.timesteps])

        # Define the constraint for inflow
        # Note: The constraints should look something like: var_charge <= max_power * (1 - mode_var)
        #       However, linopy does not support it. Thus, the somewhat more complicated version below is used.
        eq_in = (var_in <= mode_var * self.upper)
        model.add_constraints(eq_in, name=f'{self.name}_inflowflag', coords=[self.timesteps])

        return model

    def __constraint_cost_revenue(self, model):
        """Adds the constraint that the market cost and revenue are linked to the power."""

        # Define the variables
        var_in = model.variables[f'{self.name}_{self.comp_type}_{c.PF_IN}']  # inflow (buying)
        var_out = model.variables[f'{self.name}_{self.comp_type}_{c.PF_OUT}']  # outflow (selling)
        var_cost = model.variables[f'{self.name}_costs']  # costs
        var_revenue = model.variables[f'{self.name}_revenue']  # revenue
        dt_hours = pd.Series([self.dt_hours] * len(self.timesteps), index=self.timesteps)  # time delta

        # Define the constraint for costs
        eq_cost = (var_cost == var_in * dt_hours * (self.price_buy + self.grid_buy + self.levies_buy))
        model.add_constraints(eq_cost, name=f'{self.name}_costs', coords=[self.timesteps])

        # Define the constraint for revenue
        eq_revenue = (var_revenue == -var_out * dt_hours * (self.price_sell + self.grid_sell + self.levies_sell))
        model.add_constraints(eq_revenue, name=f'{self.name}_revenue', coords=[self.timesteps])

        return model
    # def define_variables(self, model, **kwargs):
    #     self.comp_type = kwargs['comp_type']
    #
    #     # Define the market power variables (need to be positive and negative due to different pricing)
    #     model.add_variables(name=f'{self.name}_{self.comp_type}_{c.PF_IN}', lower=self.lower, upper=0,
    #                         coords=[self.timesteps], integer=True)  # buying
    #     model.add_variables(name=f'{self.name}_{self.comp_type}_{c.PF_OUT}', lower=0, upper=self.upper,
    #                         coords=[self.timesteps], integer=True)  # selling
    #     # Define mode flag that decides whether the market energy is bought or sold
    #     model.add_variables(name=f'{self.name}_mode', coords=[self.timesteps], binary=True)
    #
    #     # Define the market cost and revenue variables
    #     model.add_variables(name=f'{self.name}_costs', lower=0, upper=np.inf, coords=[self.timesteps])
    #     model.add_variables(name=f'{self.name}_revenue', lower=0, upper=np.inf, coords=[self.timesteps])
    #
    #     return model

    # def define_constraints(self, model):
    #     # Add constraint that the market can either buy or sell but not both at the same time
    #     model = self.__constraint_operation_mode(model)
    #
    #     # Add constraint that the market cost and revenue are linked to the power
    #     model = self.__constraint_cost_revenue(model)
    #
    #     return model
    #
    #
    # def __constraint_operation_mode(self, model):
    #     """Adds the constraint that energy can either be bought or sold but not both at the same time."""
    #
    #     # Define the variables
    #     mode_var = model.variables[f'{self.name}_mode']  # mode variable
    #     var_in = model.variables[f'{self.name}_{self.comp_type}_{c.PF_IN}']  # inflow
    #     var_out = model.variables[f'{self.name}_{self.comp_type}_{c.PF_OUT}']  # outflow
    #
    #     # Define the constraint for inflow
    #     # Note: The constraints should look something like: var_charge <= max_power * (1 - mode_var)
    #     #       However, linopy does not support it. Thus, the somewhat more complicated version below is used.
    #     eq_in = (var_in >= -(mode_var - 1) * self.lower)
    #     model.add_constraints(eq_in, name=f'{self.name}_outflowflag', coords=[self.timesteps])
    #
    #     # Define the constraint for outflow
    #     eq_out = (var_out <= mode_var * self.upper)
    #     model.add_constraints(eq_out, name=f'{self.name}_inflowflag', coords=[self.timesteps])
    #
    #     return model
    #
    # def __constraint_cost_revenue(self, model):
    #     """Adds the constraint that the market cost and revenue are linked to the power."""
    #
    #     # Define the variables
    #     var_in = model.variables[f'{self.name}_{self.comp_type}_{c.PF_IN}']  # inflow (buying)
    #     var_out = model.variables[f'{self.name}_{self.comp_type}_{c.PF_OUT}']  # outflow (selling)
    #     var_cost = model.variables[f'{self.name}_costs']  # costs
    #     var_revenue = model.variables[f'{self.name}_revenue']  # revenue
    #     dt_hours = pd.Series([self.dt_hours] * len(self.timesteps), index=self.timesteps)  # time delta
    #
    #     # Define the constraint for costs
    #     # TODO: These are changed as that is a temp fix until the variables are switched around again.
    #     eq_cost = (var_cost == -var_in * dt_hours * (self.price_buy + self.grid_buy + self.levies_buy))
    #     model.add_constraints(eq_cost, name=f'{self.name}_costs', coords=[self.timesteps])
    #
    #     # Define the constraint for revenue
    #     eq_revenue = (var_revenue == var_out * dt_hours * (self.price_sell + self.grid_sell + self.levies_sell))
    #     model.add_constraints(eq_revenue, name=f'{self.name}_revenue', coords=[self.timesteps])
    #
    #     return model


class InflexibleLoad(LinopyComps):

    def __init__(self, name, **kwargs):
        # Call the parent class constructor
        super().__init__(name, **kwargs)

        # Get specific object attributes
        self.power = pd.Series(self.fcast[f'{self.name}_power'], index=self.timesteps, dtype='int32')

    def define_variables(self, model, **kwargs):
        comp_type = kwargs['comp_type']

        # Define the power variable
        model = self.define_electricity_variable(model, comp_type=comp_type, lower=-self.power, upper=-self.power)

        return model


class FlexibleLoad(LinopyComps):

    def __init__(self, name, **kwargs):
        # Call the parent class constructor
        super().__init__(name, **kwargs)

        # Get specific object attributes
        print(self.fcast)


class Heat(LinopyComps):

    def __init__(self, name, **kwargs):
        # Call the parent class constructor
        super().__init__(name, **kwargs)

        # Get specific object attributes
        self.heat = pd.Series(self.fcast[f'{self.name}_heat'], index=self.timesteps, dtype='int32')

    def define_variables(self, model, **kwargs):
        comp_type = kwargs['comp_type']

        # Define the power variable
        model = self.define_heat_variable(model, comp_type=comp_type, lower=-self.heat, upper=-self.heat)

        return model


class Dhw(LinopyComps):

    def __init__(self, name, **kwargs):
        # Call the parent class constructor
        super().__init__(name, **kwargs)

        # Get specific object attributes
        self.heat = list(self.fcast[f'{self.name}_dhw'].round(3))
        self.heat = [int(x * 1000) for x in self.heat]  # TODO: Fix the input data so that this is not needed anymore

    def define_variables(self, model, **kwargs):
        comp_type = kwargs['comp_type']

        # Define the power variable
        model = self.define_heat_variable(model, comp_type=comp_type, lower=-self.heat, upper=-self.heat)

        return model


class SimplePlant(LinopyComps):

    def __init__(self, name, **kwargs):
        # Call the parent class constructor
        super().__init__(name, **kwargs)

        # Get specific object attributes
        self.power = list(self.fcast[f'{self.name}_power'])
        self.controllable = self.info['sizing']['controllable']
        self.lower = [0] * len(self.power) if self.controllable else self.power

    def define_variables(self, model, **kwargs):
        comp_type = kwargs['comp_type']

        # Define the power variable
        model = self.define_electricity_variable(model, comp_type=comp_type, lower=self.lower, upper=self.power)

        return model


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


class Hp(LinopyComps):

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

        self.upper, self.lower = self.power_heat, 0

    def define_variables(self, model, **kwargs):
        self.comp_type = kwargs['comp_type']

        # Define the heat power variable (positive as it generates heat)
        model = self.define_heat_variable(model, comp_type=self.comp_type, lower=self.lower, upper=self.upper)

        # Define the electricity power variable (negative as it consumes electricity)
        model = self.define_electricity_variable(model, comp_type=self.comp_type,
                                                 lower=self.power_electricity, upper=0)

        return model

    def define_constraints(self, model):
        # Add constraint that the heat power is the electricity power times the cop
        model = self.__constraint_cop(model)

        return model

    def __constraint_cop(self, model):
        """Adds the constraint that the heat power is the electricity power times the cop."""

        # Define the variables
        var_heat = model.variables[f'{self.name}_{self.comp_type}_{c.ET_HEAT}']
        var_electricity = model.variables[f'{self.name}_{self.comp_type}_{c.ET_ELECTRICITY}']
        cop = pd.Series(self.cop_heat, index=self.timesteps)

        # Define the constraint
        eq = (var_heat + var_electricity * cop == 0)
        model.add_constraints(eq, name=f'{self.name}_cop', coords=[self.timesteps])

        return model


class Ev(LinopyComps):

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

    def define_variables(self, model, **kwargs):
        self.comp_type = kwargs['comp_type']

        # Define the power variables (need to be positive and negative due to the efficiency)
        model = self.define_electricity_variable(model, comp_type=self.comp_type, lower=self.lower, upper=0,
                                                 direction=c.PF_OUT)  # flow out of the home (charging battery)
        model = self.define_electricity_variable(model, comp_type=self.comp_type, lower=0, upper=self.upper,
                                                 direction=c.PF_IN)  # flow into the home (discharging battery)
        # Define mode flag that decides whether the battery is charging or discharging
        model = self.define_mode_flag(model, comp_type=self.comp_type)

        # Define the soc variable
        model = self.define_storage_variable(model, comp_type=self.comp_type, lower=0, upper=self.capacity)

        # Define the soc variable for the previous timestep (thus the value of self.soc[0]) as needed for constraints
        model.add_variables(name=f'{self.name}_{self.comp_type}_soc_init', lower=self.soc[0], upper=self.soc[0])

        return model

    def define_constraints(self, model):

        # Add constraint that the EV can either charge or discharge but not both at the same time
        model = self.__constraint_operation_mode(model)

        # Add constraint that adheres to the chosen charging scheme
        model = self.__constraint_charging_scheme(model)

        # Add constraint that the soc of the battery is that of the previous timestep plus dis-/charging power
        #   times the efficiency and the time delta
        model = self.__constraint_soc(model)

        return model

    def __define_power_limits_based_on_availability(self):
        """Defines the lower and upper bounds for the charging power based on the availability and v2g."""

        # Define the charging power variables (depends on availability and v2g)
        self.upper = self.charging_power * self.v2g * np.array(self.availability)
        self.lower = -self.charging_power * np.array(self.availability)

        return self.lower, self.upper

    def __constraint_operation_mode(self, model):
        """Adds the constraint that the battery can either charge or discharge but not both at the same time."""

        # Define the variables
        max_power = self.charging_power  # helper variable
        mode_var = model.variables[f'{self.name}_{self.comp_type}_mode']  # mode variable
        var_charge = model.variables[f'{self.name}_{self.comp_type}_{c.ET_ELECTRICITY}_{c.PF_OUT}']  # charging power
        var_discharge = model.variables[f'{self.name}_{self.comp_type}_{c.ET_ELECTRICITY}_{c.PF_IN}']  # discharging

        # Define the constraint for charging
        # Note: The constraints should look something like: var_charge <= max_power * (1 - mode_var)
        #       However, linopy does not support it. Thus, the somewhat more complicated version below is used.
        equation_charging = (var_charge + mode_var * self.lower >= self.lower)
        model.add_constraints(equation_charging, name=f'{self.name}_chargingflag', coords=[self.timesteps])

        # Define the constraint for discharging
        equation_discharging = (var_discharge - mode_var * self.upper <= 0)
        model.add_constraints(equation_discharging, name=f'{self.name}_dischargingflag', coords=[self.timesteps])

        return model

    def __constraint_charging_scheme(self, model):
        """Adds the constraint that the soc is always above the minimum to reach the target soc."""

        scheme = self.scheme['method']

        match scheme:
            case 'full':
                model = self.__constraint_cs_full(model)
            case 'price_sensitive':
                raise NotImplementedError(f'Charging scheme {scheme} not implemented yet.')
            case 'min_soc':
                model = self.__constraint_cs_min_soc(model)
            case 'min_soc_time':
                raise NotImplementedError(f'Charging scheme {scheme} not implemented yet.')
            case _:
                raise ValueError(f'Charging scheme {scheme} not available.')

        return model

    def __constraint_cs_full(self, model):
        """Define constraints to ensure battery is fully charged.

        Args:
            model (Object): The optimization model.

        Returns:
            Object: The updated model with new constraints.
        """
        # Define the variable for state of charge
        var_soc = model.variables[f'{self.name}_{self.comp_type}_soc']
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
        eq_soc = (var_soc >= target_soc)
        model.add_constraints(eq_soc, name=f'{self.name}_soc_scheme')

        return model

    def __constraint_cs_min_soc(self, model):
        """Define constraints to ensure a minimum state of charge is met.

        Args:
            model (Object): The optimization model.

        Returns:
            Object: The updated model with new constraints.
        """
        # Define the variable for state of charge
        var_soc = model.variables[f'{self.name}_{self.comp_type}_soc']
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

        # Define the constraint
        eq_soc = (var_soc >= target_soc)
        model.add_constraints(eq_soc, name=f'{self.name}_soc_scheme')

        return model

    def __constraint_soc(self, model):
        """Adds the constraint that the soc of the battery is that of the previous timestep plus dis-/charging power"""

        # Define the variables
        dt_hours = pd.Series([self.dt * c.SECONDS_TO_HOURS] * len(self.timesteps), index=self.timesteps)  # time in h
        efficiency = pd.Series([self.efficiency] * len(self.timesteps), index=self.timesteps)  # efficiency
        var_soc = model.variables[f'{self.name}_{self.comp_type}_soc']  # soc variable
        var_charge = model.variables[f'{self.name}_{self.comp_type}_{c.ET_ELECTRICITY}_{c.PF_OUT}']  # charging power
        var_discharge = model.variables[f'{self.name}_{self.comp_type}_{c.ET_ELECTRICITY}_{c.PF_IN}']  # discharging

        # Define the array that contains all previous socs
        var_soc_init = model.variables[f'{self.name}_{self.comp_type}_soc_init']  # current soc
        var_soc_prev = var_soc.roll(timesteps=1)  # previous soc
        # Update the first soc value with the initial soc
        var_soc_prev.lower[0] = var_soc_init.lower
        var_soc_prev.upper[0] = var_soc_init.upper
        var_soc_prev.labels[0] = var_soc_init.labels

        # Define the constraint for charging
        # Constraint: soc_new = soc_old + charge * efficiency * dt - discharge / efficiency * dt
        # Note: Everything is moved to the lhs as linopy cannot handle it otherwise
        eq = (var_soc
              - var_soc_prev
              + var_charge * efficiency * dt_hours
              + var_discharge / efficiency * dt_hours  # negative as discharging is negative
              == 0)

        model.add_constraints(eq, name=f'{self.name}_soc')

        return model

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


class SimpleStorage(LinopyComps):

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

    def define_variables(self, model, **kwargs):
        self.comp_type = kwargs['comp_type']

        # Define the power variables (need to be positive and negative due to the efficiency)
        model = self._define_power_variables(model)

        # Define mode flag that decides whether the battery is charging or discharging
        model = self.define_mode_flag(model, comp_type=self.comp_type)

        # Define the soc variable
        model = self.define_storage_variable(model, comp_type=self.comp_type, lower=0, upper=self.capacity)

        # Define the soc variable for the previous timestep (thus the value of self.soc[0]) as needed for constraints
        model.add_variables(name=f'{self.name}_{self.comp_type}_soc_init', lower=self.soc, upper=self.soc)

        return model

    def _define_power_variables(self, model: Model, energy_type: str = c.ET_ELECTRICITY) -> Model:

        # Define the power variables depending on the energy type
        match energy_type:
            case c.ET_ELECTRICITY:
                model = self.define_electricity_variable(model, comp_type=self.comp_type, lower=self.lower, upper=0,
                                                         direction=c.PF_OUT)  # flow out of the home (charging battery)
                model = self.define_electricity_variable(model, comp_type=self.comp_type, lower=0, upper=self.upper,
                                                         direction=c.PF_IN)  # flow into the home (discharging battery)
            case c.ET_HEAT:
                model = self.define_heat_variable(model, comp_type=self.comp_type, lower=self.lower, upper=0,
                                                  direction=c.PF_OUT)  # flow out of the home (charging battery)
                model = self.define_heat_variable(model, comp_type=self.comp_type, lower=0, upper=self.upper,
                                                  direction=c.PF_IN)  # flow into the home (discharging battery)
            case c.ET_COOLING:
                model = self.define_cool_variable(model, comp_type=self.comp_type, lower=self.lower, upper=0,
                                                        direction=c.PF_OUT)  # flow out of the home (charging battery)
                model = self.define_cool_variable(model, comp_type=self.comp_type, lower=0, upper=self.upper,
                                                        direction=c.PF_IN)  # flow into the home (discharging battery)
            case c.ET_H2:
                model = self.define_h2_variable(model, comp_type=self.comp_type, lower=self.lower, upper=0,
                                                direction=c.PF_OUT)
                model = self.define_h2_variable(model, comp_type=self.comp_type, lower=0, upper=self.upper,
                                                direction=c.PF_IN)
            case _:
                raise NotImplementedError(f'{energy_type} has not been implemented yet for the simple storage system.')

        return model

    def define_constraints(self, model: Model) -> Model:

        # Add constraint that the battery can either charge or discharge but not both at the same time
        model = self._constraint_operation_mode(model)

        # TODO: Limit the charging power to the own generation if the battery must not charge from grid
        model = self._constraint_power_limits(model)

        # Add constraint that the soc of the battery is that of the previous timestep plus dis-/charging power
        #   times the efficiency and the time delta
        model = self._constraint_soc(model)

        return model

    def _constraint_operation_mode(self, model: Model, energy_type: str = c.ET_ELECTRICITY) -> Model:
        """Adds the constraint that the battery can either charge or discharge but not both at the same time."""

        # Define the variables
        mode_var = model.variables[f'{self.name}_{self.comp_type}_mode']  # mode variable
        var_charge = model.variables[f'{self.name}_{self.comp_type}_{energy_type}_{c.PF_OUT}']  # charging power
        var_discharge = model.variables[f'{self.name}_{self.comp_type}_{energy_type}_{c.PF_IN}']  # discharging

        # Define the constraint for charging
        # Note: The constraints should look something like: var_discharge >= -max_power * (1 - mode_var)
        #       However, linopy does not support it. Thus, the somewhat more complicated version below is used.
        equation_charging = (var_charge + mode_var * self.lower >= self.lower)
        model.add_constraints(equation_charging, name=f'{self.name}_chargingflag', coords=[self.timesteps])

        # Define the constraint for discharging
        equation_discharging = (var_discharge - mode_var * self.upper <= 0)
        model.add_constraints(equation_discharging, name=f'{self.name}_dischargingflag', coords=[self.timesteps])

        return model

    def _constraint_power_limits(self, model: Model) -> Model:
        return model

    def _constraint_soc(self, model: Model, energy_type: str = c.ET_ELECTRICITY) -> Model:
        """Adds the constraint that the soc of the battery is that of the previous timestep plus dis-/charging power"""

        # Define the variables
        dt_hours = pd.Series([self.dt * c.SECONDS_TO_HOURS] * len(self.timesteps), index=self.timesteps)  # time in h
        efficiency = pd.Series([self.efficiency] * len(self.timesteps), index=self.timesteps)  # efficiency
        var_soc = model.variables[f'{self.name}_{self.comp_type}_soc']  # soc variable
        var_charge = model.variables[f'{self.name}_{self.comp_type}_{energy_type}_{c.PF_OUT}']  # charging power
        var_discharge = model.variables[f'{self.name}_{self.comp_type}_{energy_type}_{c.PF_IN}']  # discharging

        # Define the array that contains all previous socs
        var_soc_init = model.variables[f'{self.name}_{self.comp_type}_soc_init']  # current soc
        var_soc_prev = var_soc.roll(timesteps=1)  # previous soc
        # Update the first soc value with the initial soc
        var_soc_prev.lower[0] = var_soc_init.lower
        var_soc_prev.upper[0] = var_soc_init.upper
        var_soc_prev.labels[0] = var_soc_init.labels

        # Define the constraint for charging
        # Constraint: soc_new = soc_old + charge * efficiency * dt - discharge / efficiency * dt
        # Note: Everything is moved to the lhs as linopy cannot handle it otherwise
        eq = (var_soc
              - var_soc_prev
              + var_charge * efficiency * dt_hours
              + var_discharge / efficiency * dt_hours
              == 0)
        model.add_constraints(eq, name=f'{self.name}_soc')

        return model


class Battery(SimpleStorage):

    def __init__(self, name, **kwargs):
        # Call the parent class constructor
        super().__init__(name, **kwargs)
        self.b2g = self.info['sizing']['b2g']
        self.g2b = self.info['sizing']['g2b']


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

    def define_variables(self, model, **kwargs) -> Model:
        self.comp_type = kwargs['comp_type']

        # Define the power variables (need to be positive and negative due to the efficiency)
        model = self._define_power_variables(model, energy_type=c.ET_HEAT)

        # Define mode flag that decides whether the storage is charging or discharging
        model = self.define_mode_flag(model, comp_type=self.comp_type)

        # Define the soc variable
        model = self.define_storage_variable(model, comp_type=self.comp_type, lower=0, upper=self.capacity)

        # Define the soc variable for the previous timestep (thus the value of self.soc[0]) as needed for constraints
        model.add_variables(name=f'{self.name}_{self.comp_type}_soc_init', lower=self.soc, upper=self.soc)

        return model

    def define_constraints(self, model: Model) -> Model:

        # Add constraint that the battery can either charge or discharge but not both at the same time
        model = self._constraint_operation_mode(model, energy_type=c.ET_HEAT)

        # TODO: Limit the charging power to the own generation if the battery must not charge from grid
        model = self._constraint_power_limits(model)

        # Add constraint that the soc of the battery is that of the previous timestep plus dis-/charging power
        #   times the efficiency and the time delta
        model = self._constraint_soc(model, energy_type=c.ET_HEAT)

        return model
