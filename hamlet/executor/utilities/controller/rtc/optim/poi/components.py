__author__ = "HodaHamdy"
__credits__ = "MarkusDoepfert"
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

import math

import polars.exceptions as pl_e
import pyoptinterface as poi
from numpy import inf

import hamlet.constants as c


class POIComps:
    def __init__(self, name, timeseries, **kwargs):

        # Get the data
        self.name = name
        self.ts = timeseries
        self.info = kwargs

        # Other attributes (to be defined in subclasses)
        self.comp_type = None
        self.target = None
        self.upper = None
        self.lower = None

    def define_variables(self, model, variables, **kwargs):
        raise NotImplementedError(f'{self.name} has not been implemented yet.')

    @staticmethod
    def define_constraints(model, variables):
        pass

    @staticmethod
    def add_variable_to_model(model, variables, name, **kwargs):
        var_name = name
        lb = kwargs.get("lower", -math.inf)
        ub = kwargs.get("upper", math.inf)
        kwargs_var = {
            'name': var_name,
            'lb': lb,
            'ub': ub,
            'domain': poi.VariableDomain.Integer if kwargs.get('integer', False)
            else poi.VariableDomain.Binary
            if kwargs.get('binary', False) else poi.VariableDomain.Continuous,
        }
        variables[name] = model.add_variable(**kwargs_var)

    def define_electricity_variable(self, model, variables, comp_type, lower, upper, integer=False):
        # Define the power variable
        self.add_variable_to_model(model, variables, name=f'{self.name}_{comp_type}_{c.ET_ELECTRICITY}', lower=lower,
                                   upper=upper,
                                   integer=integer)

    def define_heat_variable(self, model, variables, comp_type, lower, upper, load_target=None, integer=False):
        # Define the power variable
        if load_target is None:
            name = f'{self.name}_{comp_type}_{c.ET_HEAT}'
        else:
            name = f'{self.name}_{comp_type}_{c.ET_HEAT}_{load_target}'
        self.add_variable_to_model(model, variables, name=name, lower=lower, upper=upper, integer=integer)

    def define_cool_variable(self, model, variables, comp_type, lower, upper, load_target=None, integer=False):
        # Define the power variable
        if load_target is None:
            name = f'{self.name}_{comp_type}_{c.ET_COOLING}'
        else:
            name = f'{self.name}_{comp_type}_{c.ET_COOLING}_{load_target}'
        self.add_variable_to_model(model, variables, name=name, lower=lower, upper=upper, integer=integer)

    def define_h2_variable(self, model, variables, comp_type, lower, upper, integer=False):
        # Define the power variable
        self.add_variable_to_model(model, variables, name=f'{self.name}_{comp_type}_{c.ET_H2}', lower=lower,
                                   upper=upper,
                                   integer=integer)

    # Subclass methods

    def _define_target_and_deviations_variables(self, model, variables):
        self.add_variable_to_model(model, variables, name=f'{self.name}_{self.comp_type}_target',
                                   lower=self.target, upper=self.target)

        # Define the deviation variable for positive and negative deviations
        # Deviation when more is charged than according to target
        name = f'{self.name}_{self.comp_type}_deviation_pos'
        if self.target <= self.upper:
            self.add_variable_to_model(model, variables, name=name, lower=0, upper=max(0, self.upper - self.target))
        else:
            raise Warning(f'Target value ({self.target}) is higher than upper limit ({self.upper}) for {name}.')
        # Deviation when less is discharged than according to target
        name = f'{self.name}_{self.comp_type}_deviation_neg'
        if self.target >= self.lower:
            self.add_variable_to_model(model, variables, name=name, lower=0, upper=max(0, self.target - self.lower))
        else:
            raise Warning(f'Target value ({self.target}) is lower than lower limit ({self.lower}) for {name}.')

    def _constraint_target_deviation(self, model, variables, energy_type: str):
        # Define the variables
        var_power = variables[f'{self.name}_{self.comp_type}_{energy_type}']
        var_target = variables[f'{self.name}_{self.comp_type}_target']
        var_deviation_pos = variables[f'{self.name}_{self.comp_type}_deviation_pos']
        var_deviation_neg = variables[f'{self.name}_{self.comp_type}_deviation_neg']

        # Define the deviation constraint
        model.add_linear_constraint(var_power - var_target - var_deviation_pos + var_deviation_neg,
                                    poi.ConstraintSense.Equal, 0, name=f'{self.name}_deviation')


class Market(POIComps):

    def __init__(self, name, **kwargs):
        # Call the parent class constructor
        super().__init__(name, **kwargs)

        # Get specific object attributes
        self.dt = kwargs['delta'].total_seconds()  # time delta in seconds
        self.market_power = int(round(kwargs['market_result'] * c.HOURS_TO_SECONDS / self.dt))  # from Wh to W
        self.balancing_power = 10000000000  # TODO: This needs to be changed to the max available balancing power

        # Get the energy type
        self.energy_type = None

    def define_variables(self, model, variables, **kwargs):
        self.energy_type = kwargs['energy_type']

        # Define the market power variable
        self.add_variable_to_model(model, variables, name=f'{self.name}_{self.energy_type}', lower=-inf, upper=inf,
                                   integer=True)

        # Define the target variable (what was previously bought/sold on the market)
        self.add_variable_to_model(model, variables, name=f'{self.name}_{self.energy_type}_target',
                                   lower=self.market_power, upper=self.market_power, integer=True)

        # Define the deviation variable for positive and negative deviations
        # Deviation when more is bought/sold on the market than according to the market
        self.add_variable_to_model(model, variables, name=f'{self.name}_{self.energy_type}_deviation_pos',
                                   lower=0, upper=self.balancing_power, integer=True)
        # Deviation when less is needed from the grid than according to the market
        self.add_variable_to_model(model, variables, name=f'{self.name}_{self.energy_type}_deviation_neg',
                                   lower=0, upper=self.balancing_power, integer=True)

    def define_constraints(self, model, variables):
        # Define the deviation constraint
        equation = (variables[f'{self.name}_{self.energy_type}']
                    - variables[f'{self.name}_{self.energy_type}_target']
                    - variables[f'{self.name}_{self.energy_type}_deviation_pos']
                    + variables[f'{self.name}_{self.energy_type}_deviation_neg'])

        model.add_linear_constraint(equation, poi.ConstraintSense.Equal, 0, name=f'{self.name}_deviation')


class InflexibleLoad(POIComps):

    def __init__(self, name, **kwargs):
        # Call the parent class constructor
        super().__init__(name, **kwargs)

        # Get specific object attributes
        self.power = self.ts[f'{self.name}_{c.ET_ELECTRICITY}'][0]

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
        self.heat = self.ts[f'{self.name}_heat'][0]

    def define_variables(self, model, variables, **kwargs):
        comp_type = kwargs['comp_type']

        # Define the power variable
        self.define_heat_variable(model, variables, comp_type=comp_type, lower=-self.heat, upper=-self.heat)


class Dhw(POIComps):

    def __init__(self, name, **kwargs):
        # Call the parent class constructor
        super().__init__(name, **kwargs)

        # Get specific object attributes
        self.dhw = self.ts[f'{self.name}_dhw'][0]

    def define_variables(self, model, variables, **kwargs):
        comp_type = kwargs['comp_type']

        # Define the power variable
        self.define_heat_variable(model, variables, comp_type=comp_type, lower=-self.dhw, upper=-self.dhw)


class SimplePlant(POIComps):

    def __init__(self, name, **kwargs):
        # Call the parent class constructor
        super().__init__(name, **kwargs)

        # Get specific object attributes
        self.power = self.ts[f'{self.name}_{c.ET_ELECTRICITY}'][0]
        self.controllable = self.info['sizing']['controllable']
        self.lower = 0 if self.controllable else self.power

    def define_variables(self, model, variables, **kwargs):
        comp_type = kwargs['comp_type']

        # Define the power variable
        self.define_electricity_variable(model, variables, comp_type=comp_type, lower=self.lower, upper=self.power)


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
        try:
            self.target = kwargs['targets'][f'{self.name}'][0]
        except pl_e.ColumnNotFoundError:
            self.target = kwargs['targets'][f'{self.name}_{c.P_HP}_{c.ET_ELECTRICITY}'][0]
        try:
            self.cop_heat = self.ts[f'{self.name}_{c.S_COP}_{c.P_HEAT}'][0] * c.COP100_TO_COP
        except pl_e.ColumnNotFoundError:
            self.cop_heat = self.ts[f'{self.name}_cop'][0]
        # TODO: Model dhw separately (as a different energy type)
        # self.cop_dhw = self.ts[f'{self.name}_{c.S_COP}_{c.P_DHW}'][0] * c.COP100_TO_COP

        # Calculate the power for heating and dhw
        # Note: The sizing power is a fallback method to ensure that there is always enough power so that the
        #  model does not fail.
        try:
            self.power_heat = max(self.info['sizing']['power'],
                                  self.ts[f'{self.name}_{c.S_POWER}_{c.ET_HEAT}_{c.P_HEAT}'][0])
        except pl_e.ColumnNotFoundError:
            try:
                self.power_heat = self.info['sizing']['power']
            except KeyError:
                self.power_heat = inf
        # self.power_dhw = max(self.info['sizing']['power'],
        #                      self.ts[f'{self.name}_{c.S_POWER}_{c.ET_HEAT}_{c.P_DHW}'][0])

        # Calculate the power for electricity
        try:
            self.power_electricity = -int(round(max(self.info['sizing']['power'] / self.cop_heat,
                                                    self.ts[f'{self.name}_{c.S_POWER}_{c.ET_ELECTRICITY}_{c.P_HEAT}'][
                                                        0])))
        except pl_e.ColumnNotFoundError:
            self.power_electricity = -inf

        self.upper, self.lower = 0, self.power_electricity

        # Ensure that target never exceeds the limits
        self.target = max(min(self.target, self.upper), self.lower)

    def define_variables(self, model, variables, **kwargs):
        self.comp_type = kwargs['comp_type']

        # Define the heat power variable (positive as it generates heat)
        self.define_heat_variable(model, variables, comp_type=self.comp_type, lower=0, upper=self.power_heat)
        #  self.define_heat_variable(model, comp_type=comp_type, lower=0, upper=self.power_dhw,
        # load_target=c.P_DHW)

        # Define the electricity power variable (negative as it consumes electricity)
        self.define_electricity_variable(model, variables, comp_type=self.comp_type,
                                         lower=self.power_electricity, upper=0)

        # Define the target and deviation variables (refers to the heat power)
        self._define_target_and_deviations_variables(model, variables)


def define_constraints(self, model, variables):
    # Add constraint that the heat power is the electricity power times the cop
    self.__constraint_cop(model, variables)

    # Define the deviation constraint that shows the relationship between the power and the target
    self._constraint_target_deviation(model, variables, energy_type=c.ET_ELECTRICITY)


def __constraint_cop(self, model, variables):
    """Add constraint that the heat power is the electricity power times the cop"""
    # Define the variables
    var_electricity = variables[f'{self.name}_{self.comp_type}_{c.ET_ELECTRICITY}']
    var_heat = variables[f'{self.name}_{self.comp_type}_{c.ET_HEAT}']

    # Add the constraint
    model.add_linear_constraint(var_heat + var_electricity * self.cop_heat, poi.ConstraintSense.Equal, 0,
                                name=f'{self.name}_cop')


class Ev(POIComps):

    def __init__(self, name, **kwargs):

        # Call the parent class constructor
        super().__init__(name, **kwargs)

        # Get specific object attributes
        self.comp_type = None
        # Workaround as it will be this only in the first instance but then name_ev_target after the first time
        # In the future the column should already be in the format to begin with
        try:
            self.target = kwargs['targets'][f'{self.name}'][0]
        except pl_e.ColumnNotFoundError:
            self.target = kwargs['targets'][f'{self.name}_{c.P_EV}_{c.ET_ELECTRICITY}'][0]
        self.availability = self.ts[f'{self.name}_availability'][0]
        self.energy = self.ts[f'{self.name}_energy_consumed'][0]
        # For now this is always charging at home. In the future this can depend on the availability column if it shows
        # the location instead of availability at home. Most of it is already prepared.
        self.capacity = self.info['sizing']['capacity']
        self.charging_power_home = self.info['sizing']['charging_home']
        self.charging_power_AC = self.info['sizing']['charging_AC']
        self.charging_power_DC = self.info['sizing']['charging_DC']
        self.charging_power = self.charging_power_home  # To be changed once more sophisticated EV modelling available
        self.efficiency = self.info['sizing']['charging_efficiency']
        self.v2g = self.info['sizing']['v2g']

        # Charging scheme
        self.scheme = self.info['charging_scheme']

        # Kwargs variables
        self.dt = kwargs['delta'].total_seconds()  # time delta in seconds
        self.soc = max(0, kwargs['socs'][f'{self.name}'][0] - self.energy)  # state of charge at timestep (energy);
        # must be greater than 0, overconsumption is assumed to be compensated elsewhere

        # Calculate the energy needed to charge to full
        self.energy_to_full = self.capacity - self.soc  # energy needed to charge to full

        # Define the charging power variables (depends on availability and v2g)
        if self.availability:
            # Set upper bound based on v2g
            if self.v2g:
                # Set lower bound by calculating max discharging power according to capacity, soc, efficiency and dt
                self.upper = int(round(min(self.charging_power,
                                           self.soc * self.efficiency / (self.dt * c.SECONDS_TO_HOURS))))
            else:
                self.upper = 0

            # Set lower bound by calculating max charging power according to capacity, soc, efficiency and dt
            self.lower = -int(round(min(self.charging_power,
                                        self.energy_to_full / self.efficiency / (self.dt * c.SECONDS_TO_HOURS))))

        else:
            self.lower, self.upper = 0, 0

        # Ensure that target never exceeds the limits
        self.target = max(min(self.target, self.upper), self.lower)

    def define_variables(self, model, variables, **kwargs):
        self.comp_type = kwargs['comp_type']

        # Define the power variable
        self.define_electricity_variable(model, variables, comp_type=self.comp_type, lower=self.lower, upper=self.upper)

        # Define the target variable
        self._define_target_and_deviations_variables(model, variables)

    def define_constraints(self, model, variables):
        # Define the deviation constraint that shows the relationship between the power and the target
        self._constraint_target_deviation(model, variables, energy_type=c.ET_ELECTRICITY)


class SimpleStorage(POIComps):

    def __init__(self, name, **kwargs):

        # Call the parent class constructor
        super().__init__(name, **kwargs)

        # Get specific object attributes
        self.comp_type = None
        try:
            self.target = kwargs['targets'][f'{self.name}'][0]
        except pl_e.ColumnNotFoundError:
            self.target = None  # Define in subclasses if column is not just name
        self.capacity = self.info['sizing']['capacity']
        self.charging_power = self.info['sizing']['power']
        self.efficiency = self.info['sizing']['efficiency']

        # Kwargs variables
        self.dt = kwargs['delta'].total_seconds()  # time delta in seconds
        self.soc = kwargs['socs'][f'{self.name}'][0]  # state of charge at timestep (energy)
        self.energy_to_full = self.capacity - self.soc  # energy needed to charge to full

        # Define the charging and discharging power variables (depend on capacity, soc, efficiency and dt)
        self.upper = int(round(min(self.charging_power,
                                   self.soc * self.efficiency / (self.dt * c.SECONDS_TO_HOURS))))
        self.lower = -int(round(min(self.charging_power,
                                    self.energy_to_full / self.efficiency / (self.dt * c.SECONDS_TO_HOURS))))

    def define_variables(self, model, variables, **kwargs):
        self.comp_type = kwargs['comp_type']

        # Define the power variable
        self._define_power_variable(model, variables)

        # Define the target variable
        self._define_target_and_deviations_variables(model, variables)

    def _define_power_variable(self, model, variables, energy_type: str = c.ET_ELECTRICITY):
        # Define the power variable depending on the energy type
        match energy_type:
            case c.ET_ELECTRICITY:
                self.define_electricity_variable(model, variables, comp_type=self.comp_type,
                                                 lower=self.lower, upper=self.upper)
            case c.ET_HEAT:
                self.define_heat_variable(model, variables, comp_type=self.comp_type, lower=self.lower, upper=self.upper)
            case c.ET_COOLING:
                self.define_cool_variable(model, variables, comp_type=self.comp_type, lower=self.lower, upper=self.upper)
            case c.ET_H2:
                self.define_h2_variable(model, variables, comp_type=self.comp_type, lower=self.lower, upper=self.upper)
            case _:
                raise NotImplementedError(f'{energy_type} has not been implemented yet for the simple storage system.')

    def define_constraints(self, model, variables):
        # Define the deviation constraint that shows the relationship between the power and the target
        self._constraint_target_deviation(model, variables, energy_type=c.ET_ELECTRICITY)

    def _limit_target(self):
        # Ensure that target never exceeds the limits
        return max(min(self.target, self.upper), self.lower)


class Battery(SimpleStorage):

    def __init__(self, name, **kwargs):
        # Call the parent class constructor
        super().__init__(name, **kwargs)

        if self.target is None:
            self.target = kwargs['targets'][f'{self.name}_{c.P_BATTERY}_{c.ET_ELECTRICITY}'][0]
        self.b2g = self.info['sizing']['b2g']
        self.g2b = self.info['sizing']['g2b']

        # Ensure that target never exceeds the limits
        self.target = self._limit_target()


class Psh(SimpleStorage):

    def __init__(self, name, **kwargs):
        # Call the parent class constructor
        super().__init__(name, **kwargs)

        if self.target is None:
            self.target = kwargs['targets'][f'{self.name}_{c.P_PSH}_{c.ET_ELECTRICITY}'][0]

        # Ensure that target never exceeds the limits
        self.target = self._limit_target()


class Hydrogen(SimpleStorage):

    def __init__(self, name, **kwargs):
        # Call the parent class constructor
        super().__init__(name, **kwargs)

        if self.target is None:
            self.target = kwargs['targets'][f'{self.name}_{c.P_HYDROGEN}_{c.ET_H2}'][0]

        # Ensure that target never exceeds the limits
        self.target = self._limit_target()

    def define_variables(self, model, variables, **kwargs):
        self.comp_type = kwargs['comp_type']

        # Define the power variable
        self._define_power_variable(model, variables, energy_type=c.ET_H2)

        # Define the target variable
        self._define_target_and_deviations_variables(model, variables)

    def define_constraints(self, model, variables):
        # Define the deviation constraint that shows the relationship between the power and the target
        self._constraint_target_deviation(model, variables, energy_type=c.ET_H2)


class HeatStorage(SimpleStorage):
    """This one is for heating. Dhw would need a separate one. Will be included in the future."""

    def __init__(self, name, **kwargs):
        # Call the parent class constructor
        super().__init__(name, **kwargs)

        if self.target is None:
            self.target = kwargs['targets'][f'{self.name}_{c.P_HEAT_STORAGE}_{c.ET_HEAT}'][0]

        # Ensure that target never exceeds the limits
        self.target = self._limit_target()

    def define_variables(self, model, variables, **kwargs):
        self.comp_type = kwargs['comp_type']

        # Define the power variable
        self._define_power_variable(model, variables, energy_type=c.ET_HEAT)

        # Define the target variable
        self._define_target_and_deviations_variables(model, variables)

    def define_constraints(self, model, variables):
        # Define the deviation constraint that shows the relationship between the power and the target
        self._constraint_target_deviation(model, variables, energy_type=c.ET_HEAT)
