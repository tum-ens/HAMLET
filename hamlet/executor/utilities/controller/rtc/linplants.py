from hamlet.constants import *
from pprint import pprint


class LinopyPlant:

    def __init__(self, name, timeseries, **kwargs):

        # Get the data
        self.name = name
        self.ts = timeseries.collect()
        self.info = kwargs

    def define_variables(self, model):
        raise NotImplementedError()

    def define_constraints(self, model):
        raise NotImplementedError()

    def define_power_variable(self, model, lower, upper):
        # Define the power variable
        model.add_variables(name=f'{self.name}_power', lower=lower, upper=upper, integer=True)

        return model


class InflexibleLoad(LinopyPlant):

    def __init__(self, name, timeseries, **kwargs):

        # Call the parent class constructor
        super().__init__(name, timeseries, **kwargs)


        # Get specific object attributes
        self.power = self.ts[f'{self.name}_power'][0]

    def define_variables(self, model):
        # Define the variables for an inflexible load

        # Define the power variable
        model = self.define_power_variable(model, lower=self.power, upper=self.power)

        return model

    def define_constraints(self, model):
        # Add constraints specific to an inflexible load
        return model


class FlexibleLoad(LinopyPlant):

    def __init__(self, name, timeseries, **kwargs):

        # Call the parent class constructor
        super().__init__(name, timeseries, **kwargs)

        # Get specific object attributes
        print(self.ts)


class Heat(LinopyPlant):

    def __init__(self, name, timeseries, **kwargs):

        # Call the parent class constructor
        super().__init__(name, timeseries, **kwargs)

        # Get specific object attributes
        print(self.ts)


class Dhw(LinopyPlant):

    def __init__(self, name, timeseries, **kwargs):

        # Call the parent class constructor
        super().__init__(name, timeseries, **kwargs)

        # Get specific object attributes
        print(self.ts)


class Pv(LinopyPlant):

    def __init__(self, name, timeseries, **kwargs):

        # Call the parent class constructor
        super().__init__(name, timeseries, **kwargs)

        # Get specific object attributes
        self.power = self.ts[f'{self.name}_power'][0]
        self.controllable = self.info['sizing']['controllable']
        self.lower = 0 if self.controllable else self.power

    def define_variables(self, model):
        # Define the variables for an inflexible load

        # Define the power variable
        model = self.define_power_variable(model, lower=self.lower, upper=self.power)

        return model

    def define_constraints(self, model):
        # Add constraints specific to an inflexible load
        return model


class Wind(LinopyPlant):

    def __init__(self, name, timeseries, **kwargs):

        # Call the parent class constructor
        super().__init__(name, timeseries, **kwargs)

        # Get specific object attributes
        print(self.ts)


class FixedGen(LinopyPlant):

    def __init__(self, name, timeseries, **kwargs):

        # Call the parent class constructor
        super().__init__(name, timeseries, **kwargs)

        # Get specific object attributes
        print(self.ts)


class Hp(LinopyPlant):

    def __init__(self, name, timeseries, **kwargs):

        # Call the parent class constructor
        super().__init__(name, timeseries, **kwargs)

        # Get specific object attributes
        print(self.ts)


class Ev(LinopyPlant):

    def __init__(self, name, timeseries, **kwargs):

        # Call the parent class constructor
        super().__init__(name, timeseries, **kwargs)

        # Get specific object attributes
        print(self.ts)
        print('Continue with the EV. Consider if you want the input data to be distance driven or simply energy consumed.')
        exit()

    def define_variables(self, model):
        pass

    def define_constraints(self, model):
        # Add constraints specific to an electric vehicle
        pass


class Battery(LinopyPlant):

    def __init__(self, name, timeseries, **kwargs):

        # Call the parent class constructor
        super().__init__(name, timeseries, **kwargs)

        # Get specific object attributes
        print(self.ts)

    def define_variables(self, model):
        pass

    def define_constraints(self, model):
        # Add constraints specific to a battery
        pass


class Psh(LinopyPlant):

    def __init__(self, name, timeseries, **kwargs):

        # Call the parent class constructor
        super().__init__(name, timeseries, **kwargs)

        # Get specific object attributes
        print(self.ts)


class Hydrogen(LinopyPlant):

    def __init__(self, name, timeseries, **kwargs):

        # Call the parent class constructor
        super().__init__(name, timeseries, **kwargs)

        # Get specific object attributes
        print(self.ts)


class HeatStorage(LinopyPlant):

    def __init__(self, name, timeseries, **kwargs):

        # Call the parent class constructor
        super().__init__(name, timeseries, **kwargs)

        # Get specific object attributes
        print(self.ts)


# class Battery(LinopyPlant):
#     def __init__(self, max_power, max_soc):
#         self.max_power = max_power
#         self.max_soc = max_soc
#
#     def define_variables(self, model):
#         model.add_variable(f'{self.name}_power', lb=0, ub=self.max_power)
#         model.add_variable(f'{self.name}_soc', lb=0, ub=self.max_soc)
#
#     def define_constraints(self, model):
#         # Add constraints specific to a battery
#         pass
