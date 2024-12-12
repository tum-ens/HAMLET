__author__ = "MarkusDoepfert"
__credits__ = "HodaHamdy"
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

from hamlet.executor.utilities.controller.fbc.fbc_base import FbcBase


class MpcBase(FbcBase):
    def __init__(self, **kwargs):
        # Call the super class
        super().__init__(**kwargs)
        # Create the model
        self.model = self.get_model(**kwargs)
        self.ems = self.ems['controller']['fbc']

        # Create the market objects
        self.market_class = self.get_market_class()
        self.market_objects = {}
        self.create_markets()

        # Create the plant objects
        self.plant_objects = {}
        self.create_plants()

        # Define the model
        self.define_variables()
        self.define_constraints()
        self.define_objective()

    def create_markets(self):
        """"""

        # Define variables from the market results and a balancing variable for each energy type
        for market in self.markets:
            # Create market object
            self.market_objects[f'{market}'] = self.market_class(name=market,
                                                                 forecasts=self.forecasts,
                                                                 timesteps=self.timesteps,
                                                                 delta=self.dt)

        return self.market_objects

    def create_plants(self):
        for plant_name, plant_data in self.plants.items():

            # Get the plant type from the plant data
            plant_type = plant_data['type']

            # Retrieve the forecast data for the plant
            cols = [col for col in self.forecasts.columns if col.startswith(plant_name)]
            forecasts = self.forecasts.select(cols)

            # Retrieve the soc data for the plant (if applicable)
            cols = [col for col in self.socs.columns if col.startswith(plant_name)]
            socs = self.socs.select(cols)

            # Get the plant class
            plant_class = self.available_plants.get(plant_type)
            if plant_class is None:
                raise ValueError(f"Unsupported plant type: {plant_name} for the chosen mpc method.")

            # Create the plant object
            self.plant_objects[plant_name] = plant_class(name=plant_name,
                                                         forecasts=forecasts,
                                                         **plant_data,
                                                         socs=socs,
                                                         delta=self.dt,
                                                         timesteps=self.timesteps,
                                                         markets=self.markets)

        return self.plant_objects

############################################## To be implemented in subclasses #########################################

    def get_model(self, **kwargs):
        raise NotImplementedError()

    def get_market_class(self):
        raise NotImplementedError()

    def define_variables(self):
        raise NotImplementedError()

    def define_constraints(self):
        raise NotImplementedError()

    def define_objective(self):
        raise NotImplementedError()
