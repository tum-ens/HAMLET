__author__ = "MarkusDoepfert"
__credits__ = "HodaHamdy"
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

import polars as pl

import hamlet.constants as c
from hamlet.executor.utilities.controller.rtc.rtc_base import RtcBase


class OptimBase(RtcBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Create the model
        self.model = self.get_model(**kwargs)
        self.ems = self.ems[c.C_CONTROLLER][c.C_RTC]

        # Obtain maximum balancing power
        # TODO: Deprecate balancing market in the equations. The relevant value is the market itself.
        #  Balancing occurs in the market section of the code.
        # TODO: self.balancing = db.get_balancing_power(market name, energy type)

        # Available plants
        self.available_plants = self.get_available_plants()

        # Create the plant objects
        self.plant_objects = self.create_plants()

        # Create the market objects
        self.market_class = self.get_market_class()
        self.market_objects = {}
        self.create_markets()

        # Define the model
        self.define_variables()
        self.define_constraints()
        self.define_objective()
        self.apply_grid_commands()

    def create_plants(self):
        """
        Create the plant objects for the optimization problem
        """
        for plant_name, plant_data in self.plants.items():

            # Get the plant type from the plant data
            plant_type = plant_data['type']

            # Retrieve the timeseries data for the plant
            cols = [col for col in self.timeseries.columns if col.startswith(plant_name)]
            timeseries = self.timeseries.select(cols)

            # Retrieve the target setpoints for the plant
            cols = [col for col in self.targets.columns if col.startswith(plant_name)]
            targets = self.targets.select(cols)

            # Retrieve the soc data for the plant (if applicable)
            cols = [col for col in self.socs.columns if col.startswith(plant_name)]
            socs = self.socs.select(cols)

            # Get the plant class
            plant_class = self.available_plants.get(plant_type)
            if plant_class is None:
                raise ValueError(f"Unsupported plant type: {plant_name}")

            # Create the plant object
            self.plant_objects[plant_name] = plant_class(name=plant_name, timeseries=timeseries, **plant_data,
                                                         targets=targets, socs=socs, delta=self.dt)

        return self.plant_objects

    def create_markets(self):
        """
        Create the market objects for the optimization problem
        """

        # Define variables from the market results and a balancing variable for each energy type
        for market in self.markets:
            # Create market object
            self.market_objects[market] = self.market_class(name=market,
                                                            timeseries=self.market,
                                                            market_result=self.market_results[market],
                                                            delta=self.dt)

        return self.market_objects

############################################## To be implemented in subclasses #########################################

    def get_model(self, **kwargs):
        raise NotImplementedError

    def get_available_plants(self):
        raise NotImplementedError

    def get_market_class(self):
        raise NotImplementedError

    def get_solution(self):
        raise NotImplementedError

    def define_variables(self):
        pass

    def define_constraints(self):
        pass

    def define_objective(self):
        pass

    def apply_grid_commands(self):
        """Apply grid commands if any"""
        pass
