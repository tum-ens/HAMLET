__author__ = "MarkusDoepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

from linopy import Model
import polars as pl
from pprint import pprint
from hamlet import constants as c
import hamlet.executor.utilities.controller.rtc.linplants as linplants


class RtcBase:
    def run(self):
        raise NotImplementedError()


class Rtc:

    def __init__(self, method='linopy', **kwargs):
        self.method = method
        self.kwargs = kwargs

        # Mapping from input string to class name
        self.class_mapping = {
            'linopy': self.Linopy,
            'rule-based': self.RuleBased
        }

    def run(self, **kwargs):
        # Return if no method is specified
        if self.method is None:
            return

        # Use the mapping to get the class
        controller_class = self.class_mapping.get(self.method.lower())

        if controller_class is None:
            raise ValueError(f"Unsupported method: {self.method}.\n"
                             f"The available methods are: {self.class_mapping.keys()}")

        return controller_class(**kwargs).run()

    class Linopy(RtcBase):
        def __init__(self, **kwargs):
            # Create the model
            self.model = Model()

            # Get the timetable and filter it to only include the rows with the current timestep
            self.timetable = kwargs[c.TN_TIMETABLE]
            self.timetable = self.timetable.filter(
                pl.col(c.TC_TIMESTAMP) == pl.col(c.TC_TIMESTEP))

            # Get the agent and other data
            self.agent = kwargs['agent']
            self.account = self.agent.account
            self.plants = self.agent.plants  # Formely known as components
            self.setpoints = self.agent.setpoints
            self.socs = self.agent.socs
            self.timeseries = self.agent.timeseries
            # Filter the timeseries to only include the rows with the current timestamp
            self.timeseries = self.timeseries.join(self.timetable, on=c.TC_TIMESTAMP, how='semi')

            # Raise warning if timeseries exceeds one row
            if len(self.timeseries.collect()) != 1:
                raise ValueError(f"Timeseries has {len(self.timeseries)} rows. It should only have 1 row for the rtc.")

            # Get the market data
            self.market = kwargs['market']

            # Available plants
            self.available_plants = {
                c.P_INFLEXIBLE_LOAD: linplants.InflexibleLoad,
                c.P_FLEXIBLE_LOAD: linplants.FlexibleLoad,
                c.P_HEAT: linplants.Heat,
                c.P_DHW: linplants.Dhw,
                c.P_PV: linplants.Pv,
                c.P_WIND: linplants.Wind,
                c.P_FIXED_GEN: linplants.FixedGen,
                c.P_HP: linplants.Hp,
                c.P_EV: linplants.Ev,
                c.P_BATTERY: linplants.Battery,
                c.P_PSH: linplants.Psh,
                c.P_HYDROGEN: linplants.Hydrogen,
                c.P_HEAT_STORAGE: linplants.HeatStorage,
            }

            # Create the plant objects
            self.plant_objects = {}
            self.create_plants()

            # Define the model
            self.define_variables()
            self.define_constraints()
            self.define_objective()

        def create_plants(self):
            for plant_name, plant_data in self.plants.items():

                # Get the plant type from the plant data
                plant_type = plant_data['type']

                # Retrieve the timeseries data for the plant
                cols = [col for col in self.timeseries.columns if col.startswith(plant_name)]
                timeseries = self.timeseries.select(cols)

                # Get the plant class
                plant_class = self.available_plants.get(plant_type)
                if plant_class is None:
                    raise ValueError(f"Unsupported plant type: {plant_name}")

                # Create the plant object
                self.plant_objects[plant_name] = plant_class(name=plant_name, timeseries=timeseries, **plant_data)

            return self.plant_objects

        def define_variables(self):
            # Define variables for each plant
            for plant_name, plant in self.plant_objects.items():
                self.model = plant.define_variables(self.model)

        def define_constraints(self):
            # Define constraints for each plant
            for plant_name, plant in self.plant_objects.items():
                plant.define_constraints(self.model)

            # Additional constraints for energy balancing, etc.
            self.add_balance_constraints()

        def add_balance_constraints(self):
            # Add balance constraints if applicable
            pass

        def define_objective(self):
            # Define the objective function here
            pass

        def run(self):

            # Solve the optimization problem
            solution = self.model.solve()

            # Check for solver success
            if not solution.success:
                raise RuntimeError("Solver failed to find a solution")

            # Process the solution into control commands and return
            control_commands = self.process_solution(solution)
            return control_commands

        def process_solution(self, solution):
            # Process the optimization solution into actionable control commands
            return control_commands

    class RuleBased(RtcBase):  # Note the change in class name

        def __init__(self, **kwargs):
            pass

        def run(self):
            print('Running Rule-Based')

