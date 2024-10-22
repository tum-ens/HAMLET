__author__ = "HodaHamdy"
__credits__ = "MarkusDoepfert"
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

import numpy as np
import pandas as pd
import polars as pl

import hamlet.constants as c


class MpcBase:
    def __init__(self, **kwargs):
        # Create the model
        self.model = self.get_model(**kwargs)

        # Store the mapping of the components to the energy types and operation modes
        self.mapping = kwargs['mapping']
        # Identify all unique energy types
        self.energy_types = set()
        for mapping in self.mapping.values():
            self.energy_types.update(mapping.keys())

        # Get the agent and other data
        self.agent = kwargs['agent']
        self.account = self.agent.account
        self.ems = self.account['ems']
        self.plants = self.agent.plants  # Formerly known as components
        self.setpoints = self.agent.setpoints
        self.forecasts = self.agent.forecasts
        self.socs = self.agent.socs

        # Get the timetable
        self.timetable = kwargs[c.TN_TIMETABLE]
        # Get the delta between timestamps
        self.dt = self.timetable[1, c.TC_TIMESTEP] - self.timetable[0, c.TC_TIMESTEP]
        # Get the current timestamp
        self.timestamp = self.timetable[0, c.TC_TIMESTAMP]
        # Get the time horizon
        self.horizon = pd.Timedelta(seconds=self.ems['controller']['mpc']['horizon'])

        # Reduce the forecast to the horizon length
        self.forecasts = self.forecasts.filter((self.timestamp < self.forecasts[c.TC_TIMESTEP])
                                               & (self.forecasts[c.TC_TIMESTEP] < self.timestamp + self.horizon))
        # Get the timesteps and the number of timesteps in the forecast
        self.timesteps = pd.Index(self.forecasts.select(c.TC_TIMESTEP).to_pandas(), name='timesteps')
        self.timesteps = pd.Index(range(len(self.timesteps)), name='timesteps')
        # self.timesteps = self.n_steps  # Use this line if the timesteps are not needed and the index is sufficient
        # Reduce the socs to the current timestamp
        self.socs = self.socs.filter(self.socs[c.TC_TIMESTAMP] == self.timestamp)

        # Get the market types
        # TODO: Still needs to be done and then adjusted in the market objects (right now the names are simply
        #  local and wholesale as this will suffice as long as there is only one market)
        # Get the market data
        # self.market = kwargs[c.TC_MARKET]
        # Get the market names and types
        self.market_names = self.timetable.select(c.TC_NAME).unique().to_series().to_list()
        self.market_types = self.timetable.select(c.TC_MARKET).unique().to_series().to_list()
        # Assign each market name to an energy type
        self.markets = {name: c.TRADED_ENERGY[mtype] for name, mtype in zip(self.market_names, self.market_types)}

        # Available plants
        self.available_plants = self.get_available_plants()

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

    def run(self):
        raise NotImplementedError()

    def get_model(self, **kwargs):
        raise NotImplementedError

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

    def get_available_plants(self):
        raise NotImplementedError

    def define_variables(self):
        raise NotImplementedError()

    def define_constraints(self):
        raise NotImplementedError()

    def define_objective(self):
        raise NotImplementedError()

    def process_solution(self):

        # Obtain the solution values
        solution = self.get_solution()

        # Update setpoints
        self.setpoints = self.update_setpoints(solution)

        # Update the agent
        self.agent.setpoints = self.setpoints

        return self.agent

    def update_setpoints(self, solution: dict):
        # Get relevant column names
        src_cols = self.__get_src_cols(solution)

        # Change the solution so that the in and out columns are computed as one
        adjusted_solution = {}
        for col in src_cols:
            # Check if the column is an in or out column
            if col.endswith(f'_{c.PF_IN}'):
                key = col.rsplit('_', 1)[0]
                val = (np.array(solution[col].round()) + np.array(solution[key + f'_{c.PF_OUT}'].round())).astype(
                    int)
            # Skip the out columns as they are already included in the in columns
            elif col.endswith(f'_{c.PF_OUT}'):
                continue
            else:
                key = col
                val = np.array(solution[col].round()).astype(int)

            # Add value to the adjusted solution
            adjusted_solution[key] = val

        # Make adjusted_solution into DataFrame
        adjusted_solution = pl.DataFrame(adjusted_solution)
        # Add timestamp column from setpoints
        adjusted_solution = adjusted_solution.hstack(self.setpoints.select(c.TC_TIMESTAMP).slice(1))

        # Update setpoints
        self.setpoints = self.setpoints.update(adjusted_solution, on=c.TC_TIMESTAMP)

        # with pl.Config(set_tbl_width_chars=400, set_tbl_cols=25, set_tbl_rows=100):
        #     print(self.model.objective)
        #     print(self.model.solution.to_pandas().to_string())
        #     print(self.setpoints)

        return self.setpoints

    def __get_src_cols(self, solution):
        # Get the relevant columns based on market names, plant names, plants and energy types
        src_cols = []
        for idx, sol_var in enumerate(solution.keys()):
            sol = sol_var
            # Check if sol starts with a market name
            if sol.startswith(tuple(self.market_objects.keys())):
                # Get the plant it starts with
                market = next((market for market in self.market_objects if market in sol))
                # Subtract the market from sol
                sol = sol.replace(f'{market}_', '')
                # Check if sol starts with an energy type
                if sol.startswith(tuple(self.energy_types)):
                    # Set src_cols to one assuming it contains a relevant column
                    src_cols.append(sol_var)
            # Check if sol starts with a plant name
            if sol.startswith(tuple(self.plant_objects.keys())):
                # Subtract the name from sol
                sol = sol.split('_', 1)[1]
                # Check if sol starts with an available plant type
                if sol.startswith(tuple(self.available_plants)):
                    # Subtract the plant from sol
                    sol = sol.split('_', 1)[1]
                    # Check if sol starts with an energy type
                    if sol.startswith(tuple(self.energy_types)):
                        # Set src_cols to one assuming it contains a relevant column
                        src_cols.append(sol_var)

        return src_cols

    def __get_src_cols2(self, solution):
        # Note: Twice as fast but currently not deployed as it does not allow underscores in market or plant names
        # Pre-compute tuples
        market_keys = tuple(self.market_objects.keys())
        plant_keys = tuple(self.plant_objects.keys())
        energy_types = tuple(self.energy_types)
        available_plants = tuple(self.available_plants)

        src_cols = []
        for sol_var in solution.keys():
            sol = sol_var

            # Market name check
            if sol.startswith(market_keys):
                market, _, remainder = sol.partition('_')
                if remainder.startswith(energy_types):
                    src_cols.append(sol_var)
                    continue  # Skip to next iteration

            # Plant name check
            if sol.startswith(plant_keys):
                plant, remainder = sol.split('_', 1)
                if remainder.startswith(available_plants):
                    _, energy_type = remainder.split('_', 1)
                    if energy_type.startswith(energy_types):
                        src_cols.append(sol_var)

        return src_cols

    def __get_src_cols3(self, solution):
        # Note: Probably fastest but does also not allow underscores in market or plant names
        # Convert the keys to sets for faster operations
        solution_keys = set(solution.keys())
        market_keys = set(self.market_objects.keys())
        plant_keys = set(self.plant_objects.keys())
        energy_types = set(self.energy_types)
        available_plants = set(self.available_plants)

        # Filter the intersections based on energy types
        market_intersection = {key for key in solution_keys
                               if (any(market in key for market in market_keys)
                                   and any(energy in key for energy in energy_types))}

        plant_intersection = {key for key in solution_keys
                              if (any(plant in key for plant in plant_keys)
                                  and any(energy in key for energy in energy_types)
                                  and any(plant in key for plant in available_plants))}

        # Combine the intersections
        src_cols = list(market_intersection | plant_intersection)

        return src_cols

    def get_market_class(self):
        raise NotImplementedError

    def get_solution(self):
        raise NotImplementedError
