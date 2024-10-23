__author__ = "HodaHamdy"
__credits__ = "MarkusDoepfert"
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

import numpy as np
import pandas as pd
import polars as pl

import hamlet.constants as c


class FbcBase:
    def __init__(self, **kwargs):

        # Store the mapping of the components to the energy types and operation modes
        self.mapping = kwargs['mapping']
        # Identify all unique energy types
        self.energy_types = set()
        for mapping in self.mapping.values():
            self.energy_types.update(mapping.keys())

        # Get the agent and other data
        self.agent = kwargs['agent']
        self.account = self.agent.account
        self.ems = self.account[c.K_EMS]
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
        self.horizon = pd.Timedelta(seconds=self.ems[c.C_CONTROLLER][c.C_FBC]['horizon'])

        # Reduce the forecast to the horizon length
        self.forecasts = self.forecasts.filter((self.timestamp < self.forecasts[c.TC_TIMESTEP])
                                               & (self.forecasts[c.TC_TIMESTEP] < self.timestamp + self.horizon))
        # Get the timesteps and the number of timesteps in the forecast
        self.timesteps = pd.Index(self.forecasts.select(c.TC_TIMESTEP).to_pandas(), name=f'{c.TC_TIMESTEP}s')
        self.timesteps = pd.Index(range(len(self.timesteps)), name=f'{c.TC_TIMESTEP}s')
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

        # To be defined in subclasses
        self.market_objects = None
        self.plant_objects = None
        self.market_class = None

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

        return self.setpoints

    def __get_src_cols(self, solution):
        # Get the relevant columns based on market names, plant names, plants and energy types
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

############################################## To be implemented in subclasses #########################################

    def run(self):
        raise NotImplementedError()

    def get_available_plants(self):
        raise NotImplementedError

    def get_solution(self):
        raise NotImplementedError()
