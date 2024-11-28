__author__ = "MarkusDoepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

# This file is in charge of handling the grids in the execution of the scenario

# Imports
import math
import polars as pl
from copy import deepcopy
import pandapower as pp
import hamlet.constants as c
from hamlet.executor.grids.grid_base import GridBase
from hamlet.executor.utilities.database.database import Database
from hamlet.executor.utilities.database.grid_db import ElectricityGridDB
from hamlet.executor.utilities.grid_restrictions.grid_regulator import GridRegulator


class Electricity(GridBase):

    def __init__(self, grid_db: ElectricityGridDB, tasks: pl.DataFrame, database: Database):

        # Call the super class
        super().__init__(grid_db=grid_db, tasks=tasks, database=database)

        # Grid object
        self.grid = deepcopy(grid_db.grid)

        # Current timestamp
        self.timestamp = self.tasks.select(c.TC_TIMESTAMP).sample(n=1).item()

        # Calculation method
        self.method = self.database.get_general_data()[c.K_GRID][c.K_GRID][c.G_ELECTRICITY]['powerflow']

        # Grid restrictions
        self.restrictions = self.database.get_general_data()[c.K_GRID][c.K_GRID][c.G_ELECTRICITY]['restrictions'][
            'apply']

    def execute(self):
        """Executes the grid."""
        # Boolean variable presents if grid status is ok (no overload)
        grid_ok = True

        # Convert trade data to grid parameters
        self.__write_grid_parameters()

        # Calculate the power flows
        self.__calculate_powerflow()

        # Execute grid restrictions
        for restriction in self.restrictions:
            kwargs = self.database.get_general_data()[c.K_GRID][c.K_GRID][c.G_ELECTRICITY]['restrictions'][restriction]
            self.grid_db, grid_ok = (GridRegulator(grid_db=self.grid_db, grid=self.grid, tasks=self.tasks,
                                                   restriction_type=restriction, database=self.database, **kwargs)
                                     .execute())

        # Generate new grid db object with results
        self.__write_result_to_grid_db()

        return self.grid_db, grid_ok

    def __write_grid_parameters(self):
        """Write grid parameters (active/reactive power of loads and generations) to grid object."""
        # write parameters for load
        load_df = self.grid.load
        load_df.fillna({'cos_phi': 1}, inplace=True)    # if cos phi data is missing, assume the phase angle is 0

        for index in load_df.index:     # iterate through all loads
            if load_df.loc[index, c.TC_ID_AGENT] is not None and load_df.loc[index, c.TC_ID_PLANT] is not None:
                # get corresponding agent database
                region = load_df.loc[index, 'zone']
                agent_db = self.database.get_agent_data(region=region, agent_type=load_df.loc[index, 'agent_type'],
                                                        agent_id=load_df.loc[index, c.TC_ID_AGENT])
                column_name = (load_df.loc[index, c.TC_ID_PLANT] + '_' + load_df.loc[index, 'load_type'] + '_' +
                               c.ET_ELECTRICITY)    # get column name for the plant in setpoints dataframe

                # calculate active power
                p_mw = (agent_db.setpoints.filter(pl.col(c.TC_TIMESTAMP) == self.timestamp).select(column_name).item() *
                        c.WH_TO_MWH)

                # assign active power to grid
                load_df.loc[index, 'p_mw'] = - p_mw

                # calculate reactive power for ac power flow
                if self.method == 'ac':
                    # convert power factor to phase angle in radians
                    phi = math.acos(load_df.loc[index, 'cos_phi'])

                    # calculate reactive power
                    q_mvar = p_mw * math.tan(phi)

                    # assign reactive power to grid
                    load_df.loc[index, 'q_mvar'] = q_mvar

        # write parameters for sgen
        sgen_df = self.grid.sgen
        sgen_df.fillna({'cos_phi': 0}, inplace=True)    # if cos phi data is missing, assume the phase angle is 0

        for index in sgen_df.index:     # iterate through all sgens
            if sgen_df.loc[index, c.TC_ID_AGENT] is not None and sgen_df.loc[index, c.TC_ID_PLANT] is not None:
                # get corresponding agent database
                region = sgen_df.loc[index, 'zone']
                agent_db = self.database.get_agent_data(region=region, agent_type=sgen_df.loc[index, 'agent_type'],
                                                        agent_id=sgen_df.loc[index, c.TC_ID_AGENT])
                column_name = (sgen_df.loc[index, c.TC_ID_PLANT] + '_' + sgen_df.loc[index, 'plant_type'] + '_' +
                               c.ET_ELECTRICITY)    # get column name for the plant in setpoints dataframe

                # calculate active power
                p_mw = (agent_db.setpoints.filter(pl.col(c.TC_TIMESTAMP) == self.timestamp).select(column_name).item() *
                        c.WH_TO_MWH)

                # assign active power to grid
                sgen_df.loc[index, 'p_mw'] = p_mw

                # calculate reactive power for ac power flow
                if self.method == 'ac':
                    # convert power factor to phase angle in radians
                    phi = math.acos(sgen_df.loc[index, 'cos_phi'])

                    # calculate reactive power
                    q_mvar = p_mw * math.tan(phi)

                    # assign reactive power to grid
                    sgen_df.loc[index, 'q_mvar'] = q_mvar

        # write load df and sgen df back to grid
        self.grid.load = load_df
        self.grid.sgen = sgen_df

    def __calculate_powerflow(self):
        """Calculates the power flows."""

        match self.method:
            case 'ac':
                pp.runpp(self.grid)
            case 'dc':
                pp.rundcpp(self.grid)
            case 'acopf':
                pp.runopp(self.grid)
            case 'dcopf':
                pp.rundcopp(self.grid)

    def __write_result_to_grid_db(self):
        """Write result from executed grid to grid db object."""
        timestamp_str = str(self.timestamp)

        for key in self.grid_db.results.keys():
            if hasattr(self.grid, key):
                result_df = deepcopy(getattr(self.grid, key))
                result_df.insert(0, c.TC_TIMESTAMP, timestamp_str)
                self.grid_db.results[key].append(result_df)
