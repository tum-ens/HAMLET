__author__ = "jiahechu"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"


import math
import polars as pl
import numpy as np
import pandas as pd
from copy import deepcopy
import pandapower as pp
from pandapower.timeseries import DFData
from pandapower.control import ConstControl
import hamlet.constants as c
from hamlet.executor.grids.grid_base import GridBase
from hamlet.executor.utilities.database.database import Database
from hamlet.executor.utilities.database.grid_db import ElectricityGridDB
from hamlet.executor.utilities.grid_restrictions.grid_regulator import GridRegulator

# This file is in charge of handling electricity grids


class Electricity(GridBase):

    def __init__(self, grid_db: ElectricityGridDB, tasks: pl.DataFrame, database: Database):

        # Call the super class
        super().__init__(grid_db=grid_db, tasks=tasks, database=database)

        # Grid object
        self.grid = deepcopy(grid_db.grid)

        # Current timestamp
        self.timestamp = self.tasks.select(c.TC_TIMESTAMP).sample(n=1).item()

        # Calculation method
        self.method = self.database.get_general_data()[c.K_GRID][c.G_ELECTRICITY]['powerflow']

        # Grid restrictions
        self.restrictions = self.database.get_general_data()[c.K_GRID][c.K_GRID][c.G_ELECTRICITY]['restrictions'][
            'apply']

    def execute(self):
        """Executes the grid."""
        # Boolean variable presents if grid status is ok (no overload)
        grid_ok = True

        # Convert trade data to grid parameters
        self._write_grid_parameters()

        # Calculate the power flows
        self._calculate_powerflow()

        # Execute grid restrictions
        for restriction in self.restrictions:
            kwargs = self.database.get_general_data()[c.K_GRID][c.K_GRID][c.G_ELECTRICITY]['restrictions'][restriction]
            self.grid_db, grid_ok = (GridRegulator(grid_db=self.grid_db, grid=self.grid, tasks=self.tasks,
                                                   restriction_type=restriction, database=self.database, **kwargs)
                                     .execute())

        # Generate new grid db object with results
        self._write_result_to_grid_db()

        return self.grid_db, grid_ok

    def _write_grid_parameters(self, is_timeseries=False):
        """Write grid parameters for a single timestamp."""
        # Process loads
        self.__process_elements(
            df=self.grid.load.copy(),
            element_name='load',
            power_sign=-1,
            type_field='load_type',
            is_timeseries=is_timeseries
        )

        # Process sgens
        self.__process_elements(
            df=self.grid.sgen.copy(),
            element_name='sgen',
            power_sign=1,
            type_field='plant_type',
            is_timeseries=is_timeseries
        )

    def _calculate_powerflow(self):
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

    def _write_result_to_grid_db(self):
        """Write result from executed grid to grid db object."""
        timestamp_str = str(self.timestamp)

        for key in self.grid_db.results.keys():
            if hasattr(self.grid, key):
                result_df = deepcopy(getattr(self.grid, key))
                result_df.insert(0, c.TC_TIMESTAMP, timestamp_str)
                self.grid_db.results[key].append(result_df)

    def __process_elements(self, df, element_name, power_sign, type_field, is_timeseries):
        """
        Process and adjust grid elements according to agent data.

        Args:
            df (pd.DataFrame): dataframe of grid element (normally load or sgen).
            element_name (string): name of grid element (normally load or sgen).
            power_sign (-1 or 1): power sign of grid element (normally -1 for load and 1 for sgen).
            type_field (string): name of the column containing plant type (currently 'load_type' for load and '
            plant_type' for sgen).
            is_timeseries (boolean): True if grid element is timeseries, False otherwise.
        """
        # prepare_data
        df.fillna({'cos_phi': 1}, inplace=True)     # if cos phi data is missing, assume the phase angle is 0
        valid_df = df[df[c.TC_ID_AGENT].notnull() & df[c.TC_ID_PLANT].notnull()]
        grouped = valid_df.groupby(c.TC_ID_AGENT)

        # iterate through all agents and get setpoints data
        for agent_id, agent_elements in grouped:
            # get agent db object from the main database
            region = agent_elements['zone'].iloc[0]
            agent_type = agent_elements['agent_type'].iloc[0]
            agent_db = self.database.get_agent_data(region=region, agent_type=agent_type, agent_id=agent_id)

            # get setpoints data from agent db object depending on whether calculating timeseries
            setpoints = agent_db.setpoints.to_pandas()
            if not is_timeseries:
                setpoints = setpoints[setpoints[c.TC_TIMESTAMP] == self.timestamp]

            # adjust numeric columns for units
            numeric_cols = setpoints.select_dtypes(include='number').columns
            setpoints[numeric_cols] *= power_sign * c.WH_TO_MWH

            # rename corresponding columns
            agent_elements['column_name'] = (
                    agent_elements[c.TC_ID_PLANT] + '_' + agent_elements[type_field] + '_' + c.ET_ELECTRICITY
            )
            columns = agent_elements['column_name'].tolist()

            # update grid data
            if is_timeseries:
                self.__add_controller_to_grid(agent_elements, setpoints, element_name)
            else:
                p_mw_values, q_mvar_values = self.__calculate_power(setpoints, columns, agent_elements)
                df.loc[agent_elements.index, 'p_mw'] = p_mw_values.values
                if q_mvar_values is not None:
                    df.loc[agent_elements.index, 'q_mvar'] = q_mvar_values

        # write result df back to the grid if not timeseries
        if not is_timeseries:
            setattr(self.grid, element_name, df)

    def __calculate_power(self, setpoints, columns, agent_elements):
        """
        Calculate active and reactive power from setpoints and columns.

        Args:
            setpoints (DataFrame): dataframe of setpoint.
            columns (list): list of column names for plants.
            agent_elements (DataFrame): dataframe of grid elements belonging to the agent.
        """

        p_mw_values = setpoints.loc[:, columns]
        if self.method == 'ac':
            phi = np.arccos(agent_elements['cos_phi'])
            q_mvar_values = p_mw_values.values * np.tan(phi)
            return p_mw_values.T, q_mvar_values.T
        else:
            return p_mw_values.T, None

    def __add_controller_to_grid(self, agent_elements, setpoints, element_name):
        """
        Add controller to the grid for timeseries calculation.

        Args:
            agent_elements (DataFrame): dataframe of grid elements belonging to the agent.
            setpoints (DataFrame): dataframe of setpoint.
            element_name (string): name of grid element (normally load or sgen).
        """
        datasource = DFData(setpoints)
        for idx, row in agent_elements.iterrows():
            column_name = row['column_name']
            # Add controller for p_mw
            ConstControl(self.grid, element=element_name, variable='p_mw', element_index=idx,
                         data_source=datasource, profile_name=[column_name])

            if self.method == 'ac':
                phi = math.acos(row['cos_phi'])
                q_mvar_series = setpoints[column_name] * np.tan(phi)
                # Assign reactive power controller
                datasource_q = DFData(pd.DataFrame({column_name: q_mvar_series}))
                ConstControl(self.grid, element=element_name, variable='q_mvar', element_index=idx,
                             data_source=datasource_q, profile_name=[column_name])

