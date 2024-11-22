__author__ = "MarkusDoepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

# This file is in charge of handling the grids in the execution of the scenario

# Imports
import os
import math
import pandas as pd
import polars as pl
import numpy as np
import time
import pytz
import logging
import traceback
import matplotlib.pyplot as plt
from datetime import datetime
from copy import deepcopy
import pandapower as pp
from pandapower.timeseries import DFData, OutputWriter
from pandapower.timeseries.run_time_series import run_timeseries
from pandapower.control import ConstControl
import networkx as nx
import hamlet.constants as c
from hamlet.executor.grids.grid_base import GridBase
from hamlet.executor.utilities.database.database import Database
from hamlet.executor.utilities.database.grid_db import ElectricityGridDB


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

    def execute(self):
        """Executes the grids"""
        # Boolean variable presents if grid status is ok (no overload)
        grid_ok = True

        # Convert trade data to grid parameters
        self.__write_grid_parameters()

        # Calculate the power flows
        self.__calculate_powerflow()

        # Generate new grid db object with results
        self.__write_result_to_grid_db()

        return self.grid_db, grid_ok

    def __write_grid_parameters(self):
        """Write grid parameters like active/reactive power of loads and generations to grid object."""
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

    def __write_grid_parameters_for_timeseries(self):
        """
        Write grid parameters like controllers of loads and generations to grid object for time series calculation.
        """
        # write timeseries controller for loads
        load_df = self.grid.load
        load_df.fillna({'cos_phi': 0}, inplace=True)    # if cos phi data is missing, assume the phase angle is 0

        # get unique agent ids and loop through them
        agents_id = load_df[c.TC_ID_AGENT].unique()

        for agent_id in agents_id:  # iterate through all agents
            loads_for_agent = load_df[load_df[c.TC_ID_AGENT] == agent_id]   # get the part of load df for this agent
            region = loads_for_agent['zone'].unique()[0]    # get the region where the agent is
            agent_type = loads_for_agent['agent_type'].unique()[0]      # get agent type

            # get agent db object
            agent_db = self.database.get_agent_data(region=region, agent_type=agent_type, agent_id=agent_id)

            # get agent setpoints as datasource for pandapower
            setpoints = agent_db.setpoints.to_pandas()
            setpoints[setpoints.select_dtypes(include=['number']).columns] *= - c.WH_TO_MWH

            # define datasource
            datasource = DFData(setpoints)

            # iterate through plants
            for load_index in loads_for_agent.index:
                column_name = (loads_for_agent.loc[load_index, c.TC_ID_PLANT] + '_' +
                               loads_for_agent.loc[load_index, 'load_type'] + '_' + c.ET_ELECTRICITY)

                # add controller
                ConstControl(self.grid, element='load', variable='p_mw', element_index=load_index,
                             data_source=datasource, profile_name=[column_name])

                # calculate reactive power for ac powerflow
                if self.method == 'ac':
                    # convert power factor to phase angle in radians
                    phi = math.acos(loads_for_agent.loc[load_index, 'cos_phi'])

                    # calculate reactive power
                    q_mvar = setpoints[column_name] * math.tan(phi)

                    # assign reactive power to grid
                    datasource_q = DFData(q_mvar)   # data source
                    ConstControl(self.grid, element='load', variable='q_mvar', element_index=load_index,
                                 data_source=datasource_q, profile_name=[column_name])      # controller

        # write timeseries controller for sgens
        sgen_df = self.grid.sgen
        sgen_df.fillna({'cos_phi': 0}, inplace=True)    # if cos phi data is missing, assume the phase angle is 0

        # get unique agent ids and loop through them
        agents_id = sgen_df[c.TC_ID_AGENT].unique()

        for agent_id in agents_id:  # iterate through all agents
            sgen_for_agent = sgen_df[sgen_df[c.TC_ID_AGENT] == agent_id]  # get the part of sgen df for this agent
            region = sgen_for_agent['zone'].unique()[0]  # get the region where the agent is
            agent_type = sgen_for_agent['agent_type'].unique()[0]  # get agent type

            # get agent db object
            agent_db = self.database.get_agent_data(region=region, agent_type=agent_type, agent_id=agent_id)

            # get agent setpoints as datasource for pandapower
            setpoints = agent_db.setpoints.to_pandas()
            setpoints[setpoints.select_dtypes(include=['number']).columns] *= c.WH_TO_MWH # / resolution

            # define datasource
            datasource = DFData(setpoints)

            # iterate through plants
            for sgen_index in sgen_for_agent.index:
                column_name = (sgen_for_agent.loc[sgen_index, c.TC_ID_PLANT] + '_' +
                               sgen_for_agent.loc[sgen_index, 'plant_type'] + '_' + c.ET_ELECTRICITY)

                # add controller
                ConstControl(self.grid, element='sgen', variable='p_mw', element_index=sgen_index,
                             data_source=datasource, profile_name=[column_name])

                # calculate reactive power for ac powerflow
                if self.method == 'ac':
                    # convert power factor to phase angle in radians
                    phi = math.acos(sgen_for_agent.loc[sgen_index, 'cos_phi'])

                    # Calculate reactive power
                    q_mvar = setpoints[column_name] * math.tan(phi)

                    # assign reactive power to grid
                    datasource_q = DFData(q_mvar)   # data source
                    ConstControl(self.grid, element='sgen', variable='q_mvar', element_index=sgen_index,
                                 data_source=datasource_q, profile_name=[column_name])      # controller

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

    def __check_overloads(self) -> bool:
        """Check if grid is overloaded."""
        # check trafo overload
        trafo_overloaded = self.grid.res_trafo['loading_percent'].max() > 100

        # check line overload
        line_overloaded = self.grid.res_line['loading_percent'].max() > 100

        # check voltage pu
        # to be implemented

        return trafo_overloaded or line_overloaded

    def __write_result_to_grid_db(self):
        """Write result from executed grid to grid db object."""
        timestamp_str = str(self.timestamp)

        for key in self.grid_db.results.keys():
            if hasattr(self.grid, key):
                result_df = deepcopy(getattr(self.grid, key))
                result_df.insert(0, c.TC_TIMESTAMP, timestamp_str)
                self.grid_db.results[key].append(result_df)
