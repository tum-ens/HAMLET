import pytz
from copy import deepcopy
import pandas as pd
import numpy as np
import polars as pl
import math
import networkx as nx
import pandapower as pp
from pandapower.timeseries import DFData, OutputWriter
from pandapower.control import ConstControl
from pandapower.timeseries.run_time_series import run_timeseries
from hamlet.executor.utilities.database.database import Database
from hamlet.executor.utilities.database.grid_db import ElectricityGridDB
import hamlet.constants as c


class EnWG14a:

    def __init__(self, grid_db: ElectricityGridDB, grid: pp.pandapowerNet, tasks: pl.DataFrame, database: Database, **kwargs):
        # grid database object
        self.grid_db = grid_db

        # tasks dataframe
        self.tasks = tasks

        # main database object
        self.database = database

        # Grid object
        self.grid = grid

        # Current timestamp
        self.timestamp = self.tasks.select(c.TC_TIMESTAMP).sample(n=1).item()

        # Calculation method
        self.method = self.database.get_general_data()[c.K_GRID][c.K_GRID][c.G_ELECTRICITY]['powerflow']

        # Further config defined in grid json file
        self.restriction_config = kwargs

    def execute(self) -> (ElectricityGridDB, bool):
        """Execute the §14a EnWG regulation."""
        # Boolean variable presents if grid status is ok (no overload)
        grid_ok = True

        # calculate variable grid fees if active
        if self.restriction_config['variable_grid_fees']['active'] and self.__check_if_update_variable_grid_fees():
            self.__calculate_variable_grid_fees()
        else:
            # set variable grid fees in grid db object to empty
            self.grid_db.restriction_commands['current_variable_grid_fees'] = {}

        # apply direct power control if grid constraints are violated and direct power control is active
        if self.restriction_config['direct_power_control']['active'] and self.__check_overloads():
            grid_ok = self.__calculate_direct_power_control()
        else:
            self.grid_db.restriction_commands['current_direct_power_control'] = {}

        return self.grid_db, grid_ok

    def __write_grid_parameters_for_timeseries(self):
        """Write grid parameters (controllers of loads and generations) to grid object for time series calculation."""
        # write timeseries controller for loads
        load_df = self.grid.load
        load_df.fillna({'cos_phi': 0}, inplace=True)  # if cos phi data is missing, assume the phase angle is 0

        # get unique agent ids and loop through them
        agents_id = load_df[c.TC_ID_AGENT].unique()

        for agent_id in agents_id:  # iterate through all agents
            loads_for_agent = load_df[load_df[c.TC_ID_AGENT] == agent_id]  # get the part of load df for this agent
            region = loads_for_agent['zone'].unique()[0]  # get the region where the agent is
            agent_type = loads_for_agent['agent_type'].unique()[0]  # get agent type

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
                    datasource_q = DFData(q_mvar)  # data source
                    ConstControl(self.grid, element='load', variable='q_mvar', element_index=load_index,
                                 data_source=datasource_q, profile_name=[column_name])  # controller

        # write timeseries controller for sgens
        sgen_df = self.grid.sgen
        sgen_df.fillna({'cos_phi': 0}, inplace=True)  # if cos phi data is missing, assume the phase angle is 0

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
            setpoints[setpoints.select_dtypes(include=['number']).columns] *= c.WH_TO_MWH  # / resolution

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
                    datasource_q = DFData(q_mvar)  # data source
                    ConstControl(self.grid, element='sgen', variable='q_mvar', element_index=sgen_index,
                                 data_source=datasource_q, profile_name=[column_name])  # controller

    def __check_if_update_variable_grid_fees(self) -> bool:
        """Check if variable grid fees should be updated for the current timestep."""
        # timestamp where simulation started
        start_ts = self.database.get_general_data()[c.K_GENERAL]['time']['start']

        # offset between current timestamp and start timestamp
        offset = (self.timestamp - start_ts.replace(tzinfo=pytz.UTC)).seconds

        # how often should variable grid fee be updated
        update = self.restriction_config['variable_grid_fees']['update']

        # check offset % update
        update_grid_fees = offset % update == 0

        return update_grid_fees

    def __calculate_variable_grid_fees(self):
        """
        Calculate variable grid fees for each node.

        Calculation involves three steps:
        1. Calculate combined loading percentage for each bus based on maximal value of the loading percentage of all
        components which the bus has to go through to reach the upper feeder.
        2. Take the average value of the results from step 1 and the combined loading calculated at last timestep.
        3. Use the resulting loading from step 2 and the base grid fee value to calculate variable grid fees, the
        following formular applies:
            G_{T,t} = \frac{G_{\text{base}} \times n_{\text{forecast}}}{\sum_{t} W_{T,t}} \times W_{T,t}
            Where:
                G_{\text{base}}: The price of fixed grid fee without applying variable grid fee tariffs as the baseline.
                T: Current timestep.
                t: Forecasted timestep.
                n_{\text{forecast}}: Number of total timesteps in the forecast horizon.

        Link: https://www.bundesnetzagentur.de/DE/Beschlusskammern/1_GZ/BK8-GZ/2022/2022_4-Steller/BK8-22-0010/BK8-22-0010-A_Festlegung_Download.pdf?__blob=publicationFile&v=5
        """
        # convert setpoints data to grid parameters (i.e. controllers)
        self.__write_grid_parameters_for_timeseries()

        # deepcopy grid db object to prevent overwrite
        grid = deepcopy(self.grid)

        # get data resolution and variable grid fees horizon and base grid fee
        general_config = self.database.get_general_data()[c.K_GENERAL]
        resolution = general_config['time']['timestep']
        horizon = self.restriction_config['variable_grid_fees']['horizon'] / resolution
        grid_fee_base = self.restriction_config['variable_grid_fees']['grid_fee_base'] * c.EUR_KWH_TO_EURe7_WH

        # generate a timestep column as index for variable grid fees, from second next timestep to calculation horizon
        timestep_column = pd.date_range(start=self.timestamp + pd.DateOffset(seconds=2 * resolution),
                                        periods=horizon - 2, freq=pd.DateOffset(seconds=resolution))

        # add output writer to grid for time series calculation
        ow = OutputWriter(grid, range(int(horizon)))
        ow.log_variable('res_line', 'loading_percent')
        ow.log_variable('res_trafo', 'loading_percent')
        ow.log_variable('res_trafo', 'p_lv_mw')
        ow.log_variable('res_bus', 'vm_pu')  # notice the voltage drop is currently not considered

        # run time series calculation
        try:
            run_timeseries(grid, range(int(horizon)))
        except pp.powerflow.LoadflowNotConverged:  # avoid stopping the simulation when load flow not converged
            print('Load flow not converged for timestamp: ', str(self.timestamp))
            return 0

        # get time series simulation results
        timeseries_results = ow.output
        trafo_loading_total = timeseries_results['res_trafo.loading_percent']
        line_loading_total = timeseries_results['res_line.loading_percent']
        trafo_lv_power = timeseries_results['res_trafo.p_lv_mw']

        # calculate variable grid fees and loading
        variable_grid_fees_for_bus = {}  # initialize
        combined_loading_for_bus = {}  # initialize

        # iterate all transformers and get transformer loading percentage
        for trafo in trafo_loading_total.columns:
            # get trafo loading percent
            trafo_loading = trafo_loading_total[trafo]

            # calculate combined loading for each bus according to line loading in this voltage level (under this trafo)
            shortest_path = self.__calculate_shortest_path_to_trafo(grid=grid,
                                                                    trafo_lv_bus_index=grid.trafo.loc[trafo, 'lv_bus'])

            # combine line loading for upper lines into a dataframe for each bus
            for bus in shortest_path.keys():
                line_loading = line_loading_total[shortest_path[bus]]

                # concat bus and line loading and take the max value per index
                combined_loading = pd.concat([trafo_loading, line_loading], axis=1).max(axis=1) / 100

                # delete the first two timesteps since they won't be considered in agents' controllers
                combined_loading = combined_loading.iloc[2:]

                # for over-generation, take the minimal value among the whole horizon to maximize incentive
                combined_loading[(combined_loading > 1) & (trafo_lv_power.iloc[2:][trafo] > 0)] = combined_loading.min()

                # assign index
                combined_loading.index = timestep_column

                # get previous combined loading and take mean value for stability
                if hasattr(self.grid_db, 'previous_combined_loading'):
                    previous_grid_fees = self.grid_db.previous_combined_loading[bus]  # read data from db
                    combined_loading = (combined_loading.to_frame(name='combined_loading')
                                        .merge(previous_grid_fees, how='left', left_index=True, right_index=True)
                                        .mean(axis=1))  # take average value

                # round up the combined loading to the next decimal
                combined_loading = np.ceil(combined_loading * 10) / 10

                # start calculation variable grid fees
                variable_grid_fees = pd.DataFrame(index=timestep_column, columns=[bus])  # initialize

                # the total sum of constant grid fee and the total sum of variable grid fee should be same
                # G_{T,t} = \frac{G_{\text{base}} \times n_{\text{forecast}}}{\sum_{t} W_{T,t}} \times W_{T,t}
                if combined_loading.sum().item() != 0:
                    # calculate a base grid fee
                    variable_grid_fees_base = grid_fee_base * len(combined_loading) / combined_loading.sum().item()
                    variable_grid_fees[bus] = variable_grid_fees_base

                    # multiply with combined loading percentage
                    variable_grid_fees = variable_grid_fees.mul(combined_loading, axis=0)

                    # if there's no overload but variable grid fee exceeds base grid fee, set variable grid fee to base
                    variable_grid_fees['combined_loading'] = combined_loading
                    variable_grid_fees[(variable_grid_fees['combined_loading'] < 1) &
                                       (variable_grid_fees[bus] > grid_fee_base)] = grid_fee_base
                    variable_grid_fees.drop(columns='combined_loading', inplace=True)

                    # maximal variable grid fee should not exceed 200% of base grid fee (according to §14a EnWG)
                    variable_grid_fees[variable_grid_fees > grid_fee_base * 2] = grid_fee_base * 2

                else:
                    # there are no available calculation for the next horizon (at the last timestep of the simulation)
                    variable_grid_fees[bus] = grid_fee_base

                # write bus results to dict
                variable_grid_fees.index.name = c.TC_TIMESTAMP
                combined_loading_for_bus[bus] = combined_loading.to_frame(name=bus)
                variable_grid_fees_for_bus[bus] = pl.from_pandas(variable_grid_fees, include_index=True)

        # update variable grid fees in grid db object
        self.grid_db.previous_combined_loading = combined_loading_for_bus
        self.grid_db.restriction_commands['current_variable_grid_fees'] = variable_grid_fees_for_bus

        # write result for saving to grid db
        # convert combined loading to dataframe
        results_df = pd.concat(list(combined_loading_for_bus.values()), axis=1)
        results_df.index.name = c.TC_TIMESTEP
        results_df.insert(0, c.TC_TIMESTAMP, self.timestamp)

        # add item for variable grid fees to grid results dict if not available
        # or delete results for the same timestep when e.g. simulated multiple iterations for the same timesteps
        if not 'res_variable_grid_fee' in self.grid_db.results:
            self.grid_db.results['res_variable_grid_fee'] = []
        elif str(self.grid_db.results['res_variable_grid_fee'][-1][c.TC_TIMESTAMP].unique()[0]) == str(self.timestamp):
            del self.grid_db.results['res_variable_grid_fee'][-1]

        self.grid_db.results['res_variable_grid_fee'].append(results_df)

    def __check_overloads(self) -> bool:
        """Check if grid is overloaded."""
        # check trafo overload
        trafo_overloaded = self.grid.res_trafo['loading_percent'].max() > 100

        # check line overload
        line_overloaded = self.grid.res_line['loading_percent'].max() > 100

        # check voltage pu
        # to be implemented

        return trafo_overloaded or line_overloaded

    def __calculate_direct_power_control(self) -> bool:
        """
        Calculate direct power control for each plant if active.

        Iterate through all buses. First detect if the combined loading at each bus is more than 100% (overload occurs).
        If yes, apply individual device control or control via EMS depends on the grid configuration.

        The returned boolean variable represents whether the direct power control result is empty or same as previous.
        If yes, it means no more iterations needs to be simulated for the current timestep. If no, there are new direct
        power control commands and the current timestep needs to be simulated again.

        Link: https://www.bundesnetzagentur.de/DE/Beschlusskammern/1_GZ/BK6-GZ/2022/BK6-22-300/Beschluss/BK6-22-300_Beschluss_Anlage1.pdf?__blob=publicationFile&v=1
        """
        # deepcopy grid db object to prevent overwrite
        grid = deepcopy(self.grid)

        # add key storing direct control command if not available
        if 'current_direct_power_control' not in self.grid_db.restriction_commands:
            self.grid_db.restriction_commands['current_direct_power_control'] = {}  # add key

        # save previous results for the current timestep to check if this timestamp need to be updated
        previous_direct_power_control = deepcopy(self.grid_db.restriction_commands['current_direct_power_control'])

        # check overload for trafo
        for trafo_index in grid.res_trafo.index:
            # get trafo power, positive or negative is important
            trafo_power = grid.res_trafo.loc[trafo_index, 'p_lv_mw']

            # get trafo loading percent
            trafo_overload = grid.res_trafo.loc[trafo_index, 'loading_percent']

            # get all buses under this trafo
            shortest_path = self.__calculate_shortest_path_to_trafo(grid=grid,
                                                                    trafo_lv_bus_index=grid.trafo.loc[trafo_index,
                                                                    'lv_bus'])

            # get line loading for upper lines into a dataframe for each bus
            for bus in shortest_path.keys():
                line_df = grid.res_line.loc[shortest_path[bus]]
                line_overload = line_df['loading_percent'].max()

                # compare trafo overload and line overload, take the bigger one
                combined_loading = max(line_overload, trafo_overload) / 100

                # perform direct power control when combined loading percentage presents an overload
                if combined_loading > 1:
                    match self.restriction_config['direct_power_control']['method']:
                        case 'individual':
                            self.grid_db.restriction_commands['current_direct_power_control'] = (
                                self.__individual_device_control(bus=bus, trafo_power=trafo_power,
                                                                 combined_loading=combined_loading))
                        case 'ems':
                            self.grid_db.restriction_commands['current_direct_power_control'] = (
                                self.__control_via_ems(bus=bus, trafo_power=trafo_power,
                                                       combined_loading=combined_loading))
                        case _:
                            raise ValueError('The given control method is not supported, currently only supports '
                                             '"individual" and "ems".')

        # save results to grid db object
        # generate a dataframe of the current result
        data_list = []
        for agent_id, plants in self.grid_db.restriction_commands['current_direct_power_control'].items():
            for plant_id, control_result in plants.items():
                data_list.append([self.timestamp, agent_id, plant_id, control_result])

        if data_list:   # only write results if there is any control commands
            # add item for variable grid fees to grid results dict if not available
            # or delete results for the same timestep when e.g. simulated multiple iterations for the same timesteps
            if not 'res_direct_power_control' in self.grid_db.results:
                self.grid_db.results['res_direct_power_control'] = []
            elif (str(self.grid_db.results['res_direct_power_control'][-1][c.TC_TIMESTAMP].unique()[0]) == str(
                    self.timestamp)):
                del self.grid_db.results['res_direct_power_control'][-1]

            results_df = pd.DataFrame(data_list, columns=[c.TC_TIMESTAMP, c.TC_ID_AGENT, c.TC_ID_PLANT,
                                                          'control_result'])
            self.grid_db.results['res_direct_power_control'].append(results_df)

        # compare the previous and current results to check if another iteration is needed
        grid_ok = previous_direct_power_control == self.grid_db.restriction_commands['current_direct_power_control']

        # delete current direct power control results if no more iteration is needed for the current timestep
        if grid_ok:
            self.grid_db.restriction_commands['current_direct_power_control'] = {}

        return grid_ok

    def __individual_device_control(self, bus, trafo_power, combined_loading) -> dict:
        """
        Apply individual device control to the given bus under the given transformer power and combined loading.

        First check the transformer power and the bus power. Only apply individual device control if the transformer is
        importing power (i.e. the overload is due to the over-consumption) and the bus is consuming power, or if the
        transformer is exporting power (i.e. the overload is due to the over-generation) and the bus is generating
        power. Otherwise, the individual device control is not necessary since the bus is not really contributing to the
        overload.

        For over-consumption, the power reduction command will be individually sent to the following devices at the bus
        in the given order:
            1. batteries with a power consumption (charging) > threshold (controlled first since no influence on
            household comfort)
            2. EVs with a power consumption (charging) > threshold (controlled second since charging time is not
            important as long as it's enough charged before next trip)
            3. heat pumps with a power consumption > threshold (controlled last since it directly influence the
            household heating demand)

        For over-generation, the power reduction command will be individually sent to the following devices at the bus
        in the given order for the same reason as in over-consumption:
            1. batteries with a power generation (discharging) > threshold
            2. EVs with a power generation (discharging) > threshold

        Attributes:
            bus: bus index.
            trafo_power: power at the transformer.
            combined_loading: combined loading at the bus.

        Returns:
            control_target: result of the individual device control command, key: agent id, values (dict): {plant id:
            target power}.
        """
        # deepcopy grid db object to prevent overwrite
        grid = deepcopy(self.grid)

        # deepcopy current direct power control results as target to write
        control_target = deepcopy(self.grid_db.restriction_commands['current_direct_power_control'])

        # get minimal power to be guaranteed
        threshold = self.restriction_config['direct_power_control']['threshold']

        # get total power at the bus
        total_p_at_bus = grid.res_bus.loc[bus, 'p_mw']

        # power to be reduced to bring the loading to < 100%
        p_mw_to_be_reduced = total_p_at_bus * (1 - 1 / combined_loading)

        # already reduce power by 14a
        p_mw_14a_reduced = 0

        # check if sgen or load need to be adjusted
        if trafo_power < 0 < total_p_at_bus:  # too much load, load needs to be reduced
            # count number of relevant 14a devices at bus
            load_df = deepcopy(grid.load[grid.load['bus'] == bus][(grid.load['load_type'] == c.P_EV) |
                                                                  (grid.load['load_type'] == c.P_HP)])
            load_df = load_df[load_df['p_mw'] > threshold]

            # check if battery is charging
            battery_df = deepcopy(grid.sgen[grid.sgen['bus'] == bus][grid.sgen['plant_type'] == c.P_BATTERY])
            battery_df = battery_df[battery_df['p_mw'] < - threshold]  # take only batteries which are charging

            # perform individual device control
            # priority: battery -> ev -> hp
            for battery_index in battery_df.index:  # control battery
                if p_mw_14a_reduced < p_mw_to_be_reduced:  # perform control if power to be reduced is not met yet
                    # get agent id and plant id for the battery
                    agent_id = battery_df.loc[battery_index, c.TC_ID_AGENT]
                    plant_id = battery_df.loc[battery_index, c.TC_ID_PLANT]

                    # can only reduce as much as possible
                    battery_reduction = min(p_mw_to_be_reduced - p_mw_14a_reduced,
                                            abs(battery_df.loc[battery_index, 'p_mw']) - threshold)

                    # add reduction to reduced power
                    p_mw_14a_reduced += battery_reduction

                    # write reduction to target dict
                    if agent_id not in control_target.keys():
                        control_target[agent_id] = {}

                    control_target[agent_id][plant_id] = int((battery_df.loc[battery_index, 'p_mw'] +
                                                              battery_reduction) * c.MWH_TO_WH)

            for ev_index in load_df[load_df['load_type'] == c.P_EV].index:  # control ev
                if p_mw_14a_reduced < p_mw_to_be_reduced:  # perform control if power to be reduced is not met yet
                    # get agent id and plant id for the ev
                    agent_id = load_df.loc[ev_index, c.TC_ID_AGENT]
                    plant_id = load_df.loc[ev_index, c.TC_ID_PLANT]

                    # can only reduce as much as possible
                    ev_reduction = min(p_mw_to_be_reduced - p_mw_14a_reduced, load_df.loc[ev_index, 'p_mw'] - threshold)

                    # add reduction to reduced power
                    p_mw_14a_reduced += ev_reduction

                    # write reduction to target dict
                    if agent_id not in control_target.keys():
                        control_target[agent_id] = {}

                    control_target[agent_id][plant_id] = int((ev_reduction - load_df.loc[ev_index, 'p_mw']) *
                                                             c.MWH_TO_WH)

            for hp_index in load_df[load_df['load_type'] == c.P_HP].index:  # control heat pump
                if p_mw_14a_reduced < p_mw_to_be_reduced:  # perform control if power to be reduced is not met yet
                    # get agent id and plant id for the hp
                    agent_id = load_df.loc[hp_index, c.TC_ID_AGENT]
                    plant_id = load_df.loc[hp_index, c.TC_ID_PLANT]

                    # check if hp power > 0.011, if yes apply scaling factor (according to 14a)
                    if load_df.loc[hp_index, 'p_mw'] <= 0.011:  # hp power <= 11 kW, reduce to threshold
                        hp_reduction = min(p_mw_to_be_reduced - p_mw_14a_reduced, load_df.loc[hp_index, 'p_mw'] -
                                           threshold)
                    else:  # hp power > 11 kW, reduce to hp power * scaling factor (0.4)
                        hp_reduction = min(p_mw_to_be_reduced - p_mw_14a_reduced, load_df.loc[hp_index, 'p_mw'] * 0.6)

                    # hp power can only be reduced to minimal possible power with feasibility
                    if hp_reduction > load_df.loc[hp_index, 'p_mw'] - load_df.loc[hp_index, 'hp_min_control']:
                        hp_reduction = load_df.loc[hp_index, 'p_mw'] - load_df.loc[hp_index, 'hp_min_control']

                    p_mw_14a_reduced += hp_reduction

                    # write reduction to target dict
                    if agent_id not in control_target.keys():
                        control_target[agent_id] = {}

                    control_target[agent_id][plant_id] = int((hp_reduction - load_df.loc[hp_index, 'p_mw']) *
                                                             c.MWH_TO_WH)

        elif trafo_power > 0 > total_p_at_bus:  # too much generation, generation needs to be reduced
            # count number of 14a devices at bus
            # take feed in ev
            ev_df = deepcopy(grid.load[grid.load['bus'] == bus][grid.load['load_type'] == c.P_EV])
            ev_df = ev_df[ev_df['p_mw'] < - threshold]  # check if ev is feeding in to the grid

            # take discharging battery
            sgen_df = deepcopy(grid.sgen[grid.sgen['bus'] == bus][(grid.sgen['plant_type'] == c.P_BATTERY)])
            sgen_df = sgen_df[sgen_df['p_mw'] > threshold]  # take only batteries which are discharging

            # perform individual device control
            # priority: battery -> ev
            for battery_index in sgen_df[sgen_df['plant_type'] == c.P_BATTERY].index:  # control battery
                if p_mw_14a_reduced < p_mw_to_be_reduced:
                    agent_id = sgen_df.loc[battery_index, c.TC_ID_AGENT]
                    plant_id = sgen_df.loc[battery_index, c.TC_ID_PLANT]

                    battery_reduction = min(p_mw_to_be_reduced - p_mw_14a_reduced, sgen_df.loc[battery_index, 'p_mw'] -
                                            threshold)

                    p_mw_14a_reduced += battery_reduction

                    # write reduction to target dict
                    if agent_id not in control_target.keys():
                        control_target[agent_id] = {}

                    control_target[agent_id][plant_id] = int((sgen_df.loc[battery_index, 'p_mw'] - battery_reduction)
                                                             * c.MWH_TO_WH)

            for ev_index in ev_df.index:  # control ev
                if p_mw_14a_reduced < p_mw_to_be_reduced:
                    agent_id = ev_df.loc[ev_index, c.TC_ID_AGENT]
                    plant_id = ev_df.loc[ev_index, c.TC_ID_PLANT]

                    ev_reduction = min(p_mw_to_be_reduced - p_mw_14a_reduced, abs(ev_df.loc[ev_index, 'p_mw']) -
                                       threshold)

                    p_mw_14a_reduced += ev_reduction

                    # write reduction to target dict
                    if agent_id not in control_target.keys():
                        control_target[agent_id] = {}

                    control_target[agent_id][plant_id] = int(-(ev_df.loc[ev_index, 'p_mw'] + ev_reduction) *
                                                             c.MWH_TO_WH)

        return control_target

    def __control_via_ems(self, bus, trafo_power, combined_loading) -> dict:
        """
        Apply control via EMS to the given bus under the given transformer power and combined loading.

        First check the transformer power and the bus power. Only apply control via EMS if the transformer is importing
        power (i.e. the overload is due to the over-consumption) and the bus is consuming power, or if the transformer
        is exporting power (i.e. the overload is due to the over-generation) and the bus is generating power. Otherwise,
        the control via EMS is not necessary since the bus is not really contributing to the overload, and this function
        directly returns the same results as stored previously in grid database.

        During control via EMS, the total power reduction command among all devices will be sent to the EMS by taking
        the smaller value between the power to be reduced to bring the combined loading < 100% and the maximal reducible
        power according to §14a. The EMS can freely distribute the power reduction among all devices. For
        over-consumption, batteries with a power consumption (charging) > threshold, EVs with a power consumption
        (charging) > threshold, and heat pumps with a power consumption > threshold are considered for calculating the
        maximal reducible power according to §14a. For over-generation, batteries with a power generation (discharging)
        > threshold and EVs with a power generation (discharging) > threshold are considered.

        Attributes:
            bus: bus index.
            trafo_power: power at the transformer.
            combined_loading: combined loading at the bus.

        Returns:
            control_target: result of the control via EMS command, key: agent id, values (dict): {plant id:
            target power}.
        """
        # deepcopy grid db object to prevent overwrite
        grid = deepcopy(self.grid)

        # deepcopy current direct power control results as target to write
        control_target = self.grid_db.restriction_commands['current_direct_power_control']

        # get minimal power to be guaranteed
        threshold = self.restriction_config['direct_power_control']['threshold']

        # get total power at the bus
        total_p_at_bus = grid.res_bus.loc[bus, 'p_mw']

        # initialize reducible power for each agent
        reducible_power_for_agent = {}

        # power to be reduced to bring the loading to < 100%
        p_mw_to_be_reduced = total_p_at_bus * (1 - 1 / combined_loading)

        # check if sgen or load need to be adjusted
        if trafo_power < 0 < total_p_at_bus:  # too much load, load needs to be reduced
            # count number of 14a devices at bus
            load_df = deepcopy(grid.load[grid.load['bus'] == bus][(grid.load['load_type'] == c.P_EV) |
                                                                  (grid.load['load_type'] == c.P_HP)])
            load_df = load_df[load_df['p_mw'] > threshold]

            # check if battery is charging
            battery_df = deepcopy(grid.sgen[grid.sgen['bus'] == bus][grid.sgen['plant_type'] == c.P_BATTERY])
            battery_df = battery_df[battery_df['p_mw'] < - threshold]  # take only batteries which are charging

            # get all agent who has 14a relevant plants at bus
            all_agents = pd.concat([load_df[c.TC_ID_AGENT], battery_df[c.TC_ID_AGENT]], axis=0).unique()

            for agent in all_agents:  # get reducible power for each agent
                # total reducible battery power
                battery_reduction = (abs(battery_df[battery_df[c.TC_ID_AGENT] == agent]['p_mw'].sum()) -
                                     len(battery_df[battery_df[c.TC_ID_AGENT] == agent]) * threshold)

                # total reducible ev power
                ev_reduction = (load_df[load_df[c.TC_ID_AGENT] == agent][load_df['load_type'] == c.P_EV]['p_mw'].sum() -
                                len(load_df[load_df[c.TC_ID_AGENT] == agent][load_df['load_type'] == c.P_EV]) *
                                threshold)

                # total reducible hp power
                # calculate reducible power for hp with power <= 11 kW
                hp_reduction_normal = (load_df[load_df[c.TC_ID_AGENT] == agent][(load_df['load_type'] == c.P_HP) &
                                                                                (load_df['p_mw'] <= 0.011)]['p_mw']
                                       .sum() - len(
                    load_df[load_df[c.TC_ID_AGENT] == agent][(load_df['load_type'] == c.P_HP) &
                                                             (load_df['p_mw'] <= 0.011)]) * threshold)

                # calculate reducible power for hp with power > 11 kW using a scaling factor 0.4
                hp_reduction_scalar = (load_df[load_df[c.TC_ID_AGENT] == agent][(load_df['load_type'] == c.P_HP) &
                                                                                (load_df['p_mw'] > 0.011)]['p_mw'].sum()
                                       * 0.6)

                # hp power can only be reduced to minimal possible power with feasibility
                hp_reduction_max = (
                        load_df[load_df[c.TC_ID_AGENT] == agent][load_df['load_type'] == c.P_HP]['p_mw'].sum() -
                        load_df[load_df[c.TC_ID_AGENT] == agent][load_df['load_type'] == c.P_HP]
                        ['hp_min_control'].sum())

                hp_reduction = min(hp_reduction_normal + hp_reduction_scalar, hp_reduction_max)

                # calculate total reduction
                total_reduction = battery_reduction + ev_reduction + hp_reduction
                reducible_power_for_agent[agent] = total_reduction

            power_reduction_at_bus = - min(sum(reducible_power_for_agent.values()), p_mw_to_be_reduced)

        elif trafo_power > 0 > total_p_at_bus:  # too much generation, generation needs to be reduced
            # count number of 14a devices and pv at bus
            # take feed in ev
            ev_df = deepcopy(grid.load[grid.load['bus'] == bus][grid.load['load_type'] == c.P_EV])
            ev_df = ev_df[ev_df['p_mw'] < - threshold]  # check if ev is feeding in to the grid

            # take discharging battery and pv
            sgen_df = deepcopy(grid.sgen[grid.sgen['bus'] == bus][(grid.sgen['plant_type'] == c.P_BATTERY)])
            sgen_df = sgen_df[sgen_df['p_mw'] > threshold]  # take only batteries which are discharging

            # get all agent who has 14a plants at bus
            all_agents = pd.concat([sgen_df[c.TC_ID_AGENT], ev_df[c.TC_ID_AGENT]], axis=0).unique()

            for agent in all_agents:  # get reducible power for each agent
                # total reducible battery power
                battery_reduction = (
                        sgen_df[sgen_df[c.TC_ID_AGENT] == agent][sgen_df['plant_type'] == c.P_BATTERY]['p_mw']
                        .sum() - len(sgen_df[sgen_df[c.TC_ID_AGENT] == agent][sgen_df['plant_type']
                                                                              == c.P_BATTERY]) * threshold)

                # total reducible ev power
                ev_reduction = (abs(ev_df[ev_df[c.TC_ID_AGENT] == agent]['p_mw'].sum()) -
                                len(ev_df[ev_df[c.TC_ID_AGENT] == agent]) * threshold)

                # total reducible power at bus
                total_reduction = battery_reduction + ev_reduction
                reducible_power_for_agent[agent] = total_reduction

            power_reduction_at_bus = min(sum(reducible_power_for_agent.values()), p_mw_to_be_reduced)

        else:  # not necessary to control this bus
            return control_target

        # calculate percent of each agent
        for agent in reducible_power_for_agent.keys():
            # calculate total power for the agent
            total_p_agent = int((grid.load[grid.load[c.TC_ID_AGENT] == agent]['p_mw'].sum() -
                                 grid.sgen[grid.sgen[c.TC_ID_AGENT] == agent]['p_mw'].sum()) * c.MWH_TO_WH)

            # write control result to target dict
            if agent not in control_target.keys():
                control_target[agent] = {}

            reduced_p_agent = int(reducible_power_for_agent[agent] / sum(reducible_power_for_agent.values()) *
                                  power_reduction_at_bus * c.MWH_TO_WH)  # distribute power reduction to each agent

            control_target[agent]['ems'] = total_p_agent + reduced_p_agent

        return control_target

    @staticmethod
    def __calculate_shortest_path_to_trafo(grid, trafo_lv_bus_index: int) -> dict:
        """
        Calculate the shortest path between a transformer and each bus connected under this transformer.

        Attributes:
            grid: pandapower grid object.
            trafo_lv_bus_index: index of the transformer low voltage bus.

        Returns:
            shortest_path: dictionary, keys are bus index and values are lists of line index (path).

        """
        # get which region the trafo is at
        region = grid.bus.loc[trafo_lv_bus_index, 'zone']

        # get all buses in this v-level
        buses_under_trafo_df = grid.bus[grid.bus['zone'] == region]

        # calculate the shortest path between each bus and trafo
        line_df = grid.line

        # translate grid object to networkx object to perform the shortest path algorithm
        abstract_grid = pp.topology.create_nxgraph(grid)
        shortest_path = {}
        for index in buses_under_trafo_df.index:
            path_by_line = []
            path_by_bus = nx.shortest_path(abstract_grid, index, trafo_lv_bus_index)

            # find line index between nodes in the shortest path
            for i in range(len(path_by_bus) - 1):
                try:
                    path_by_line.append(line_df[((line_df['from_bus'] == path_by_bus[i]) &
                                                 (line_df['to_bus'] == path_by_bus[i + 1])) |
                                                ((line_df['from_bus'] == path_by_bus[i + 1]) &
                                                 (line_df['to_bus'] == path_by_bus[i]))].index.item())
                    shortest_path[index] = path_by_line
                except ValueError:
                    pass

        return shortest_path
