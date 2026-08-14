import os
import pandas as pd
from copy import deepcopy
import hamlet.functions as f
import hamlet.constants as c
from hamlet.analyzer.data_processor_base import DataProcessorBase


class AgentDataProcessor(DataProcessorBase):
    def __init__(self, path: dict, config: dict):
        super().__init__(path=path, config=config, name_subdirectory='agents')

    def process_all_meters_data(self):
        """
        Summarize meter data at each plant for all agents across scenarios.

        Returns:
            dict: A dictionary with scenario names as keys and summarized meter data as Pandas DataFrames:
                  Each DataFrame contains:
                  - Plant-specific power data (e.g., 'hp_heat', 'hp_power', etc.).
                  - Charging and discharging splits for bi-directional meters.
                  - A 'total_power' column for the sum of all meter data.
                  The DataFrame is indexed by timestamps.
        """
        results_summary = {}

        # Iterate through each scenario
        for scenario_name, scenario_path in self.path.items():
            scenario_meters = {}
            timestamps = None
            last_recorded_row = -1

            # Process all agents within the scenario
            for agent_type in f.get_all_subdirectories(scenario_path):
                for agent in f.get_all_subdirectories(os.path.join(scenario_path, agent_type)):
                    # Read meter data for the agent
                    meters = pd.read_feather(os.path.join(scenario_path, agent_type, agent, 'meters.ft'))
                    agent_timestamps = meters.pop(c.TC_TIMESTAMP)  # Extract timestamps and remove the column

                    # One time axis for the whole scenario, taken from the first agent and checked
                    # against every other. It used to be reassigned per agent and read after the
                    # loop, so the index of every plot came from whichever agent the filesystem
                    # yielded last -- that is `os.listdir` order (see !202).
                    if timestamps is None:
                        timestamps = agent_timestamps
                    elif not agent_timestamps.equals(timestamps):
                        raise ValueError(
                            f"agent '{agent}' of scenario '{scenario_name}' carries different meter "
                            f"timestamps from the agents read before it, so there is no single time "
                            f"axis to plot them on")

                    # The meters table is allocated for the whole forecast horizon and the rows
                    # past the simulated end are never written, so they are all-zero padding and
                    # have to go -- otherwise every plot ends in a flat run of zeros, and the first
                    # padding row reads as one enormous negative flow because the meters are
                    # cumulative.
                    #
                    # Found as the last row carrying any reading, NOT as `meters.abs().idxmax()`.
                    # That returned the row at which some meter *peaked*, which is the last row
                    # only while a meter is still rising at the end: an agent whose meters have all
                    # flattened by then -- one owning only PV and a battery, after sunset -- lost
                    # the remaining timesteps silently. Every agent of every shipped scenario
                    # happens to own a continuously-rising load, which is the only reason the old
                    # rule returned the right answer.
                    recorded = meters.ne(0).any(axis=1)
                    if recorded.any():
                        agent_last_row = recorded[recorded].index.max()
                        last_recorded_row = max(last_recorded_row, agent_last_row)
                        meters = meters.loc[:agent_last_row, :]

                    # Process meter readings
                    for meter_name in meters.columns:
                        plant_key = '_'.join(meter_name.split('_')[1:])
                        energy_type = meter_name.split('_')[-1]
                        if energy_type not in scenario_meters:
                            scenario_meters[energy_type] = {}

                        if plant_key not in scenario_meters[energy_type]:
                            scenario_meters[energy_type][plant_key] = {}

                        # Store the difference in meter readings (time series)
                        scenario_meters[energy_type][plant_key][meter_name] = meters[meter_name].diff()

            if last_recorded_row < 0:
                raise ValueError(
                    f"every meter of every agent in scenario '{scenario_name}' reads zero for the "
                    f"whole run, so there is nothing to plot")

            # Combine all agent data for each meter type
            plant_data = {}
            for energy_type, energy_data in scenario_meters.items():
                plant_data[energy_type] = {plant_key: pd.concat(readings.values(), axis=1) for plant_key, readings in
                                           energy_data.items()}

            # Aggregate data for plotting
            summarized_data = {key: pd.DataFrame() for key in plant_data.keys()}
            for energy_type, energy_data in plant_data.items():
                for plant_key, readings in energy_data.items():
                    aggregated_reading = readings.sum(axis=1)

                    # Split into charging and discharging if applicable
                    if (aggregated_reading > 0).any() and (aggregated_reading < 0).any():
                        discharging = deepcopy(aggregated_reading)
                        discharging[discharging < 0] = 0
                        charging = deepcopy(aggregated_reading)
                        charging[charging > 0] = 0
                        summarized_data[energy_type][f"{plant_key}_discharging"] = discharging
                        summarized_data[energy_type][f"{plant_key}_charging"] = charging
                    else:
                        summarized_data[energy_type][plant_key] = aggregated_reading

                # Add total power and finalize DataFrame. An agent that stopped recording before
                # the scenario did contributes NaN for the rows it does not cover, which is what
                # `fillna` is for; the frame is then cut to the scenario's own last recorded row so
                # every energy type shares one length and one time axis.
                summarized_data[energy_type].fillna(0, inplace=True)
                summarized_data[energy_type]['total'] = summarized_data[energy_type].sum(axis=1)
                summarized_data[energy_type] = summarized_data[energy_type].iloc[:last_recorded_row + 1]
                summarized_data[energy_type].index = pd.DatetimeIndex(
                    timestamps.iloc[:last_recorded_row + 1])
                summarized_data[energy_type] /= 1000  # Convert to kilowatts

            # Store summarized data for the scenario
            results_summary[scenario_name] = summarized_data

        return results_summary
