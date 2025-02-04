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

            # Process all agents within the scenario
            for agent_type in f.get_all_subdirectories(scenario_path):
                for agent in f.get_all_subdirectories(os.path.join(scenario_path, agent_type)):
                    # Read meter data for the agent
                    meters = pd.read_feather(os.path.join(scenario_path, agent_type, agent, 'meters.ft'))
                    timestamps = meters.pop(c.TC_TIMESTAMP)  # Extract timestamps and remove the column

                    max_index = meters.abs().idxmax().max()  # get the index of maximum
                    meters = meters.loc[:max_index, :]
                    timestamps = timestamps.loc[:max_index:]

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

                # Add total power and finalize DataFrame
                summarized_data[energy_type].fillna(0, inplace=True)
                summarized_data[energy_type]['total'] = summarized_data[energy_type].sum(axis=1)
                summarized_data[energy_type].index = pd.DatetimeIndex(timestamps)
                summarized_data[energy_type] /= 1000  # Convert to kilowatts

            # Store summarized data for the scenario
            results_summary[scenario_name] = summarized_data

        return results_summary
