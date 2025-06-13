import os
import pandas as pd
import pandapower as pp
import pandapower.plotting.plotly as pplotly
import hamlet.constants as c
from hamlet.analyzer.data_processor_base import DataProcessorBase


class GridDataProcessor(DataProcessorBase):
    def __init__(self, path: dict, config: dict):
        super().__init__(path=path, config=config, name_subdirectory='grids')

        # Initialize specific grid paths for active grids in the configuration.
        self.specific_grid_path = {}

        for scenario_name, grid_config in config['grids'].items():
            for grid_type, grid_details in grid_config.items():
                # Ensure the grid type exists in the specific path dictionary
                if grid_type not in self.specific_grid_path:
                    self.specific_grid_path[grid_type] = {}

                # Add the grid path if the grid type is active
                if grid_details['active']:
                    self.specific_grid_path[grid_type][scenario_name] = os.path.join(path[scenario_name], 'grids',
                                                                                     grid_type)

    def process_electricity_transformer_loading(self):
        """
        Calculate directional (feed-in/export) transformer loading over time for all scenarios.

        Returns:
            dict: A dictionary with scenario names as keys and DataFrames as values. Each DataFrame contains:
                  - `loading_percent_pos`: Positive (feed-in) transformer loading as a percentage.
                  - `loading_percent_neg`: Negative (export) transformer loading as a percentage.
                  - `max_cap_pos`: Maximum positive capacity, set to 100.
                  - `max_cap_neg`: Maximum negative capacity, set to -100.
                  DataFrames are indexed by timestamps.
        """
        results_summary = {}

        for scenario_name, scenario_path in self.specific_grid_path['electricity'].items():
            # Load and process ext_grid data
            ext_grid_path = os.path.join(scenario_path, 'res_ext_grid.csv')
            res_ext_grid = pd.read_csv(ext_grid_path, index_col=1).rename(columns={'Unnamed: 0': 'ext_grid'})[
                ['ext_grid', 'p_mw']]
            ext_grid_data = []

            for ext_grid in res_ext_grid['ext_grid'].unique():
                grid_data = res_ext_grid[res_ext_grid['ext_grid'] == ext_grid]
                ext_grid_data.append(grid_data.rename(columns={'p_mw': ext_grid})[ext_grid] * 1000)

            ext_grid_df = pd.concat(ext_grid_data, axis=1)

            # Create binary masks for positive and negative loadings
            binary_ext = ext_grid_df.iloc[:, 0].copy()
            binary_ext[binary_ext > 0] = 1
            binary_ext[binary_ext <= 0] = -1

            # Load and process transformer data
            trafo_path = os.path.join(scenario_path, 'res_trafo.csv')
            res_trafo = pd.read_csv(trafo_path, index_col=1).rename(columns={'Unnamed: 0': 'trafo'})[
                ['trafo', 'loading_percent']]
            res_trafo['loading_percent'] *= binary_ext

            # Drop unused columns and ensure unique timestamps
            res_trafo = res_trafo.drop(columns=['trafo'])
            res_trafo.index = pd.DatetimeIndex(res_trafo.index)
            res_trafo = res_trafo[~res_trafo.index.duplicated(keep='last')]

            # Add total loading for the scenario
            results_summary[scenario_name] = res_trafo.rename(columns={'loading_percent': scenario_name})

        return results_summary

    def process_electricity_grid_topology(self):
        """
        Process grid topology and extract detailed information for buses and lines.

        Returns:
            dict: A dictionary with scenario names as keys and Pandapower network objects (ppnet) as values.
                  Each network object includes additional metadata for buses and lines.
        """
        results_summary = {}

        for scenario_name, scenario_path in self.specific_grid_path['electricity'].items():
            # Load grid data into a Pandapower network object
            ppnet = pp.from_excel(os.path.join(scenario_path, 'electricity.xlsx'))

            # Initialize geodata for visualization
            ppnet.line_geodata = pd.DataFrame(columns=ppnet.line_geodata.columns)
            ppnet.bus_geodata = pd.DataFrame(columns=ppnet.bus_geodata.columns)
            ppnet = pp.plotting.create_generic_coordinates(ppnet, library='igraph')

            # Adjust specific bus positions for visual clarity
            ppnet.bus_geodata.loc[ppnet.ext_grid['bus'].iloc[0].item(), 'x'] -= 5
            ppnet.bus_geodata.loc[ppnet.trafo['lv_bus'].iloc[0].item(), 'x'] -= 5

            # Load bus result data
            res_bus = pd.read_csv(os.path.join(scenario_path, 'res_bus.csv')).rename(
                columns={'Unnamed: 0': 'bus'}).drop_duplicates(
                subset=['bus', c.TC_TIMESTAMP], keep='last')

            # Initialize metadata columns for buses
            ppnet.bus['agent_description'] = 'no agents at bus'
            ppnet.bus['plant_description'] = 'no plants at bus'
            ppnet.bus['power_description'] = 'no power at bus'

            for bus_index in ppnet.bus.index:
                # Calculate total power at the bus
                power_at_bus = res_bus[res_bus['bus'] == bus_index]['p_mw'].abs().sum()
                ppnet.bus.loc[bus_index, 'power_description'] = power_at_bus * 1000

                # Skip if no loads or generators are connected
                load_at_bus = ppnet.load[ppnet.load['bus'] == bus_index]
                sgen_at_bus = ppnet.sgen[ppnet.sgen['bus'] == bus_index]
                if load_at_bus.empty and sgen_at_bus.empty:
                    continue

                # Extract agent information
                agents = pd.concat([
                    load_at_bus[['agent_type', c.TC_ID_AGENT]],
                    sgen_at_bus[['agent_type', c.TC_ID_AGENT]]
                ])
                agents_unique = agents.drop_duplicates(keep='first')
                agents_unique['full_text'] = agents_unique[c.TC_ID_AGENT] + ' (' + agents_unique['agent_type'] + ')'
                ppnet.bus.loc[bus_index, 'agent_description'] = ', '.join(agents_unique['full_text'].tolist())

                # Count and describe plants connected to the bus
                plants_str = []
                for load_type in load_at_bus['load_type'].unique():
                    plants_str.append(f"{load_type} ({len(load_at_bus[load_at_bus['load_type'] == load_type])})")

                for plant_type in sgen_at_bus['plant_type'].unique():
                    plants_str.append(f"{plant_type} ({len(sgen_at_bus[sgen_at_bus['plant_type'] == plant_type])})")

                ppnet.bus.loc[bus_index, 'plant_description'] = ', '.join(plants_str)

            # Reset transformer high-voltage bus power to zero
            ppnet.bus.loc[ppnet.trafo['hv_bus'].item(), 'power_description'] = 0

            # Add metadata for lines
            ppnet.line['average_loading_percent'] = 0
            ppnet.line['overload_count'] = 0

            # Store processed network
            results_summary[scenario_name] = ppnet

        return results_summary
