import os
import pandas as pd
import pandapower as pp
from copy import deepcopy
import hamlet.functions as f
import hamlet.constants as c
from hamlet.analyzer.data_processor_base import DataProcessorBase


class GridDataProcessor(DataProcessorBase):
    def __init__(self, path: dict, config: dict):
        super().__init__(path=path, config=config)

    def process_transformer_loading(self):
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

        for scenario_name, scenario_path in self.path.items():
            # Load and process ext_grid data
            ext_grid_path = os.path.join(scenario_path, 'electricity/res_ext_grid.csv')
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
            trafo_path = os.path.join(scenario_path, 'electricity/res_trafo.csv')
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
