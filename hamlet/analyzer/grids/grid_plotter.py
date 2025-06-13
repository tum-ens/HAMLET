import matplotlib.pyplot as plt
import os
import pandas as pd
import pandapower.plotting.plotly as pplotly
import hamlet.constants as c
from hamlet.analyzer.plotter_base import PlotterBase, decorator_plot_function


class GridPlotter(PlotterBase):
    def __init__(self, path: dict, config: dict, data_processor):
        super().__init__(path=path, config=config, data_processor=data_processor, name_subdirectory='grids')

        # Initialize specific grid paths for active grids in the configuration.
        self.specific_grid_path = {}

        for scenario_name, grid_config in config['grids'].items():
            for grid_type, grid_details in grid_config.items():
                # Ensure the grid type exists in the specific path dictionary
                if grid_type not in self.specific_grid_path:
                    self.specific_grid_path[grid_type] = {}

                # Add the grid path if the grid type is active
                if grid_details['active']:
                    self.specific_grid_path[grid_type][scenario_name] = os.path.join(path[scenario_name], grid_type)

    def plot_all(self, **kwargs):
        """Plot all results for active grids based on configuration."""
        for grid_type, path_dict in self.specific_grid_path.items():
            if path_dict:
                plot_functions = [func for func in dir(self) if callable(getattr(self, func)) and
                                  func.startswith('plot_' + grid_type)]

                # Execute all plot functions for the grid type
                for func in plot_functions:
                    getattr(self, func)(**kwargs)

    @decorator_plot_function
    def plot_electricity_transformer_loading(self, **kwargs):
        """
        Generate a plot of transformer loading percentages for all scenarios.

        Description:
            Combines transformer loading data from all scenarios and plots it on a single graph.
            Each scenario's data is plotted as a separate line, showing loading percentages over time.

        Returns:
            matplotlib.figure.Figure: The figure containing the plot of transformer loading percentages.
        """
        trafo_loading_dict = super().get_plotting_data(data_name='electricity_transformer_loading')

        # Combine transformer loading data from all scenarios
        trafo_loading = pd.concat(trafo_loading_dict.values(), axis=1)

        # Create a single plot
        fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
        trafo_loading.plot(
            ax=ax,
            xlabel='',
            ylabel='Loading Percentage [%]',
            title='Transformer Loading Percentage'
        )

        # Tighten layout and display the plot
        fig.tight_layout()
        plt.show()

        return fig

    @decorator_plot_function
    def plot_electricity_grid_topology(self, **kwargs):
        """
        Plot grid topology for all scenarios as interactive html file.

        Returns:
            dict: A dictionary with scenario names as keys and Plotly Figure objects as values.
        """
        result_figs = {}
        grid_topology = super().get_plotting_data(data_name='electricity_grid_topology')

        # Iterate through each scenario
        for scenario_name, scenario_path in self.path.items():
            ppnet = grid_topology[scenario_name]

            # Read line result data
            res_line = pd.read_csv(os.path.join(scenario_path, 'res_line.csv')).rename(
                columns={'Unnamed: 0': 'line'}).drop_duplicates(subset=['line', c.TC_TIMESTAMP], keep='last')

            # Update line metadata in the network
            for line_index in ppnet.line.index:
                line_loading = res_line[res_line['line'] == line_index]['loading_percent']
                ppnet.line.loc[line_index, 'average_loading_percent'] = line_loading.mean()
                ppnet.line.loc[line_index, 'overload_count'] = (line_loading > 100).sum()

            # Generate info functions for buses and lines
            info_bus = pd.Series(
                index=ppnet.bus.index,
                data=(
                        'Agents: ' + ppnet.bus['agent_description'] + '<br>' +
                        'Plants: ' + ppnet.bus['plant_description'] + '<br>' +
                        'Power: ' + ppnet.bus['power_description'].astype(str)
                )
            )

            info_line = pd.Series(
                index=ppnet.line.index,
                data=(
                        'ID: ' + ppnet.line.index.astype(str) + '<br>' +
                        'Average loading percent: ' + ppnet.line['average_loading_percent'].astype(str) + '<br>' +
                        'Overload count: ' + ppnet.line['overload_count'].astype(str)
                )
            )

            # Create traces for Plotly visualization
            line_trace = pplotly.create_line_trace(
                ppnet,
                width=1.7,
                cmap_vals=ppnet.line.average_loading_percent,
                cmap=True,
                cbar_title='Average loading percent (%)',
                infofunc=info_line,
                cmin=0,
                cmax=100
            )

            ext_grid_trace = pplotly.create_bus_trace(
                ppnet,
                ppnet.ext_grid.bus.values,
                patch_type="square",
                size=17
            )

            bus_trace = pplotly.create_bus_trace(
                ppnet,
                color='black',
                infofunc=info_bus,
                size=6
            )

            # Combine traces and draw the figure
            fig = pplotly.draw_traces(bus_trace + line_trace + ext_grid_trace, showlegend=False, auto_open=False)
            fig.show()

            # Store the figure in the results dictionary
            result_figs[scenario_name] = fig

        return result_figs
