import matplotlib.pyplot as plt
from hamlet.analyzer.plotter_base import PlotterBase, decorator_plot_function


class AgentPlotter(PlotterBase):
    def __init__(self, path: dict, config: dict, data_processor):
        super().__init__(path=path, config=config, data_processor=data_processor, name_subdirectory='agents')

    @decorator_plot_function
    def plot_all_meters_data(self, **kwargs):
        """
        Generate area plots of meter data for all scenarios.

        Description:
            Creates an area plot for each scenario, showing power data for all plants.
            A line plot overlays the total power. Each plot is labeled with the scenario name.

        Returns:
            fig (matplotlib.figure.Figure): The figure containing the plots for all scenarios.
        """
        all_meters_data = super().get_plotting_data(data_name='all_meters_data')
        result_figs = {}

        for scenario_name, scenario_data in all_meters_data.items():
            # Determine the number of subplots
            num_energy_type = len(scenario_data)
            fig, axes = plt.subplots(
                nrows=num_energy_type,
                ncols=1,
                figsize=(10, 4 * num_energy_type),
                layout="constrained"
            )

            # Ensure axes is iterable, even for single subplot
            axes = axes if num_energy_type > 1 else [axes]

            # Plot data for each scenario
            for ax, (energy_type, meters_df) in zip(axes, scenario_data.items()):
                # Filter and plot the data
                meters_df.drop(columns='total').plot.area(ax=ax)
                meters_df['total'].plot.line(
                    ax=ax, color='black', linewidth=2, linestyle='--', label='total'
                )

                # Set axis labels, title, and legend
                ax.set(xlabel='', ylabel=f'{energy_type} [kW]', title=scenario_name)
                ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=1)

            fig.tight_layout()
            plt.show()
            result_figs[scenario_name] = fig

        return fig
