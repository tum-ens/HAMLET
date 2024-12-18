import matplotlib.pyplot as plt
from hamlet.analyzer.plotter_base import PlotterBase


class AgentPlotter(PlotterBase):
    def __init__(self, path: dict, config: dict, data: dict):
        super().__init__(path=path, config=config, data=data)

    def plot_all_meters_data(self):
        """
        Generate area plots of meter data for all scenarios.

        Description:
            Creates an area plot for each scenario, showing power data for all plants.
            A line plot overlays the total power. Each plot is labeled with the scenario name.

        Returns:
            fig (matplotlib.figure.Figure): The figure containing the plots for all scenarios.
        """
        all_meters_data = self.data['all_meters_data']
        reference_columns = list(all_meters_data.values())[0].columns

        # Determine the number of subplots
        num_scenarios = len(all_meters_data)
        fig, axes = plt.subplots(
            nrows=num_scenarios,
            ncols=1,
            figsize=(10, 4 * num_scenarios),
            layout="constrained"
        )

        # Ensure axes is iterable, even for single subplot
        axes = axes if num_scenarios > 1 else [axes]

        # Plot data for each scenario
        for ax, (scenario_name, meters_df) in zip(axes, all_meters_data.items()):
            # Filter and plot the data
            meters_df = meters_df[reference_columns]
            meters_df.drop(columns='total_power').plot.area(ax=ax)
            meters_df['total_power'].plot.line(
                ax=ax, color='black', linewidth=2, linestyle='--', label='Total Power'
            )

            # Set axis labels, title, and legend
            ax.set(xlabel='', ylabel='Power [kW]', title=scenario_name)
            ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=1)

        fig.tight_layout()
        plt.show()

        return fig
