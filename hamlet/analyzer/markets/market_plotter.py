import matplotlib.pyplot as plt
import pandas as pd
from hamlet.analyzer.plotter_base import PlotterBase


class MarketPlotter(PlotterBase):
    def __init__(self, path: dict, config: dict, data: dict):
        super().__init__(path=path, config=config, data=data, name_subdirectory='markets')

    @PlotterBase.decorator_plot_function
    def plot_total_balancing(self, **kwargs):
        """
        Generate bar plots for total balancing data across all scenarios and markets.

        Description:
            For each scenario and its associated markets, this method creates stacked bar plots
            to visualize the cost and revenue data for each transaction type. Each market's
            data is plotted on a separate subplot.

        Returns:
            dict: A dictionary where keys are scenario names, and values are matplotlib Figure objects.
                  Each figure contains the plots for all markets in the scenario.
        """
        total_balancing = self.data['total_balancing']
        result_figs = {}

        # Iterate through all scenarios
        for scenario_name, scenario_data in total_balancing.items():
            market_count = len(scenario_data)
            fig, axes = plt.subplots(
                nrows=market_count,
                ncols=1,
                figsize=(6, 4 * market_count),
                layout="constrained"
            )

            # Ensure axes is iterable, even if there is only one market
            axes = axes if market_count > 1 else [axes]

            # Plot data for each market
            for ax, (market_name, balancing_df) in zip(axes, scenario_data.items()):
                # Plot stacked bar chart for cost and revenue
                balancing_df.plot.bar(ax=ax, width=0.8, stacked=True, label=balancing_df.columns)

                # Plot total balancing as a dotted line
                total_balancing_sum = balancing_df.sum(axis=1)
                total_balancing_sum.plot(ax=ax, style='.', label='Total', color='black')

                # Annotate the total values above each bar
                for idx, value in enumerate(total_balancing_sum):
                    ax.annotate(
                        int(value),
                        (idx, value),
                        textcoords="offset points",
                        xytext=(0, 7),  # Offset annotation above the bar
                        ha='center'
                    )

                # Set labels, title, and grid
                ax.set(
                    xlabel='Transaction Type',
                    ylabel='Cost / Revenue [€]',
                    title=market_name.replace(self.path[scenario_name] + '/', '')
                )
                ax.legend(loc='lower left', ncol=1)
                ax.set_axisbelow(True)
                ax.yaxis.grid(True, linestyle='--', which='major')

            # Adjust layout and store the figure
            fig.tight_layout()
            plt.show()
            result_figs[scenario_name] = fig

        return result_figs

