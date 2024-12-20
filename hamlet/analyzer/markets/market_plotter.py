import matplotlib.pyplot as plt
import pandas as pd
import hamlet.constants as c
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
                figsize=(8, 6 * market_count),
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

    @PlotterBase.decorator_plot_function
    def plot_agent_balancing(self, **kwargs):
        """
        Plot agent balancing data for all scenarios.

        Description:
            Generates stacked bar charts for cost and revenue for each agent in all scenarios, with a dot plot overlay
            for total balancing.

        Args:
            **kwargs: Additional arguments to customize the plot.

        Returns:
            dict: A dictionary where keys are scenario names and values are matplotlib Figure objects.
        """

        agent_balancing_data = self.data['agent_balancing']

        num_scenarios = len(agent_balancing_data)
        fig, axes = plt.subplots(
            nrows=num_scenarios,
            ncols=1,
            figsize=(8, 6 * num_scenarios),
            layout="constrained"
        )

        # Ensure axes is iterable even for a single subplot
        axes = axes if num_scenarios > 1 else [axes]

        for ax, (scenario_name, balancing_df) in zip(axes, agent_balancing_data.items()):
            # Compute total balancing and scale values
            balancing_df['total_balancing'] = balancing_df[c.TC_PRICE_OUT] - balancing_df[c.TC_PRICE_IN]
            stacked_balancing = balancing_df['total_balancing'].unstack(level=c.TC_TYPE_TRANSACTION) / 1e7

            # Plot stacked bar chart for cost and revenue
            stacked_balancing.plot.bar(ax=ax, width=0.8, stacked=True)

            # Plot total balancing as a dotted line
            total_balancing_sum = stacked_balancing.sum(axis=1)
            total_balancing_sum.plot(ax=ax, style='.', label='Total', color='black')

            # Annotate total values above bars
            for idx, value in enumerate(total_balancing_sum):
                ax.annotate(
                    int(value),
                    (idx, value),
                    textcoords="offset points",
                    xytext=(0, 7),
                    ha='center'
                )

            # Set labels, title, grid, and rotate x-axis labels
            ax.set(
                xlabel='Agent ID',
                ylabel='Cost / Revenue [€]',
                title=scenario_name
            )
            ax.legend(loc='best')
            ax.yaxis.grid(True, linestyle='--', which='major')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center')

        # Adjust layout and show the figure
        fig.tight_layout()
        plt.show()

        return fig

    def plot_average_balancing(self, market_only=False, **kwargs):
        """
        Plot average balancing data for all scenarios. Generates line plots of average pricing data for each market
        across scenarios. Optionally filters for market-only data.

        Args:
            market_only (bool): If True, only include market data in the plot.
            **kwargs: Additional arguments to customize the plot.

        Returns:
            matplotlib.figure.Figure: A figure containing the plots for all scenarios.
        """
        average_pricing_data = self.data['average_pricing']
        num_scenarios = len(average_pricing_data)

        fig, axes = plt.subplots(
            nrows=num_scenarios,
            ncols=1,
            figsize=(8, 6 * num_scenarios),
            layout="constrained"
        )

        # Ensure axes is iterable even for a single subplot
        axes = axes if num_scenarios > 1 else [axes]

        for ax, (scenario_name, scenario_data) in zip(axes, average_pricing_data.items()):
            market_transactions_list = []

            # Aggregate data for each market in the scenario
            for market_name, market_transactions in scenario_data.items():
                market_transactions = market_transactions.unstack(level=c.TC_TYPE_TRANSACTION)

                # Filter for market-only data if specified
                if market_only:
                    market_transactions = market_transactions[c.TT_MARKET]

                # Sum transactions and prepare for plotting
                market_transactions = market_transactions.sum(axis=1)
                market_transactions.rename(market_name.replace(self.path[scenario_name] + '/', ''), inplace=True)
                market_transactions_list.append(market_transactions)

            # Combine data from all markets
            aggregated_transactions = pd.concat(market_transactions_list, axis=1)

            # Plot aggregated data
            aggregated_transactions.plot(ax=ax)

            # Set labels, title, and grid
            ax.set(
                xlabel='Timesteps',
                ylabel='Average price [€/kWh]',
                title=scenario_name
            )
            ax.legend(loc='lower left', ncol=1)
            ax.yaxis.grid(True, linestyle='--', which='major')

        # Adjust layout and show the figure
        fig.tight_layout()
        plt.show()

        return fig
