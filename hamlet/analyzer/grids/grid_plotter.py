import matplotlib.pyplot as plt
import pandas as pd
from hamlet.analyzer.plotter_base import PlotterBase


class GridPlotter(PlotterBase):
    def __init__(self, path: dict, config: dict, data: dict):
        super().__init__(path=path, config=config, data=data)

    def plot_transformer_loading(self):
        """
        Generate a plot of transformer loading percentages for all scenarios.

        Description:
            Combines transformer loading data from all scenarios and plots it on a single graph.
            Each scenario's data is plotted as a separate line, showing loading percentages over time.

        Returns:
            matplotlib.figure.Figure: The figure containing the plot of transformer loading percentages.
        """
        trafo_loading_dict = self.data['transformer_loading']

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

