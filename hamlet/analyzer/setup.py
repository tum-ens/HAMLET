__author__ = "TUM-Doepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "TUM-Doepfert"
__email__ = "markus.doepfert@tum.de"
__status__ = "Development"

import os
import hamlet.functions as f
from hamlet.analyzer.agents.agent_data_processor import AgentDataProcessor
from hamlet.analyzer.agents.agent_plotter import AgentPlotter
from hamlet.analyzer.markets.market_data_processor import MarketDataProcessor
from hamlet.analyzer.markets.market_plotter import MarketPlotter
from hamlet.analyzer.grids.grid_data_processor import GridDataProcessor
from hamlet.analyzer.grids.grid_plotter import GridPlotter
import matplotlib.pyplot as plt
plt.style.use(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'plots.mplstyle'))


class Analyzer:

    def __init__(self, path: dict):
        """Initializes the analyzer object."""
        self.results_path = path

        self.general = {}  # general information

        self.config = {}    # configurations

        self.tasks = {'agents': {'data_processor': AgentDataProcessor, 'plotter': AgentPlotter},
                      'grids': {'data_processor': GridDataProcessor, 'plotter': GridPlotter},
                      'markets': {'data_processor': MarketDataProcessor, 'plotter': MarketPlotter}}    # relevant tasks

    def setup(self):
        """Set up the analyzer before plotting."""
        for key, value in self.results_path.items():
            # load general information
            self.general[key] = f.load_file(os.path.join(value, 'general', 'general.json'))

            # load configurations
            self.config[key] = {}
            self.config[key]['setup'] = f.load_file(os.path.join(value, 'config', 'config_setup.yaml'))
            self.config[key]['markets'] = f.load_file(os.path.join(value, 'config', 'config_markets.yaml'))
            self.config[key]['grids'] = f.load_file(os.path.join(value, 'config', 'config_grid.yaml'))

    def plot(self, focus=None, save_path=None, **kwargs):
        """
        Execute the analyzer, process results, and generate plots for specified tasks.

        Args:
            focus (str, optional): A specific task key to focus on. Must be one of: 'agents', 'grids', or 'markets'.
            If None, all tasks are processed.
            save_path (str, optional): Directory path to save the generated plots. If None, plots are not saved.
            **kwargs: Additional arguments passed to the data processor and plotter.

        Description:
            This method initializes the analyzer and processes results for either all tasks or the task specified by
            `focus`. For each task, it constructs result paths, processes the data using the task's data processor, and
            generates plots using the task's plotter. If `save_path` is provided, the plots are saved as PDF files in
            the specified directory.

        Notes:
            - Valid values for `focus` are:
                - 'agents': Uses `AgentDataProcessor` and `AgentPlotter`.
                - 'grids': Uses `GridDataProcessor` and `GridPlotter`.
                - 'markets': Uses `MarketDataProcessor` and `MarketPlotter`.
            - If `focus` does not match these keys, no task will be processed.

        Returns:
            None
        """
        self.setup()    # set up the analyzer

        # iterate through all tasks
        for task_key in self.tasks.keys():
            # perform task only if no specific focus is given (None) or the given focus matches the current task
            if not focus or task_key == focus:
                # get path for the current focus
                path = {}
                for path_key, value in self.results_path.items():
                    path[path_key] = os.path.join(value, task_key)

                # process results data for the current aspect
                data = self.tasks[task_key]['data_processor'](path=path, config=self.config).process(**kwargs)

                # plot results data for the current aspect
                figures = self.tasks[task_key]['plotter'](path=path, config=self.config, data=data).plot(**kwargs)

                # save all figures if a save path is given
                if save_path:
                    # create directory if not exist
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                    # save all results
                    for name, figure in figures.items():
                        if isinstance(figure, dict):
                            for subname, subfigure in figure.items():
                                subfigure.savefig(os.path.join(save_path, subname + '_' + name + '.pdf'))
                        else:
                            figure.savefig(os.path.join(save_path, name + '.pdf'))
