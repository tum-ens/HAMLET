__author__ = "jiahechu"
__credits__ = ""
__license__ = ""
__maintainer__ = "TUM-Doepfert"
__email__ = "markus.doepfert@tum.de"

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

        self.config = {'setup': {}, 'markets': {}, 'grids': {}}    # configurations

        # Set up the analyzer before plotting
        for key, value in self.results_path.items():
            # load general information
            self.general[key] = f.load_file(os.path.join(value, 'general', 'general.json'))

            # load configurations
            self.config['setup'][key] = f.load_file(os.path.join(value, 'config', 'setup.yaml'))
            self.config['markets'][key] = f.load_file(os.path.join(value, 'config', 'markets.yaml'))
            self.config['grids'][key] = f.load_file(os.path.join(value, 'config', 'grids.yaml'))

        # init plotters
        self.agents = AgentPlotter(
            path=path,
            config=self.config,
            data_processor=AgentDataProcessor(path=path, config=self.config)
        )

        self.grids = GridPlotter(
            path=path,
            config=self.config,
            data_processor=GridDataProcessor(path=path, config=self.config)
        )

        self.markets = MarketPlotter(
            path=path,
            config=self.config,
            data_processor=MarketDataProcessor(path=path, config=self.config)
        )

    def plot_all(self, save_path=None, **kwargs):
        kwargs['save_path'] = save_path
        for plotter in [self.agents, self.grids, self.markets]:
            # plot results data for the current aspect
            plotter.plot_all(**kwargs)
