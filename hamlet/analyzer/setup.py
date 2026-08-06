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

    def __init__(self, path: dict, allow_incompatible_scenario: bool = False):
        """Initializes the analyzer object.

        Args:
            path: mapping of name -> results folder.
            allow_incompatible_scenario: plot results whose scenario format this version of
                HAMLET does not read. Off by default; the executor copies the scenario into the
                results folder, so results carry the stamp of the scenario that produced them.
        """
        self.results_path = path

        self.general = {}  # general information

        self.config = {'setup': {}, 'markets': {}, 'grids': {}}    # configurations

        # Load the general information and refuse any results produced from a scenario format
        # this version does not read. Plotting is where wrong numbers become figures, so the
        # check belongs here too and not only in the executor. Every folder is checked before
        # anything else is loaded, because comparing several runs is the normal case and an
        # unreadable one should be reported up front rather than after the others have loaded
        for key, value in self.results_path.items():
            self.general[key] = f.load_file(os.path.join(value, 'general', 'general.json'))
            f.check_scenario_format(self.general[key], value, allow_incompatible_scenario)

        # Set up the analyzer before plotting
        for key, value in self.results_path.items():
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
