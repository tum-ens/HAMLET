import sys
sys.path.append("..")  # Add the parent directory to the Python path for execution outside of an IDE
sys.path.append("./")  # Add the current directory to the Python path for execution in VSCode
from hamlet import Analyzer

# Path to the results folder (relative or absolute)
path = {'single_market': "../05_results/example_single_market"}

# Create the analyzer object
sim = Analyzer(path)

# Plot all available plots
sim.plot_all(save_path=None)    # save path = None: not saving any fig

# Plot all available plots under one aspect
sim.agents.plot_all(save_path='../05_results/example_single_market/plots/agents')   # save_path = str, save to the given path
sim.markets.plot_all(save_path=None)
sim.grids.plot_all(save_path='../05_results/example_single_market/plots/grids')

# Plot a specific single plot
sim.markets.plot_total_balancing(save_path='../05_results/example_single_market/plots/grids')
sim.agents.plot_all_meters_data(save_path=None)
