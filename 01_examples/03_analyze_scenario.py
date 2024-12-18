import sys
sys.path.append("..")  # Add the parent directory to the Python path for execution outside of an IDE
sys.path.append("./")  # Add the current directory to the Python path for execution in VSCode
from hamlet import Analyzer

# Path to the results folder (relative or absolute)
path = {'test': "../05_results/example_grid"}

# Create the analyzer object
sim = Analyzer(path)

# Plot the general analysis (more details in the documentation)
sim.plot(save_path='../05_results/example_grid/plots')
