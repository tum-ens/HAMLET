import sys
sys.path.append("..")  # Add the parent directory to the Python path for execution outside an IDEimport sys
sys.path.append("./")  # Add the current directory to the Python path for execution in VSCode
from hamlet import Executor

# Path to the scenario folder (relative or absolute)
path = "./04_scenarios/example_single_market"

# Create the executor object
sim = Executor(path, num_workers=1)

# Run the simulation
sim.run()
