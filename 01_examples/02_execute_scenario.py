import sys
sys.path.append("..")  # Add the parent directory to the Python path for execution outside of an IDE
from hamlet import Executor

# Path to the scenario folder (relative or absolute)
path = "../04_scenarios/example_single_market"

# Create the executor object
sim = Executor(path)

# Run the simulation
sim.run()
