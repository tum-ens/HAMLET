import sys
sys.path.append("..")  # Add the parent directory to the Python path for execution outside an IDEimport sys
sys.path.append("./")  # Add the current directory to the Python path for execution in VSCode
from hamlet import Executor

if __name__ == "__main__":
    # Path to the scenario folder (relative or absolute)
    path = "../04_scenarios/example_single_market"

    # Create the executor object
    sim = Executor(path)

    # Run the simulation
    sim.run()
