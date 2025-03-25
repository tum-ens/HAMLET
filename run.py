import sys
sys.path.append("..")  # Add the parent directory to the Python path for execution outside an IDEimport sys
sys.path.append("./")  # Add the current directory to the Python path for execution in VSCode
from hamlet import Creator, Executor, Analyzer


def create_scenario(path: str, name: str = None):
    Creator(path, name=name).new_scenario_from_configs()
    # Creator(path).new_scenario_from_grids(name=name)
    # Creator(path).new_scenario_from_files(name=name)


def run_scenario(path: str):
    Executor(path, num_workers=1).run()


def analyze_scenario(path: str):
    Analyzer(path).plot_general_analysis()


if __name__ == '__main__':
    # This is a minimal non-working example on how you would use the framework.
    # For working examples, please refer to the "examples" folder.
    path = 'path to config folder'
    create_scenario(path)
    path = 'path to scenario folder'
    run_scenario(path)
    path = 'path to results folder'
    analyze_scenario(path)

