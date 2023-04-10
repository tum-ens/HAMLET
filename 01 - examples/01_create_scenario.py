from hamlet import Scenario
from time import time

start = time()
path = "../02 - config/example_big"     # relative or absolute path to the config folder

scenario = Scenario(path=path)
scenario.new_scenario_from_configs()
# scenario.new_scenario_from_grids()
# scenario.new_scenario_from_files()
print(f'Elapsed time: {time() - start:.2f} seconds')
