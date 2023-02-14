from lama import Scenario
from time import time

start = time()
path = "../02 - config/example_small"     # relative or absolute path to the config folder

scenario = Scenario(path=path)
# scenario.new_scenario_from_configs()
scenario.new_scenario_from_grids()
print(f'Elapsed time: {time() - start:.2f} seconds')
