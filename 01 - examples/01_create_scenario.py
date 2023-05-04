from hamlet import Creator

path = "../02 - config/example_small"     # relative or absolute path to the config folder

simulation = Creator(path=path)
# simulation.new_scenario_from_configs()
# simulation.new_scenario_from_grids()
simulation.new_scenario_from_files()
