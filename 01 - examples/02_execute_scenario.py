from hamlet import Executor


path = "../04 - scenarios/example_single_market_test"     # relative or absolute path to the scenario folder

simulation = Executor(path, num_workers=1)

simulation.run()

