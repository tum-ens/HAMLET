import sys
sys.path.append("..")  # Add the parent directory to the Python path for execution outside an IDEimport sys
sys.path.append("./")  # Add the current directory to the Python path for execution in VSCode
from hamlet import Executor


list_scenarios = ["individual", "prosumer", "consumer", "mixed"]
list_seasons = ["summer", "transition", "winter"]
list_group_size = ["5", "10", "15"]
list_forecasting_methods = ["naive", "perfect"]

# Path to the scenario folder (relative or absolute)


for scenario in list_scenarios:
    if scenario == "individual":
        for season in list_seasons:
            for forecasting_method in list_forecasting_methods:
                path = "../04_scenarios/scenario_pv_60_hp_60_ev_60_" + scenario + "_" + season + "_" + forecasting_method

                # Create the executor object
                sim = Executor(path, num_workers=1)

                # Run the simulation
                sim.run()

    else:
        for season in list_seasons:
            for group in list_group_size:
                for forecasting_method in list_forecasting_methods:

                    path = "../04_scenarios/scenario_pv_60_hp_60_ev_60_" + scenario + "_" + group + "_" + season + "_" + forecasting_method

                    # Create the executor object
                    sim = Executor(path, num_workers=1)

                    # Run the simulation
                    sim.run()




