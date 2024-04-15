import sys
sys.path.append("..")  # Add the parent directory to the Python path for execution outside of an IDE
sys.path.append("./")  # Add the current directory to the Python path for execution in VSCode
from hamlet import Creator

# Path to the scenario folder (relative or absolute)
list_scenarios = ["individual", "prosumer", "consumer", "mixed"]
list_seasons = ["summer", "transition", "winter"]
list_group_size = ["5", "10", "15"]
list_forecasting_methods = ["naive", "perfect"]


for scenario in list_scenarios:
    if scenario == "individual":
        for season in list_seasons:
            for forecasting_method in list_forecasting_methods:
                path = "../02_config/scenario_pv_60_hp_60_ev_60_" + scenario + "_" + season + "_" + forecasting_method

                # Create the creator object
                sim = Creator(path=path)

                # Create the scenario
                sim.new_scenario_from_grids()

                # Alternative methods to create the scenario:
                # sim.new_scenario_from_grids()
                # sim.new_scenario_from_files()

    else:
        for season in list_seasons:
            for group in list_group_size:
                for forecasting_method in list_forecasting_methods:

                    path = "../02_config/scenario_pv_60_hp_60_ev_60_" + scenario + "_" + group + "_" + season + "_" + forecasting_method

                    # Create the creator object
                    sim = Creator(path=path)

                    # Create the scenario
                    sim.new_scenario_from_grids()

                    # Alternative methods to create the scenario:
                    # sim.new_scenario_from_grids()
                    # sim.new_scenario_from_files()
