import pandas as pd
import sys
sys.path.append("..")  # Add the parent directory to the Python path for execution outside of an IDE
sys.path.append("./")  # Add the current directory to the Python path for execution in VSCode
from hamlet import Analyzer
from datetime import datetime

list_scenarios = ["individual", "prosumer", "consumer", "mixed"]
list_seasons = ["summer", "transition", "winter"]
list_group_size = ["5", "10", "15"]
list_forecasting_methods = ["naive", "perfect"]

# Path to the scenario folder (relative or absolute)

# datetime(year, month, day, hour, minute, second, microsecond), Summer, Transition, Winter
# Sample data (replace with your actual dataframe)

starts = {
    'timestamp': [
        '2019-07-26T00:00:00.000Z',
        '2019-09-27T00:00:00.000Z',
        '2019-11-15T00:00:00.000Z',
    ]
}

starting_date_df = pd.DataFrame(starts)

# Convert 'timestamp' column to datetime64[ns, UTC]
starting_date_df['timestamp'] = pd.to_datetime(starting_date_df['timestamp'])

for scenario in list_scenarios:
    if scenario == "individual":
        idx = 0
        for season in list_seasons:
            starting_date = starting_date_df['timestamp'][idx]
            for forecasting_method in list_forecasting_methods:
                path = "../05_results/scenario_pv_60_hp_60_ev_60_" + scenario + "_" + season + "_" + forecasting_method
                # Create the analyzer object
                sim = Analyzer(path)

                sim.plot_general_analysis(starting_date)
            idx = idx + 1

    else:
        idx = 0
        for season in list_seasons:
            starting_date = starting_date_df['timestamp'][idx]
            for group in list_group_size:
                for forecasting_method in list_forecasting_methods:

                    path = "../05_results/scenario_pv_60_hp_60_ev_60_" + scenario + "_" + group + "_" + season + "_" + forecasting_method

                    # Create the analyzer object
                    sim = Analyzer(path)

                    sim.plot_general_analysis(starting_date)

            idx = idx + 1
