from hamlet import Creator
from hamlet import functions as f
import os
import ruamel.yaml as yaml
import pandas as pd
from tqdm import tqdm

# Set the grid
grid = 'Stadt'  # options: Land, Dorf, Vorstadt, Stadt

# TODO: If it is possible to fix the issue that the simulations starts on Tuesday and not Monday.
# Moved the day back for all scenarios. Check tomorrow if that works. --> No it does not.

# TODO: Don't forget to change the commented out parts in the setup.py file of the creator (for markets)

# Info about the scenarios: name and the typical week (1=winter, 2=transition, 3=summer)
info = {
    'names': {
        'Land': 'Countryside',
        'Dorf': 'Rural',
        'Vorstadt': 'Suburban',
        'Stadt': 'Urban',
    },
    'weeks': {
        'week1': {
            'week': 1,
            'season': 'transition'},
        'week2': {
            'week': 2,
            'season': 'winter'
        },
        'week3': {
            'week': 3,
            'season': 'summer'
        },
    },
}

# Initialize yaml
yaml = yaml.YAML()

# Source path
src = os.path.join(os.path.dirname(os.path.abspath(__file__)), grid)

# Get all scenarios from the files in the folder
scenarios = next(os.walk(src))[2]
counter = 0
# Loop through scenarios
for scenario in tqdm(scenarios, desc="Processing Scenarios", unit="scenario"):
    # raise Warning('Make sure you really want to run this ;)')

    # Create a separate scenario folder for each scenario
    scenario_name = scenario.rsplit(".", 1)[0]
    grid_name = info['names'][grid]
    season = info['weeks'][scenario_name.rsplit("_", 1)[-1]]['season']
    week = info['weeks'][scenario_name.rsplit("_", 1)[-1]]['week']
    # Replace first and last part of scenario name with grid and season
    scenario_name = f"{grid_name}_{'_'.join(scenario_name.split('_')[1:-1])}_{season}"
    scenario_folder = os.path.join(src, scenario_name)

    # Copy the config files in the config folder to the scenario folder
    f.copy_folder(os.path.join(src, "configs"), scenario_folder)

    # Change the start date and weather file in the config file
    with open(os.path.join(scenario_folder, f'config_general_{week}.yaml')) as config_file:
        config = yaml.load(config_file)
    # config['simulation']['location']['weather'] = info['weather']
    with open(os.path.join(scenario_folder, 'config_general.yaml'), 'w') as config_file:
        yaml.dump(config, config_file)

    # Delete the config_general files that are not needed
    os.remove(os.path.join(scenario_folder, 'config_general_1.yaml'))
    os.remove(os.path.join(scenario_folder, 'config_general_2.yaml'))
    os.remove(os.path.join(scenario_folder, 'config_general_3.yaml'))

    # Copy the scenario file to the scenario folder and rename it to "electricity.xlsx"
    f.copy_file(os.path.join(src, scenario), os.path.join(scenario_folder, "electricity.xlsx"))

    # Create the scenario
    simulation = Creator(path=scenario_folder)
    simulation.new_scenario_from_grids()

    #counter += 1
    #if counter == 3:
    #    exit()

