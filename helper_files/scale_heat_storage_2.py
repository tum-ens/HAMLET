"""Sizes the heat storage for every user with a factor
Note: This is done instead of increasing the heat pump size.
File needs to be in the folder above the topology folder."""

import pandas as pd
import os
import warnings
from pprint import pprint

# Suppress pandas warnings
warnings.filterwarnings("ignore")

# Path to the scenarios folder
path = './urbs_scenarios'

# Factor the heat storages are sized with
factor = 4

# Get a list of all scenario files
files = next(os.walk(path))[2]

# Keep only the files that contain the target string (to get _timestamps out)
target = 'pv_'
files = [file for file in files if target in file]

# Loop through files
for file in files:
    # Get the week of the file name
    week = file.rsplit('_', 1)[-1].split('.')[0]

    # Load the Excel file as a pandas dataframe
    df = pd.read_excel(os.path.join(path, file), sheet_name='sgen')

    # Loop through the rows
    for idx, row in df.iterrows():
        # Target string
        target = 'plant_type:heat_storage'

        # Check if target string is in the description column
        if target in row['description']:
            # Get the user
            user = row['bus']

            # Get the description and replace the hp size
            description = row['description']

            # Find the string part between capacity: and ,
            target = 'capacity:'
            hs_energy = description.split(target)[1].split(',')[0]
            hs_energy_new = float(hs_energy) * factor
            # Replace the power and energy
            description = description.replace(f'{target}{hs_energy}', f'{target}{hs_energy_new}')

            # Replace the description
            df.loc[idx, 'description'] = description

    # Replace the dataframe with the sheet in the Excel file
    with pd.ExcelWriter(os.path.join(path, file), engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name='sgen', index=False)


