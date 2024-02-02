"""Adds batteries to the users that have a pv system installed."""
import pandas as pd
import os
import numpy as np
from pprint import pprint
import warnings

# Suppress pandas warnings
warnings.filterwarnings("ignore")

# Share that has a battery
share = 0.75  # according to https://arxiv.org/ftp/arxiv/papers/2203/2203.06762.pdf

# Path to the scenarios folder
path = './Szenarien/Vorstadt'

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
    df = pd.read_excel(os.path.join(path, file), sheet_name='sgen', index_col=0)

    # Create new column that contains the power of the pv system
    target = 'plant_type:pv'
    power_col = 'power:'
    col = 'description'
    df_pv = df[df[col].str.contains(target)]
    for idx, row in df_pv.iterrows():
        if target in row[col]:
            df_pv.loc[idx, 'power_pv'] = float(row[col].split(power_col)[1].split(',')[0])
        else:
            df_pv.loc[idx, 'power_pv'] = np.nan

    # Group by bus and sum the power_pv column
    df_pv = df_pv.groupby('bus').sum()

    # Check if df_pv is empty
    if df_pv.empty:
        continue

    # Round the power_pv column to 4 decimals
    df_pv['power_pv'] = df_pv['power_pv'].round(4)

    # Reduce the number of rows by 1 - share randomly
    df_pv = df_pv.sample(frac=share, random_state=1)

    # Create a dictionary from the buses and power_pv
    pv_dict = df_pv['power_pv'].to_dict()

    # Loop through the rows to implement the batteries in the original dataframe
    for idx, row in df.iterrows():
        # Target string
        target = 'plant_type:battery'

        # Check if target string is in the description column and if the bus is in the pv_dict
        if target in row['description'] and row['bus'] in pv_dict:
            # Get the description and size the battery
            description = row['description']

            # Find the string part between power: and ,
            target = 'power:'
            bat_power = description.split(target)[1].split(',')[0]
            bat_power_new = pv_dict[row['bus']]
            # Replace the power
            description = description.replace(f'{target}{bat_power}', f'{target}{bat_power_new}')

            # Find the string part between capacity: and ,
            target = 'capacity:'
            bat_energy = description.split(target)[1].split(',')[0]
            bat_energy_new = pv_dict[row['bus']]
            # Replace the energy
            description = description.replace(f'{target}{bat_energy}', f'{target}{bat_energy_new}')

            # Replace the description
            df.loc[idx, 'description'] = description


    # Replace the dataframe with the sheet in the Excel file
    with pd.ExcelWriter(os.path.join(path, file), engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name='sgen')