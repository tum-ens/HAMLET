"""Sizes the heat pumps for every user in the simulation based on the heat demand of the user with a factor.
File needs to be in the folder above the topology folder."""

import pandas as pd
import os
import warnings
from pprint import pprint

# Suppress pandas warnings
warnings.filterwarnings("ignore")

raise Warning('Cancelled. Instead of adjusting the heat pumps, the heat storage will be increased in size.')

# Path to the scenarios folder
path = './Dorf'

# Factor the heat pumps are sized with (0.5 = peak * 0.5)
factor = 0.5  # represents COP to some extent

# Get a list of all scenario files
files = next(os.walk(path))[2]

# Keep only the files that contain the target string (to get _timestamps out)
target = 'hp_1.0' # TODO: Change back to pv_
files = [file for file in files if target in file]

# Loop through files
for file in files:
    # Get the week of the file name
    week = file.rsplit('_', 1)[-1].split('.')[0]

    # Load the Excel file as a pandas dataframe
    df = pd.read_excel(os.path.join(path, file), sheet_name='load')

    # Create a new dataframe that contains only the heat demands
    target = 'load_type:heat'
    col = 'description'
    df_heat = df[df[col].str.contains(target)]

    # Create a new column that contains the file name that is in the description after file_add:
    target = 'file_add:'
    col = 'description'
    df_heat[f'file'] = df_heat[col].str.split(target).str[1]
    df_heat[f'file'] = df_heat[f'file'].str.split(',').str[0]

    # Create a dictionary with the keys being all the unique values of the bus column and the values being 0
    max_user = dict.fromkeys(df_heat['bus'].unique(), 0)

    # Loop through the file row and get the peak power for each file for each user
    for idx, row in df_heat.iterrows():
        # Get the file name and user
        csv_file = row['file']
        user = row['bus']

        # Load file as pandas dataframe
        df_file = pd.read_csv(os.path.join(path, 'heat', csv_file))

        # Compute the peak power
        peak_power = df_file['heat'].max()

        # Add the peak power to the dictionary if it is bigger than the current value
        max_user[user] = max(max_user[user], peak_power)

    # Loop through the user dictionary and size the heat pumps
    hp_user = {}
    for user, peak_power in max_user.items():
        # Size the heat pump
        hp_user[user] = round(peak_power * factor / 1e6, 4)  # MW

    # Replace the current hp size with the new size
    for idx, row in df.iterrows():
        # Target string
        target = 'load_type:hp'

        # Check if target string is in the description column
        if target in row['description']:
            # Get the user
            user = row['bus']

            # Get the description and replace the hp size
            description = row['description']

            # Find the string part between power: and ,
            target = 'power:'
            hp_size = description.split(target)[1].split(',')[0]

            # If hp_size is smaller than new hp size, replace it
            if float(hp_size) < hp_user[user]:
                # Replace the hp size
                description = description.replace(f'{target}{hp_size}', f'{target}{hp_user[user]}')
                description = description.replace(f'demand:{hp_size}', f'demand:{hp_user[user]}')

                # Replace the description
                df.loc[idx, 'description'] = description

    # Replace the dataframe with the sheet in the Excel file
    with pd.ExcelWriter(os.path.join(path, file), engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name='load', index=False)
    print(f'{file} done')
    exit()

