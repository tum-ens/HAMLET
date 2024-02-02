"""Script goes through each of the heat urbs files and computes the heat demand and peak power for that week.
    The script then saves the results in a csv file."""

import pandas as pd
import os
import numpy as np

# Path to the scenarios folder
path = './urbs_scenarios'

# Get a list of all files
files = next(os.walk(path))[2]

# Keep only the files that contain the target string
target = 'pv_1.0_hp_1.0_ev_1.0'
files = [file for file in files if target in file]

df_info = pd.DataFrame()

# Loop through files
for file in files:
    # Get the week of the file name
    week = file.rsplit('_', 1)[-1].split('.')[0]

    # Load the Excel file as a pandas dataframe
    df = pd.read_excel(os.path.join(path, file), sheet_name='load')

    # Keep only the rows where target string is in the description column
    target = 'load_type:heat'
    col = 'description'
    df = df[df[col].str.contains(target)]

    # Create a new column that contains the file name that is in the description after file_add:
    target = 'file_add:'
    col = 'description'
    df[f'{week}_file'] = df[col].str.split(target).str[1]
    df[f'{week}_file'] = df[f'{week}_file'].str.split(',').str[0]

    # Create new dataframe with only the bus and file column
    df = df[['bus', f'{week}_file']]

    # Concat the df with the df_info
    df_info = pd.concat([df_info, df], axis=0, join='outer', ignore_index=True)
    # df_info = df_info.set_index('bus')
    #df_info = pd.concat([df_info, df], axis=0, how='inner')


    # Add columns for heat demand and peak power
    df_info[f'{week}_heat_demand'] = np.nan
    df_info[f'{week}_peak_power'] = np.nan

    # Loop through the file row and get the heat demand and peak power for each file
    for file in df[f'{week}_file']:
        # Load file as pandas dataframe
        df_file = pd.read_csv(os.path.join(path, 'heat', file))

        # Compute the heat demand and peak power
        heat_demand = round(df_file['heat'].sum() * 0.25 / 1000, 1)  # kWh
        peak_power = round(df_file['heat'].max() / 1000, 1)  # kW

        # Add the heat demand and peak power to the dataframe
        df_info.loc[df_info[f'{week}_file'] == file, f'{week}_heat_demand'] = heat_demand
        df_info.loc[df_info[f'{week}_file'] == file, f'{week}_peak_power'] = peak_power

# Get one row for each bus but keep the values of all the rows that are not nan
df_info = df_info.groupby('bus').agg(lambda x: x.dropna().tolist()).apply(pd.Series.explode).reset_index()

# Make the bus column the index
df_info = df_info.set_index('bus')

# Save the dataframe as a csv file
df_info.to_csv(os.path.join(path, f'Vorstadt_heat_info2.csv'))
