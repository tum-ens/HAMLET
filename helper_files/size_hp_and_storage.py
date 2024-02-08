"""Sizes the heat pumps for every user in the simulation based on the heat demand of the user with a factor.
File needs to be in the folder above the topology folder.
File was used to size the heat pumps for Claudio's scenarios where I doubled the size of the hps, limited the maximum
heat demand to that value and sized the storage systems to two hours of the hp size."""

import pandas as pd
import os
import warnings
from pprint import pprint
import matplotlib.pyplot as plt
from tqdm import tqdm

# Suppress pandas warnings
warnings.filterwarnings("ignore")

raise Warning('Watch out. This file changes the Excel file so it should only be run once as otherwise you start to '
              'multiply the values again and again and again. If you have to run it again, set the factors back to 1.')

# Path to the scenarios folder
file = 'electricity.xlsx'

# Load the Excel file as a pandas dataframe
df_load = pd.read_excel(file, sheet_name='load', index_col=0)
df_sgen = pd.read_excel(file, sheet_name='sgen', index_col=0)

# Create a new dataframe that contains only the heat demands
target = 'load_type:heat'
col = 'description'
df_heat = df_load[df_load[col].str.contains(target)]

# Create a new column that contains the file name that is in the description after file_add:
target = 'file_add:'
col = 'description'
df_heat[f'file'] = df_heat[col].str.split(target).str[1]
# print(df_heat.to_string())
# Create series with the file names and the bus as the index
df_heat[f'file'] = df_heat[f'file'].str.split(',').str[0]

# Create a new dataframe that contains only the hp rows
target = 'load_type:hp'
col = 'description'
df_hp = df_load[df_load[col].str.contains(target)]

# Create a new column that contains the hp power that is in the description after power:
target = 'power:'
col = 'description'
df_hp[f'power'] = df_hp[col].str.split(target).str[1]
df_hp[f'power'] = df_hp[f'power'].str.split(',').str[0]

# Resize the heat pumps
factor = 2
df_hp[f'power_new'] = round(df_hp[f'power'].astype(float) * factor, 4)
# Replace the description
df_hp['description'] = df_hp.apply(lambda row: row['description'].replace(target + str(row['power']),
                                                                          target + str(row['power_new'])),
                                   axis=1)

# Merge the dataframes to contain both the power info as well as the files that need to be changed
df_info = pd.merge(df_hp, df_heat[['bus', 'file']], how='left', on=['bus'])

# Replace the rows in df_load with df_hp
df_load.loc[df_hp.index, 'description'] = df_hp['description']

# Merge the hp new_power column with the sgen dataframe
df_sgen = pd.merge(df_sgen, df_hp[['bus', 'power_new']], how='left', on=['bus'])
# Rename column to hp_power
df_sgen.rename(columns={'power_new': 'hp_power'}, inplace=True)


# Filter the sgen dataframe to only contain the heat-storage rows
target = 'plant_type:heat-storage'
col = 'description'
df_storage = df_sgen[df_sgen[col].str.contains(target)]

# Replace the power and capacity of the heat-storage in the description with the according hp_power times the factor
target = 'power:'
col = 'description'
factor = 1.5
df_storage['power'] = df_storage[col].str.split(target).str[1]
df_storage[f'power'] = df_storage[f'power'].str.split(',').str[0]
df_storage['power_new'] = (df_storage['hp_power'].astype(float) * factor).round(4)
df_storage['description'] = df_storage.apply(lambda row: row['description'].replace(target + str(row['power']),
                                                                                    target + str(row['power_new'])),
                                             axis=1)
target = 'capacity:'
factor = 2
df_storage['capacity'] = df_storage[col].str.split(target).str[1]
df_storage[f'capacity'] = df_storage[f'capacity'].str.split(',').str[0]
df_storage['capacity_new'] = (df_storage['hp_power'].astype(float) * factor).round(4)
df_storage['description'] = df_storage.apply(lambda row: row['description'].replace(target + str(row['capacity']),
                                                                                    target + str(row['capacity_new'])),
                                             axis=1)
df_storage['power_diff'] = df_storage['power_new'] / df_storage['power'].astype(float)
df_storage['cap_diff'] = df_storage['capacity_new'] / df_storage['capacity'].astype(float)

# Replace the rows in df_sgen with df_storage
df_sgen.loc[df_storage.index, 'description'] = df_storage['description']

# Drop the hp_power column
df_sgen.drop(columns=['hp_power'], inplace=True)

# Replace the dataframe with the sheet in the Excel file
with pd.ExcelWriter(file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    df_load.to_excel(writer, sheet_name='load', index=True)
    df_sgen.to_excel(writer, sheet_name='sgen', index=True)

# Loop through the file row and get the peak power for each file for each user
for idx, row in tqdm(df_info.iterrows(), total=df_info.shape[0]):

    # Get the file name, user and power (in W)
    csv_file = row['file']
    user = row['bus']
    peak_power = int(row['power_new'] * 1e6)

    # Load file as pandas dataframe
    df_file = pd.read_csv(os.path.join('C:/Users/ge23nur/Documents/Python Scripts/HAMLET/03_input_data/agents/sfh/heat',
                                       csv_file), index_col=0)

    # Clip the peak power to peak_power
    df_file['heat'] = df_file['heat'].clip(upper=peak_power)

    # Save the file
    df_file.to_csv(os.path.join('C:/Users/ge23nur/Documents/Python Scripts/HAMLET/03_input_data/agents/sfh/heat',
                                csv_file))

