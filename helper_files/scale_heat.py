"""Scales the heat files to the given factor for each topology."""

import pandas as pd
import os
import numpy as np

# Path to the scenarios folder
path = './Szenarien'

# Get a list of all topologies
topologies = next(os.walk(path))[1]

# Factors the heat files are divided by
factors = {
    'Land': 2.1,
    'Dorf': 1.9,
    'Stadt': 1.9,
    'Vorstadt': 1.9,
}

# Loop through topologies
for topology in topologies:
    print(topology)
    # Get a list of all files in the topology folder
    files = next(os.walk(os.path.join(path, topology)))[2]

    # Keep only the files that contain the target string
    target = 'pv_'
    files = [file for file in files if target in file]

    # Loop through files
    for file in files:
        # Get the week of the file name
        week = file.rsplit('_', 1)[-1].split('.')[0]

        # Load the Excel file as a pandas dataframe
        df = pd.read_excel(os.path.join(path, topology, file), sheet_name='load')

        # Keep only the rows where target string is in the description column
        target = 'load_type:heat'
        col = 'description'
        df = df[df[col].str.contains(target)]

        # Create a new column that contains the file name that is in the description after file_add:
        target = 'file_add:'
        col = 'description'
        df[f'file'] = df[col].str.split(target).str[1]
        df[f'file'] = df[f'file'].str.split(',').str[0]

        # Create new dataframe with only the bus and file column
        df = df[['bus', f'file']]

        # Loop through the file row and get the heat demand and peak power for each file
        for file in df[f'file']:
            # Load file as pandas dataframe
            df_file = pd.read_csv(os.path.join(path, topology, 'heat', file), index_col=0)

            # Scale the heat demand
            df_file['heat'] = round(df_file['heat'] / factors[topology]).astype(int)

            # Save the file
            df_file.to_csv(os.path.join(path, topology, 'heat', file))

