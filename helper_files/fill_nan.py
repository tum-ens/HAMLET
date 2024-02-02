"""Checks each input file for NaN values and fills them using backward fill.
Was necessary since some file had no values."""

import os
import pandas as pd

# Get all files in the folder
# path = os.path.dirname(os.path.abspath(__file__))
parent_path = './03_input_data/agents/sfh'
folders = next(os.walk(parent_path))[1]

# Loop through folders
for folder in folders:
    print(folder)
    # Get all files in the folder
    path = os.path.join(parent_path, folder)
    files = next(os.walk(path))[2]

    # Loop through files
    nan_counter = 0
    for file in files:
        # Get the file path
        file_path = os.path.join(path, file)

        # Load the file as a pandas dataframe
        df = pd.read_csv(file_path)

        # Check for NaN values
        if df.isnull().values.any():
            # Count the NaN values
            nan_counter += df.isnull().sum().sum()

            # Fill NaN values using backward and forward fill
            df = df.bfill()
            df = df.ffill()

            # Save the file
            df.to_csv(file_path, index=False)

    # Print the number of NaN values
    print(f'{folder}: {nan_counter}')