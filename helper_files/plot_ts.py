"""Combines all csv files and plots the time series of the data (as sum and divided by 1000).
Needs to be in the folder above the component."""
import pandas as pd
import os
import matplotlib.pyplot as plt

# Get all files in the directory
component = 'pv'
col = 'power'  # Column to plot
path = f'./{component}'
files = os.listdir(path)

# Create a dataframe with all the data
df = pd.DataFrame()
for file in files:
    # Read the file
    df_temp = pd.read_csv(path + '/' + file, index_col=0)

    # Add the data to the dataframe
    df = pd.concat([df, df_temp], axis=1)

# Sum all the columns that are named the same
df = df.groupby(df.columns, axis="columns").sum()
df /= 1000  # Convert to kW

# Convert the index to datetime
df.index = pd.to_datetime(df.index, unit='s')

# Sort the index
df = df.sort_index().dropna()

# Plot the data
df[col].plot()
plt.ylabel(col)
plt.xlabel('Time')
plt.title(f'{component} {col}')
plt.show()
