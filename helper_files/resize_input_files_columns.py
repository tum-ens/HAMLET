"""File to resize the columns of the input files since some of them have the wrong units"""


import os
import pandas as pd

# Path to the files
path = "./03_input_data/agents/sfh/hp"

# Files in the folder (only the .csv files)
files = next(os.walk(path))[2]
files = [f for f in files if f.endswith(".csv")]

# Columns to rename
factors = {
    "COP100_heat": 100
}

# Rename the columns
for file in files:
    # Read the file
    df = pd.read_csv(os.path.join(path, file), index_col=0)

    # Resize the columns
    for key, val in factors.items():
        df[key] *= val

    # Save the file
    df.to_csv(os.path.join(path, file))
print(df)
exit()


