"""File to rename the columns of the input files since there are new column names to be used"""

"""
F2tUOIcR2cekkYD_P_heat_heat
FJvTvVQRJ836P70_energy_consumed
"""

import os
import pandas as pd

# Path to the files
path = "./03_input_data/agents/sfh/ev"

# Files in the folder (only the .csv files)
files = next(os.walk(path))[2]
files = [f for f in files if f.endswith(".csv")]

# Columns to rename
rename = {
    "cop": "COP100_heat",
    "demand_Wh": "energy_consumed"
}

# Rename the columns
for file in files:
    # Read the file
    df = pd.read_csv(os.path.join(path, file), index_col=0)

    # Rename the columns
    df = df.rename(columns=rename)

    # Save the file
    df.to_csv(os.path.join(path, file))
print(df)
exit()


