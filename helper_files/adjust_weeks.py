"""Replaces week 3 with week 4. Needs to be in the folder above the scenarios of a topology."""
import os

path = './urbs_scenarios'

raise Warning('The following code will delete all files that contain the target string and rename all files that '
              'contain the old string and replace it with new string. It has already been run and should not be run.')

# Get all scenarios from the files in the folder
scenarios = next(os.walk(path))[2]

# Get all files that contain the target string
target = 'week3'
files = [file for file in scenarios if target in file]

# Delete all files that contain the target string
for file in files:
    os.remove(os.path.join(path, file))

# Rename all files that contain the old string and replace it with new string
old = 'week4'
new = 'week3'
files = [file for file in scenarios if old in file]
for file in files:
    # Get the new file name
    new_file = file.replace(old, new)
    # Rename the file
    os.rename(os.path.join(path, file), os.path.join(path, new_file))