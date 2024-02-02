"""This script is used to copy the parts that are the same for all scenarios to the scenario folders to speed up the
scenario creation. Thus only one scenario needs to have everything"""

import os
import shutil
import time

source_path = './source'  # Replace with your source path
target_path = './04 - scenarios/urban'  # Replace with your target path
suffixes = ['summer', 'transition', 'winter']
files_to_copy = [
    './general/grid.json',
    './general/retailer.ft',
    './general/timetable.ft'
]
folders_to_copy = [
    './markets',
    './retailers'
]

start = time.perf_counter()
# Iterate through the suffixes to find the matching folders
for suffix in suffixes:
    source_subfolders = [f for f in os.listdir(source_path) if f.endswith(suffix)]
    target_subfolders = [f for f in os.listdir(target_path) if f.endswith(suffix)]

    # Copy the specified files and folders
    for source_subfolder in source_subfolders:
        source_full_path = os.path.join(source_path, source_subfolder)
        for target_subfolder in target_subfolders:
            target_full_path = os.path.join(target_path, target_subfolder)

            # Copy files
            for file_rel_path in files_to_copy:
                src_file = os.path.join(source_full_path, file_rel_path)
                dest_file = os.path.join(target_full_path, file_rel_path)
                dest_folder = os.path.dirname(dest_file)
                os.makedirs(dest_folder, exist_ok=True)  # Create destination directory if it doesn't exist
                shutil.copy(src_file, dest_file)
                # print(f"Copied {src_file} to {dest_file}")

            # Copy subfolders
            for folder_rel_path in folders_to_copy:
                src_folder = os.path.join(source_full_path, folder_rel_path)
                dest_folder = os.path.join(target_full_path, folder_rel_path)
                shutil.copytree(src_folder, dest_folder, dirs_exist_ok=True)
                # print(f"Copied folder {src_folder} to {dest_folder}")

end = time.perf_counter()
print(f"Finished in {end - start} seconds")
