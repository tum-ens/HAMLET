import os
import shutil
import time
import json
import pandas as pd
import polars as pl
from ruamel.yaml import YAML
from typing import Callable


# Contains all functions that are shared among the classes and used universally


def create_folder(path: str, delete: bool = True) -> None:
    """Creates a folder at the given path

    Args:
        path: path to the folder
        delete: if True, the folder will be deleted if it already exists

    Returns:
        None
    """

    # Create main folder if does not exist
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        if delete:
            shutil.rmtree(path)
            os.makedirs(path)
    time.sleep(0.01)


def copy_folder(src: str, dst: str, only_files: bool = False, delete: bool = True) -> None:
    """Copies a folder to another location

    Args:
        src: path to the copy
        dst: path to the folder
        only_files: if True, only the files will be copied and no subfolders
        delete: if True, the folder will be deleted if it already exists

    Returns:
        None
    """

    # Check if only files should be copied
    if only_files:
        # Create the destination folder if it does not exist
        os.makedirs(dst, exist_ok=True)
        # Get a list of all files in the source folder
        files = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
        # Iterate through the list of files and copy them to the destination folder
        for file in files:
            src_file = os.path.join(src, file)
            dst_file = os.path.join(dst, file)
            shutil.copy(src_file, dst_file)
    else:
        # Check if the folder exists
        if not os.path.exists(dst):
            shutil.copytree(src, dst)
        else:
            if delete:
                shutil.rmtree(dst)
                shutil.copytree(src, dst)
        time.sleep(0.01)


def load_file(path: str, index: int = 0, df: str = 'pandas', parse_dates: bool | list | None = None,
              method: str = 'lazy') -> object:
    # Find the file type
    file_type = path.rsplit('.', 1)[-1]

    # Load the file
    if file_type == 'yaml' or file_type == 'yml':
        with open(path) as file:
            file = YAML().load(file)
    elif file_type == 'json':
        with open(path) as file:
            file = json.load(file)
    elif file_type == 'csv':
        if df == 'pandas':
            file = pd.read_csv(path, parse_dates=parse_dates, index_col=index)
        elif df == 'polars':
            if method == 'lazy':
                file = pl.scan_csv(path, try_parse_dates=parse_dates)
            elif method == 'eager':
                file = pl.read_csv(path, try_parse_dates=parse_dates)
        else:
            raise ValueError(f'Dataframe type "{df}" not supported')
    elif file_type == 'xlsx':
        if df == 'pandas':
            file = pd.ExcelFile(path)
        elif df == 'polars':
            file = pl.read_excel(path)
        else:
            raise ValueError(f'Dataframe type "{df}" not supported')
    elif file_type == 'ft':
        if df == 'pandas':
            file = pd.read_feather(path)
        elif df == 'polars':
            if method == 'lazy':
                file = pl.scan_ipc(path, memory_map=False)
            elif method == 'eager':
                # Workaround for polars bug
                with pl.StringCache():
                    file = pl.read_ipc(path, memory_map=False)
        else:
            raise ValueError(f'Dataframe type "{df}" not supported')
    else:
        raise ValueError(f'File type "{file_type}" not supported')

    return file


def save_file(path: str, data, index: bool = True, df: str = 'pandas') -> None:
    # Find the file type
    file_type = path.rsplit('.', 1)[-1]

    # Save the file
    if file_type == 'yaml' or file_type == 'yml':
        with open(path, 'w') as file:
            YAML().dump(data, file)
    elif file_type == 'json':
        with open(path, 'w') as file:
            json.dump(data, file, indent=4)
    elif file_type == 'csv':
        if df == 'pandas':
            data.to_csv(path, index=index)
        elif df == 'polars':
            data.write_csv(path)
        else:
            raise ValueError(f'Dataframe type "{df}" not supported')
    elif file_type == 'xlsx':
        if df == 'pandas':
            data.to_excel(path, index=index)
        elif df == 'polars':
            data.write_excel(path)
        else:
            raise ValueError(f'Dataframe type "{df}" not supported')
    elif file_type == 'ft':
        if df == 'pandas':
            data.reset_index(inplace=True)
            data.to_feather(path)
        elif df == 'polars':
            data.write_ipc(path, compression='lz4')
    else:
        raise ValueError(f'File type "{file_type}" not supported')


def loop_folder(src: str, struct: dict, folder: str, func: Callable, **kwargs) -> dict:
    """Loads the agent data from all the scenario files and saves them in the same structure"""

    # Create an empty dictionary to store the data
    data = {}

    # Load the data from the scenario files
    for name, key in struct.items():
        # Add the agent data to the data structure
        data[name] = func(path=os.path.join(src, key, folder), **kwargs)

    return data


def add_nested_data(path: str, df: str = 'pandas', parse_dates: bool | list | None = None, method: str = 'lazy') \
        -> dict:
    """loops through the path and adds the agent files to the data structure

    Args:
        path: path to the agents' folder
        df: dataframe type
        parse_dates: list of columns to parse as dates
        method: method to load the data (polars only)

    Returns:
        None

    """
    # Create an empty dictionary to store the files
    data = {}

    # Loop through each item in the specified path
    for item in os.listdir(path):
        # Get the full path of the current item
        item_path = os.path.join(path, item)

        # If the current item is a file
        if os.path.isfile(item_path):
            # Extract the file name without the extension
            file_name = os.path.splitext(item)[0]

            # Load the file using the given function and add it to the dictionary
            # Note: dataframes are loaded as polars and lazily to save memory
            data[file_name] = load_file(item_path, df=df, method=method, parse_dates=parse_dates)

        # If the current item is a directory
        elif os.path.isdir(item_path):
            # Use the directory name as the key for a new nested dictionary
            folder_name = item

            # Recursively call the function to add files and folders to the nested dictionary
            data[folder_name] = add_nested_data(item_path)

    return data


def get_all_subdirectories(path_directory):
    """Get names of all subdirectories."""
    subdirectories = [name for name in os.listdir(path_directory) if os.path.isdir(os.path.join(path_directory,
                                                                                                name))]
    if len(subdirectories) > 0:
        return subdirectories  # Return the name of the first subdirectory
    else:
        return None  # No subdirectories found
