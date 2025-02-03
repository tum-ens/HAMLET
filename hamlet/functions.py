__author__ = "MarkusDoepfert"
__credits__ = "jiahechu"
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"


import json
import os
import random
import shutil
import string
import time

import pandas as pd
import polars as pl
import numpy as np
from ruamel.yaml import YAML
from typing import Callable

import hamlet.constants as c

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

    # Create the folder if it does not exist
    folder = os.path.dirname(path)
    if not os.path.exists(folder):
        os.makedirs(folder)

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


def calculate_timedelta(target_df, reference_ts, by=c.TC_TIMESTAMP):
    """
    Calculate time difference (timedelta) between current timestep and datetime index of given polars data/lazyframe.

    The returned dataframe has an additional 'timedelta' column = current time step - datetime index. If 'timedelta'
    is positive, the index is earlier than current ts. If negative, the index is later than current ts. The column used
    as datetime index should be named as c.TC_TIMESTAMP.

    FUNCTION SPECIFICALLY DESIGNED FOR DATA PROCESSING IN POLARS

    Args:
        target_df: dataframe or lazyframe to be calculated.
        reference_ts: reference human time in datetime format.
        by: column name of a time column to be calculated (usually c.TC_TIMESTAMP or c.TC_TIMESTEP).

    Returns:
        target_df: same as the input target_df with an additional 'timedelta' column.
    """

    # get time info from original dataframe
    datetime_index = target_df.select(by)
    dtype = datetime_index.dtypes[0]

    # generate a new column with current timestep
    target_df = target_df.with_columns(pl.lit(reference_ts)
                                       .alias('current')
                                       .cast(pl.Datetime(time_unit=dtype.time_unit, time_zone=dtype.time_zone)))

    # calculate timedelta
    target_df = target_df.with_columns((pl.col('current') - pl.col(by)).alias('timedelta'))
    target_df = target_df.drop('current')   # delete column with same current ts value

    return target_df


def calculate_time_resolution(target_df, by=c.TC_TIMESTAMP):
    """
    Calculate the time resolution of the given dataframe according to the given column.

    Args:
        target_df: dataframe or lazyframe to be calculated.
        by: column name of a time column to be calculated (usually c.TC_TIMESTAMP or c.TC_TIMESTEP).

    """
    # randomly choose a value in timestamp column
    reference_ts = target_df.select(by).sample(n=1).item()

    # calculate time resolution
    target = calculate_timedelta(target_df=target_df, reference_ts=reference_ts, by=by)
    target = target.filter(pl.col('timedelta') != 0)  # delete the row for the current ts
    target = target.with_columns(abs(pl.col('timedelta')))  # set timedelta to absolute value
    resolution = target.select(pl.min('timedelta')).item()  # the smallest timedelta is the resolution

    # return resolution in seconds
    return resolution.seconds


def slice_dataframe_between_times(target_df, reference_ts, duration: int, unit='second', by=c.TC_TIMESTAMP):
    """
    Slice the given pl data/lazyframe to the given duration to the reference time step.

    The reference ts and timestep column in target data should be in human time. The column used as datetime index
    should be named as c.TC_TIMESTAMP. In this function, when slicing data in the future, the reference timestep won't
    be considered. When slicing data from the past, the reference timestep will be considered.

    FUNCTION SPECIFICALLY DESIGNED FOR DATA PROCESSING IN POLARS

    Args:
        target_df: dataframe or lazyframe to be sliced.
        reference_ts: reference time step in datetime format.
        duration: duration to the reference time step. Could be positive (after reference ts) or negative (before
        reference ts).
        unit: unit of the duration in 'second', 'minute', 'hour' or 'day'.
        by: column name of a time column to be calculated (usually c.TC_TIMESTAMP or c.TC_TIMESTEP).

    Returns:
        sliced_df: sliced dataframe or lazyframe between reference ts and duration.
    """

    # add timedelta as a column
    target_df = calculate_timedelta(target_df=target_df, reference_ts=reference_ts, by=by)

    # convert duration to second
    converter = {'second': 1,
                 'minute': c.MINUTES_TO_SECONDS,
                 'hour': c.HOURS_TO_SECONDS,
                 'day': c.DAYS_TO_SECONDS}      # factors to multiply when converting the corresponding unit to second
    duration = duration * converter[unit]

    if duration > 0:    # slice data in the future, reference (current) timestep will not be included
        filter_conditions = ((pl.col('timedelta') + pl.duration(seconds=duration) > 0) &
                             (pl.col('timedelta') <= 0))
    elif duration < 0:   # slice data from the past, reference (current) timestep will be included
        filter_conditions = ((pl.col('timedelta') + pl.duration(seconds=duration) < 0) &
                             (pl.col('timedelta') >= 0))
    else:   # get data at current timestep
        filter_conditions = pl.col('timedelta') == 0

    sliced_df = target_df.filter(filter_conditions)
    sliced_df = sliced_df.drop('timedelta')     # delete unnecessary column

    return sliced_df


def gen_ids(n: int = 1, length: int = 15, prefix: str = '', suffix: str = '', only_integers: bool = False) \
        -> list[str] | str:
    """
    Generate random unique IDs with optional prefix, suffix, and integer-only option.

    This function generates a specified number of unique random IDs, each of a given length.
    Optionally, a prefix and/or suffix can be added to each ID. IDs can be made of only integers if desired.

    Args:
        n (int): The number of IDs to generate. Defaults to 1.
        length (int): The length of the random part of each ID. Defaults to 15.
        prefix (str): A prefix string to add to the start of each ID. Defaults to an empty string.
        suffix (str): A suffix string to add to the end of each ID. Defaults to an empty string.
        only_integers (bool): If True, the IDs will consist only of integers. Defaults to False.

    Returns:
        list[str] | str: A list of generated IDs if n > 1, otherwise a single ID as a string.
    """
    id_set = set()
    characters = string.digits if only_integers else string.ascii_letters + string.digits

    while len(id_set) < n:
        new_id = ''.join(random.choices(characters, k=length))
        id_set.add(prefix + new_id + suffix)

    ids = list(id_set)
    return ids[0] if n == 1 else ids


def enforce_schema(schema: dict, df: pl.DataFrame, threshold: int = 20_000) -> pl.DataFrame:
    """
    Dynamically selects the best method to enforce schema based on DataFrame size.
    """

    def enforce_schema_eager(schema: dict, df: pl.DataFrame) -> pl.DataFrame:
        """
        Enforces schema using eager execution for small DataFrames.
        """
        for col, dtype in schema.items():
            if col in df.columns and df[col].dtype != dtype:
                try:
                    df = df.with_columns(df[col].cast(dtype))
                except Exception as e:
                    raise ValueError(
                        f"Failed to cast column '{col}' to type '{dtype}'. Error: {e}"
                    )
        return df

    def enforce_schema_lazy(schema: dict, df: pl.DataFrame) -> pl.DataFrame:
        """
        Enforces schema using lazy evaluation for large DataFrames.
        """
        lazy_df = df.lazy()
        for col, dtype in schema.items():
            if col in df.columns and df[col].dtype != dtype:
                lazy_df = lazy_df.with_columns(pl.col(col).cast(dtype))
        return lazy_df.collect()

    if len(df) < threshold:  # Threshold for choosing method
        return enforce_schema_eager(schema, df)
    else:
        return enforce_schema_lazy(schema, df)


def add_info_from_col(df: pd.DataFrame, col: str, drop: bool = False, sep: str = ',', key_val_sep: str = ':') -> \
        pd.DataFrame:
    """
    Adds information from a column to the dataframe.

    The column should contain a string with the following format:
    'key1:value1,key2:value2,...,keyN:valueN'

    Parameters:
        df (pd.DataFrame): The input dataframe
        col (str): The column containing the information
        drop (bool): Whether to drop the column containing the information
        sep (str): The separator between key-value pairs
        key_val_sep (str): The separator between keys and values

    Returns:
        pd.DataFrame: The dataframe with the information added as separate columns

    Alternative method:
        # Split the strings into separate key-value pairs
        df['parsed'] = df[col].apply(lambda x: dict(tuple(i.split(key_val_sep)) for i in x.split(sep)))

        # Get all the keys from the parsed strings
        keys = set().union(*df['parsed'].apply(lambda x: x.keys()))

        # Create separate columns for each key-value pair and fill with np.nan if not present
        for key in keys:
            df[key] = df['parsed'].apply(
                lambda x: x.get(key, np.nan) if x.get(key, None) in ['NaN', 'nan'] else x.get(key, np.nan))

        # Drop the original column and the intermediate parsed column
        df.drop(columns=['col', 'parsed'], inplace=True)
    """

    # Turn the column into a dictionary
    info = df[col].to_dict()

    # Loop through the dictionary and create entries for the key-value pairs
    for idx, val in info.items():
        # Split the key-value pairs

        key_value_pairs = val.split(sep)
        # Create a dictionary for the key-value pairs
        info[idx] = dict()

        for key_value_pair in key_value_pairs:
            # Split the key and value
            key, value = key_value_pair.split(key_val_sep)

            # Add the key-value pair to the dictionary and convert them to the desired data type
            try:
                info[idx][key] = int(value)
            except ValueError:
                try:
                    info[idx][key] = float(value)
                except ValueError:
                    info[idx][key] = str(value)

    # Create a dataframe from the dictionary
    df = df.join(pd.DataFrame(info).T)

    # Fill empty values and cells with string NaN and nan with NaN
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df = df.replace(["NaN", "nan"], np.nan, regex=True)

    # Drop the original column
    if drop:
        df.drop(columns=col, inplace=True)

    return df
