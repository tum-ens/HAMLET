import os
import shutil
import time
import json
import pandas as pd
import polars as pl
from ruamel.yaml import YAML

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


def load_file(path: str, index: int = 0, df: str = 'pandas') -> object:
    file_type = path.rsplit('.', 1)[-1]
    if file_type == 'yaml' or file_type == 'yml':
        with open(path) as file:
            file = YAML().load(file)
    elif file_type == 'json':
        with open(path) as file:
            file = json.load(file)
    elif file_type == 'csv':
        if df == 'pandas':
            file = pd.read_csv(path, index_col=index)
        elif df == 'polars':
            file = pl.read_csv(path)
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
            file = pl.read_ipc(path)
        else:
            raise ValueError(f'Dataframe type "{df}" not supported')
    else:
        raise ValueError(f'File type "{file_type}" not supported')

    return file


def save_file(path: str, data, index: bool = True, df: str = 'pandas') -> None:
    file_type = path.rsplit('.', 1)[-1]

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