"""Takes a list of the 100% scenarios and derives the other scenarios from them based on the following rules:
    - sort households by inflexible_load
    - take every nth household's out of the list in descending order of inflexible_load, i.e. in 75 % scenario take
        every 4th household starting from the top"""

import pandas as pd
import os
import numpy as np
from pprint import pprint
from fractions import Fraction
from hamlet import functions as f
from tqdm import tqdm

def check_step_size(stepsize: float):
    if stepsize <= 0 or stepsize >= 1:
        raise ValueError('Step size must be between 0 and 1')

    if 1 % stepsize != 0:
        print('Warning: Step size does not divide 1 evenly. This may lead to unexpected results.')


def load_sheets_as_df(path: str):
    # Dictionaries to store the dataframes
    dict_grids = {}
    dict_original = {}

    # Load grid file if not already loaded
    grid = f.load_file(path=path)

    # Load sheets from grid file
    for sheet in grid.sheet_names:
        # Load the dataframe from the sheet
        dict_grids[sheet] = grid.parse(sheet, index_col=0)
        dict_original[sheet] = dict_grids[sheet].copy()

        # Add the info in the description column as new columns
        try:
            dict_grids[sheet] = add_info_from_col(df=dict_grids[sheet], col='description', drop=True)
        except (AttributeError, KeyError):
            pass

    return dict_grids, dict_original


def add_info_from_col(df: pd.DataFrame, col: str, drop: bool = False, sep: str = ',', key_val_sep: str = ':') \
        -> pd.DataFrame:
    """Adds information from a column to the dataframe

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


def get_owner_indices(series, percentage):
    # Check if the percentage is valid, otherwise return empty list
    if percentage == 1:
        return []

    # Convert percentage to fraction and get the denominator and numerator for the batch size and the reduction
    batch_size = Fraction(percentage).limit_denominator().denominator
    reduce_batch_by = batch_size - Fraction(percentage).limit_denominator().numerator

    # Split the series index into batches
    batches = [series.index[i:i + batch_size] for i in range(0, len(series.index), batch_size)]

    # Reduce each batch by the number reduce_batch_by, starting with the first value
    indices_to_reduce = []
    for batch in batches:
        indices_to_reduce.extend(batch[:reduce_batch_by])

    return indices_to_reduce


def reduce_dataframe(df: pd.DataFrame, idx: list, targets: list, idx_col: str = 'owner', target_col: str = 'load_type') \
        -> pd.DataFrame:

    # Iterate through the targets and filter the DataFrame
    for target in targets:
        # Identify the rows that match the target value in the target column and are in the list of indices
        try:
            condition = (df[idx_col].isin(idx)) & (df[target_col] == target)
        except KeyError:
            continue

        # Find the index of rows that match the condition
        rows_to_reduce = df[condition].index

        # Drop the rows identified by the index
        df = df.drop(rows_to_reduce)

    return df


def keep_by_indices(df: pd.DataFrame, indices: list) -> pd.DataFrame:
    return df.loc[indices]


def save_excel(path: str, data: dict):
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        for key, df in data.items():
            df.to_excel(writer, sheet_name=key)


def run(path: str, scenarios: list, stepsize: float):

    # Check if the step size is valid (between 0 and 1 and modulus of 1)
    check_step_size(stepsize)

    # Calculate the total number of iterations
    total_iterations = len(scenarios) * (Fraction(stepsize).limit_denominator().denominator + 1) ** 3

    # Create a tqdm object for the progress bar
    pbar = tqdm(total=total_iterations, desc="Processing scenarios", leave=True, unit=" scenarios", position=0)

    # Loop through the scenarios
    for scenario in scenarios:
        # Split the scenario name into the grid and the scenario type
        str_parts = scenario.split('_')
        share_pv_orig = float(str_parts[2])
        share_hp_orig = float(str_parts[4])
        share_ev_orig = float(str_parts[6])

        # Load the scenario
        scenario_path = os.path.join(path, scenario)
        scenario_dict, scenario_orig = load_sheets_as_df(path=scenario_path)

        # Get the load and the generation dataframes
        df_load = scenario_dict['load']  # relevant columns: load_type
        df_gen = scenario_dict['sgen']   # relevant columns: plant_type

        # Sort the households by inflexible load
        sorted_bus = df_load[df_load.load_type == 'inflexible_load'].sort_values(by='demand', ascending=False).bus

        # Get the original share of pv
        share_pv = share_pv_orig
        while share_pv >= 0:

            # Create copies of the original dataframes and the original scenario
            pv_load = df_load.copy()
            pv_gen = df_gen.copy()
            pv_orig = scenario_orig.copy()
            pv_load_orig = scenario_orig['load'].copy()
            pv_gen_orig = scenario_orig['sgen'].copy()

            # Get the indices of the owners to reduce
            indices = get_owner_indices(series=sorted_bus, percentage=share_pv)

            # Reduce the load and generation dataframes by the indices and the targets
            pv_gen = reduce_dataframe(df=pv_gen, idx=indices, targets=['pv', 'battery'], target_col='plant_type')

            # Keep only the rows with the indices in the original dataframes
            pv_gen_orig = keep_by_indices(df=pv_gen_orig, indices=pv_gen.index)

            # Update the scenario with the new dataframes
            pv_orig['sgen'] = pv_gen_orig

            # Save the scenario with the new name
            new_scenario_name = f'{str_parts[0]}_pv_{round(share_pv, 2)}_hp_{share_hp_orig}_ev_{share_ev_orig}_{str_parts[7]}'
            new_scenario_path = os.path.join(path, new_scenario_name)

            # Save the scenario
            save_excel(path=new_scenario_path, data=pv_orig)

            # Get the original share of hp
            share_hp = share_hp_orig
            while share_hp >= 0:

                # Create copies of the original dataframes and the original scenario with the reduced pv
                hp_load = pv_load.copy()
                hp_gen = pv_gen.copy()
                hp_orig = pv_orig.copy()
                hp_load_orig = pv_load_orig.copy()
                hp_gen_orig = pv_gen_orig.copy()

                # Get the indices of the owners to reduce
                indices = get_owner_indices(series=sorted_bus, percentage=share_hp)

                # Reduce the load and generation dataframes by the indices and the targets
                hp_load = reduce_dataframe(df=hp_load, idx=indices, targets=['heat', 'hp'], target_col='load_type')
                hp_gen = reduce_dataframe(df=hp_gen, idx=indices, targets=['heat_storage'], target_col='plant_type')

                # Keep only the rows with the indices in the original dataframes
                hp_load_orig = keep_by_indices(df=hp_load_orig, indices=hp_load.index)
                hp_gen_orig = keep_by_indices(df=hp_gen_orig, indices=hp_gen.index)

                # Update the scenario with the new dataframes
                hp_orig['load'] = hp_load_orig
                hp_orig['sgen'] = hp_gen_orig

                # Save the scenario with the new name
                new_scenario_name = f'{str_parts[0]}_pv_{round(share_pv, 2)}_hp_{round(share_hp, 2)}_ev_{share_ev_orig}_{str_parts[7]}'
                new_scenario_path = os.path.join(path, new_scenario_name)

                # Save the scenario
                save_excel(path=new_scenario_path, data=hp_orig)

                # Get the original share of ev
                share_ev = share_ev_orig
                while share_ev >= 0:

                    # Create copies of the original dataframes and the original scenario with the reduced pv and hp
                    ev_load = hp_load.copy()
                    ev_gen = hp_gen.copy()
                    ev_orig = hp_orig.copy()
                    ev_load_orig = hp_load_orig.copy()
                    ev_gen_orig = hp_gen_orig.copy()

                    # Get the indices of the owners to reduce
                    indices = get_owner_indices(series=sorted_bus, percentage=share_ev)

                    # Reduce the load and generation dataframes by the indices and the targets
                    ev_load = reduce_dataframe(df=ev_load, idx=indices, targets=['ev'], target_col='load_type')

                    # Keep only the rows with the indices in the original dataframes
                    ev_load_orig = keep_by_indices(df=ev_load_orig, indices=ev_load.index)

                    # Update the scenario with the new dataframes
                    ev_orig['load'] = ev_load_orig

                    # Save the scenario with the new name
                    new_scenario_name = (f'{str_parts[0]}_pv_{round(share_pv, 2)}_hp_{round(share_hp, 2)}'
                                         f'_ev_{round(share_ev, 2)}_{str_parts[7]}')
                    new_scenario_path = os.path.join(path, new_scenario_name)

                    # Save the scenario
                    save_excel(path=new_scenario_path, data=ev_orig)

                    # Update the progress bar
                    pbar.update(1)

                    # Update the share of ev
                    share_ev -= stepsize

                # Update the share of hp
                share_hp -= stepsize

            # Update the share of pv
            share_pv -= stepsize

    # Close the progress bar when done
    pbar.close()


if __name__ == '__main__':

    # Define step size
    step_size = 0.2

    # Define the path to the scenarios folder
    path_scenarios = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'urbs_scenarios')

    # Get all the scenarios
    scenarios = next(os.walk(path_scenarios))[2]
    scenarios = [scenario for scenario in scenarios if 'pv_' in scenario]

    # Derive the scenarios
    run(path=path_scenarios, scenarios=scenarios, stepsize=step_size)


