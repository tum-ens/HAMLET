__author__ = "TUM-Doepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "TUM-Doepfert"
__email__ = "markus.doepfert@tum.de"
__status__ = "Development"

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib import rc
from matplotlib import rcParams
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
import datetime
import pickle
import tqdm
import re
import copy
import warnings
import itertools
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib import rc
from matplotlib import rcParams
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
import datetime
import pickle
import tqdm
import re
import copy
import warnings
import itertools
import polars as pl
import os
from datetime import datetime
from hamlet import functions as f
from hamlet import constants as c
import math
from datetime import timedelta
from pathlib import PurePath
import plotly.graph_objects as go
import plotly.express as px
import ast
import plotly.express as px
import plotly.io as plt_io



class Analyzer:

    def __init__(self, path: str):
        """Initializes the analyzer object."""
        self.path = path

        market_path = os.path.join(self.path, "market/lem/lem_continuos")
        self.market_path = market_path
        path = PurePath(path)
        last_part = path.name
        path_scenario = "../04_scenarios"
        combined_path = os.path.join(path_scenario, last_part)

        self.path_scenario = os.path.abspath(combined_path)
        # Load general information and configuration
        self.general = f.load_file(os.path.join(self.path_scenario, 'general', 'general.json'))
        self.config = f.load_file(os.path.join(self.path_scenario, 'config', 'config_setup.yaml'))
        self.config_markets = f.load_file(os.path.join(self.path_scenario, 'config', 'config_markets.yaml'))

        # Load timetable
        self.timetable = f.load_file(os.path.join(self.path_scenario, 'general', 'timetable.ft'),
                                     df='pandas')

        # Load scenario structure
        self.structure = self.general['structure']
        self.name = os.path.basename(path_scenario)
        self.EPSILON = 1e-10

        # Set the results path
        #self.path_results = os.path.join(self.config['paths']['results'], self.name)

    def plot_general_analysis(self, starting_date):
        """Plots the general analysis."""

        error_matrix, estimation_matrix = self.calculate_error(starting_date)

        #self.plot_error(error_matrix, estimation_matrix)

        #self.calculate_error_forecats()

    def calculate_error(self, starting_date):

        error_matrix = pd.DataFrame(columns=['Agent', 'Label', 'MAPE', 'RMSE', 'MAE', 'NRMSE', 'MASE', 'CV', 'MAAPE', 'MAAPE2'])
        estimation_matrix = pd.DataFrame(columns=['Timestamp', 'Agent', 'Label', 'Overestimation', 'Underestimation', 'Benefits', 'Benefits_Market', 'Benefits_Balancing', 'Costs', 'Costs_Market', 'Costs_Balancing', 'Costs_Fees', 'Balance'])

        for region in self.structure.keys():
            # initialize RegionDB object
            self.region_path = os.path.join(os.path.dirname(self.path), self.structure[region])

            markets_types = f.get_all_subdirectories(os.path.join(self.region_path, 'markets'))
            for markets_type in markets_types:
                path_markts_type = os.path.join(self.region_path, 'markets', markets_type)
                markets = f.get_all_subdirectories(os.path.join(self.region_path, 'markets', markets_type))
                for market in markets:

                    self.market_path = os.path.join(path_markts_type, market)

                    market_transactions = f.load_file(path=os.path.join(self.market_path, 'market_transactions.csv'), df='pandas')

                    # create a Boolean mask for the rows to remove
                    mask_balancing = market_transactions[c.TC_TYPE_TRANSACTION] != c.TT_BALANCING
                    mask_market = market_transactions[c.TC_TYPE_TRANSACTION] != c.TT_MARKET
                    mask_grid = market_transactions[c.TC_TYPE_TRANSACTION] != c.TT_GRID
                    mask_levies = market_transactions[c.TC_TYPE_TRANSACTION] != c.TT_LEVIES

                    # select only the rows that contain 'balancing'
                    market_transactions_cleared = market_transactions[~mask_balancing]
                    market_transactions_market = market_transactions[~mask_market]
                    market_transactions_grid = market_transactions[~mask_grid]
                    market_transactions_levies = market_transactions[~mask_levies]

                    agents_types = f.get_all_subdirectories(os.path.join(self.region_path, 'agents'))
                    for agents_type in agents_types:
                        # register agents for each type
                        path_agents_type = os.path.join(self.region_path, 'agents', agents_type)
                        agents = f.get_all_subdirectories(os.path.join(self.region_path, 'agents', agents_type))
                        if agents:
                            for agent in agents:
                                if agents_type == c.A_AGGREGATOR:

                                    agents_path = os.path.join(path_agents_type, agent)
                                    account = f.load_file(path=os.path.join(agents_path, 'account.json'))
                                    aggregated_agents = ast.literal_eval(account["general"]["aggregated_agents"])
                                    grid_and_levies_price = self.config_markets['lem_continuous']['pricing']['retailer']['grid']['fixed']['local'][1] + self.config_markets['lem_continuous']['pricing']['retailer']['levies']['fixed']['price'][1]

                                    for aggregated_agent in aggregated_agents:
                                        path_agents_sfh = os.path.join(self.region_path, 'agents', c.A_SFH)
                                        aggregated_agents_path = os.path.join(path_agents_sfh, aggregated_agent)

                                        meters = f.load_file(path=os.path.join(aggregated_agents_path, 'meters.ft'), df='pandas')

                                        meters = meters.loc[~(meters.iloc[:, 1:] == 0).all(axis=1)]

                                        for column in meters:
                                            if column != "timestamp":
                                                meters[column] = self.cumulative_to_normal(list(meters[column]))

                                        # removes day 1
                                        meters = meters.loc[meters['timestamp'] >= starting_date]

                                        filtered_meters = meters.copy()

                                        filtered_meters[c.TC_TIMESTAMP] = filtered_meters[c.TC_TIMESTAMP] - timedelta(minutes=15)

                                        for timestamp in filtered_meters[c.TC_TIMESTAMP]:
                                            filtered_market_transactions = market_transactions_cleared[(pd.to_datetime(market_transactions_cleared[c.TC_TIMESTEP]) == timestamp) & (market_transactions_cleared['id_agent'] == agent)]
                                            market_transactions_market_time = market_transactions_market[(pd.to_datetime(market_transactions_market[c.TC_TIMESTEP]) == timestamp)]
                                            filtered_meters_time = filtered_meters[(pd.to_datetime(filtered_meters[c.TC_TIMESTAMP]) == timestamp)]

                                            balancing_in = filtered_market_transactions.fillna(0)[
                                                               c.TC_ENERGY_IN] * c.WH_TO_KWH
                                            balancing_in_agent = (balancing_in/len(aggregated_agents)).sum()

                                            balancing_out = filtered_market_transactions.fillna(0)[
                                                                c.TC_ENERGY_OUT] * c.WH_TO_KWH
                                            balancing_out_agent = (balancing_out/len(aggregated_agents)).sum()


                                            if market_transactions_market_time[c.TC_PRICE_PU_OUT].notna().any():
                                                price_sell = market_transactions_market_time[c.TC_PRICE_PU_OUT].dropna().iloc[0] / c.EUR_KWH_TO_EURe7_WH

                                            else:
                                                price_sell = self.config_markets['lem_continuous']['pricing']['retailer']['energy']['fixed']['price'][0]

                                            if market_transactions_market_time[c.TC_PRICE_PU_IN].notna().any():
                                                price_buy = market_transactions_market_time[c.TC_PRICE_PU_IN].dropna().iloc[0] / c.EUR_KWH_TO_EURe7_WH

                                            else:
                                                price_buy = self.config_markets['lem_continuous']['pricing']['retailer']['energy']['fixed']['price'][1]

                                            filtered_meters_time = filtered_meters_time.iloc[:, 1:]
                                            sum_meters_time = filtered_meters_time.sum(axis=1)
                                            sum_meters_time = sum_meters_time.item() * c.WH_TO_KWH

                                            if sum_meters_time > 0:
                                                costs_market = 0
                                                costs_balancing = balancing_in_agent * \
                                                                  self.config_markets['lem_continuous']['pricing'][
                                                                      'retailer']['balancing']['fixed']['price'][1]
                                                costs_fees = 0
                                                benefits_balancing = balancing_out_agent * \
                                                                     self.config_markets['lem_continuous']['pricing'][
                                                                         'retailer']['balancing']['fixed']['price'][0]
                                                benefits_market = (sum_meters_time - balancing_out_agent) * price_sell

                                                costs_in = costs_market + costs_balancing + costs_fees
                                                benefits_out = benefits_balancing + benefits_market
                                                balance = benefits_out - costs_in

                                            elif sum_meters_time < 0:
                                                sum_meters_time = -sum_meters_time
                                                costs_market = (sum_meters_time - balancing_in_agent) * price_buy
                                                costs_balancing = balancing_in_agent * self.config_markets['lem_continuous']['pricing']['retailer']['balancing']['fixed']['price'][1]
                                                costs_fees = sum_meters_time * grid_and_levies_price
                                                benefits_balancing = balancing_out_agent * self.config_markets['lem_continuous']['pricing']['retailer']['balancing']['fixed']['price'][0]
                                                benefits_market = 0

                                                costs_in = costs_market + costs_balancing + costs_fees
                                                benefits_out = benefits_balancing + benefits_market
                                                balance = benefits_out - costs_in


                                            elif sum_meters_time == 0:
                                                costs_market = 0
                                                costs_balancing = balancing_in_agent * \
                                                                  self.config_markets['lem_continuous']['pricing'][
                                                                      'retailer']['balancing']['fixed']['price'][1]
                                                costs_fees = 0
                                                benefits_balancing = balancing_out_agent * \
                                                                     self.config_markets['lem_continuous']['pricing'][
                                                                         'retailer']['balancing']['fixed']['price'][0]
                                                benefits_market = 0

                                                costs_in = costs_market + costs_balancing + costs_fees
                                                benefits_out = benefits_balancing + benefits_market
                                                balance = benefits_out - costs_in

                                            overestimation = balancing_in_agent.sum()
                                            underestimation = balancing_out_agent.sum()

                                            if filtered_market_transactions.empty:
                                                underestimation = 0
                                                overestimation = 0

                                            label = "aggregated"
                                            estimation_error_row = [timestamp, aggregated_agent, label, overestimation, underestimation, benefits_out, benefits_market, benefits_balancing, costs_in, costs_market, costs_balancing, costs_fees, balance]

                                            estimation_matrix.loc[len(estimation_matrix.index)] = estimation_error_row


                                    meters = f.load_file(path=os.path.join(agents_path, 'meters.ft'), df='pandas')
                                    plants = f.load_file(path=os.path.join(agents_path, 'plants.json'))

                                    meters = meters.loc[~(meters.iloc[:, 1:] == 0).all(axis=1)]

                                    for column in meters:
                                        if column != "timestamp":
                                            meters[column] = self.cumulative_to_normal(list(meters[column]))

                                    # removes day 1
                                    meters = meters.loc[meters['timestamp'] >= starting_date]

                                    meters_renamed = meters.copy()

                                    meters_renamed.rename(columns={col: plant['type'] for col, plant in plants.items()},
                                                          inplace=True)

                                    filtered_meters = meters_renamed.copy()

                                    filtered_meters[c.TC_TIMESTAMP] = filtered_meters[c.TC_TIMESTAMP] - timedelta(minutes=15)

                                    label = 'Aggregator'
                                    # create our custom_dark theme from the plotly_dark template
                                    plt_io.templates["custom"] = plt_io.templates["simple_white"]

                                    # create our custom_dark theme from the plotly_dark template
                                    plt_io.templates["custom"] = plt_io.templates["simple_white"]

                                    plt_io.templates['custom']['layout']['yaxis']['showgrid'] = True
                                    plt_io.templates['custom']['layout']['xaxis']['showgrid'] = True
                                    plt_io.templates['custom']['layout']['font']['size'] = 45
                                    plt_io.templates['custom']['layout']['legend']['y'] = 7000
                                    plt_io.templates['custom']['layout']['legend']['yanchor'] = 'bottom'
                                    plt_io.templates['custom']['layout']['legend']['orientation'] = 'h'
                                    plt_io.templates['custom']['layout']['legend']['xanchor'] = 'right'
                                    plt_io.templates['custom']['layout']['legend']['x'] = 1
                                    plt_io.templates['custom']['layout']['legend']['font_size'] = 40

                                    grph = filtered_meters.copy()

                                    column_groups = grph.columns.str.split('_').str[1] +"_"+ grph.columns.str.split('_').str[2]

                                    # Group columns by the common prefix and sum them
                                    grph = grph.groupby(column_groups, axis=1).sum()

                                    grph[c.TC_TIMESTAMP] = filtered_meters[c.TC_TIMESTAMP]


                                    fig = px.line(grph, x=c.TC_TIMESTAMP, y=grph.columns)
                                    fig.add_trace(go.Scatter(x=grph[c.TC_TIMESTAMP], y=grph.sum(numeric_only=True, axis=1), fill='tozeroy',
                                                             mode='lines', line_color='rgb(173,216,230)', name='net_amount'
                                                             ))
                                    fig.update_layout(yaxis_title='Wh per 15 minutes', font=dict(size=20),
                                                      legend=dict(y=0.5, traceorder='reversed', font_size=18))
                                    fig.layout.template = 'custom'
                                    fig.update_layout(legend_title_text='')
                                    fig.update_layout(legend={
                                        "font": {
                                            "family": "Arial, monospace",
                                            "size": 25
                                        }})

                                    fig.update_layout(legend=dict(
                                        orientation="h",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="right",
                                        x=1
                                    ))

                                    fig.update_layout(
                                        legend=dict(
                                            bordercolor="Black",
                                            borderwidth=2
                                        )
                                    )
                                    fig.show()



                                    energy_error_all = []

                                    for timestamp in filtered_meters[c.TC_TIMESTAMP]:
                                        # selecting rows based on condition
                                        filtered_market_transactions = market_transactions_cleared[(pd.to_datetime(market_transactions_cleared[c.TC_TIMESTEP]) == timestamp) & (market_transactions_cleared['id_agent'] == agent)]
                                        filtered_market_transactions_market = market_transactions_market[(pd.to_datetime(market_transactions_market[c.TC_TIMESTEP]) == timestamp) & (market_transactions_market['id_agent'] == agent)]
                                        filtered_market_transactions_grid = market_transactions_grid[(pd.to_datetime(market_transactions_grid[c.TC_TIMESTEP]) == timestamp) & (market_transactions_grid['id_agent'] == agent)]
                                        filtered_market_transactions_levies = market_transactions_levies[(pd.to_datetime(market_transactions_levies[c.TC_TIMESTEP]) == timestamp) & (market_transactions_levies['id_agent'] == agent)]

                                        overestimation = filtered_market_transactions.fillna(0)[c.TC_ENERGY_IN] * c.WH_TO_KWH
                                        underestimation = filtered_market_transactions.fillna(0)[c.TC_ENERGY_OUT] * c.WH_TO_KWH

                                        underestimation = underestimation.sum()
                                        overestimation = overestimation.sum()

                                        if filtered_market_transactions.empty:
                                            underestimation = 0
                                            overestimation = 0

                                        costs_market = (filtered_market_transactions_market.fillna(0)[c.TC_ENERGY_IN] * c.WH_TO_KWH * (filtered_market_transactions_market.fillna(0)[c.TC_PRICE_PU_IN] / c.EUR_KWH_TO_EURe7_WH)).sum()
                                        costs_balancing = (filtered_market_transactions.fillna(0)[c.TC_ENERGY_IN] * c.WH_TO_KWH * (filtered_market_transactions.fillna(0)[c.TC_PRICE_PU_IN] / c.EUR_KWH_TO_EURe7_WH)).sum()
                                        costs_fees = (filtered_market_transactions_grid.fillna(0)[c.TC_ENERGY_IN] * c.WH_TO_KWH * (filtered_market_transactions_grid.fillna(0)[c.TC_PRICE_PU_IN] / c.EUR_KWH_TO_EURe7_WH)).sum() + (filtered_market_transactions_levies.fillna(0)[c.TC_ENERGY_IN] * c.WH_TO_KWH * (filtered_market_transactions_levies.fillna(0)[c.TC_PRICE_PU_IN] / c.EUR_KWH_TO_EURe7_WH)).sum()
                                        benefits_balancing = (filtered_market_transactions.fillna(0)[c.TC_ENERGY_OUT] * c.WH_TO_KWH * (filtered_market_transactions.fillna(0)[c.TC_PRICE_PU_OUT] / c.EUR_KWH_TO_EURe7_WH)).sum()
                                        benefits_market = (filtered_market_transactions_market.fillna(0)[c.TC_ENERGY_OUT] * c.WH_TO_KWH * (filtered_market_transactions_market.fillna(0)[c.TC_PRICE_PU_OUT] / c.EUR_KWH_TO_EURe7_WH)).sum()

                                        costs_in = costs_market + costs_balancing + costs_fees
                                        benefits_out = benefits_balancing + benefits_market
                                        balance = benefits_out - costs_in

                                        estimation_error_row = [timestamp, agent, label, overestimation,
                                                                underestimation, benefits_out, benefits_market,
                                                                benefits_balancing, costs_in, costs_market,
                                                                costs_balancing, costs_fees, balance]

                                        estimation_matrix.loc[len(estimation_matrix.index)] = estimation_error_row

                                        energy_error = filtered_market_transactions.fillna(0)[c.TC_ENERGY_IN] * c.WH_TO_KWH + \
                                                       filtered_market_transactions.fillna(0)[c.TC_ENERGY_OUT] * c.WH_TO_KWH

                                        if energy_error is None:
                                            energy_error = 0

                                        energy_error = energy_error.sum()

                                        energy_error_all.append(energy_error)

                                    filtered_meters = filtered_meters.iloc[:, 1:]

                                    sum_meters = filtered_meters.sum(axis=1)
                                    sum_meters = [i * c.WH_TO_KWH for i in sum_meters]

                                    # Define the function to return the MAPE values
                                    def calculate_mape_error(sum_meters, energy_error_all) -> float:

                                        # Convert actual and predicted
                                        # to numpy array data type if not already
                                        if not all([isinstance(sum_meters, np.ndarray),
                                                    isinstance(energy_error_all, np.ndarray)]):
                                            sum_meters, energy_error_all = np.array(sum_meters), np.array(
                                                energy_error_all)

                                        sum_meters, energy_error_all = remove_zeros_and_corresponding_values(sum_meters,
                                                                                                             energy_error_all)

                                        # Calculate the MAPE value and return
                                        return np.mean(np.abs(energy_error_all / sum_meters)) * 100

                                    def calculate_rmse_error(sum_meters, energy_error_all) -> float:
                                        # Convert actual and predicted
                                        # to numpy array data type if not already
                                        if not all([isinstance(sum_meters, np.ndarray),
                                                    isinstance(energy_error_all, np.ndarray)]):
                                            sum_meters, energy_error_all = np.array(sum_meters), np.array(
                                                energy_error_all)

                                        mse = np.square(energy_error_all).mean()

                                        rmse = math.sqrt(mse)

                                        return rmse

                                    def calculate_mae_error(sum_meters, energy_error_all) -> float:
                                        # Convert actual and predicted
                                        # to numpy array data type if not already
                                        if not all([isinstance(sum_meters, np.ndarray),
                                                    isinstance(energy_error_all, np.ndarray)]):
                                            sum_meters, energy_error_all = np.array(sum_meters), np.array(
                                                energy_error_all)

                                        mae = energy_error_all.mean()

                                        return mae

                                    def calculate_nrmse_error(sum_meters, energy_error_all) -> float:
                                        # Convert actual and predicted
                                        # to numpy array data type if not already
                                        if not all([isinstance(sum_meters, np.ndarray),
                                                    isinstance(energy_error_all, np.ndarray)]):
                                            sum_meters, energy_error_all = np.array(sum_meters), np.array(
                                                energy_error_all)

                                        sum_meters, energy_error_all = remove_zeros_and_corresponding_values(sum_meters,
                                                                                                             energy_error_all)

                                        nrmse = np.square((np.mean(np.square(energy_error_all / sum_meters)) * 100))

                                        return nrmse

                                    def calculate_mase_error(sum_meters, energy_error_all) -> float:
                                        # Convert actual and predicted
                                        # to numpy array data type if not already
                                        if not all([isinstance(sum_meters, np.ndarray),
                                                    isinstance(energy_error_all, np.ndarray)]):
                                            sum_meters, energy_error_all = np.array(sum_meters), np.array(
                                                energy_error_all)

                                        mase = np.mean([energy_error_all / (
                                                abs(sum_meters[i] - sum_meters[i - 1]) / len(sum_meters) - 1) for i in
                                                        range(1, len(sum_meters))])
                                        return round(mase, 2)

                                    def calculate_cv_error(sum_meters, energy_error_all) -> float:
                                        # Convert actual and predicted
                                        # to numpy array data type if not already
                                        if not all([isinstance(sum_meters, np.ndarray),
                                                    isinstance(energy_error_all, np.ndarray)]):
                                            sum_meters, energy_error_all = np.array(sum_meters), np.array(
                                                energy_error_all)

                                        cv = 100 * (np.sqrt(np.mean(np.square(energy_error_all))) / np.mean(sum_meters))

                                        return round(cv, 2)

                                    def calculate_maape_error(sum_meters, energy_error_all) -> float:
                                        # Convert actual and predicted
                                        # to numpy array data type if not already
                                        if not all([isinstance(sum_meters, np.ndarray),
                                                    isinstance(energy_error_all, np.ndarray)]):
                                            sum_meters, energy_error_all = np.array(sum_meters), np.array(
                                                energy_error_all)

                                        sum_meters, energy_error_all = remove_zeros_and_corresponding_values(sum_meters,
                                                                                                             energy_error_all)

                                        maape = np.mean(np.arctan(np.abs(energy_error_all / sum_meters)))

                                        return round(maape, 2)


                                    def calculate_maape2_error(sum_meters, energy_error_all) -> float:
                                        # Convert actual and predicted
                                        # to numpy array data type if not already
                                        if not all([isinstance(sum_meters, np.ndarray),
                                                    isinstance(energy_error_all, np.ndarray)]):
                                            sum_meters, energy_error_all = np.array(sum_meters), np.array(
                                                energy_error_all)

                                        maape = np.mean(np.arctan(np.abs(energy_error_all / (sum_meters + self.EPSILON))))

                                        return maape


                                    def remove_zeros_and_corresponding_values(array1, array2):
                                        # Step 1: Find indices of zero values in array1
                                        zero_indices = np.where(array1 == 0)[0]

                                        # Step 2: Remove corresponding values from array2
                                        for idx in reversed(zero_indices):
                                            array2 = np.delete(array2, idx)

                                        # Remove zero values from array1
                                        for idx in reversed(zero_indices):
                                            array1 = np.delete(array1, idx)

                                        return array1, array2

                                    #tran = market_transactions[(market_transactions['id_agent'] == agent) | (
                                    #            market_transactions['id_agent'] == "retailer")]
                                    #trans = market_transactions[market_transactions['id_agent'] == agent]

                                    if not all([isinstance(sum_meters, np.ndarray),
                                                isinstance(energy_error_all, np.ndarray)]):
                                        sum_meters, energy_error_all = np.array(sum_meters), np.array(
                                            energy_error_all)

                                    mape = calculate_mape_error(sum_meters, energy_error_all)

                                    rmse = calculate_rmse_error(sum_meters, energy_error_all)

                                    mae = calculate_mae_error(sum_meters, energy_error_all)

                                    nrmse = calculate_nrmse_error(sum_meters, energy_error_all)

                                    mase = calculate_mase_error(sum_meters, energy_error_all)

                                    cv = calculate_cv_error(sum_meters, energy_error_all)

                                    maape = calculate_maape_error(sum_meters, energy_error_all)

                                    maape2 = calculate_maape2_error(sum_meters, energy_error_all)

                                    error_row = [agent, label, mape, rmse, mae, nrmse, mase, cv, maape, maape2]

                                    error_matrix.loc[len(error_matrix.index)] = error_row



                                else:

                                    agents_path = os.path.join(path_agents_type, agent)
                                    account = f.load_file(path=os.path.join(agents_path, 'account.json'))

                                    if not isinstance(account["general"]["aggregated_by"], str):

                                        meters = f.load_file(path=os.path.join(agents_path, 'meters.ft'), df='pandas')
                                        plants = f.load_file(path=os.path.join(agents_path, 'plants.json'))

                                        meters = meters.loc[~(meters.iloc[:, 1:] == 0).all(axis=1)]

                                        for column in meters:
                                            if column != "timestamp":
                                                meters[column] = self.cumulative_to_normal(list(meters[column]))

                                        # removes day 1
                                        meters = meters.loc[meters['timestamp'] >= starting_date]

                                        meters_renamed = meters.copy()

                                        meters_renamed.rename(columns={col: plant['type'] for col, plant in plants.items()}, inplace=True)

                                        # Check if there are other columns except 'inflexible-load'
                                        if len(meters_renamed.columns) <= 2:
                                            label = 'consumer'
                                        else:
                                            label = 'prosumer'

                                        filtered_meters = meters_renamed.copy()

                                        filtered_meters[c.TC_TIMESTAMP] = filtered_meters[c.TC_TIMESTAMP] - timedelta(minutes=15)

                                        #title = label.capitalize() + ' ' + 'Prodution and Consumption'

                                        # create our custom_dark theme from the plotly_dark template
                                        plt_io.templates["custom"] = plt_io.templates["simple_white"]

                                        # create our custom_dark theme from the plotly_dark template
                                        plt_io.templates["custom"] = plt_io.templates["simple_white"]

                                        plt_io.templates['custom']['layout']['yaxis']['showgrid'] = True
                                        plt_io.templates['custom']['layout']['xaxis']['showgrid'] = True
                                        plt_io.templates['custom']['layout']['font']['size'] = 45
                                        plt_io.templates['custom']['layout']['legend']['y'] = 7000
                                        plt_io.templates['custom']['layout']['legend']['yanchor'] = 'bottom'
                                        plt_io.templates['custom']['layout']['legend']['orientation'] = 'h'
                                        plt_io.templates['custom']['layout']['legend']['xanchor'] = 'right'
                                        plt_io.templates['custom']['layout']['legend']['x'] = 1
                                        plt_io.templates['custom']['layout']['legend']['font_size'] = 40

                                        grph = filtered_meters.copy()

                                        column_groups = grph.columns.str.split('_').str[1] + "_" + \
                                                        grph.columns.str.split('_').str[2]

                                        # Group columns by the common prefix and sum them
                                        grph = grph.groupby(column_groups, axis=1).sum()

                                        grph[c.TC_TIMESTAMP] = filtered_meters[c.TC_TIMESTAMP]

                                        fig = px.line(grph, x=c.TC_TIMESTAMP, y=grph.columns)
                                        fig.add_trace(
                                            go.Scatter(x=grph[c.TC_TIMESTAMP], y=grph.sum(numeric_only=True, axis=1),
                                                       fill='tozeroy',
                                                       mode='lines', line_color='rgb(173,216,230)', name='net_amount'
                                                       ))
                                        fig.update_layout(yaxis_title='Wh per 15 minutes', font=dict(size=20),
                                                          legend=dict(y=0.5, traceorder='reversed', font_size=18))
                                        fig.layout.template = 'custom'
                                        fig.update_layout(legend_title_text='')
                                        fig.update_layout(legend={
                                            "font": {
                                                "family": "Arial, monospace",
                                                "size": 25
                                            }})

                                        fig.update_layout(legend=dict(
                                            orientation="h",
                                            yanchor="bottom",
                                            y=1.02,
                                            xanchor="right",
                                            x=1
                                        ))

                                        fig.update_layout(
                                            legend=dict(
                                                bordercolor="Black",
                                                borderwidth=2
                                            )
                                        )

                                        fig.show()



                                        energy_error_all = []

                                        for timestamp in filtered_meters[c.TC_TIMESTAMP]:
                                            # selecting rows based on condition
                                            filtered_market_transactions = market_transactions_cleared[(pd.to_datetime(
                                                market_transactions_cleared[c.TC_TIMESTEP]) == timestamp) & (
                                                                                                                   market_transactions_cleared[
                                                                                                                       'id_agent'] == agent)]
                                            filtered_market_transactions_market = market_transactions_market[(
                                                                                                                         pd.to_datetime(
                                                                                                                             market_transactions_market[
                                                                                                                                 c.TC_TIMESTEP]) == timestamp) & (
                                                                                                                         market_transactions_market[
                                                                                                                             'id_agent'] == agent)]
                                            filtered_market_transactions_grid = market_transactions_grid[(
                                                                                                                     pd.to_datetime(
                                                                                                                         market_transactions_grid[
                                                                                                                             c.TC_TIMESTEP]) == timestamp) & (
                                                                                                                     market_transactions_grid[
                                                                                                                         'id_agent'] == agent)]

                                            filtered_market_transactions_levies = market_transactions_levies[(
                                                                                                                                  pd.to_datetime(
                                                                                                                                      market_transactions_levies[
                                                                                                                                          c.TC_TIMESTEP]) == timestamp) & (
                                                                                                                                  market_transactions_levies[
                                                                                                                                      'id_agent'] == agent)]

                                            overestimation = filtered_market_transactions.fillna(0)[
                                                                  c.TC_ENERGY_IN] * c.WH_TO_KWH
                                            underestimation = filtered_market_transactions.fillna(0)[
                                                                 c.TC_ENERGY_OUT] * c.WH_TO_KWH

                                            underestimation = underestimation.sum()
                                            overestimation = overestimation.sum()

                                            if filtered_market_transactions.empty:
                                                underestimation = 0
                                                overestimation = 0

                                            costs_market = (filtered_market_transactions_market.fillna(0)[
                                                                c.TC_ENERGY_IN] * c.WH_TO_KWH * (
                                                                        filtered_market_transactions_market.fillna(0)[
                                                                            c.TC_PRICE_PU_IN] / c.EUR_KWH_TO_EURe7_WH)).sum()
                                            costs_balancing = (filtered_market_transactions.fillna(0)[
                                                                   c.TC_ENERGY_IN] * c.WH_TO_KWH * (
                                                                           filtered_market_transactions.fillna(0)[
                                                                               c.TC_PRICE_PU_IN] / c.EUR_KWH_TO_EURe7_WH)).sum()
                                            costs_fees = (filtered_market_transactions_grid.fillna(0)[
                                                              c.TC_ENERGY_IN] * c.WH_TO_KWH * (
                                                                      filtered_market_transactions_grid.fillna(0)[
                                                                          c.TC_PRICE_PU_IN] / c.EUR_KWH_TO_EURe7_WH)).sum() + (
                                                                     filtered_market_transactions_levies.fillna(0)[
                                                                         c.TC_ENERGY_IN] * c.WH_TO_KWH * (
                                                                                 filtered_market_transactions_levies.fillna(
                                                                                     0)[
                                                                                     c.TC_PRICE_PU_IN] / c.EUR_KWH_TO_EURe7_WH)).sum()
                                            benefits_balancing = (filtered_market_transactions.fillna(0)[
                                                                      c.TC_ENERGY_OUT] * c.WH_TO_KWH * (
                                                                              filtered_market_transactions.fillna(0)[
                                                                                  c.TC_PRICE_PU_OUT] / c.EUR_KWH_TO_EURe7_WH)).sum()
                                            benefits_market = (filtered_market_transactions_market.fillna(0)[
                                                                   c.TC_ENERGY_OUT] * c.WH_TO_KWH * (
                                                                           filtered_market_transactions_market.fillna(
                                                                               0)[
                                                                               c.TC_PRICE_PU_OUT] / c.EUR_KWH_TO_EURe7_WH)).sum()

                                            costs_in = costs_market + costs_balancing + costs_fees
                                            benefits_out = benefits_balancing + benefits_market
                                            balance = benefits_out - costs_in

                                            estimation_error_row = [timestamp, agent, label, overestimation,
                                                                    underestimation, benefits_out, benefits_market,
                                                                    benefits_balancing, costs_in, costs_market,
                                                                    costs_balancing, costs_fees, balance]

                                            estimation_matrix.loc[len(estimation_matrix.index)] = estimation_error_row

                                            energy_error = filtered_market_transactions.fillna(0)[c.TC_ENERGY_IN] * c.WH_TO_KWH + filtered_market_transactions.fillna(0)[c.TC_ENERGY_OUT] * c.WH_TO_KWH

                                            if energy_error is None:
                                                energy_error = 0

                                            energy_error = energy_error.sum()

                                            energy_error_all.append(energy_error)

                                        filtered_meters = filtered_meters.iloc[:, 1:]

                                        sum_meters = filtered_meters.sum(axis=1)
                                        sum_meters = [i * c.WH_TO_KWH for i in sum_meters]

                                        # Define the function to return the MAPE values
                                        def calculate_mape_error(sum_meters, energy_error_all) -> float:

                                            # Convert actual and predicted
                                            # to numpy array data type if not already
                                            if not all([isinstance(sum_meters, np.ndarray),
                                                        isinstance(energy_error_all, np.ndarray)]):
                                                sum_meters, energy_error_all = np.array(sum_meters), np.array(energy_error_all)

                                            sum_meters, energy_error_all = remove_zeros_and_corresponding_values(sum_meters, energy_error_all)

                                                # Calculate the MAPE value and return
                                            return round(np.mean(np.abs(energy_error_all / sum_meters)) * 100, 2)

                                        def calculate_rmse_error(sum_meters, energy_error_all) -> float:
                                            # Convert actual and predicted
                                            # to numpy array data type if not already
                                            if not all([isinstance(sum_meters, np.ndarray),
                                                        isinstance(energy_error_all, np.ndarray)]):
                                                sum_meters, energy_error_all = np.array(sum_meters), np.array(energy_error_all)

                                            mse = np.square(energy_error_all).mean()

                                            rmse = math.sqrt(mse)

                                            return rmse

                                        def calculate_mae_error(sum_meters, energy_error_all) -> float:
                                            # Convert actual and predicted
                                            # to numpy array data type if not already
                                            if not all([isinstance(sum_meters, np.ndarray),
                                                        isinstance(energy_error_all, np.ndarray)]):
                                                sum_meters, energy_error_all = np.array(sum_meters), np.array(energy_error_all)

                                            mae = energy_error_all.mean()

                                            return mae

                                        def calculate_nrmse_error(sum_meters, energy_error_all) -> float:
                                            # Convert actual and predicted
                                            # to numpy array data type if not already
                                            if not all([isinstance(sum_meters, np.ndarray),
                                                        isinstance(energy_error_all, np.ndarray)]):
                                                sum_meters, energy_error_all = np.array(sum_meters), np.array(energy_error_all)

                                            sum_meters, energy_error_all = remove_zeros_and_corresponding_values(sum_meters,energy_error_all)

                                            nrmse = np.square((np.mean(np.square(energy_error_all / sum_meters)) * 100))

                                            return nrmse

                                        def calculate_mase_error(sum_meters, energy_error_all) -> float:
                                            # Convert actual and predicted
                                            # to numpy array data type if not already
                                            if not all([isinstance(sum_meters, np.ndarray),
                                                        isinstance(energy_error_all, np.ndarray)]):
                                                sum_meters, energy_error_all = np.array(sum_meters), np.array(energy_error_all)

                                            mase = np.mean([energy_error_all / (
                                                        abs(sum_meters[i] - sum_meters[i - 1]) / len(sum_meters) - 1) for i in
                                                               range(1, len(sum_meters))])
                                            return round(mase, 2)

                                        def calculate_cv_error(sum_meters, energy_error_all) -> float:
                                            # Convert actual and predicted
                                            # to numpy array data type if not already
                                            if not all([isinstance(sum_meters, np.ndarray),
                                                        isinstance(energy_error_all, np.ndarray)]):
                                                sum_meters, energy_error_all = np.array(sum_meters), np.array(energy_error_all)

                                            cv = 100 * (np.sqrt(np.mean(np.square(energy_error_all))) / np.mean(sum_meters))

                                            return round(cv, 2)

                                        def calculate_maape_error(sum_meters, energy_error_all) -> float:
                                            # Convert actual and predicted
                                            # to numpy array data type if not already
                                            if not all([isinstance(sum_meters, np.ndarray),
                                                        isinstance(energy_error_all, np.ndarray)]):
                                                sum_meters, energy_error_all = np.array(sum_meters), np.array(
                                                    energy_error_all)

                                            sum_meters, energy_error_all = remove_zeros_and_corresponding_values(sum_meters, energy_error_all)

                                            maape = np.mean(np.arctan(np.abs(energy_error_all / sum_meters)))

                                            return round(maape, 2)

                                        def calculate_maape2_error(sum_meters, energy_error_all) -> float:
                                            # Convert actual and predicted
                                            # to numpy array data type if not already
                                            if not all([isinstance(sum_meters, np.ndarray),
                                                        isinstance(energy_error_all, np.ndarray)]):
                                                sum_meters, energy_error_all = np.array(sum_meters), np.array(
                                                    energy_error_all)

                                            maape = np.mean(
                                                np.arctan(np.abs(energy_error_all / (sum_meters + self.EPSILON))))

                                            return maape

                                        def remove_zeros_and_corresponding_values(array1, array2):
                                            # Step 1: Find indices of zero values in array1
                                            zero_indices = np.where(array1 == 0)[0]

                                            # Step 2: Remove corresponding values from array2
                                            for idx in reversed(zero_indices):
                                                array2 = np.delete(array2, idx)

                                            # Remove zero values from array1
                                            for idx in reversed(zero_indices):
                                                array1 = np.delete(array1, idx)

                                            return array1, array2

                                        #tran = market_transactions[(market_transactions['id_agent'] == agent) | (market_transactions['id_agent'] == "retailer")]
                                        #trans = market_transactions[market_transactions['id_agent'] == agent]

                                        if not all([isinstance(sum_meters, np.ndarray),
                                                    isinstance(energy_error_all, np.ndarray)]):
                                            sum_meters, energy_error_all = np.array(sum_meters), np.array(
                                                energy_error_all)

                                        mape = calculate_mape_error(sum_meters, energy_error_all)

                                        rmse = calculate_rmse_error(sum_meters, energy_error_all)

                                        mae = calculate_mae_error(sum_meters, energy_error_all)

                                        nrmse = calculate_nrmse_error(sum_meters, energy_error_all)

                                        mase = calculate_mase_error(sum_meters, energy_error_all)

                                        cv = calculate_cv_error(sum_meters, energy_error_all)

                                        maape = calculate_maape_error(sum_meters, energy_error_all)

                                        maape2 = calculate_maape2_error(sum_meters, energy_error_all)

                                        error_row = [agent, label, mape, rmse, mae, nrmse, mase, cv, maape, maape2]

                                        error_matrix.loc[len(error_matrix.index)] = error_row


                        #energy_in_all_agents = market_transactions_market.fillna(0)[c.TC_ENERGY_IN] * c.WH_TO_KWH + market_transactions_cleared.fillna(0)[c.TC_ENERGY_IN] * c.WH_TO_KWH

                        #energy_out_all_agents = market_transactions_market.fillna(0)[c.TC_ENERGY_OUT] + \
                        #                        market_transactions_cleared.fillna(0)[c.TC_ENERGY_OUT]

        f.save_file(path=os.path.join(self.path, 'general', 'error_matrix.csv'), data=error_matrix, df='pandas')
        f.save_file(path=os.path.join(self.path, 'general', 'estimation_matrix.csv'), data=estimation_matrix, df='pandas')

        return error_matrix, estimation_matrix

    def plot_error(self, error_matrix, estimation_matrix) -> None:
        """plots the flow within the market over time

        Args:

        Returns:
            None

        """
        print("*** CREATING PLOT OF Error ***")
        plt.close("all")

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

        # Fixing random state for reproducibility
        np.random.seed(19680801)

        # get the data and remove nan
        my_array = np.array(error_matrix["MAPE"])

        new_array = my_array[np.logical_not(np.isnan(my_array))]


        # plot violin plot
        axs[0].violinplot(new_array,
                          showmeans=False,
                          showmedians=True)
        axs[0].set_title('Violin plot')

        # plot box plot
        axs[1].boxplot(new_array)
        axs[1].set_title('Box plot')

        # adding horizontal grid lines
        for ax in axs:
            ax.yaxis.grid(True)
            #ax.set_xticks([y + 1 for y in range(len(new_array))], labels=['MAPE'])
            ax.set_xlabel('MAPE')
            ax.set_ylabel('Observed values')

        plt.show()

        # Group by ID and sum the errors
        #Opportunity_Cost_For_Not_Selling = estimation_matrix.groupby('Agent')['Opportunity_Cost_For_Not_Selling'].sum()
        #Opportunity_Cost_For_Not_Buying = estimation_matrix.groupby('Agent')['Opportunity_Cost_For_Not_Buying'].sum()

        #weight_counts = {
        #    "Opportunity_Cost_For_Not_Selling": np.array(Opportunity_Cost_For_Not_Selling.values),
        #    "Opportunity_Cost_For_Not_Buying": np.array(Opportunity_Cost_For_Not_Buying.values),
        #}
        #width = 0.5

        #fig, ax = plt.subplots()
        #bottom = np.zeros(39)

        #for boolean, weight_count in weight_counts.items():
        #    p = ax.bar(np.array(Opportunity_Cost_For_Not_Selling.index.values), weight_count, width, label=boolean, bottom=bottom)
        #    bottom += weight_count

        #ax.set_title("Costs of the Forecast Error")
        #ax.legend(loc="upper right")

        #plt.show()


        overestimation_sum = list(estimation_matrix.groupby('Agent')['Overestimation'].sum())
        underestimation_sum = list(estimation_matrix.groupby('Agent')['Underestimation'].sum())
        bar_positions = list(estimation_matrix['Agent'].unique())
        underestimation_sum_minus = [-x for x in underestimation_sum]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=bar_positions, y=underestimation_sum,
                             base=underestimation_sum_minus,
                             marker_color='crimson',
                             name='Underestimation'))
        fig.add_trace(go.Bar(x=bar_positions, y=overestimation_sum,
                             base=0,
                             marker_color='lightslategrey',
                             name='Overestimation'
                             ))

        fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
        fig.show()

        # set a minus in front of underestimation
        estimation_matrix['Underestimation'] = -estimation_matrix['Underestimation']

        # Group by ID and sum the errors
        overestimation_sum = estimation_matrix.groupby('Agent')['Overestimation'].sum()
        underestimation_sum = estimation_matrix.groupby('Agent')['Underestimation'].sum()

        # Create a bar plot
        fig, ax = plt.subplots()
        bar_width = 0.35
        bar_positions = range(len(overestimation_sum))

        # Bar for overestimation (red)
        ax.bar(bar_positions, overestimation_sum, bar_width, color='red', label='Overestimation')

        # Bar for underestimation (blue)
        ax.bar(bar_positions, underestimation_sum, bar_width, color='blue',
               label='Underestimation')

        # Set labels and title
        ax.set_xlabel('ID of the Agent')
        ax.set_ylabel('Sum of Errors (kWh)')
        ax.set_title('Sum of Errors by Agent ID and Estimation Type')
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(overestimation_sum.index, rotation='vertical')  # Rotate agent IDs vertically

        # Add legend
        ax.legend()

        # Show the plot
        plt.show()

        #plt.savefig("Over-and-Underestimation.png")

        data = estimation_matrix.groupby(['Agent', 'Label'])['Balance'].sum().reset_index()

        aggregated_positions = data.loc[data['Label'] == 'aggregated'].index.tolist()
        colors = ['lightslategray', ] * len(bar_positions)
        #colors[aggregated_positions] = 'crimson'
        for index in aggregated_positions:
            colors[index] = 'crimson'

        fig = go.Figure(data=[go.Bar(
            x=list(data['Agent']),
            y=list(data['Balance']),
            marker_color=colors  # marker color can be a single color value or an iterable
        )])
        fig.update_layout(title_text='Balances Distribution')
        fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
        #fig.update_layout(
        #    title='Balances Distribution',
        #    xaxis_tickfont_size=14,
        #    yaxis=dict(
        #        title='Euro (millions)',
        #        titlefont_size=16,
        #        tickfont_size=14,
        #)
        fig.show()


        #colors = ['lightslategray', ] * 5
        #colors[1] = 'crimson'

        #fig = go.Figure(data=[go.Bar(
        #    x=['Feature A', 'Feature B', 'Feature C',
        #       'Feature D', 'Feature E'],
        #    y=[20, 14, 23, 25, 22],
        #    marker_color=colors  # marker color can be a single color value or an iterable
        #)])
        #fig.update_layout(title_text='Least Used Feature')


        for region in self.structure.keys():
            # initialize RegionDB object
            self.region_path = os.path.join(os.path.dirname(self.path), self.structure[region])

            markets_types = f.get_all_subdirectories(os.path.join(self.region_path, 'markets'))
            for markets_type in markets_types:
                path_markts_type = os.path.join(self.region_path, 'markets', markets_type)
                markets = f.get_all_subdirectories(os.path.join(self.region_path, 'markets', markets_type))
                for market in markets:

                    self.market_path = os.path.join(path_markts_type, market)

                    #self.retailer = f.load_file(path=os.path.join(self.market_path, 'retailer.ft'), df='polars')

                    market_transactions = f.load_file(path=os.path.join(self.market_path, 'market_transactions.csv'), df='pandas')

                    agents_types = f.get_all_subdirectories(os.path.join(self.region_path, 'agents'))
                    for agents_type in agents_types:
                        # register agents for each type
                        path_agents_type = os.path.join(self.region_path, 'agents', agents_type)
                        agents = f.get_all_subdirectories(os.path.join(self.region_path, 'agents', agents_type))
                        if agents:
                            for agent in agents:

                                #agent_data = estimation_matrix[(estimation_matrix['Agent'] == agent)]
                                #x = agent_data['Timestamp']

                                #y_over = agent_data['Overestimation']
                                #y_under = agent_data['Underestimation']

                                #plt.step(x, y_over + 2, label='Overestimation')
                                #plt.plot(x, y_over, 'o--', color='grey', alpha=0.3)

                                #plt.step(x, y_under, where='mid', label='Underestimation')
                                #plt.plot(x, y_under, 'o--', color='grey', alpha=0.3)


                                #plt.grid(axis='x', color='0.95')
                                #plt.legend(title='Amount of Energy Error:')
                                #plt.title('Over- and Underestimation per Agent over all Timestamps')
                                #plt.show()



                                #years = [1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003,
                                #         2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012]
                                agent_data = estimation_matrix[(estimation_matrix['Agent'] == agent)]
                                x = list(agent_data['Timestamp'])

                                y = list(agent_data['Balance'])

                                fig = go.Figure()
                                #fig.add_trace(go.Bar(x=years,
                                #                     y=[219, 146, 112, 127, 124, 180, 236, 207, 236, 263,
                                #                        350, 430, 474, 526, 488, 537, 500, 439],
                                #                     name='Rest of world',
                                #                     marker_color='rgb(55, 83, 109)'
                                #                     ))
                                fig.add_trace(go.Bar(x=x,
                                                     y=y,
                                                     name='Total Costs',
                                                     marker_color='rgb(26, 118, 255)'
                                                     ))

                                fig.update_layout(
                                    title='Total Costs of Agents during Simulation',
                                    xaxis_tickfont_size=14,
                                    yaxis=dict(
                                        title='Euro ()',
                                        titlefont_size=16,
                                        tickfont_size=14,
                                    ),
                                    legend=dict(
                                        x=0,
                                        y=1.0,
                                        bgcolor='rgba(255, 255, 255, 0)',
                                        bordercolor='rgba(255, 255, 255, 0)'
                                    ),
                                    barmode='group',
                                    bargap=0.15,  # gap between bars of adjacent location coordinates.
                                    #bargroupgap=0.1  # gap between bars of the same location coordinate.
                                )
                                fig.show()





    def calculate_error_forecats(self):

        for region in self.structure.keys():
            # initialize RegionDB object
            self.region_path = os.path.join(os.path.dirname(self.path), self.structure[region])

            markets_types = f.get_all_subdirectories(os.path.join(self.region_path, 'markets'))
            for markets_type in markets_types:
                path_markts_type = os.path.join(self.region_path, 'markets', markets_type)
                markets = f.get_all_subdirectories(os.path.join(self.region_path, 'markets', markets_type))
                for market in markets:

                    self.market_path = os.path.join(path_markts_type, market)

                    #self.retailer = f.load_file(path=os.path.join(self.market_path, 'retailer.ft'), df='polars')

                    market_transactions = f.load_file(path=os.path.join(self.market_path, 'market_transactions.csv'), df='pandas')

                    agents_types = f.get_all_subdirectories(os.path.join(self.region_path, 'agents'))
                    for agents_type in agents_types:
                        # register agents for each type
                        path_agents_type = os.path.join(self.region_path, 'agents', agents_type)
                        agents = f.get_all_subdirectories(os.path.join(self.region_path, 'agents', agents_type))
                        if agents:
                            for agent in agents:
                                sub_agents = f.get_all_subdirectories(os.path.join(self.region_path, 'agents', agents_type, agent))
                                agents_path = os.path.join(path_agents_type,agent)

                                meters = f.load_file(path=os.path.join(agents_path, 'meters.ft'), df='pandas')
                                forecasts = f.load_file(path=os.path.join(agents_path, 'forecasts_all.ft'), df='pandas')

                                meters = meters.loc[~(meters.iloc[:, 1:] == 0).all(axis=1)]

                                for column in meters:
                                    if column != "timestamp":
                                        meters[column] = self.cumulative_to_normal(list(meters[column]))

                                # only keep those forecasts where timestep and timestamp are the same
                                filtered_forecasts = forecasts.loc[forecasts['timestamp'] == forecasts['timestep']]

                                # remove all the rows where all the meters are zero
                                filtered_meters = meters.loc[~(meters.iloc[:, 1:] == 0).all(axis=1)]

                                # adapt the forecasts timestamps to the meters ones
                                filtered_forecasts = filtered_forecasts[filtered_forecasts['timestamp'].isin(filtered_meters['timestamp'])]

                                # adapt the meters timestamps to the forecat ones
                                filtered_meters = filtered_meters[filtered_meters['timestamp'].isin(filtered_forecasts['timestamp'])]

                                # remove the timestamps
                                filtered_meters = filtered_meters.iloc[:, 1:]
                                filtered_forecasts = filtered_forecasts.iloc[:, 2:]

                                # Get the column names from the meters and add "_power" to them
                                power_columns = [col + "_power" for col in filtered_meters.columns]

                                # Filter out the power_columns that are not present in filtered_forecasts
                                power_columns = [col for col in power_columns if col in filtered_forecasts.columns]

                                # Extract the corresponding columns from filtered_meters
                                forecasts_subset_power = filtered_forecasts[power_columns]

                                forecasts_subset_power.columns = forecasts_subset_power.columns.str.replace('_power', '')

                                # Get the column names from the meters and add "_energy_consumed" to them
                                battery_columns = [col + "_energy_consumed" for col in filtered_meters.columns]

                                # Filter out the battery_columns that are not present in filtered_forecasts
                                battery_columns = [col for col in battery_columns if col in filtered_forecasts.columns]

                                # Extract the corresponding columns from filtered_meters
                                forecasts_subset_battery = filtered_forecasts[battery_columns]

                                forecasts_subset_battery.columns = forecasts_subset_battery.columns.str.replace('_energy_consumed', '')

                                # Get the column names from the meters and add "heat" to them
                                heat_columns = [col + "_heat" for col in filtered_meters.columns]

                                # Filter out the heat_columns that are not present in filtered_forecasts
                                heat_columns = [col for col in heat_columns if col in filtered_forecasts.columns]

                                # Extract the corresponding columns from filtered_meters
                                forecasts_subset_heat = filtered_forecasts[heat_columns]

                                forecasts_subset_heat.columns = forecasts_subset_heat.columns.str.replace('_heat', '')

                                # add them together
                                forecasts_subset = pd.concat([forecasts_subset_heat, forecasts_subset_battery, forecasts_subset_power], axis=1)

                                # Keep only the columns in meters_subset that were found in forecasts_subset
                                meters_subset = filtered_meters[forecasts_subset.columns]

                                # sum them up
                                sum_meters = meters_subset.sum(axis=1)
                                sum_forecasts = forecasts_subset.sum(axis=1)

                                # Define the function to return the MAPE values
                                def calculate_mape(actual, predicted) -> float:

                                    # Convert actual and predicted
                                    # to numpy array data type if not already
                                    if not all([isinstance(actual, np.ndarray),
                                                isinstance(predicted, np.ndarray)]):
                                        actual, predicted = np.array(actual), np.array(predicted)

                                        # Calculate the MAPE value and return
                                    return round(np.mean(np.abs((actual - predicted) / actual)) * 100, 2)

                                def calculate_rmse(actual, predicted) -> float:
                                    # Convert actual and predicted
                                    # to numpy array data type if not already
                                    if not all([isinstance(actual, np.ndarray),
                                                isinstance(predicted, np.ndarray)]):
                                        actual, predicted = np.array(actual), np.array(predicted)

                                    mse = np.square(np.subtract(actual, predicted)).mean()

                                    rmse = math.sqrt(mse)

                                    return rmse

                                def calculate_mae(actual, predicted) -> float:
                                    # Convert actual and predicted
                                    # to numpy array data type if not already
                                    if not all([isinstance(actual, np.ndarray),
                                                isinstance(predicted, np.ndarray)]):
                                        actual, predicted = np.array(actual), np.array(predicted)

                                    n = len(actual)
                                    sum = 0

                                    # for loop for iteration
                                    for i in range(n):
                                        sum += abs(actual[i] - predicted[i])

                                    mae = sum / n

                                    return mae

                                def compute_difference_and_sum(actual, predicted):
                                    # Compute the symmetric difference (unique elements in either list)
                                    diff = actual - predicted

                                    # Separate positive and negative values
                                    positive_values = [x for x in diff if x > 0]
                                    negative_values = [x for x in diff if x < 0]

                                    # Calculate the sum of positive and negative values
                                    sum_positive = sum(positive_values)
                                    sum_negative = sum(negative_values)

                                    return sum_positive, sum_negative

                                mape = calculate_mape(sum_meters,sum_forecasts)

                                rmse = calculate_rmse(sum_meters,sum_forecasts)

                                mae = calculate_mae(sum_meters,sum_forecasts)

                                error_row = [agent, mape, rmse, mae]

                                sum_underestimation, sum_overestimation = compute_difference_and_sum(sum_meters, sum_forecasts)

                                #error_matrix.loc[len(error_matrix.index)] = error_row

    def pairwise(self, iterable):
        # Helper function to create pairs of adjacent elements
        a, b = iter(iterable), iter(iterable)
        next(b, None)  # Advance b by one step
        for x, y in zip(a, b):
            yield x, y

    def cumulative(self, cumulative):
        yield cumulative[0]
        for a, b in self.pairwise(cumulative):
            yield b - a

    def cumulative_to_normal(self, list_cumulative):
        # Get the values from the generator
        values = list(self.cumulative(list_cumulative))

        listd = []

        for val in values:
            listd.append(val)

        return listd

    def _error(self, actual: np.ndarray, predicted: np.ndarray):
        """ Simple error """
        return actual - predicted

    def _percentage_error(self, actual: np.ndarray, predicted: np.ndarray):
        """
        Percentage error
        Note: result is NOT multiplied by 100
        """
        return self._error(actual, predicted) / (actual + self.EPSILON)

    def _naive_forecasting(self, actual: np.ndarray, seasonality: int = 1):
        """ Naive forecasting method which just repeats previous samples """
        return actual[:-seasonality]

    def _relative_error(self,actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
        """ Relative Error """
        if benchmark is None or isinstance(benchmark, int):
            # If no benchmark prediction provided - use naive forecasting
            if not isinstance(benchmark, int):
                seasonality = 1
            else:
                seasonality = benchmark
            return self._error(actual[seasonality:], predicted[seasonality:]) / \
                (self._error(actual[seasonality:], self._naive_forecasting(actual, seasonality)) + self.EPSILON)

        return self._error(actual, predicted) / (self._error(actual, benchmark) + self.EPSILON)

    def _bounded_relative_error(self,actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
        """ Bounded Relative Error """
        if benchmark is None or isinstance(benchmark, int):
            # If no benchmark prediction provided - use naive forecasting
            if not isinstance(benchmark, int):
                seasonality = 1
            else:
                seasonality = benchmark

            abs_err = np.abs(self._error(actual[seasonality:], predicted[seasonality:]))
            abs_err_bench = np.abs(self._error(actual[seasonality:], self._naive_forecasting(actual, seasonality)))
        else:
            abs_err = np.abs(self._error(actual, predicted))
            abs_err_bench = np.abs(self._error(actual, benchmark))

        return abs_err / (abs_err + abs_err_bench + self.EPSILON)

    def _geometric_mean(self,a, axis=0, dtype=None):
        """ Geometric mean """
        if not isinstance(a, np.ndarray):  # if not an ndarray object attempt to convert it
            log_a = np.log(np.array(a, dtype=dtype))
        elif dtype:  # Must change the default dtype allowing array type
            if isinstance(a, np.ma.MaskedArray):
                log_a = np.log(np.ma.asarray(a, dtype=dtype))
            else:
                log_a = np.log(np.asarray(a, dtype=dtype))
        else:
            log_a = np.log(a)
        return np.exp(log_a.mean(axis=axis))

    def mse(self,actual: np.ndarray, predicted: np.ndarray):
        """ Mean Squared Error """
        return np.mean(np.square(self._error(actual, predicted)))

    def rmse(self,actual: np.ndarray, predicted: np.ndarray):
        """ Root Mean Squared Error """
        return np.sqrt(self.mse(actual, predicted))

    def nrmse(self,actual: np.ndarray, predicted: np.ndarray):
        """ Normalized Root Mean Squared Error """
        return self.rmse(actual, predicted) / (actual.max() - actual.min())

    def me(self,actual: np.ndarray, predicted: np.ndarray):
        """ Mean Error """
        return np.mean(self._error(actual, predicted))

    def mae(self,actual: np.ndarray, predicted: np.ndarray):
        """ Mean Absolute Error """
        return np.mean(np.abs(self._error(actual, predicted)))

    def mad(self,actual: np.ndarray, predicted: np.ndarray):
        """ Mean Absolute Deviation """
        error = self._error(actual, predicted)
        return np.mean(np.abs(error - np.mean(error)))

    def gmae(self,actual: np.ndarray, predicted: np.ndarray):
        """ Geometric Mean Absolute Error """
        return self._geometric_mean(np.abs(self._error(actual, predicted)))

    def mdae(self,actual: np.ndarray, predicted: np.ndarray):
        """ Median Absolute Error """
        return np.median(np.abs(self._error(actual, predicted)))

    def mpe(self,actual: np.ndarray, predicted: np.ndarray):
        """ Mean Percentage Error """
        return np.mean(self._percentage_error(actual, predicted))

    def mape(self,actual: np.ndarray, predicted: np.ndarray):
        """
        Mean Absolute Percentage Error
        Properties:
            + Easy to interpret
            + Scale independent
            - Biased, not symmetric
            - Undefined when actual[t] == 0
        Note: result is NOT multiplied by 100
        """
        return np.mean(np.abs(self._percentage_error(actual, predicted)))

    def mdape(self,actual: np.ndarray, predicted: np.ndarray):
        """
        Median Absolute Percentage Error
        Note: result is NOT multiplied by 100
        """
        return np.median(np.abs(self._percentage_error(actual, predicted)))

    def smape(self,actual: np.ndarray, predicted: np.ndarray):
        """
        Symmetric Mean Absolute Percentage Error
        Note: result is NOT multiplied by 100
        """
        return np.mean(2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)) + self.EPSILON))

    def smdape(self,actual: np.ndarray, predicted: np.ndarray):
        """
        Symmetric Median Absolute Percentage Error
        Note: result is NOT multiplied by 100
        """
        return np.median(2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)) + self.EPSILON))

    def maape(self,actual: np.ndarray, predicted: np.ndarray):
        """
        Mean Arctangent Absolute Percentage Error
        Note: result is NOT multiplied by 100
        """
        return np.mean(np.arctan(np.abs((actual - predicted) / (actual + self.EPSILON))))

    def mase(self,actual: np.ndarray, predicted: np.ndarray, seasonality: int = 1):
        """
        Mean Absolute Scaled Error
        Baseline (benchmark) is computed with naive forecasting (shifted by @seasonality)
        """
        return self.mae(actual, predicted) / self.mae(actual[seasonality:], self._naive_forecasting(actual, seasonality))

    def std_ae(self,actual: np.ndarray, predicted: np.ndarray):
        """ Normalized Absolute Error """
        __mae = self.mae(actual, predicted)
        return np.sqrt(np.sum(np.square(self._error(actual, predicted) - __mae)) / (len(actual) - 1))

    def std_ape(self,actual: np.ndarray, predicted: np.ndarray):
        """ Normalized Absolute Percentage Error """
        __mape = self.mape(actual, predicted)
        return np.sqrt(np.sum(np.square(self._percentage_error(actual, predicted) - __mape)) / (len(actual) - 1))

    def rmspe(self,actual: np.ndarray, predicted: np.ndarray):
        """
        Root Mean Squared Percentage Error
        Note: result is NOT multiplied by 100
        """
        return np.sqrt(np.mean(np.square(self._percentage_error(actual, predicted))))

    def rmdspe(self,actual: np.ndarray, predicted: np.ndarray):
        """
        Root Median Squared Percentage Error
        Note: result is NOT multiplied by 100
        """
        return np.sqrt(np.median(np.square(self._percentage_error(actual, predicted))))

    def rmsse(self,actual: np.ndarray, predicted: np.ndarray, seasonality: int = 1):
        """ Root Mean Squared Scaled Error """
        q = np.abs(self._error(actual, predicted)) / self.mae(actual[seasonality:], self._naive_forecasting(actual, seasonality))
        return np.sqrt(np.mean(np.square(q)))

    def inrse(self,actual: np.ndarray, predicted: np.ndarray):
        """ Integral Normalized Root Squared Error """
        return np.sqrt(np.sum(np.square(self._error(actual, predicted))) / np.sum(np.square(actual - np.mean(actual))))

    def rrse(self,actual: np.ndarray, predicted: np.ndarray):
        """ Root Relative Squared Error """
        return np.sqrt(np.sum(np.square(actual - predicted)) / np.sum(np.square(actual - np.mean(actual))))

    def mre(self,actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
        """ Mean Relative Error """
        return np.mean(self._relative_error(actual, predicted, benchmark))

    def rae(self,actual: np.ndarray, predicted: np.ndarray):
        """ Relative Absolute Error (aka Approximation Error) """
        return np.sum(np.abs(actual - predicted)) / (np.sum(np.abs(actual - np.mean(actual))) + self.EPSILON)

    def mrae(self,actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
        """ Mean Relative Absolute Error """
        return np.mean(np.abs(self._relative_error(actual, predicted, benchmark)))

    def mdrae(self,actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
        """ Median Relative Absolute Error """
        return np.median(np.abs(self._relative_error(actual, predicted, benchmark)))

    def gmrae(self,actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
        """ Geometric Mean Relative Absolute Error """
        return self._geometric_mean(np.abs(self._relative_error(actual, predicted, benchmark)))

    def mbrae(self,actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
        """ Mean Bounded Relative Absolute Error """
        return np.mean(self._bounded_relative_error(actual, predicted, benchmark))

    def umbrae(self,actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
        """ Unscaled Mean Bounded Relative Absolute Error """
        __mbrae = self.mbrae(actual, predicted, benchmark)
        return __mbrae / (1 - __mbrae)

    def mda(self,actual: np.ndarray, predicted: np.ndarray):
        """ Mean Directional Accuracy """
        return np.mean((np.sign(actual[1:] - actual[:-1]) == np.sign(predicted[1:] - actual[:-1])).astype(int))