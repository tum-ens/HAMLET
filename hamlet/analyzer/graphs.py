

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
import plotly.io as plt_io
import ast
import sys
import os.path as path
import statistics
from datetime import datetime
sys.path.append("..")  # Add the parent directory to the Python path for execution outside an IDEimport sys
sys.path.append("./")  # Add the current directory to the Python path for execution in VSCode

# create our custom_dark theme from the plotly_dark template
plt_io.templates["custom"] = plt_io.templates["simple_white"]

plt_io.templates['custom']['layout']['yaxis']['showgrid'] = True
plt_io.templates['custom']['layout']['xaxis']['showgrid'] = True
plt_io.templates['custom']['layout']['font']['size'] = 20
plt_io.templates['custom']['layout']['legend']['y'] = 1.02
plt_io.templates['custom']['layout']['legend']['yanchor'] = 'bottom'
plt_io.templates['custom']['layout']['legend']['orientation'] = 'h'
plt_io.templates['custom']['layout']['legend']['xanchor'] = 'right'
plt_io.templates['custom']['layout']['legend']['x'] = 1
plt_io.templates['custom']['layout']['legend']['font_size'] = 40




# Path to the scenario folder (relative or absolute)
list_scenarios = ["individual", "prosumer", "consumer", "mixed"]
list_seasons = ["summer", "transition", "winter"]
list_group_size = ["5", "10", "15"]
list_forecasting_methods = ["naive", "perfect"]

two_levels_up = path.abspath(path.join(__file__, "..", "..", ".."))

class ndict(dict):
    def __getitem__(self, key):
        if key in self: return self.get(key)
        return self.setdefault(key, ndict())



## COSTS PLOTS ##


for scenario in list_scenarios:
    if not scenario == 'individual':
        dict_season = ndict({'balance': ndict({'summer': {}, 'transition': {}, 'winter': {}}),
                             'median': ndict({'summer': {}, 'transition': {}, 'winter': {}})})
        dict_group_aggregated = ndict({'balance': ndict({'summer': {}, 'transition': {}, 'winter': {}}),
                                       'median': ndict({'summer': {}, 'transition': {}, 'winter': {}})})
        dict_group_consumer = ndict({'balance': ndict({'summer': {}, 'transition': {}, 'winter': {}}),
                                     'median': ndict({'summer': {}, 'transition': {}, 'winter': {}})})
        dict_group_prosumer = ndict({'balance': ndict({'summer': {}, 'transition': {}, 'winter': {}}),
                                     'median': ndict({'summer': {}, 'transition': {}, 'winter': {}})})
        for season in list_seasons:
            group_list_naive = np.array([])
            aggregated_list_naive = np.array([])
            consumers_list_naive = np.array([])
            prosumers_list_naive = np.array([])

            group_list_perfect = np.array([])
            aggregated_list_perfect = np.array([])
            consumers_list_perfect = np.array([])
            prosumers_list_perfect = np.array([])

            median_naiv = np.array([])
            median_perf = np.array([])

            for group in list_group_size:

                for forecasting_method in list_forecasting_methods:

                    if forecasting_method == "naive":

                        path = "05_results/scenario_pv_60_hp_60_ev_60_" + scenario + "_" + group + "_" + season + "_" + forecasting_method
                        path_individual = "05_results/scenario_pv_60_hp_60_ev_60_individual_" + season + "_" + forecasting_method

                        estimation_matrix_individual = f.load_file(path=os.path.join(two_levels_up, path_individual, 'general', 'estimation_matrix.csv'),df='pandas')
                        estimation_matrix = f.load_file(path=os.path.join(two_levels_up, path, 'general', 'estimation_matrix.csv'), df='pandas')

                        # Filter rows with desired labels
                        estimation_matrix = estimation_matrix[
                            estimation_matrix['Label'].isin(['aggregated', 'consumer', 'prosumer'])]
                        estimation_matrix_individual = estimation_matrix_individual[
                            estimation_matrix_individual['Label'].isin(['aggregated', 'consumer', 'prosumer'])]


                        estimation_matrix_agent = estimation_matrix.groupby('Agent').agg({'Timestamp': 'first',
                                                  'Label': 'first',
                                                  'Underestimation': 'sum',
                                                  'Overestimation': 'sum',
                                                  'Benefits': 'sum',
                                                  'Benefits_Market': 'sum',
                                                    'Benefits_Balancing': 'sum',
                                                    'Costs': 'sum',
                                                     'Costs_Market': 'sum',
                                                     'Costs_Balancing': 'sum',
                                                     'Costs_Fees': 'sum',
                                                  'Balance': 'sum'})

                        if group == '5':
                            estimation_matrix_individual = estimation_matrix_individual.groupby('Agent').agg({'Timestamp': 'first',
                                                                                              'Label': 'first',
                                                                                              'Underestimation': 'sum',
                                                                                              'Overestimation': 'sum',
                                                                                              'Benefits': 'sum',
                                                                                              'Benefits_Market': 'sum',
                                                                                                'Benefits_Balancing': 'sum',
                                                                                                'Costs': 'sum',
                                                                                                 'Costs_Market': 'sum',
                                                                                                 'Costs_Balancing': 'sum',
                                                                                                 'Costs_Fees': 'sum',
                                                                                              'Balance': 'sum'})

                            group_list_naive = np.append(group_list_naive, estimation_matrix_individual['Balance'].mean())
                            median_naiv = np.append(median_naiv, statistics.median(estimation_matrix_individual['Balance']))

                        aggregated_list_naive = np.append(aggregated_list_naive,
                                                          estimation_matrix_agent.groupby('Label').get_group(
                                                              'aggregated')['Balance'].mean())
                        consumers_list_naive = np.append(consumers_list_naive,
                                                         estimation_matrix_agent.groupby('Label').get_group('consumer')[
                                                             'Balance'].mean())
                        prosumers_list_naive = np.append(prosumers_list_naive,
                                                         estimation_matrix_agent.groupby('Label').get_group('prosumer')[
                                                             'Balance'].mean())

                        group_list_naive = np.append(group_list_naive, estimation_matrix_agent['Balance'].mean())
                        median_naiv = np.append(median_naiv,statistics.median(estimation_matrix_agent['Balance']))


                    if forecasting_method == "perfect":
                        path = "05_results/scenario_pv_60_hp_60_ev_60_" + scenario + "_" + group + "_" + season + "_" + forecasting_method
                        path_individual = "05_results/scenario_pv_60_hp_60_ev_60_individual_" + season + "_" + forecasting_method

                        estimation_matrix_individual = f.load_file(path=os.path.join(two_levels_up, path_individual, 'general', 'estimation_matrix.csv'), df='pandas')
                        estimation_matrix = f.load_file(path=os.path.join(two_levels_up, path, 'general', 'estimation_matrix.csv'),df='pandas')

                        # Filter rows with desired labels
                        estimation_matrix = estimation_matrix[
                            estimation_matrix['Label'].isin(['aggregated', 'consumer', 'prosumer'])]
                        estimation_matrix_individual = estimation_matrix_individual[
                            estimation_matrix_individual['Label'].isin(['aggregated', 'consumer', 'prosumer'])]

                        estimation_matrix_agent = estimation_matrix.groupby('Agent').agg({'Timestamp': 'first',
                                                  'Label': 'first',
                                                  'Underestimation': 'sum',
                                                  'Overestimation': 'sum',
                                                  'Benefits': 'sum',
                                                  'Benefits_Market': 'sum',
                                                    'Benefits_Balancing': 'sum',
                                                    'Costs': 'sum',
                                                     'Costs_Market': 'sum',
                                                     'Costs_Balancing': 'sum',
                                                     'Costs_Fees': 'sum',
                                                  'Balance': 'sum'})

                        if group == '5':
                            estimation_matrix_individual = estimation_matrix_individual.groupby('Agent').agg(
                                {'Timestamp': 'first',
                                 'Label': 'first',
                                  'Underestimation': 'sum',
                                  'Overestimation': 'sum',
                                  'Benefits': 'sum',
                                  'Benefits_Market': 'sum',
                                    'Benefits_Balancing': 'sum',
                                    'Costs': 'sum',
                                     'Costs_Market': 'sum',
                                     'Costs_Balancing': 'sum',
                                     'Costs_Fees': 'sum',
                                  'Balance': 'sum'})

                            group_list_perfect = np.append(group_list_perfect,
                                                         estimation_matrix_individual['Balance'].mean())

                            median_perf = np.append(median_perf, statistics.median(estimation_matrix_individual['Balance']))

                        group_list_perfect = np.append(group_list_perfect, estimation_matrix_agent['Balance'].mean())

                        median_perf = np.append(median_perf, statistics.median(estimation_matrix_agent['Balance']))

                        aggregated_list_perfect = np.append(aggregated_list_perfect,
                                                            estimation_matrix_agent.groupby('Label').get_group(
                                                                'aggregated')['Balance'].mean())
                        consumers_list_perfect = np.append(consumers_list_perfect,
                                                           estimation_matrix_agent.groupby('Label').get_group(
                                                               'consumer')[
                                                               'Balance'].mean())
                        prosumers_list_perfect = np.append(prosumers_list_perfect,
                                                           estimation_matrix_agent.groupby('Label').get_group(
                                                               'prosumer')[
                                                               'Balance'].mean())

            dict_season['balance'][season] = [- i + j for i, j in zip(group_list_naive, group_list_perfect)]
            dict_group_aggregated['balance'][season] = [- i + j for i, j in zip(aggregated_list_naive, aggregated_list_perfect)]
            dict_group_consumer['balance'][season] = [- i + j for i, j in zip(consumers_list_naive, consumers_list_perfect)]
            dict_group_prosumer['balance'][season] = [- i + j for i, j in zip(prosumers_list_naive, prosumers_list_perfect)]

            dict_season['median'][season] = [- i + j for i, j in zip(median_naiv, median_perf)]

        x1 = np.array(['1', '5', '10', '15'])
        x = np.array(['5', '10', '15'])

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=x1, y=dict_season['median']['summer'], name="summer",
                                 text=["tweak line smoothness<br>with 'smoothing' in line object"],
                                 hoverinfo='text+name'))
        fig.add_trace(go.Scatter(x=x1, y=dict_season['median']['winter'], name="winter",
                                 text=["tweak line smoothness<br>with 'smoothing' in line object"],
                                 hoverinfo='text+name'))
        fig.add_trace(go.Scatter(x=x1, y=dict_season['median']['transition'], name="transition",
                                 text=["tweak line smoothness<br>with 'smoothing' in line object"],
                                 hoverinfo='text+name'))

        fig.update_traces(hoverinfo='text+name', mode='lines+markers')
        res = scenario[0].upper() + scenario[1:]
        title = res + '-Group Forecast Error Costs per Agent'
        # Edit the layout
        fig.layout.template = 'custom'
        fig.update_layout(title=title,
                          xaxis_title='Group Size of the Aggregator',
                          yaxis_title='Median Forecast Error Costs (EUR)')

        fig.show()

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=x1, y=dict_season['balance']['summer'], name="summer",
                                 text=["tweak line smoothness<br>with 'smoothing' in line object"],
                                 hoverinfo='text+name'))
        fig.add_trace(go.Scatter(x=x1, y=dict_season['balance']['winter'], name="winter",
                                 text=["tweak line smoothness<br>with 'smoothing' in line object"],
                                 hoverinfo='text+name'))
        fig.add_trace(go.Scatter(x=x1, y=dict_season['balance']['transition'], name="transition",
                                 text=["tweak line smoothness<br>with 'smoothing' in line object"],
                                 hoverinfo='text+name'))

        fig.update_traces(hoverinfo='text+name', mode='lines+markers')
        res = scenario[0].upper() + scenario[1:]
        title = res + '-Group Forecast Error Costs per Agent'
        # Edit the layout
        fig.layout.template = 'custom'
        fig.update_layout(title=title,
                          xaxis_title='Group Size of the Aggregator',
                          yaxis_title='Mean Forecast Error Costs (EUR)')

        fig.show()

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=x, y=dict_group_aggregated['balance']['summer'], name="summer",
                                 text=["tweak line smoothness<br>with 'smoothing' in line object"],
                                 hoverinfo='text+name'))
        fig.add_trace(go.Scatter(x=x, y=dict_group_aggregated['balance']['winter'], name="winter",
                                 text=["tweak line smoothness<br>with 'smoothing' in line object"],
                                 hoverinfo='text+name'))
        fig.add_trace(go.Scatter(x=x, y=dict_group_aggregated['balance']['transition'], name="transition",
                                 text=["tweak line smoothness<br>with 'smoothing' in line object"],
                                 hoverinfo='text+name'))

        fig.update_traces(hoverinfo='text+name', mode='lines+markers')
        res = scenario[0].upper() + scenario[1:]
        title = res + '-Group Forecast Error Costs for Aggregated Agents per Agent'
        # Edit the layout
        fig.layout.template = 'custom'
        fig.update_layout(title=title,
                          xaxis_title='Group Size of the Aggregator',
                          yaxis_title='Mean Forecast Error Costs (EUR)')

        fig.show()

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=x, y=dict_group_consumer['balance']['summer'], name="summer",
                                 text=["tweak line smoothness<br>with 'smoothing' in line object"],
                                 hoverinfo='text+name'))
        fig.add_trace(go.Scatter(x=x, y=dict_group_consumer['balance']['winter'], name="winter",
                                 text=["tweak line smoothness<br>with 'smoothing' in line object"],
                                 hoverinfo='text+name'))
        fig.add_trace(go.Scatter(x=x, y=dict_group_consumer['balance']['transition'], name="transition",
                                 text=["tweak line smoothness<br>with 'smoothing' in line object"],
                                 hoverinfo='text+name'))

        fig.update_traces(hoverinfo='text+name', mode='lines+markers')
        res = scenario[0].upper() + scenario[1:]
        title = res + '-Group Forecast Error Costs for Not Aggregated Consumers'
        # Edit the layout
        fig.layout.template = 'custom'
        fig.update_layout(title=title,
                          xaxis_title='Group Size of the Aggregator',
                          yaxis_title='Mean Forecast Error Costs (EUR)')

        fig.show()

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=x, y=dict_group_prosumer['balance']['summer'], name="summer",
                                 text=["tweak line smoothness<br>with 'smoothing' in line object"],
                                 hoverinfo='text+name'))
        fig.add_trace(go.Scatter(x=x, y=dict_group_prosumer['balance']['winter'], name="winter",
                                 text=["tweak line smoothness<br>with 'smoothing' in line object"],
                                 hoverinfo='text+name'))
        fig.add_trace(go.Scatter(x=x, y=dict_group_prosumer['balance']['transition'], name="transition",
                                 text=["tweak line smoothness<br>with 'smoothing' in line object"],
                                 hoverinfo='text+name'))

        fig.update_traces(hoverinfo='text+name', mode='lines+markers')
        res = scenario[0].upper() + scenario[1:]
        title = res + '-Group Forecast Error Costs for Not Aggregated Prosumers per Agent'
        # Edit the layout
        fig.layout.template = 'custom'
        fig.update_layout(title=title,
                          xaxis_title='Group Size of the Aggregator',
                          yaxis_title='Mean Forecast Error Costs (EUR)')

        fig.show()


# create our custom_dark theme from the plotly_dark template
plt_io.templates["custom"] = plt_io.templates["simple_white"]

plt_io.templates['custom']['layout']['yaxis']['showgrid'] = True
plt_io.templates['custom']['layout']['xaxis']['showgrid'] = True
plt_io.templates['custom']['layout']['font']['size'] = 20

plt_io.templates['custom']['layout']['legend']['y'] = 1.02
plt_io.templates['custom']['layout']['legend']['yanchor'] = 'bottom'
plt_io.templates['custom']['layout']['legend']['orientation'] = 'h'
plt_io.templates['custom']['layout']['legend']['xanchor'] = 'right'
plt_io.templates['custom']['layout']['legend']['x'] = 1
plt_io.templates['custom']['layout']['legend']['font_size'] = 35

df = pd.DataFrame(columns=['Agent', 'Label', 'Season', 'Group'])
for season in list_seasons:

    for forecasting_method in list_forecasting_methods:

        if forecasting_method == "naive":

            path_individual = "05_results/scenario_pv_60_hp_60_ev_60_individual_" + season + "_" + forecasting_method

            estimation_matrix_individual = f.load_file(
                path=os.path.join(two_levels_up, path_individual, 'general', 'estimation_matrix.csv'),
                df='pandas')

            estimation_matrix_individual = estimation_matrix_individual.groupby('Agent').agg(
                {'Timestamp': 'first',
                 'Label': 'first',
                 'Underestimation': 'sum',
                 'Overestimation': 'sum',
                 'Benefits': 'sum',
                 'Benefits_Market': 'sum',
                 'Benefits_Balancing': 'sum',
                 'Costs': 'sum',
                 'Costs_Market': 'sum',
                 'Costs_Balancing': 'sum',
                 'Costs_Fees': 'sum',
                 'Balance': 'sum'}).reset_index()

            naive = list(estimation_matrix_individual['Balance'])


        if forecasting_method == "perfect":

            path_individual = "05_results/scenario_pv_60_hp_60_ev_60_individual_" + season + "_" + forecasting_method

            estimation_matrix_individual = f.load_file(
                path=os.path.join(two_levels_up, path_individual, 'general', 'estimation_matrix.csv'),
                df='pandas')

            estimation_matrix_individual = estimation_matrix_individual.groupby('Agent').agg(
                {'Timestamp': 'first',
                 'Label': 'first',
                 'Underestimation': 'sum',
                 'Overestimation': 'sum',
                 'Benefits': 'sum',
                 'Benefits_Market': 'sum',
                 'Benefits_Balancing': 'sum',
                 'Costs': 'sum',
                 'Costs_Market': 'sum',
                 'Costs_Balancing': 'sum',
                 'Costs_Fees': 'sum',
                 'Balance': 'sum'}).reset_index()

            perfect = list(estimation_matrix_individual['Balance'])

    balance = [- i + j for i, j in zip(naive, perfect)]

    group1 = '1'
    grouped_df_individual = estimation_matrix_individual[['Agent', 'Label']]
    grouped_df_individual["balance"] = balance
    grouped_df_individual["Season"] = season
    grouped_df_individual["Group"] = group1
    df = pd.concat([df, grouped_df_individual], ignore_index=True)

fig = px.bar(df, x="Group", y="balance", color="Label",facet_col="Season",category_orders={"Season": ["summer", "transition", "winter"]})
title = scenario.capitalize() + '-Group ' + 'Forecast Error Costs for Individual Trading per Agent'
fig.layout.template = 'custom'
fig.update_layout(
    title=title, yaxis_title="Amount in Euro")

fig.show()





for scenario in list_scenarios:
    if not scenario == 'individual':
        for season in list_seasons:

            for group in list_group_size:

                for forecasting_method in list_forecasting_methods:

                    if forecasting_method == "naive":
                        path = "05_results/scenario_pv_60_hp_60_ev_60_" + scenario + "_" + group + "_" + season + "_" + forecasting_method

                        estimation_matrix = f.load_file(
                            path=os.path.join(two_levels_up, path, 'general', 'estimation_matrix.csv'), df='pandas')

                        estimation_matrix['Total_Error'] = estimation_matrix['Overestimation'] + estimation_matrix[
                            'Underestimation']


                        # Filter rows with desired labels
                        filtered_df = estimation_matrix[
                            estimation_matrix['Label'].isin(['aggregated', 'consumer', 'prosumer'])]

                        filtered_df['Label'] = filtered_df['Label'].str.capitalize()

                        filtered_df['Combined'] = filtered_df['Label'] + ' ' + filtered_df['Agent']

                        # Group and aggregate by Timestamp and Agent_ID
                        grouped_df = filtered_df.groupby(['Timestamp', 'Combined'])['Total_Error'].sum().reset_index()

                        # Pivot the dataframe
                        pivoted_df = grouped_df.pivot(index='Timestamp', columns='Combined', values='Total_Error')

                        pivoted_df = pivoted_df.T

                        #pivoted_df = pivoted_df.sort_index()

                        #pivoted_df.index = pd.to_datetime(pivoted_df.index, format='%Y-%m-%d %H:%M:%S%z')

                        #pls = pivoted_df.index.to_pydatetime()

                        fig = go.Figure(data=go.Heatmap(
                            z=pivoted_df,
                            x=pivoted_df.columns,
                            y=pivoted_df.index,
                            colorscale='Viridis'))

                        title ='Heatmap' + ' for the ' + scenario.capitalize() + ' Szenario in  '+ season.capitalize()  + ' for an Aggregator of  '+ group

                        fig.update_layout(
                            title=title)

                        fig.update_layout(
                            font=dict(size=30)
                        )

                        fig.update_yaxes(tickfont=dict(size= 15))

                        fig.show()

                        h0= 5

    else:
        for season in list_seasons:

            for forecasting_method in list_forecasting_methods:

                if forecasting_method == "naive":

                    path_individual = "05_results/scenario_pv_60_hp_60_ev_60_individual_" + season + "_" + forecasting_method

                    estimation_matrix = f.load_file(
                        path=os.path.join(two_levels_up, path_individual, 'general', 'estimation_matrix.csv'), df='pandas')

                    estimation_matrix['Total_Error'] = estimation_matrix['Overestimation'] + estimation_matrix['Underestimation']

                    # Filter rows with desired labels
                    filtered_df = estimation_matrix[estimation_matrix['Label'].isin(['aggregated', 'consumer', 'prosumer'])]

                    filtered_df['Label'] = filtered_df['Label'].str.capitalize()

                    filtered_df['Combined'] = filtered_df['Label'] + ' ' + filtered_df['Agent']

                    # Group and aggregate by Timestamp and Agent_ID
                    grouped_df = filtered_df.groupby(['Timestamp', 'Combined'])['Total_Error'].sum().reset_index()

                    # Pivot the dataframe
                    pivoted_df = grouped_df.pivot(index='Timestamp', columns='Combined', values='Total_Error')

                    pivoted_df = pivoted_df.T

                    #pivoted_df = pivoted_df.sort_index()

                    #pivoted_df.index = pd.to_datetime(pivoted_df.index, format='%Y-%m-%d %H:%M:%S%z')

                    #pivoted_df.index = pivoted_df.index.to_pydatetime()


                    fig = go.Figure(data=go.Heatmap(
                        z=pivoted_df,
                        x=pivoted_df.columns,
                        y=pivoted_df.index,
                        colorscale='Viridis'))

                    title = 'Heatmap' + ' for the ' + scenario.capitalize() + '-Szenario in '+ season.capitalize()

                    fig.update_layout(
                        title=title)

                    fig.update_layout(
                        font=dict(size=30)
                    )

                    fig.update_yaxes(tickfont=dict(size= 15))


                    fig.show()


# create our custom_dark theme from the plotly_dark template
plt_io.templates["custom"] = plt_io.templates["simple_white"]

plt_io.templates['custom']['layout']['yaxis']['showgrid'] = True
plt_io.templates['custom']['layout']['xaxis']['showgrid'] = True
plt_io.templates['custom']['layout']['font']['size'] = 35
plt_io.templates['custom']['layout']['legend']['y'] = 1.2
plt_io.templates['custom']['layout']['legend']['yanchor'] = 'bottom'
plt_io.templates['custom']['layout']['legend']['orientation'] = 'h'
plt_io.templates['custom']['layout']['legend']['xanchor'] = 'right'
plt_io.templates['custom']['layout']['legend']['x'] = 1
plt_io.templates['custom']['layout']['legend']['font_size'] = 35



Measurements = ['Benefits_Market', 'Benefits_Balancing', 'Costs_Market', 'Costs_Balancing', 'Costs_Fees']
for scenario in list_scenarios:
    if not scenario == 'individual':
        df = pd.DataFrame(columns=['Group', 'Benefits/Costs subcategories', 'Season', 'Benefits/Costs in Euro'])
        for season in list_seasons:

            for group in list_group_size:

                for forecasting_method in list_forecasting_methods:

                    if forecasting_method == "naive":
                        path = "05_results/scenario_pv_60_hp_60_ev_60_" + scenario + "_" + group + "_" + season + "_" + forecasting_method
                        path_individual = "05_results/scenario_pv_60_hp_60_ev_60_individual_" + season + "_" + forecasting_method

                        estimation_matrix_individual = f.load_file(
                            path=os.path.join(two_levels_up, path_individual, 'general', 'estimation_matrix.csv'),
                            df='pandas')
                        estimation_matrix = f.load_file(
                            path=os.path.join(two_levels_up, path, 'general', 'estimation_matrix.csv'), df='pandas')

                        # Filter rows with desired labels
                        estimation_matrix = estimation_matrix[
                            estimation_matrix['Label'].isin(['aggregated', 'consumer', 'prosumer'])]
                        estimation_matrix_individual = estimation_matrix_individual[
                            estimation_matrix_individual['Label'].isin(['aggregated', 'consumer', 'prosumer'])]

                        if group == '5':
                            for measure in Measurements:
                                amount = estimation_matrix_individual[measure].mean()
                                if "Costs" in measure:
                                    amount = -amount

                                group1 = '1'
                                df_row = [group1, measure, season, amount]

                                df.loc[len(df.index)] = df_row

                        for measure in Measurements:

                            amount = estimation_matrix[measure].mean()

                            if "Costs" in measure:
                                amount = -amount

                            df_row = [group, measure, season, amount]

                            df.loc[len(df.index)] = df_row

        fig = px.bar(df, x="Group", y="Benefits/Costs in Euro", color="Benefits/Costs subcategories", barmode="relative", facet_col="Season",
                     category_orders={"Season": ["summer", "transition", "winter"]})
        fig.layout.template = 'custom'
        col = 1
        for season in list_seasons:
            x3 = 0.5
            for group in list_group_size:

                for forecasting_method in list_forecasting_methods:

                    if forecasting_method == "naive":
                        path = "05_results/scenario_pv_60_hp_60_ev_60_" + scenario + "_" + group + "_" + season + "_" + forecasting_method
                        path_individual = "05_results/scenario_pv_60_hp_60_ev_60_individual_" + season + "_" + forecasting_method

                        estimation_matrix_individual = f.load_file(
                            path=os.path.join(two_levels_up, path_individual, 'general', 'estimation_matrix.csv'),
                            df='pandas')
                        estimation_matrix = f.load_file(
                            path=os.path.join(two_levels_up, path, 'general', 'estimation_matrix.csv'), df='pandas')

                        # Filter rows with desired labels
                        estimation_matrix = estimation_matrix[
                            estimation_matrix['Label'].isin(['aggregated', 'consumer', 'prosumer'])]
                        estimation_matrix_individual = estimation_matrix_individual[
                            estimation_matrix_individual['Label'].isin(['aggregated', 'consumer', 'prosumer'])]

                        amount = estimation_matrix['Balance'].mean()

                        x4 = x3 + 1
                        fig.add_shape(
                            type='line', line={'color': 'rgb(50, 171, 96)', 'width': 4},
                            xref="x",
                            yref="y",
                            x0=x3,
                            x1=x4,
                            y0=amount,
                            y1=amount,
                            row=1,
                            col=col
                        )
                        x3 = x3 + 1

                        if group == '5':

                            amount = estimation_matrix_individual['Balance'].mean()

                            x1 = -0.5
                            x2 = 0.5
                            if season == "summer":
                                fig.add_shape(
                                    name="Balance",
                                    showlegend=True,
                                    type='line', line={'color': 'rgb(50, 171, 96)','width': 4},
                                    xref="x",
                                    yref="y",
                                    x0=x1,
                                    x1=x2,
                                    y0=amount,
                                    y1=amount,
                                    row=1,
                                    col=col
                                )
                            else:
                                fig.add_shape(
                                    name="Balance",
                                    type='line', line={'color': 'rgb(50, 171, 96)', 'width': 4},
                                    xref="x",
                                    yref="y",
                                    x0=x1,
                                    x1=x2,
                                    y0=amount,
                                    y1=amount,
                                    row=1,
                                    col=col
                                )

            col = col + 1

        title = scenario.capitalize() + '-Group ' + 'Benefits/Costs per Agent per Timestamp over all Groups at Naive Forecasting'

        fig.update_layout(yaxis_title="Amount in Euro"
        )
        fig.show()


Measurements = ['Benefits_Market', 'Benefits_Balancing', 'Costs_Market', 'Costs_Balancing', 'Costs_Fees']
for scenario in list_scenarios:
    if not scenario == 'individual':
        df = pd.DataFrame(columns=['Group', 'Benefits/Costs subcategories', 'Season', 'Benefits/Costs in Euro'])
        for season in list_seasons:

            for group in list_group_size:

                for forecasting_method in list_forecasting_methods:

                    if forecasting_method == "perfect":
                        path = "05_results/scenario_pv_60_hp_60_ev_60_" + scenario + "_" + group + "_" + season + "_" + forecasting_method
                        path_individual = "05_results/scenario_pv_60_hp_60_ev_60_individual_" + season + "_" + forecasting_method

                        estimation_matrix_individual = f.load_file(
                            path=os.path.join(two_levels_up, path_individual, 'general', 'estimation_matrix.csv'),
                            df='pandas')
                        estimation_matrix = f.load_file(
                            path=os.path.join(two_levels_up, path, 'general', 'estimation_matrix.csv'), df='pandas')

                        # Filter rows with desired labels
                        estimation_matrix = estimation_matrix[
                            estimation_matrix['Label'].isin(['aggregated', 'consumer', 'prosumer'])]
                        estimation_matrix_individual = estimation_matrix_individual[
                            estimation_matrix_individual['Label'].isin(['aggregated', 'consumer', 'prosumer'])]

                        if group =='5':
                            for measure in Measurements:
                                amount = estimation_matrix_individual[measure].mean()
                                if "Costs" in measure:
                                    amount = -amount
                                group1 = '1'
                                df_row = [group1, measure, season, amount]

                                df.loc[len(df.index)] = df_row


                        for measure in Measurements:

                            amount = estimation_matrix[measure].mean()
                            if "Costs" in measure:
                                amount = -amount

                            df_row = [group, measure, season, amount]

                            df.loc[len(df.index)] = df_row


        fig = px.bar(df, x="Group", y="Benefits/Costs in Euro", color="Benefits/Costs subcategories", barmode="relative", facet_col="Season",
                     category_orders={"Season": ["summer", "transition", "winter"]})
        fig.layout.template = 'custom'
        col = 1
        for season in list_seasons:
            x3 = 0.5
            for group in list_group_size:

                for forecasting_method in list_forecasting_methods:

                    if forecasting_method == "perfect":
                        path = "05_results/scenario_pv_60_hp_60_ev_60_" + scenario + "_" + group + "_" + season + "_" + forecasting_method
                        path_individual = "05_results/scenario_pv_60_hp_60_ev_60_individual_" + season + "_" + forecasting_method

                        estimation_matrix_individual = f.load_file(
                            path=os.path.join(two_levels_up, path_individual, 'general', 'estimation_matrix.csv'),
                            df='pandas')
                        estimation_matrix = f.load_file(
                            path=os.path.join(two_levels_up, path, 'general', 'estimation_matrix.csv'), df='pandas')

                        # Filter rows with desired labels
                        estimation_matrix = estimation_matrix[
                            estimation_matrix['Label'].isin(['aggregated', 'consumer', 'prosumer'])]
                        estimation_matrix_individual = estimation_matrix_individual[
                            estimation_matrix_individual['Label'].isin(['aggregated', 'consumer', 'prosumer'])]

                        amount = estimation_matrix['Balance'].mean()

                        x4 = x3 + 1
                        fig.add_shape(
                            type='line', line={'color': 'rgb(50, 171, 96)', 'width': 4},
                            xref="x",
                            yref="y",
                            x0=x3,
                            x1=x4,
                            y0=amount,
                            y1=amount,
                            row=1,
                            col=col
                        )
                        x3 = x3 + 1

                        if group == '5':

                            amount = estimation_matrix_individual['Balance'].mean()

                            x1 = -0.5
                            x2 = 0.5
                            if season == "summer":
                                fig.add_shape(
                                    name="Balance",
                                    showlegend=True,
                                    type='line', line={'color': 'rgb(50, 171, 96)', 'width': 4},
                                    xref="x",
                                    yref="y",
                                    x0=x1,
                                    x1=x2,
                                    y0=amount,
                                    y1=amount,
                                    row=1,
                                    col=col
                                )
                            else:
                                fig.add_shape(
                                    name="Balance",
                                    type='line', line={'color': 'rgb(50, 171, 96)', 'width': 4},
                                    xref="x",
                                    yref="y",
                                    x0=x1,
                                    x1=x2,
                                    y0=amount,
                                    y1=amount,
                                    row=1,
                                    col=col
                                )
                            #x1 = x1 + 4
                            #x2 = x2 + 4
            col = col + 1

        title = scenario.capitalize() + '-Group ' + 'Benefits/Costs per Agent per Timestamp over all Groups at Perfect Forecasting'

        fig.update_layout(
            title=title, yaxis_title="Amount in Euro")
        fig.show()

list_scenarios = ["consumer", "prosumer", "mixed"]
Measurements = ['Balance']
for scenario in list_scenarios:
    if not scenario == 'individual':
        df = pd.DataFrame(columns=['Agent', 'Label', 'Season', 'Group'])
        for season in list_seasons:

            for group in list_group_size:

                for forecasting_method in list_forecasting_methods:

                    if forecasting_method == "naive":
                        path = "05_results/scenario_pv_60_hp_60_ev_60_" + scenario + "_" + group + "_" + season + "_" + forecasting_method
                        path_individual = "05_results/scenario_pv_60_hp_60_ev_60_individual_" + season + "_" + forecasting_method

                        estimation_matrix_individual = f.load_file(
                            path=os.path.join(two_levels_up, path_individual, 'general', 'estimation_matrix.csv'),
                            df='pandas')
                        estimation_matrix = f.load_file(
                            path=os.path.join(two_levels_up, path, 'general', 'estimation_matrix.csv'), df='pandas')

                        estimation_matrix = estimation_matrix.groupby('Agent').agg({'Timestamp': 'first',
                                                                                          'Label': 'first',
                                                                                          'Underestimation': 'sum',
                                                                                          'Overestimation': 'sum',
                                                                                          'Benefits': 'sum',
                                                                                          'Benefits_Market': 'sum',
                                                                                          'Benefits_Balancing': 'sum',
                                                                                          'Costs': 'sum',
                                                                                          'Costs_Market': 'sum',
                                                                                          'Costs_Balancing': 'sum',
                                                                                          'Costs_Fees': 'sum',
                                                                                          'Balance': 'sum'}).reset_index()

                        if group == '5':
                            estimation_matrix_individual = estimation_matrix_individual.groupby('Agent').agg(
                                {'Timestamp': 'first',
                                 'Label': 'first',
                                 'Underestimation': 'sum',
                                 'Overestimation': 'sum',
                                 'Benefits': 'sum',
                                 'Benefits_Market': 'sum',
                                 'Benefits_Balancing': 'sum',
                                 'Costs': 'sum',
                                 'Costs_Market': 'sum',
                                 'Costs_Balancing': 'sum',
                                 'Costs_Fees': 'sum',
                                 'Balance': 'sum'}).reset_index()

                            print(estimation_matrix_individual[estimation_matrix_individual['Label'].isin(['consumer'])]["Balance"].sum())

                        # Filter rows with desired labels
                        filtered_df = estimation_matrix[
                            estimation_matrix['Label'].isin(['aggregated', 'consumer', 'prosumer'])]

                        print(estimation_matrix[estimation_matrix['Label'].isin(['consumer', 'aggregated'])][
                                  "Balance"].sum())
                        # Filter rows with desired labels
                        filtered_df_individual = estimation_matrix_individual[
                            estimation_matrix_individual['Label'].isin(['aggregated', 'consumer', 'prosumer'])]



                        if group == '5':
                            group1 = '1'
                            grouped_df_individual = filtered_df_individual[['Agent', 'Label', 'Balance']]
                            grouped_df_individual["Season"] = season
                            grouped_df_individual["Group"] = group1
                            df = pd.concat([df, grouped_df_individual], ignore_index=True)

                        grouped_df = filtered_df[['Agent', 'Label', 'Balance']]
                        grouped_df["Season"] = season
                        grouped_df["Group"] = group
                        df = pd.concat([df, grouped_df], ignore_index=True)

        #fig = px.bar(df, x="Group", y="Balance", color="Label", facet_row="Agent",facet_row_spacing=0.001, facet_col="Season",
        #             category_orders={"Season": ["summer", "transition", "winter"]})

        fig = px.bar(df, x="Group", y="Balance", color="Label",facet_col="Season",category_orders={"Season": ["summer", "transition", "winter"]})
        title = scenario.capitalize() + '-Group ' + 'Total Amount of Balances over all Groups at Naive Forecasting'
        fig.layout.template = 'custom'
        fig.update_layout(
            title=title, yaxis_title="Amount in Euro")

        fig.update_layout(yaxis_range=[0, -1200])

        fig.show()


Measurements = ['Balance']
for scenario in list_scenarios:
    if not scenario == 'individual':
        df = pd.DataFrame(columns=['Agent', 'Label', 'Season', 'Group'])
        for season in list_seasons:

            for group in list_group_size:

                for forecasting_method in list_forecasting_methods:

                    if forecasting_method == "naive":
                        path = "05_results/scenario_pv_60_hp_60_ev_60_" + scenario + "_" + group + "_" + season + "_" + forecasting_method
                        path_individual = "05_results/scenario_pv_60_hp_60_ev_60_individual_" + season + "_" + forecasting_method

                        estimation_matrix_individual = f.load_file(
                            path=os.path.join(two_levels_up, path_individual, 'general', 'estimation_matrix.csv'),
                            df='pandas')
                        estimation_matrix = f.load_file(
                            path=os.path.join(two_levels_up, path, 'general', 'estimation_matrix.csv'), df='pandas')

                        estimation_matrix = estimation_matrix.groupby('Agent').agg({'Timestamp': 'first',
                                                                                          'Label': 'first',
                                                                                          'Underestimation': 'sum',
                                                                                          'Overestimation': 'sum',
                                                                                          'Benefits': 'sum',
                                                                                          'Benefits_Market': 'sum',
                                                                                          'Benefits_Balancing': 'sum',
                                                                                          'Costs': 'sum',
                                                                                          'Costs_Market': 'sum',
                                                                                          'Costs_Balancing': 'sum',
                                                                                          'Costs_Fees': 'sum',
                                                                                          'Balance': 'sum'}).reset_index()

                        if group == '5':
                            estimation_matrix_individual = estimation_matrix_individual.groupby('Agent').agg(
                                {'Timestamp': 'first',
                                 'Label': 'first',
                                 'Underestimation': 'sum',
                                 'Overestimation': 'sum',
                                 'Benefits': 'sum',
                                 'Benefits_Market': 'sum',
                                 'Benefits_Balancing': 'sum',
                                 'Costs': 'sum',
                                 'Costs_Market': 'sum',
                                 'Costs_Balancing': 'sum',
                                 'Costs_Fees': 'sum',
                                 'Balance': 'sum'}).reset_index()

                        # Filter rows with desired labels
                        filtered_df = estimation_matrix[
                            estimation_matrix['Label'].isin(['aggregated', 'consumer', 'prosumer'])]

                        filtered_df['Balance'] = filtered_df['Balance'] + filtered_df['Costs_Fees']

                        # Filter rows with desired labels
                        filtered_df_individual = estimation_matrix_individual[
                            estimation_matrix_individual['Label'].isin(['aggregated', 'consumer', 'prosumer'])]

                        filtered_df_individual['Balance'] = filtered_df_individual['Balance'] + filtered_df_individual['Costs_Fees']

                        if group == '5':
                            group1 = '1'
                            grouped_df_individual = filtered_df_individual[['Agent', 'Label', 'Balance']]
                            grouped_df_individual["Season"] = season
                            grouped_df_individual["Group"] = group1
                            df = pd.concat([df, grouped_df_individual], ignore_index=True)

                        grouped_df = filtered_df[['Agent', 'Label', 'Balance']]
                        grouped_df["Season"] = season
                        grouped_df["Group"] = group
                        df = pd.concat([df, grouped_df], ignore_index=True)

        #fig = px.bar(df, x="Group", y="Balance", color="Label", facet_row="Agent",facet_row_spacing=0.001, facet_col="Season",
        #             category_orders={"Season": ["summer", "transition", "winter"]})

        fig = px.bar(df, x="Group", y="Balance", color="Label",facet_col="Season",category_orders={"Season": ["summer", "transition", "winter"]})
        title = scenario.capitalize() + '-Group ' + 'Total Amount of Balances at Naive Forecasting if no Fees exist'
        fig.layout.template = 'custom'
        fig.update_layout(
            title=title, yaxis_title="Amount Amount in Euro")

        fig.show()


Measurements = ['Balance']
for scenario in list_scenarios:
    if not scenario == 'individual':
        df = pd.DataFrame(columns=['Agent', 'Label', 'Season', 'Group'])
        for season in list_seasons:

            for group in list_group_size:

                for forecasting_method in list_forecasting_methods:

                    if forecasting_method == "naive":
                        path = "05_results/scenario_pv_60_hp_60_ev_60_" + scenario + "_" + group + "_" + season + "_" + forecasting_method
                        path_individual = "05_results/scenario_pv_60_hp_60_ev_60_individual_" + season + "_" + forecasting_method

                        estimation_matrix_individual = f.load_file(
                            path=os.path.join(two_levels_up, path_individual, 'general', 'estimation_matrix.csv'),
                            df='pandas')
                        estimation_matrix = f.load_file(
                            path=os.path.join(two_levels_up, path, 'general', 'estimation_matrix.csv'), df='pandas')

                        estimation_matrix = estimation_matrix.groupby('Label').agg({'Timestamp': 'first',
                                                                                          'Underestimation': 'sum',
                                                                                          'Overestimation': 'sum',
                                                                                          'Benefits': 'sum',
                                                                                          'Benefits_Market': 'sum',
                                                                                          'Benefits_Balancing': 'sum',
                                                                                          'Costs': 'sum',
                                                                                          'Costs_Market': 'sum',
                                                                                          'Costs_Balancing': 'sum',
                                                                                          'Costs_Fees': 'sum',
                                                                                          'Balance': 'sum'}).reset_index()

                        if group == '5':
                            estimation_matrix_individual = estimation_matrix_individual.groupby('Label').agg(
                                {'Timestamp': 'first',
                                 'Underestimation': 'sum',
                                 'Overestimation': 'sum',
                                 'Benefits': 'sum',
                                 'Benefits_Market': 'sum',
                                 'Benefits_Balancing': 'sum',
                                 'Costs': 'sum',
                                 'Costs_Market': 'sum',
                                 'Costs_Balancing': 'sum',
                                 'Costs_Fees': 'sum',
                                 'Balance': 'sum'}).reset_index()

                        # Filter rows with desired labels
                        filtered_df = estimation_matrix[
                            estimation_matrix['Label'].isin(['aggregated', 'consumer', 'prosumer'])]

                        # Filter rows with desired labels
                        filtered_df_individual = estimation_matrix_individual[
                            estimation_matrix_individual['Label'].isin(['aggregated', 'consumer', 'prosumer'])]

                        filtered_df_aggre = estimation_matrix[estimation_matrix['Label'].isin(['Aggregator'])]

                        filtered_df.loc[filtered_df['Label'] == 'aggregated', 'Balance'] = filtered_df.loc[filtered_df['Label'] == 'aggregated', 'Balance'].sum() + filtered_df.loc[filtered_df['Label'] == 'aggregated', 'Costs_Fees'].sum() - filtered_df_aggre['Costs_Fees'].sum()

                        if group == '5':
                            group1 = '1'
                            grouped_df_individual = filtered_df_individual[['Label', 'Balance']]
                            grouped_df_individual["Season"] = season
                            grouped_df_individual["Group"] = group1
                            df = pd.concat([df, grouped_df_individual], ignore_index=True)

                        grouped_df = filtered_df[['Label', 'Balance']]
                        grouped_df["Season"] = season
                        grouped_df["Group"] = group
                        df = pd.concat([df, grouped_df], ignore_index=True)

        #fig = px.bar(df, x="Group", y="Balance", color="Label", facet_row="Agent",facet_row_spacing=0.001, facet_col="Season",
        #             category_orders={"Season": ["summer", "transition", "winter"]})

        fig = px.bar(df, x="Group", y="Balance", color="Label",facet_col="Season",category_orders={"Season": ["summer", "transition", "winter"]})
        title = scenario.capitalize() + '-Group ' + 'Total Amount of Balances at Naive Forecasting if no Fees exist for the Aggregated Agents'
        fig.layout.template = 'custom'
        fig.update_layout(
            title=title, yaxis_title="Amount in Euro")

        fig.show()

Measurements = ['Total_Error']
for scenario in list_scenarios:
    if not scenario == 'individual':
        df = pd.DataFrame(columns=['Group', 'Total_Error', 'Season', 'Amount of Errors in KWh'])
        for season in list_seasons:

            for group in list_group_size:

                for forecasting_method in list_forecasting_methods:

                    if forecasting_method == "naive":
                        path = "05_results/scenario_pv_60_hp_60_ev_60_" + scenario + "_" + group + "_" + season + "_" + forecasting_method
                        path_individual = "05_results/scenario_pv_60_hp_60_ev_60_individual_" + season + "_" + forecasting_method

                        estimation_matrix_individual = f.load_file(
                            path=os.path.join(two_levels_up, path_individual, 'general', 'estimation_matrix.csv'),
                            df='pandas')
                        estimation_matrix = f.load_file(
                            path=os.path.join(two_levels_up, path, 'general', 'estimation_matrix.csv'), df='pandas')

                        # Filter rows with desired labels
                        estimation_matrix = estimation_matrix[
                            estimation_matrix['Label'].isin(['aggregated', 'consumer', 'prosumer'])]
                        estimation_matrix_individual = estimation_matrix_individual[
                            estimation_matrix_individual['Label'].isin(['aggregated', 'consumer', 'prosumer'])]

                        estimation_matrix['Total_Error'] = estimation_matrix['Overestimation'] + estimation_matrix[
                            'Underestimation']

                        estimation_matrix_individual['Total_Error'] = estimation_matrix_individual['Overestimation'] + estimation_matrix_individual[
                            'Underestimation']

                        if group =='5':
                            for measure in Measurements:
                                amount = estimation_matrix_individual[measure].sum()
                                group1 = '1'
                                df_row = [group1, measure, season, amount]

                                df.loc[len(df.index)] = df_row

                        for measure in Measurements:

                            amount = estimation_matrix[measure].sum()

                            df_row = [group, measure, season, amount]

                            df.loc[len(df.index)] = df_row



        fig = px.bar(df, x="Group", y="Amount of Errors in KWh", color="Total_Error", barmode="relative", facet_col="Season",
                     category_orders={"Season": ["summer", "transition", "winter"]})
        #fig.update_traces(mode='lines+markers')
        title = scenario.capitalize() + '-Group ' + 'Total Amount of Errors in KWh over all Groups at Naive Forecasting'
        fig.layout.template = 'custom'
        fig.update_layout(
            title=title, yaxis_title="Amount in KWh")

        fig.show()


plt_io.templates["custom"] = plt_io.templates["simple_white"]

plt_io.templates['custom']['layout']['yaxis']['showgrid'] = True
plt_io.templates['custom']['layout']['xaxis']['showgrid'] = True
plt_io.templates['custom']['layout']['font']['size'] = 25
plt_io.templates['custom']['layout']['legend']['y'] = 1.06
plt_io.templates['custom']['layout']['legend']['yanchor'] = 'bottom'
plt_io.templates['custom']['layout']['legend']['orientation'] = 'h'
plt_io.templates['custom']['layout']['legend']['xanchor'] = 'right'
plt_io.templates['custom']['layout']['legend']['x'] = 1
plt_io.templates['custom']['layout']['legend']['font_size'] = 28
Measurements = ['Total_Error']
df = pd.DataFrame(columns=['Group Size', 'Season', 'Total Error Costs in Euro', "Scenario"])
for scenario in list_scenarios:
    if not scenario == 'individual':

        for season in list_seasons:

            for group in list_group_size:

                for forecasting_method in list_forecasting_methods:

                    if forecasting_method == "naive":
                        path = "05_results/scenario_pv_60_hp_60_ev_60_" + scenario + "_" + group + "_" + season + "_" + forecasting_method
                        path_individual = "05_results/scenario_pv_60_hp_60_ev_60_individual_" + season + "_" + forecasting_method

                        estimation_matrix_individual = f.load_file(
                            path=os.path.join(two_levels_up, path_individual, 'general', 'estimation_matrix.csv'),
                            df='pandas')
                        estimation_matrix = f.load_file(
                            path=os.path.join(two_levels_up, path, 'general', 'estimation_matrix.csv'), df='pandas')

                        # Filter rows with desired labels
                        estimation_matrix = estimation_matrix[
                            estimation_matrix['Label'].isin(['aggregated', 'consumer', 'prosumer'])]
                        estimation_matrix_individual = estimation_matrix_individual[
                            estimation_matrix_individual['Label'].isin(['aggregated', 'consumer', 'prosumer'])]

                        estimation_matrix = estimation_matrix.groupby('Agent').agg(
                            {'Timestamp': 'first',
                             'Label': 'first',
                             'Underestimation': 'sum',
                             'Overestimation': 'sum',
                             'Benefits': 'sum',
                             'Benefits_Market': 'sum',
                             'Benefits_Balancing': 'sum',
                             'Costs': 'sum',
                             'Costs_Market': 'sum',
                             'Costs_Balancing': 'sum',
                             'Costs_Fees': 'sum',
                             'Balance': 'sum'}).reset_index()

                        estimation_matrix_individual = estimation_matrix_individual.groupby('Agent').agg(
                            {'Timestamp': 'first',
                             'Label': 'first',
                             'Underestimation': 'sum',
                             'Overestimation': 'sum',
                             'Benefits': 'sum',
                             'Benefits_Market': 'sum',
                             'Benefits_Balancing': 'sum',
                             'Costs': 'sum',
                             'Costs_Market': 'sum',
                             'Costs_Balancing': 'sum',
                             'Costs_Fees': 'sum',
                             'Balance': 'sum'}).reset_index()

                        estimation_matrix['Total_Error'] = estimation_matrix['Overestimation'] + estimation_matrix[
                            'Underestimation']


                        estimation_matrix_individual['Total_Error'] = estimation_matrix_individual['Overestimation'] + estimation_matrix_individual[
                            'Underestimation']

                        estimation_matrix['Total_Error'] = estimation_matrix['Total_Error'] * 0.01

                        estimation_matrix_individual['Total_Error'] = estimation_matrix_individual['Total_Error'] * 0.01

                        if group =='5':
                            for measure in Measurements:
                                amount = estimation_matrix_individual[measure].mean()
                                group1 = '1'
                                scen = scenario.capitalize() + "-Group"
                                df_row = [group1, season, amount, scen]

                                df.loc[len(df.index)] = df_row

                        for measure in Measurements:

                            amount = estimation_matrix[measure].mean()
                            scen = scenario.capitalize() + "-Group"
                            df_row = [group, season, amount, scen]

                            df.loc[len(df.index)] = df_row



fig = px.line(df, x="Group Size", y='Total Error Costs in Euro', color="Scenario", facet_col="Season", markers=True,
             category_orders={"Season": ["summer", "transition", "winter"]})

title = 'Mean Error Costs in Euro over all Groups at Naive Forecasting'
fig.layout.template = 'custom'
f#ig.update_layout(yaxis_title="Total Forecast Error Costs [Euro]")

fig.for_each_annotation(lambda a: a.update(text=a.text.replace("Season=", "")))
#fig.update_layout(
#    title=title, yaxis_title="Amount in KWh")
#fig.update_xaxes(categoryorder='array')
fig.update_layout(yaxis_title="Average Forecast Error Costs [Euro]")

#fig.update_layout(yaxis_range=[0, 380])
#fig.update_layout(autosize=False,
 #                 width=1600,
#                  height=1000)
fig.update_layout(legend_title_text='')
fig.update_layout(legend={
    "font": {
        "family": "Arial, monospace",
        "size": 35
    }})

fig.update_layout(
    legend=dict(
        bordercolor="Black",
        borderwidth=2
    )
)

fig.show()


plt_io.templates["custom"] = plt_io.templates["simple_white"]

plt_io.templates['custom']['layout']['yaxis']['showgrid'] = True
plt_io.templates['custom']['layout']['xaxis']['showgrid'] = True
plt_io.templates['custom']['layout']['font']['size'] = 32
plt_io.templates['custom']['layout']['legend']['y'] = 1.05
plt_io.templates['custom']['layout']['legend']['yanchor'] = 'bottom'
plt_io.templates['custom']['layout']['legend']['orientation'] = 'h'
plt_io.templates['custom']['layout']['legend']['xanchor'] = 'right'
plt_io.templates['custom']['layout']['legend']['x'] = 1
plt_io.templates['custom']['layout']['legend']['font_size'] = 30


for scenario in list_scenarios:
    if not scenario == 'individual':
        df = pd.DataFrame(columns=['Label', 'Total_Error', 'Season', 'Group'])
        for season in list_seasons:

            for group in list_group_size:

                for forecasting_method in list_forecasting_methods:

                    if forecasting_method == "naive":
                        path = "05_results/scenario_pv_60_hp_60_ev_60_" + scenario + "_" + group + "_" + season + "_" + forecasting_method
                        path_individual = "05_results/scenario_pv_60_hp_60_ev_60_individual_" + season + "_" + forecasting_method

                        estimation_matrix_individual = f.load_file(
                            path=os.path.join(two_levels_up, path_individual, 'general', 'estimation_matrix.csv'),
                            df='pandas')
                        estimation_matrix = f.load_file(
                            path=os.path.join(two_levels_up, path, 'general', 'estimation_matrix.csv'), df='pandas')

                        estimation_matrix['Total_Error'] = estimation_matrix['Overestimation'] + estimation_matrix[
                            'Underestimation']

                        # Filter rows with desired labels
                        filtered_df = estimation_matrix[
                            estimation_matrix['Label'].isin(['aggregated', 'consumer', 'prosumer'])]

                        # Group and aggregate by Label
                        grouped_df = filtered_df.groupby(['Label'])['Total_Error'].sum().reset_index()

                        estimation_matrix_individual['Total_Error'] = estimation_matrix_individual['Overestimation'] + estimation_matrix_individual[
                            'Underestimation']

                        # Filter rows with desired labels
                        filtered_df_individual = estimation_matrix_individual[
                            estimation_matrix_individual['Label'].isin(['aggregated', 'consumer', 'prosumer'])]

                        # Group and aggregate by Label
                        grouped_df_individual = filtered_df_individual.groupby(['Label'])['Total_Error'].sum().reset_index()

                        if group == '5':

                            group1 = '1'
                            grouped_df_individual["Season"] = season
                            grouped_df_individual["Group"] = group1
                            df = pd.concat([df, grouped_df_individual], ignore_index=True)

                        grouped_df["Season"] = season
                        grouped_df["Group"] = group
                        df = pd.concat([df, grouped_df], ignore_index=True)


        fig = px.bar(df, x="Group", y="Total_Error", color="Label", barmode="relative", facet_col="Season",
                     category_orders={"Season": ["summer", "transition", "winter"]})

        title = scenario.capitalize() + '-Group ' + 'Total Amount of Errors in KWh over all Groups at Naive Forecasting'
        fig.layout.template = 'custom'
        fig.for_each_annotation(lambda a: a.update(text=a.text.replace("Season=", "")))
        #fig.update_layout(
        #    title=title, yaxis_title="Amount in KWh")
        #fig.update_xaxes(categoryorder='array')
        fig.update_layout(yaxis_title="Amount in KWh")
        fig.update_layout(yaxis_range=[0, 380])
        fig.update_layout(autosize=False,
                          width=900,
                          height=1000)
        fig.update_layout(legend_title_text='')
        fig.update_layout(legend={
            "font": {
                "family": "Arial, monospace",
                "size": 35
            }})

        fig.update_layout(
            legend=dict(
                bordercolor="Black",
                borderwidth=2
            )
        )

        fig.show()


for scenario in list_scenarios:
    if not scenario == 'individual':
        df = pd.DataFrame(columns=['Agent', 'Label', 'Season', 'Group'])
        for season in list_seasons:

            for group in list_group_size:

                for forecasting_method in list_forecasting_methods:

                    if forecasting_method == "naive":
                        path = "05_results/scenario_pv_60_hp_60_ev_60_" + scenario + "_" + group + "_" + season + "_" + forecasting_method
                        path_individual = "05_results/scenario_pv_60_hp_60_ev_60_individual_" + season + "_" + forecasting_method

                        estimation_matrix_individual = f.load_file(
                            path=os.path.join(two_levels_up, path_individual, 'general',
                                              'estimation_matrix.csv'),
                            df='pandas')
                        estimation_matrix = f.load_file(
                            path=os.path.join(two_levels_up, path, 'general', 'estimation_matrix.csv'),
                            df='pandas')

                        estimation_matrix = estimation_matrix.groupby('Agent').agg({'Timestamp': 'first',
                                                                                    'Label': 'first',
                                                                                    'Underestimation': 'sum',
                                                                                    'Overestimation': 'sum',
                                                                                    'Benefits': 'sum',
                                                                                    'Benefits_Market': 'sum',
                                                                                    'Benefits_Balancing': 'sum',
                                                                                    'Costs': 'sum',
                                                                                    'Costs_Market': 'sum',
                                                                                    'Costs_Balancing': 'sum',
                                                                                    'Costs_Fees': 'sum',
                                                                                    'Balance': 'sum'}).reset_index()

                        estimation_matrix_individual = estimation_matrix_individual.groupby('Agent').agg(
                            {'Timestamp': 'first',
                             'Label': 'first',
                             'Underestimation': 'sum',
                             'Overestimation': 'sum',
                             'Benefits': 'sum',
                             'Benefits_Market': 'sum',
                             'Benefits_Balancing': 'sum',
                             'Costs': 'sum',
                             'Costs_Market': 'sum',
                             'Costs_Balancing': 'sum',
                             'Costs_Fees': 'sum',
                             'Balance': 'sum'}).reset_index()

                        # Filter rows with desired labels
                        filtered_df_individual = estimation_matrix_individual[
                            estimation_matrix_individual['Label'].isin(['aggregated', 'consumer', 'prosumer'])]

                        # Filter rows with desired labels
                        filtered_df = estimation_matrix[
                            estimation_matrix['Label'].isin(['aggregated', 'consumer', 'prosumer'])]

                        filtered_df['Total_Error'] = filtered_df[
                                                                          'Overestimation'] + \
                                                                      filtered_df[
                                                                          'Underestimation']

                        filtered_df_individual['Total_Error'] = filtered_df_individual[
                                                                          'Overestimation'] + \
                                                                      filtered_df_individual[
                                                                          'Underestimation']



                        if group == '5':
                            group1 = '1'
                            grouped_df_individual = filtered_df_individual[['Agent', 'Label', 'Total_Error']]
                            grouped_df_individual["Season"] = season
                            grouped_df_individual["Group"] = group1
                            df = pd.concat([df, grouped_df_individual], ignore_index=True)

                        grouped_df = filtered_df[['Agent', 'Label', 'Total_Error']]
                        grouped_df["Season"] = season
                        grouped_df["Group"] = group
                        df = pd.concat([df, grouped_df], ignore_index=True)

        # fig = px.bar(df, x="Group", y="Balance", color="Label", facet_row="Agent",facet_row_spacing=0.001, facet_col="Season",
        #             category_orders={"Season": ["summer", "transition", "winter"]})

        fig = px.bar(df, x="Group", y="Total_Error", color="Label", facet_col="Season",color_discrete_sequence=['#EF553B','#636efa', '#00cc96'],
                     category_orders={"Season": ["summer", "transition", "winter"]})
        title = scenario.capitalize() + '-Group ' + 'Total Amount of Errors in KWh over all Groups at Naive Forecasting'
        fig.layout.template = 'custom'
        fig.for_each_annotation(lambda a: a.update(text=a.text.replace("Season=", "")))
        #fig.update_layout(
        #    title=title, yaxis_title="Amount in KWh")
        #fig.update_xaxes(categoryorder='array')
        fig.update_layout(yaxis_title="Amount in KWh")
        fig.update_layout(yaxis_range=[0, 380])
        fig.update_layout(autosize=False,
                          width=900,
                          height=1000)
        fig.update_layout(legend_title_text='')
        fig.update_layout(legend={
            "font": {
                "family": "Arial, monospace",
                "size": 35
            }})

        fig.update_layout(
            legend=dict(
                bordercolor="Black",
                borderwidth=2
            )
        )

        fig.show()



for scenario in list_scenarios:
    if not scenario == 'individual':
        df = pd.DataFrame(columns=['Agent', 'Label', 'Season', 'Group'])
        for season in list_seasons:

            for group in list_group_size:

                for forecasting_method in list_forecasting_methods:

                    if forecasting_method == "naive":
                        path = "05_results/scenario_pv_60_hp_60_ev_60_" + scenario + "_" + group + "_" + season + "_" + forecasting_method
                        path_individual = "05_results/scenario_pv_60_hp_60_ev_60_individual_" + season + "_" + forecasting_method

                        estimation_matrix_individual = f.load_file(
                            path=os.path.join(two_levels_up, path_individual, 'general',
                                              'estimation_matrix.csv'),
                            df='pandas')
                        estimation_matrix = f.load_file(
                            path=os.path.join(two_levels_up, path, 'general', 'estimation_matrix.csv'),
                            df='pandas')

                        estimation_matrix = estimation_matrix.groupby('Agent').agg({'Timestamp': 'first',
                                                                                    'Label': 'first',
                                                                                    'Underestimation': 'sum',
                                                                                    'Overestimation': 'sum',
                                                                                    'Benefits': 'sum',
                                                                                    'Benefits_Market': 'sum',
                                                                                    'Benefits_Balancing': 'sum',
                                                                                    'Costs': 'sum',
                                                                                    'Costs_Market': 'sum',
                                                                                    'Costs_Balancing': 'sum',
                                                                                    'Costs_Fees': 'sum',
                                                                                    'Balance': 'sum'}).reset_index()

                        estimation_matrix_individual = estimation_matrix_individual.groupby('Agent').agg(
                            {'Timestamp': 'first',
                             'Label': 'first',
                             'Underestimation': 'sum',
                             'Overestimation': 'sum',
                             'Benefits': 'sum',
                             'Benefits_Market': 'sum',
                             'Benefits_Balancing': 'sum',
                             'Costs': 'sum',
                             'Costs_Market': 'sum',
                             'Costs_Balancing': 'sum',
                             'Costs_Fees': 'sum',
                             'Balance': 'sum'}).reset_index()

                        # Filter rows with desired labels
                        filtered_df_individual = estimation_matrix_individual[
                            estimation_matrix_individual['Label'].isin(['aggregated', 'consumer', 'prosumer'])]

                        # Filter rows with desired labels
                        filtered_df = estimation_matrix[
                            estimation_matrix['Label'].isin(['aggregated', 'consumer', 'prosumer'])]

                        filtered_df['Total_Error'] = filtered_df[
                                                                          'Overestimation'] + \
                                                                      filtered_df[
                                                                          'Underestimation']

                        filtered_df_individual['Total_Error'] = filtered_df_individual[
                                                                          'Overestimation'] + \
                                                                      filtered_df_individual[
                                                                          'Underestimation']



                        if group == '5':
                            group1 = '1'
                            grouped_df_individual = filtered_df_individual[['Agent', 'Label', 'Total_Error']]
                            grouped_df_individual["Season"] = season
                            grouped_df_individual["Group"] = group1
                            df = pd.concat([df, grouped_df_individual], ignore_index=True)

                        grouped_df = filtered_df[['Agent', 'Label', 'Total_Error']]
                        grouped_df["Season"] = season
                        grouped_df["Group"] = group
                        df = pd.concat([df, grouped_df], ignore_index=True)

        df.loc[df['Label'] == 'consumer', 'Label'] = 'xxosumer'

        custom_order = ['1', '5', '10', '15']

        # Convert the 'group' column to a categorical data type with the custom order
        df['Group'] = pd.Categorical(df['Group'], categories=custom_order, ordered=True)

        # Sort the DataFrame by 'label' and then by 'group'
        df = df.sort_values(by=['Group', 'Label'], ascending=[True, False])

        df.loc[df['Label'] == 'xxosumer', 'Label'] = 'consumer'

        # fig = px.bar(df, x="Group", y="Balance", color="Label", facet_row="Agent",facet_row_spacing=0.001, facet_col="Season",
        #             category_orders={"Season": ["summer", "transition", "winter"]})

        fig = px.bar(df, x="Group", y="Total_Error", color="Label", facet_col="Season",color_discrete_sequence=['#636efa','#EF553B', '#00cc96'],
                     category_orders={"Season": ["summer", "transition", "winter"]})
        title = scenario.capitalize() + '-Group ' + 'Total Amount of Errors in KWh over all Groups at Naive Forecasting'
        fig.layout.template = 'custom'
        fig.for_each_annotation(lambda a: a.update(text=a.text.replace("Season=", "")))
        #fig.update_layout(
        #    title=title, yaxis_title="Amount in KWh")
        #fig.update_xaxes(categoryorder='array')
        fig.update_layout(yaxis_title="Amount in KWh")
        fig.update_layout(yaxis_range=[0, 380])
        fig.update_layout(autosize=False,
                          width=900,
                          height=1000)
        fig.update_layout(legend_title_text='')
        fig.update_layout(legend={
            "font": {
                "family": "Arial, monospace",
                "size": 35
            }})

        fig.update_layout(
            legend=dict(
                bordercolor="Black",
                borderwidth=2
            )
        )

        fig.show()



Measurements = ['Benefits_Market', 'Benefits_Balancing', 'Costs_Market', 'Costs_Balancing', 'Costs_Fees']
for scenario in list_scenarios:
    if not scenario == 'individual':
        df = pd.DataFrame(columns=['Group', 'Legend', 'Season', 'Benefits/Costs in Euro'])
        for season in list_seasons:

            for group in list_group_size:

                for forecasting_method in list_forecasting_methods:

                    if forecasting_method == "naive":
                        path = "05_results/scenario_pv_60_hp_60_ev_60_" + scenario + "_" + group + "_" + season + "_" + forecasting_method
                        path_individual = "05_results/scenario_pv_60_hp_60_ev_60_individual_" + season + "_" + forecasting_method

                        estimation_matrix_individual = f.load_file(
                            path=os.path.join(two_levels_up, path_individual, 'general', 'estimation_matrix.csv'),
                            df='pandas')
                        estimation_matrix = f.load_file(
                            path=os.path.join(two_levels_up, path, 'general', 'estimation_matrix.csv'), df='pandas')

                        estimation_matrix_individual = estimation_matrix_individual.groupby('Agent').agg(
                            {'Timestamp': 'first',
                             'Label': 'first',
                             'Underestimation': 'sum',
                             'Overestimation': 'sum',
                             'Benefits': 'sum',
                             'Benefits_Market': 'sum',
                             'Benefits_Balancing': 'sum',
                             'Costs': 'sum',
                             'Costs_Market': 'sum',
                             'Costs_Balancing': 'sum',
                             'Costs_Fees': 'sum',
                             'Balance': 'sum'})

                        estimation_matrix = estimation_matrix.groupby('Agent').agg(
                            {'Timestamp': 'first',
                             'Label': 'first',
                             'Underestimation': 'sum',
                             'Overestimation': 'sum',
                             'Benefits': 'sum',
                             'Benefits_Market': 'sum',
                             'Benefits_Balancing': 'sum',
                             'Costs': 'sum',
                             'Costs_Market': 'sum',
                             'Costs_Balancing': 'sum',
                             'Costs_Fees': 'sum',
                             'Balance': 'sum'})

                        # Filter rows with desired labels
                        estimation_matrix = estimation_matrix[
                            estimation_matrix['Label'].isin(['aggregated', 'consumer', 'prosumer'])]
                        estimation_matrix_individual = estimation_matrix_individual[
                            estimation_matrix_individual['Label'].isin(['aggregated', 'consumer', 'prosumer'])]

                        if group =='5':

                            for measure in Measurements:
                                amount = estimation_matrix_individual[measure].mean()
                                if "Costs" in measure:
                                    amount = -amount
                                group1 = '1'
                                df_row = [group1, measure, season, amount]

                                df.loc[len(df.index)] = df_row


                        for measure in Measurements:

                            amount = estimation_matrix[measure].mean()
                            if "Costs" in measure:
                                amount = -amount

                            df_row = [group, measure, season, amount]

                            df.loc[len(df.index)] = df_row


        fig = px.bar(df, x="Group", y="Benefits/Costs in Euro", color="Legend", barmode="relative", facet_col="Season",
                     category_orders={"": ["Summer", "Transition", "Winter"]})
        fig.for_each_annotation(lambda a: a.update(text=a.text.replace("Season=", "")))
        #df = {'col1': ['DarkSlateGrey', 'green'], 'col2': ['red', 'blue']}
        #fig.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')), opacity=0.7)
        fig.layout.template = 'custom'
        col = 1
        for season in list_seasons:
            x3 = 0.5
            for group in list_group_size:

                for forecasting_method in list_forecasting_methods:

                    if forecasting_method == "naive":
                        path = "05_results/scenario_pv_60_hp_60_ev_60_" + scenario + "_" + group + "_" + season + "_" + forecasting_method
                        path_individual = "05_results/scenario_pv_60_hp_60_ev_60_individual_" + season + "_" + forecasting_method

                        estimation_matrix_individual = f.load_file(
                            path=os.path.join(two_levels_up, path_individual, 'general', 'estimation_matrix.csv'),
                            df='pandas')
                        estimation_matrix = f.load_file(
                            path=os.path.join(two_levels_up, path, 'general', 'estimation_matrix.csv'), df='pandas')

                        estimation_matrix_individual = estimation_matrix_individual.groupby('Agent').agg(
                            {'Timestamp': 'first',
                             'Label': 'first',
                             'Underestimation': 'sum',
                             'Overestimation': 'sum',
                             'Benefits': 'sum',
                             'Benefits_Market': 'sum',
                             'Benefits_Balancing': 'sum',
                             'Costs': 'sum',
                             'Costs_Market': 'sum',
                             'Costs_Balancing': 'sum',
                             'Costs_Fees': 'sum',
                             'Balance': 'sum'})

                        estimation_matrix = estimation_matrix.groupby('Agent').agg(
                            {'Timestamp': 'first',
                             'Label': 'first',
                             'Underestimation': 'sum',
                             'Overestimation': 'sum',
                             'Benefits': 'sum',
                             'Benefits_Market': 'sum',
                             'Benefits_Balancing': 'sum',
                             'Costs': 'sum',
                             'Costs_Market': 'sum',
                             'Costs_Balancing': 'sum',
                             'Costs_Fees': 'sum',
                             'Balance': 'sum'})

                        # Filter rows with desired labels
                        estimation_matrix = estimation_matrix[
                            estimation_matrix['Label'].isin(['aggregated', 'consumer', 'prosumer'])]
                        estimation_matrix_individual = estimation_matrix_individual[
                            estimation_matrix_individual['Label'].isin(['aggregated', 'consumer', 'prosumer'])]

                        amount = estimation_matrix['Balance'].mean()

                        x4 = x3 + 1
                        fig.add_shape(
                            type='line', line={'color': 'Gray', 'width': 4},
                            xref="x",
                            yref="y",
                            x0=x3,
                            x1=x4,
                            y0=amount,
                            y1=amount,
                            row=1,
                            col=col,
                            opacity=1
                        )
                        x3 = x3 + 1

                        if group == '5':


                            amount = estimation_matrix_individual['Balance'].mean()

                            x1 = -0.5
                            x2 = 0.5
                            if season == "summer":
                                fig.add_shape(
                                    name="Balance",
                                    showlegend=True,
                                    type='line', line={'color': 'Gray', 'width': 4},
                                    xref="x",
                                    yref="y",
                                    x0=x1,
                                    x1=x2,
                                    y0=amount,
                                    y1=amount,
                                    row=1,
                                    col=col,
                                    opacity=1
                                )
                            else:
                                fig.add_shape(
                                    name="Balance",
                                    type='line', line={'color': 'Gray', 'width': 4},
                                    xref="x",
                                    yref="y",
                                    x0=x1,
                                    x1=x2,
                                    y0=amount,
                                    y1=amount,
                                    row=1,
                                    col=col,
                                    opacity=1
                                )
            col = col + 1

        title = scenario.capitalize() + '-Group ' + 'Benefits/Costs per Agent over all Groups at Perfect Forecasting'
        fig.update_layout(yaxis_range=[-33, 11])
        fig.update_layout(yaxis_title="Amount in Euro")
        fig.update_layout(autosize=False,
                          width=1200,
                          height=1000)
        fig.update_layout(legend = {
            "font": {
                "family": "Arial, monospace",
                "size": 25
            }})
        fig.show()






Measurements = ['Balance']
for scenario in list_scenarios:
    if not scenario == 'individual':
        df = pd.DataFrame(columns=['Agent', 'Label', 'Season', 'Group'])
        for season in list_seasons:

            for group in list_group_size:

                for forecasting_method in list_forecasting_methods:

                    if forecasting_method == "naive":
                        path = "05_results/scenario_pv_60_hp_60_ev_60_" + scenario + "_" + group + "_" + season + "_" + forecasting_method
                        path_individual = "05_results/scenario_pv_60_hp_60_ev_60_individual_" + season + "_" + forecasting_method

                        estimation_matrix_individual = f.load_file(
                            path=os.path.join(two_levels_up, path_individual, 'general', 'estimation_matrix.csv'),
                            df='pandas')
                        estimation_matrix = f.load_file(
                            path=os.path.join(two_levels_up, path, 'general', 'estimation_matrix.csv'), df='pandas')

                        estimation_matrix = estimation_matrix.groupby('Label').agg({'Timestamp': 'first',
                                                                                          'Underestimation': 'sum',
                                                                                          'Overestimation': 'sum',
                                                                                          'Benefits': 'sum',
                                                                                          'Benefits_Market': 'sum',
                                                                                          'Benefits_Balancing': 'sum',
                                                                                          'Costs': 'sum',
                                                                                          'Costs_Market': 'sum',
                                                                                          'Costs_Balancing': 'sum',
                                                                                          'Costs_Fees': 'sum',
                                                                                          'Balance': 'sum'}).reset_index()

                        if group == '5':
                            estimation_matrix_individual = estimation_matrix_individual.groupby('Label').agg(
                                {'Timestamp': 'first',
                                 'Underestimation': 'sum',
                                 'Overestimation': 'sum',
                                 'Benefits': 'sum',
                                 'Benefits_Market': 'sum',
                                 'Benefits_Balancing': 'sum',
                                 'Costs': 'sum',
                                 'Costs_Market': 'sum',
                                 'Costs_Balancing': 'sum',
                                 'Costs_Fees': 'sum',
                                 'Balance': 'sum'}).reset_index()

                        # Filter rows with desired labels
                        filtered_df = estimation_matrix[
                            estimation_matrix['Label'].isin(['aggregated', 'consumer', 'prosumer'])]

                        # Filter rows with desired labels
                        filtered_df_individual = estimation_matrix_individual[
                            estimation_matrix_individual['Label'].isin(['aggregated', 'consumer', 'prosumer'])]

                        filtered_df_aggre = estimation_matrix[estimation_matrix['Label'].isin(['Aggregator'])]

                        filtered_df.loc[filtered_df['Label'] == 'aggregated', 'Balance'] = filtered_df.loc[filtered_df['Label'] == 'aggregated', 'Balance'].sum() + filtered_df.loc[filtered_df['Label'] == 'aggregated', 'Costs_Fees'].sum() - filtered_df_aggre['Costs_Fees'].sum()

                        if group == '5':
                            group1 = '1'
                            grouped_df_individual = filtered_df_individual[['Label', 'Balance']]
                            grouped_df_individual["Season"] = season
                            grouped_df_individual["Group"] = group1
                            df = pd.concat([df, grouped_df_individual], ignore_index=True)

                        grouped_df = filtered_df[['Label', 'Balance']]
                        grouped_df["Season"] = season
                        grouped_df["Group"] = group
                        df = pd.concat([df, grouped_df], ignore_index=True)

        #fig = px.bar(df, x="Group", y="Balance", color="Label", facet_row="Agent",facet_row_spacing=0.001, facet_col="Season",
        #             category_orders={"Season": ["summer", "transition", "winter"]})

        fig = px.bar(df, x="Group", y="Balance", color="Label",facet_col="Season",category_orders={"Season": ["summer", "transition", "winter"]})
        title = scenario.capitalize() + '-Group ' + 'Total Amount of Balances at Naive Forecasting if no Fees exist for the Aggregated Agents'
        fig.layout.template = 'custom'
        fig.update_layout(yaxis_title="Amount in Euro")

        fig.show()

Measurements = ['Balance']
for scenario in list_scenarios:
    if not scenario == 'individual':
        df = pd.DataFrame(columns=['Agent', 'Label', 'Season', 'Group'])
        for season in list_seasons:

            for group in list_group_size:

                for forecasting_method in list_forecasting_methods:

                    if forecasting_method == "naive":
                        path = "05_results/scenario_pv_60_hp_60_ev_60_" + scenario + "_" + group + "_" + season + "_" + forecasting_method
                        path_individual = "05_results/scenario_pv_60_hp_60_ev_60_individual_" + season + "_" + forecasting_method

                        estimation_matrix_individual = f.load_file(
                            path=os.path.join(two_levels_up, path_individual, 'general', 'estimation_matrix.csv'),
                            df='pandas')
                        estimation_matrix = f.load_file(
                            path=os.path.join(two_levels_up, path, 'general', 'estimation_matrix.csv'), df='pandas')

                        estimation_matrix = estimation_matrix.groupby('Agent').agg({'Timestamp': 'first',
                                                                                          'Label': 'first',
                                                                                          'Underestimation': 'sum',
                                                                                          'Overestimation': 'sum',
                                                                                          'Benefits': 'sum',
                                                                                          'Benefits_Market': 'sum',
                                                                                          'Benefits_Balancing': 'sum',
                                                                                          'Costs': 'sum',
                                                                                          'Costs_Market': 'sum',
                                                                                          'Costs_Balancing': 'sum',
                                                                                          'Costs_Fees': 'sum',
                                                                                          'Balance': 'sum'}).reset_index()

                        if group == '5':
                            estimation_matrix_individual = estimation_matrix_individual.groupby('Agent').agg(
                                {'Timestamp': 'first',
                                 'Label': 'first',
                                 'Underestimation': 'sum',
                                 'Overestimation': 'sum',
                                 'Benefits': 'sum',
                                 'Benefits_Market': 'sum',
                                 'Benefits_Balancing': 'sum',
                                 'Costs': 'sum',
                                 'Costs_Market': 'sum',
                                 'Costs_Balancing': 'sum',
                                 'Costs_Fees': 'sum',
                                 'Balance': 'sum'}).reset_index()

                        # Filter rows with desired labels
                        filtered_df = estimation_matrix[
                            estimation_matrix['Label'].isin(['aggregated', 'consumer', 'prosumer'])]


                        # Filter rows with desired labels
                        filtered_df_individual = estimation_matrix_individual[
                            estimation_matrix_individual['Label'].isin(['aggregated', 'consumer', 'prosumer'])]



                        if group == '5':
                            group1 = '1'
                            grouped_df_individual = filtered_df_individual[['Agent', 'Label', 'Balance']]
                            grouped_df_individual["Season"] = season
                            grouped_df_individual["Group"] = group1
                            df = pd.concat([df, grouped_df_individual], ignore_index=True)

                        grouped_df = filtered_df[['Agent', 'Label', 'Balance']]
                        grouped_df["Season"] = season
                        grouped_df["Group"] = group
                        df = pd.concat([df, grouped_df], ignore_index=True)

        #fig = px.bar(df, x="Group", y="Balance", color="Label", facet_row="Agent",facet_row_spacing=0.001, facet_col="Season",
        #             category_orders={"Season": ["summer", "transition", "winter"]})

        fig = px.bar(df, x="Group", y="Balance", color="Label",facet_col="Season",color_discrete_sequence=['#EF553B','#636efa', '#00cc96'], category_orders={"Season": ["summer", "transition", "winter"]})
        title = scenario.capitalize() + '-Group ' + 'Total Amount of Balances over all Groups at Naive Forecasting'
        fig.layout.template = 'custom'
        fig.update_layout(yaxis_title="Amount in Euro")
        fig.for_each_annotation(lambda a: a.update(text=a.text.replace("Season=", "")))

        fig.update_layout(yaxis_range=[-1070, 30])
        fig.update_layout(autosize=False,
                          width=900,
                          height=1000)
        fig.update_layout(legend_title_text='')
        fig.update_layout(legend={
            "font": {
                "family": "Arial, monospace",
                "size": 35
            }})

        fig.update_layout(
            legend=dict(
                bordercolor="Black",
                borderwidth=2
            )
        )

        fig.show()


Measurements = ['Balance']
for scenario in list_scenarios:
    if not scenario == 'individual':
        df = pd.DataFrame(columns=['Agent', 'Label', 'Season', 'Group'])
        for season in list_seasons:

            for group in list_group_size:

                for forecasting_method in list_forecasting_methods:

                    if forecasting_method == "naive":
                        path = "05_results/scenario_pv_60_hp_60_ev_60_" + scenario + "_" + group + "_" + season + "_" + forecasting_method
                        path_individual = "05_results/scenario_pv_60_hp_60_ev_60_individual_" + season + "_" + forecasting_method

                        estimation_matrix_individual = f.load_file(
                            path=os.path.join(two_levels_up, path_individual, 'general', 'estimation_matrix.csv'),
                            df='pandas')
                        estimation_matrix = f.load_file(
                            path=os.path.join(two_levels_up, path, 'general', 'estimation_matrix.csv'), df='pandas')

                        estimation_matrix = estimation_matrix.groupby('Agent').agg({'Timestamp': 'first',
                                                                                          'Label': 'first',
                                                                                          'Underestimation': 'sum',
                                                                                          'Overestimation': 'sum',
                                                                                          'Benefits': 'sum',
                                                                                          'Benefits_Market': 'sum',
                                                                                          'Benefits_Balancing': 'sum',
                                                                                          'Costs': 'sum',
                                                                                          'Costs_Market': 'sum',
                                                                                          'Costs_Balancing': 'sum',
                                                                                          'Costs_Fees': 'sum',
                                                                                          'Balance': 'sum'}).reset_index()

                        if group == '5':
                            estimation_matrix_individual = estimation_matrix_individual.groupby('Agent').agg(
                                {'Timestamp': 'first',
                                 'Label': 'first',
                                 'Underestimation': 'sum',
                                 'Overestimation': 'sum',
                                 'Benefits': 'sum',
                                 'Benefits_Market': 'sum',
                                 'Benefits_Balancing': 'sum',
                                 'Costs': 'sum',
                                 'Costs_Market': 'sum',
                                 'Costs_Balancing': 'sum',
                                 'Costs_Fees': 'sum',
                                 'Balance': 'sum'}).reset_index()

                        # Filter rows with desired labels
                        filtered_df = estimation_matrix[
                            estimation_matrix['Label'].isin(['aggregated', 'consumer', 'prosumer'])]


                        # Filter rows with desired labels
                        filtered_df_individual = estimation_matrix_individual[
                            estimation_matrix_individual['Label'].isin(['aggregated', 'consumer', 'prosumer'])]



                        if group == '5':
                            group1 = '1'
                            grouped_df_individual = filtered_df_individual[['Agent', 'Label', 'Balance']]
                            grouped_df_individual["Season"] = season
                            grouped_df_individual["Group"] = group1
                            df = pd.concat([df, grouped_df_individual], ignore_index=True)

                        grouped_df = filtered_df[['Agent', 'Label', 'Balance']]
                        grouped_df["Season"] = season
                        grouped_df["Group"] = group
                        df = pd.concat([df, grouped_df], ignore_index=True)

        #fig = px.bar(df, x="Group", y="Balance", color="Label", facet_row="Agent",facet_row_spacing=0.001, facet_col="Season",
        #             category_orders={"Season": ["summer", "transition", "winter"]})

        df.loc[df['Label'] == 'consumer', 'Label'] = 'rosumer'

        custom_order = ['1', '5', '10', '15']

        # Convert the 'group' column to a categorical data type with the custom order
        df['Group'] = pd.Categorical(df['Group'], categories=custom_order, ordered=True)

        # Sort the DataFrame by 'label' and then by 'group'
        df = df.sort_values(by=['Label', 'Group'], ascending=[False, True])

        df.loc[df['Label'] == 'rosumer', 'Label'] = 'consumer'


        fig = px.bar(df, x="Group", y="Balance", color="Label",facet_col="Season",category_orders={"Season": ["summer", "transition", "winter"]})
        title = scenario.capitalize() + '-Group ' + 'Total Amount of Balances over all Groups at Naive Forecasting'
        fig.layout.template = 'custom'
        fig.update_layout(yaxis_title="Amount in Euro")

        fig.for_each_annotation(lambda a: a.update(text=a.text.replace("Season=", "")))

        fig.update_layout(yaxis_range=[-1070, 30])
        fig.update_layout(autosize=False,
                          width=900,
                          height=1000)
        fig.update_layout(legend_title_text='')
        fig.update_layout(legend={
            "font": {
                "family": "Arial, monospace",
                "size": 35
            }})

        fig.update_layout(
            legend=dict(
                bordercolor="Black",
                borderwidth=2
            ))

        fig.show()


Measurements = ['Balance']
for scenario in list_scenarios:
    if not scenario == 'individual':
        df = pd.DataFrame(columns=['Agent', 'Label', 'Season', 'Group'])
        for season in list_seasons:

            for group in list_group_size:

                for forecasting_method in list_forecasting_methods:

                    if forecasting_method == "naive":
                        path = "05_results/scenario_pv_60_hp_60_ev_60_" + scenario + "_" + group + "_" + season + "_" + forecasting_method
                        path_individual = "05_results/scenario_pv_60_hp_60_ev_60_individual_" + season + "_" + forecasting_method

                        estimation_matrix_individual = f.load_file(
                            path=os.path.join(two_levels_up, path_individual, 'general', 'estimation_matrix.csv'),
                            df='pandas')
                        estimation_matrix = f.load_file(
                            path=os.path.join(two_levels_up, path, 'general', 'estimation_matrix.csv'), df='pandas')

                        estimation_matrix = estimation_matrix.groupby('Label').agg({'Timestamp': 'first',
                                                                                          'Underestimation': 'sum',
                                                                                          'Overestimation': 'sum',
                                                                                          'Benefits': 'sum',
                                                                                          'Benefits_Market': 'sum',
                                                                                          'Benefits_Balancing': 'sum',
                                                                                          'Costs': 'sum',
                                                                                          'Costs_Market': 'sum',
                                                                                          'Costs_Balancing': 'sum',
                                                                                          'Costs_Fees': 'sum',
                                                                                          'Balance': 'sum'}).reset_index()

                        if group == '5':
                            estimation_matrix_individual = estimation_matrix_individual.groupby('Label').agg(
                                {'Timestamp': 'first',
                                 'Underestimation': 'sum',
                                 'Overestimation': 'sum',
                                 'Benefits': 'sum',
                                 'Benefits_Market': 'sum',
                                 'Benefits_Balancing': 'sum',
                                 'Costs': 'sum',
                                 'Costs_Market': 'sum',
                                 'Costs_Balancing': 'sum',
                                 'Costs_Fees': 'sum',
                                 'Balance': 'sum'}).reset_index()

                        # Filter rows with desired labels
                        filtered_df = estimation_matrix[
                            estimation_matrix['Label'].isin(['aggregated', 'consumer', 'prosumer'])]

                        # Filter rows with desired labels
                        filtered_df_individual = estimation_matrix_individual[
                            estimation_matrix_individual['Label'].isin(['aggregated', 'consumer', 'prosumer'])]

                        filtered_df_aggre = estimation_matrix[estimation_matrix['Label'].isin(['Aggregator'])]

                        filtered_df.loc[filtered_df['Label'] == 'aggregated', 'Balance'] = filtered_df.loc[filtered_df['Label'] == 'aggregated', 'Balance'].sum() + filtered_df.loc[filtered_df['Label'] == 'aggregated', 'Costs_Fees'].sum() - filtered_df_aggre['Costs_Fees'].sum()

                        if group == '5':
                            group1 = '1'
                            grouped_df_individual = filtered_df_individual[['Label', 'Balance']]
                            grouped_df_individual["Season"] = season
                            grouped_df_individual["Group"] = group1
                            df = pd.concat([df, grouped_df_individual], ignore_index=True)

                        grouped_df = filtered_df[['Label', 'Balance']]
                        grouped_df["Season"] = season
                        grouped_df["Group"] = group
                        df = pd.concat([df, grouped_df], ignore_index=True)

        #fig = px.bar(df, x="Group", y="Balance", color="Label", facet_row="Agent",facet_row_spacing=0.001, facet_col="Season",
        #             category_orders={"Season": ["summer", "transition", "winter"]})

        df.loc[df['Label'] == 'prosumer', 'Label'] = 'xxosumer'

        custom_order = ['1', '5', '10', '15']

        # Convert the 'group' column to a categorical data type with the custom order
        df['Group'] = pd.Categorical(df['Group'], categories=custom_order, ordered=True)

        # Sort the DataFrame by 'label' and then by 'group'
        df = df.sort_values(by=['Group', 'Label'], ascending=[True, False])

        df.loc[df['Label'] == 'xxosumer', 'Label'] = 'prosumer'

        fig = px.bar(df, x="Group", y="Balance", color="Label",facet_col="Season",color_discrete_sequence=['#EF553B','#636efa', '#00cc96'],category_orders={"Season": ["summer", "transition", "winter"]})
        title = scenario.capitalize() + '-Group ' + 'Total Amount of Balances at Naive Forecasting if no Fees exist for the Aggregated Agents'
        fig.layout.template = 'custom'
        fig.for_each_annotation(lambda a: a.update(text=a.text.replace("Season=", "")))

        fig.update_layout(yaxis_range=[-1070, 30])
        fig.update_layout(autosize=False,
                          width=900,
                          height=1000)
        fig.update_layout(legend_title_text='')
        fig.update_layout(legend={
            "font": {
                "family": "Arial, monospace",
                "size": 35
            }})

        fig.update_layout(
            legend=dict(
                bordercolor="Black",
                borderwidth=2
            ))
        fig.update_layout(yaxis_title="Amount in Euro")

        fig.show()

## ERROR PLOTS ##



error_list = ["MAE", "RMSE","NRMSE", "MAPE", "MAAPE2"]
#error_list = ["RMSE"]

for error in error_list:

    for scenario in list_scenarios:
        if not scenario == 'individual':

            dict_season = ndict({'mean': ndict({'summer': {}, 'transition': {}, 'winter': {}}),
                                 'median': ndict({'summer': {}, 'transition': {}, 'winter': {}})})
            dict_group_aggregator = ndict({'mean': ndict({'summer': {}, 'transition': {}, 'winter': {}}),
                                           'median': ndict({'summer': {}, 'transition': {}, 'winter': {}})})
            dict_group_consumer = ndict({'mean': ndict({'summer': {}, 'transition': {}, 'winter': {}}),
                                         'median': ndict({'summer': {}, 'transition': {}, 'winter': {}})})
            dict_group_prosumer = ndict({'mean': ndict({'summer': {}, 'transition': {}, 'winter': {}}),
                                         'median': ndict({'summer': {}, 'transition': {}, 'winter': {}})})
            d = {"MAE": [], "RMSE": [], "NRMSE": [], "MAPE": [], "MAAPE2": []}
            df = pd.DataFrame(data=d)

            for season in list_seasons:
                group_list_mean = np.array([])
                aggregator_list_mean = np.array([])
                consumers_list_mean = np.array([])
                prosumers_list_mean = np.array([])

                group_list_median = np.array([])
                aggregator_list_median = np.array([])
                consumers_list_median = np.array([])
                prosumers_list_median = np.array([])

                for group in list_group_size:

                    for forecasting_method in list_forecasting_methods:
                        if forecasting_method == "naive":

                            path = "05_results/scenario_pv_60_hp_60_ev_60_" + scenario + "_" + group + "_" + season + "_" + forecasting_method
                            path_individual = "05_results/scenario_pv_60_hp_60_ev_60_individual_" + season + "_" + forecasting_method


                            error_matrix_individual = f.load_file(path=os.path.join(two_levels_up, path_individual, 'general', 'error_matrix.csv'), df='pandas')
                            error_matrix = f.load_file(path=os.path.join(two_levels_up, path, 'general', 'error_matrix.csv'), df='pandas')

                            error_matrix = error_matrix.fillna(0)
                            error_matrix[error] = error_matrix[error].abs()


                            if group == '5':
                                my_list = list(error_matrix_individual[error])
                                group_list_mean = np.append(group_list_mean, sum(my_list) / len(my_list))
                                group_list_median = np.append(group_list_median, statistics.median(my_list))


                            consumers_list_mean = np.append(consumers_list_mean,
                                                       error_matrix.groupby('Label').get_group('consumer')[error].mean())
                            consumers_list_median = np.append(consumers_list_median,
                                                            statistics.median(error_matrix.groupby('Label').get_group('consumer')[
                                                                error]))
                            prosumers_list_mean = np.append(prosumers_list_mean,
                                                       error_matrix.groupby('Label').get_group('prosumer')[error].mean())
                            prosumers_list_median = np.append(prosumers_list_median,
                                                            statistics.median(error_matrix.groupby('Label').get_group('prosumer')[
                                                                error]))

                            value_to_add = float(error_matrix.groupby('Label').get_group('Aggregator')[error].mean())
                            error_matrix = error_matrix[
                                error_matrix['Label'].isin(['consumer', 'prosumer'])]

                            value_to_add_single = value_to_add/int(''.join(filter(str.isdigit, group)))
                            aggregator_list_mean = np.append(aggregator_list_mean, value_to_add_single)

                            n = int(''.join(filter(str.isdigit, group)))
                            my_list = list(error_matrix[error])
                            my_list.extend([value_to_add_single] * n)
                            group_list_mean = np.append(group_list_mean, sum(my_list) / len(my_list))
                            #value_to_add = float(error_matrix.groupby('Label').get_group('Aggregator')[error].median())
                            value_to_add_single = value_to_add / int(''.join(filter(str.isdigit, group)))
                            aggregator_list_median = np.append(aggregator_list_mean, value_to_add_single)

                            n = int(''.join(filter(str.isdigit, group)))
                            my_list = list(error_matrix[error])
                            my_list.extend([value_to_add_single] * n)
                            group_list_median = np.append(group_list_median, statistics.median(my_list))

                dict_season['mean'][season] = group_list_mean
                dict_group_aggregator['mean'][season] = aggregator_list_mean
                dict_group_consumer['mean'][season] = consumers_list_mean
                dict_group_prosumer['mean'][season] = prosumers_list_mean

                dict_season['median'][season] = group_list_median
                dict_group_aggregator['median'][season] = aggregator_list_median
                dict_group_consumer['median'][season] = consumers_list_median
                dict_group_prosumer['median'][season] = prosumers_list_median


            x1 = np.array(['1', '5', '10', '15'])
            x = np.array(['5', '10', '15'])
            measures = ['mean', 'median']

            for measure in measures:
                if error in ["MAAPE2", "MAPE", "NRMSE"]:
                    yaxis_title = measure.capitalize() + " " + error.capitalize() + ' [%]'
                else:
                    yaxis_title = measure.capitalize() + " " + error.capitalize()

                fig = go.Figure()

                # line_shape='spline'

                #fig.add_trace(go.Scatter(x=x1, y=dict_season[measure]['summer'], name="summer", marker=dict(color='rgb(158,202,225)', size=15), line=dict(color='rgb(158,202,225)', width=6)))
                #fig.add_trace(go.Scatter(x=x1, y=dict_season[measure]['winter'], name="winter", marker=dict(color='rgb(49,130,189)', size=15), line=dict(color='rgb(49,130,189)', width=6)))
                #fig.add_trace(go.Scatter(x=x1, y=dict_season[measure]['transition'], name='transition', marker=dict(color='#64A0C8', size=15), line=dict(color='#64A0C8', width=6)))
                fig.add_trace(go.Scatter(x=x1, y=dict_season[measure]['summer'], name="summer", marker=dict(color='orange', size=10), line=dict(color='orange', width=4)))
                fig.add_trace(go.Scatter(x=x1, y=dict_season[measure]['transition'], name="transition", marker=dict(color='green', size=10), line=dict(color='green', width=4)))
                fig.add_trace(go.Scatter(x=x1, y=dict_season[measure]['winter'], name='winter', marker=dict(color='rgb(49,130,189)', size=10), line=dict(color='rgb(49,130,189)', width=4)))


                fig.update_traces(hoverinfo='text+name', mode='lines+markers')
                res = scenario[0].upper() + scenario[1:]
                title = res + '-Group Forecast Error for the Aggregator'
                # Edit the layout
                fig.layout.template = 'custom'
                if error == "MAAPE2":
                    fig.update_layout(yaxis_range=[-0.01, 0.23])
                    fig.update_layout(xaxis_title='Group Size of the Aggregator', yaxis_title = 'Average MAAPE [%]'
                                      )

                if error == "RMSE":
                    fig.update_layout(yaxis_range=[-0.01, 0.257])
                    fig.update_layout(xaxis_title='Group Size of the Aggregator', yaxis_title = 'Average RMSE [kWh]')

                if error == "MAE":
                    fig.update_layout(yaxis_range=[-0.01, 0.11])
                    fig.update_layout(xaxis_title='Group Size of the Aggregator', yaxis_title = 'Average MAE [kWh]'
                                      )

                fig.update_layout(autosize=False,
                                  width=800,
                                  height=800)
                fig.update_layout(legend={
                    "font": {
                        "family": "Arial, monospace",
                        "size": 33
                    }})
                fig.show()

## ERROR PLOTS ##

error_list = ["MAAPE2", "MAE", "RMSE"]

for error in error_list:

    for scenario in list_scenarios:
        if scenario == 'individual':
            fig = go.Figure()
            for season in list_seasons:

                path_individual = "05_results/scenario_pv_60_hp_60_ev_60_individual_" + season + "_naive"

                error_matrix_individual = f.load_file(
                    path=os.path.join(two_levels_up, path_individual, 'general', 'error_matrix.csv'), df='pandas')

                my_list = list(error_matrix_individual[error])
                if season == "summer":
                    list_summer = np.array(my_list)
                    summer = np.array(error_matrix_individual["Label"])
                if season == "transition":
                    list_trans = np.array(my_list)
                    trans = np.array(error_matrix_individual["Label"])
                if season == "winter":
                    list_winter = np.array(my_list)
                    winter = np.array(error_matrix_individual["Label"])

                fig.add_trace(go.Box(
                    y=my_list,
                    name=season.capitalize(),
                    boxpoints='outliers',  # only outliers
                    marker_color='rgb(107,174,214)',
                    line_color='rgb(107,174,214)'
                ))

            title_text = error.capitalize() + " Error Boxplot"
            fig.update_layout(title_text=title_text)
            fig.show()

            df = pd.DataFrame({'Season': ['Summer'] * len(list_summer) + ['Transition'] * len(list_trans) + [
                'Winter'] * len(list_winter),
                               'value': np.concatenate([list_summer, list_trans, list_winter], 0),
                               'color': np.concatenate([summer, trans, winter], 0)}
                              )

            fig = px.strip(df,
                           x='Season',
                           y='value',
                           color='color',
                           stripmode='overlay')

            fig.add_trace(go.Box(y=df.query('Season == "Summer"')['value'], name='Summer'))
            fig.add_trace(go.Box(y=df.query('Season == "Transition"')['value'], name='Transition'))
            fig.add_trace(go.Box(y=df.query('Season == "Winter"')['value'], name='Winter'))

            fig.update_layout(autosize=False,
                              width=900,
                              height=900,
                              legend={'traceorder': 'normal'})
            if error == "MAAPE2":
                yaxis_title = "MAAPE [%]"

            if error == "NRMSE":
                yaxis_title = "NRMSE [%]"

            if error == "MAE":
                yaxis_title = "MAE [kWh]"

            if error == "RMSE":
                yaxis_title = "RMSE [kWh]"

            fig.update_layout(yaxis_title=yaxis_title)
            fig.update_layout(
                legend_title_text='Legend',
                legend_title_font=dict(family='Arial', size=33)
            )
            fig.update_layout(
                font=dict(size=30)
            )
            fig.update_layout(
                legend=dict(

                    font=dict(

                        size=33
                    )
                )
            )



            fig.show()


for scenario in list_scenarios:
    if scenario == 'individual':
        dict_season = ndict({'balance': ndict({'summer': {}, 'transition': {}, 'winter': {}}),
                             'median': ndict({'summer': {}, 'transition': {}, 'winter': {}})})
        dict_group_aggregated = ndict({'balance': ndict({'summer': {}, 'transition': {}, 'winter': {}}),
                                       'median': ndict({'summer': {}, 'transition': {}, 'winter': {}})})
        dict_group_consumer = ndict({'balance': ndict({'summer': {}, 'transition': {}, 'winter': {}}),
                                     'median': ndict({'summer': {}, 'transition': {}, 'winter': {}})})
        dict_group_prosumer = ndict({'balance': ndict({'summer': {}, 'transition': {}, 'winter': {}}),
                                     'median': ndict({'summer': {}, 'transition': {}, 'winter': {}})})
        for season in list_seasons:
            group_list_naive = np.array([])
            aggregated_list_naive = np.array([])
            consumers_list_naive = np.array([])
            prosumers_list_naive = np.array([])

            group_list_perfect = np.array([])
            aggregated_list_perfect = np.array([])
            consumers_list_perfect = np.array([])
            prosumers_list_perfect = np.array([])

            median_naiv = np.array([])
            median_perf = np.array([])

            for forecasting_method in list_forecasting_methods:

                if forecasting_method == "naive":

                    path_individual = "05_results/scenario_pv_60_hp_60_ev_60_individual_" + season + "_" + forecasting_method

                    estimation_matrix_individual = f.load_file(path=os.path.join(two_levels_up, path_individual, 'general', 'estimation_matrix.csv'),df='pandas')

                    estimation_matrix_individual = estimation_matrix_individual[
                        estimation_matrix_individual['Label'].isin(['aggregated', 'consumer', 'prosumer'])]

                    estimation_matrix_individual = estimation_matrix_individual.groupby('Agent').agg({'Timestamp': 'first',
                                                                                      'Label': 'first',
                                                                                      'Underestimation': 'sum',
                                                                                      'Overestimation': 'sum',
                                                                                      'Benefits': 'sum',
                                                                                      'Benefits_Market': 'sum',
                                                                                        'Benefits_Balancing': 'sum',
                                                                                        'Costs': 'sum',
                                                                                         'Costs_Market': 'sum',
                                                                                         'Costs_Balancing': 'sum',
                                                                                         'Costs_Fees': 'sum',
                                                                                      'Balance': 'sum'})

                    group_list_naive = np.append(group_list_naive, estimation_matrix_individual['Balance'])
                    #median_naiv = np.append(median_naiv, statistics.median(estimation_matrix_individual['Balance']))

                if forecasting_method == "perfect":
                    path_individual = "05_results/scenario_pv_60_hp_60_ev_60_individual_" + season + "_" + forecasting_method

                    estimation_matrix_individual = f.load_file(path=os.path.join(two_levels_up, path_individual, 'general', 'estimation_matrix.csv'), df='pandas')

                    estimation_matrix_individual = estimation_matrix_individual[
                        estimation_matrix_individual['Label'].isin(['aggregated', 'consumer', 'prosumer'])]

                    estimation_matrix_individual = estimation_matrix_individual.groupby('Agent').agg(
                        {'Timestamp': 'first',
                         'Label': 'first',
                          'Underestimation': 'sum',
                          'Overestimation': 'sum',
                          'Benefits': 'sum',
                          'Benefits_Market': 'sum',
                            'Benefits_Balancing': 'sum',
                            'Costs': 'sum',
                             'Costs_Market': 'sum',
                             'Costs_Balancing': 'sum',
                             'Costs_Fees': 'sum',
                          'Balance': 'sum'})

                    group_list_perfect = np.append(group_list_perfect,
                                                 estimation_matrix_individual['Balance'])

                    #median_perf = np.append(median_perf, statistics.median(estimation_matrix_individual['Balance']))

            if season == "summer":
                #list_summer = np.array(my_list)
                summer = np.array(estimation_matrix_individual["Label"])
            if season == "transition":
                #list_trans = np.array(my_list)
                trans = np.array(estimation_matrix_individual["Label"])
            if season == "winter":
                #list_winter = np.array(my_list)
                winter = np.array(estimation_matrix_individual["Label"])

            dict_season['balance'][season] = np.array([- i + j for i, j in zip(group_list_naive, group_list_perfect)])
            pitn = str(season) + str(statistics.median(dict_season['balance'][season]))
            print(pitn)



        df = pd.DataFrame({'Season': ['Summer'] * len(dict_season['balance']['summer']) + ['Transition'] * len(dict_season['balance']['transition']) + [
            'Winter'] * len(dict_season['balance']['winter']),
                           'value': np.concatenate([dict_season['balance']['summer'], dict_season['balance']['transition'], dict_season['balance']['winter']], 0),
                           'color': np.concatenate([summer, trans, winter], 0)}
                          )

        fig = px.strip(df,
                       x='Season',
                       y='value',
                       color='color',
                       stripmode='overlay')

        fig.add_trace(go.Box(y=df.query('Season == "Summer"')['value'], name='Summer'))
        fig.add_trace(go.Box(y=df.query('Season == "Transition"')['value'], name='Transition'))
        fig.add_trace(go.Box(y=df.query('Season == "Winter"')['value'], name='Winter'))



        yaxis_title = 'Forecast Error Costs in Euro'


        fig.update_layout(autosize=False,
                          width=900,
                          height=900,
                          legend={'traceorder': 'normal'})


        fig.update_layout(yaxis_title=yaxis_title)
        fig.update_layout(
            legend_title_text='Legend',
            legend_title_font=dict(family='Arial', size=33)
        )
        fig.update_layout(
            font=dict(size=30)
        )
        fig.update_layout(
            legend=dict(

                font=dict(

                    size=33
                )
            )
        )

        fig.show()

for scenario in list_scenarios:
    if scenario == 'individual':
        dict_season = ndict({'balance': ndict({'summer': {}, 'transition': {}, 'winter': {}}),
                             'median': ndict({'summer': {}, 'transition': {}, 'winter': {}})})
        dict_group_aggregated = ndict({'balance': ndict({'summer': {}, 'transition': {}, 'winter': {}}),
                                       'median': ndict({'summer': {}, 'transition': {}, 'winter': {}})})
        dict_group_consumer = ndict({'balance': ndict({'summer': {}, 'transition': {}, 'winter': {}}),
                                     'median': ndict({'summer': {}, 'transition': {}, 'winter': {}})})
        dict_group_prosumer = ndict({'balance': ndict({'summer': {}, 'transition': {}, 'winter': {}}),
                                     'median': ndict({'summer': {}, 'transition': {}, 'winter': {}})})
        for season in list_seasons:
            group_list_naive = np.array([])
            aggregated_list_naive = np.array([])
            consumers_list_naive = np.array([])
            prosumers_list_naive = np.array([])

            group_list_perfect = np.array([])
            aggregated_list_perfect = np.array([])
            consumers_list_perfect = np.array([])
            prosumers_list_perfect = np.array([])

            median_naiv = np.array([])
            median_perf = np.array([])

            for forecasting_method in list_forecasting_methods:

                if forecasting_method == "naive":

                    path_individual = "05_results/scenario_pv_60_hp_60_ev_60_individual_" + season + "_" + forecasting_method

                    estimation_matrix_individual = f.load_file(path=os.path.join(two_levels_up, path_individual, 'general', 'estimation_matrix.csv'),df='pandas')

                    estimation_matrix_individual = estimation_matrix_individual[
                        estimation_matrix_individual['Label'].isin(['aggregated', 'consumer', 'prosumer'])]

                    estimation_matrix_individual = estimation_matrix_individual.groupby('Agent').agg({'Timestamp': 'first',
                                                                                      'Label': 'first',
                                                                                      'Underestimation': 'sum',
                                                                                      'Overestimation': 'sum',
                                                                                      'Benefits': 'sum',
                                                                                      'Benefits_Market': 'sum',
                                                                                        'Benefits_Balancing': 'sum',
                                                                                        'Costs': 'sum',
                                                                                         'Costs_Market': 'sum',
                                                                                         'Costs_Balancing': 'sum',
                                                                                         'Costs_Fees': 'sum',
                                                                                      'Balance': 'sum'})


                    estimation_matrix_individual['Total_Error'] = estimation_matrix_individual['Overestimation'] + \
                                                                  estimation_matrix_individual[
                                                                      'Underestimation']

                    estimation_matrix_individual['Total_Error'] = estimation_matrix_individual['Total_Error'] * 0.01

                    group_list_naive = np.append(group_list_naive, estimation_matrix_individual['Total_Error'])

            if season == "summer":
                #list_summer = np.array(my_list)
                summer = np.array(estimation_matrix_individual["Label"])
            if season == "transition":
                #list_trans = np.array(my_list)
                trans = np.array(estimation_matrix_individual["Label"])
            if season == "winter":
                #list_winter = np.array(my_list)
                winter = np.array(estimation_matrix_individual["Label"])

            dict_season['balance'][season] = np.array(group_list_naive)
            pitn = str(season) + str(statistics.median(group_list_naive))
            print(pitn)



        df = pd.DataFrame({'Season': ['Summer'] * len(dict_season['balance']['summer']) + ['Transition'] * len(dict_season['balance']['transition']) + [
            'Winter'] * len(dict_season['balance']['winter']),
                           'value': np.concatenate([dict_season['balance']['summer'], dict_season['balance']['transition'], dict_season['balance']['winter']], 0),
                           'color': np.concatenate([summer, trans, winter], 0)}
                          )

        fig = px.strip(df,
                       x='Season',
                       y='value',
                       color='color',
                       stripmode='overlay')

        fig.add_trace(go.Box(y=df.query('Season == "Summer"')['value'], name='Summer'))
        fig.add_trace(go.Box(y=df.query('Season == "Transition"')['value'], name='Transition'))
        fig.add_trace(go.Box(y=df.query('Season == "Winter"')['value'], name='Winter'))

        yaxis_title = 'Forecast Error Costs in Euro'

        fig.update_layout(autosize=False,
                          width=900,
                          height=900,
                          legend={'traceorder': 'normal'})

        fig.update_layout(yaxis_title=yaxis_title)
        fig.update_layout(
            legend_title_text='Legend',
            legend_title_font=dict(family='Arial', size=33)
        )
        fig.update_layout(
            font=dict(size=30)
        )
        fig.update_layout(
            legend=dict(

                font=dict(

                    size=33
                )
            )
        )

        fig.show()




Measurements = ['Benefits_Market', 'Benefits_Balancing', 'Costs_Market', 'Costs_Balancing', 'Costs_Fees']
for scenario in list_scenarios:
    if not scenario == 'individual':
        df = pd.DataFrame(columns=['Group', 'Benefits/Costs subcategories', 'Season', 'Benefits/Costs in Euro'])
        for season in list_seasons:

            for group in list_group_size:

                for forecasting_method in list_forecasting_methods:

                    if forecasting_method == "naive":
                        path = "05_results/scenario_pv_60_hp_60_ev_60_" + scenario + "_" + group + "_" + season + "_" + forecasting_method
                        path_individual = "05_results/scenario_pv_60_hp_60_ev_60_individual_" + season + "_" + forecasting_method

                        estimation_matrix_individual = f.load_file(
                            path=os.path.join(two_levels_up, path_individual, 'general', 'estimation_matrix.csv'),
                            df='pandas')
                        estimation_matrix = f.load_file(
                            path=os.path.join(two_levels_up, path, 'general', 'estimation_matrix.csv'), df='pandas')

                        estimation_matrix_individual = estimation_matrix_individual.groupby('Agent').agg(
                            {'Timestamp': 'first',
                             'Label': 'first',
                             'Underestimation': 'sum',
                             'Overestimation': 'sum',
                             'Benefits': 'sum',
                             'Benefits_Market': 'sum',
                             'Benefits_Balancing': 'sum',
                             'Costs': 'sum',
                             'Costs_Market': 'sum',
                             'Costs_Balancing': 'sum',
                             'Costs_Fees': 'sum',
                             'Balance': 'sum'})

                        estimation_matrix = estimation_matrix.groupby('Agent').agg(
                            {'Timestamp': 'first',
                             'Label': 'first',
                             'Underestimation': 'sum',
                             'Overestimation': 'sum',
                             'Benefits': 'sum',
                             'Benefits_Market': 'sum',
                             'Benefits_Balancing': 'sum',
                             'Costs': 'sum',
                             'Costs_Market': 'sum',
                             'Costs_Balancing': 'sum',
                             'Costs_Fees': 'sum',
                             'Balance': 'sum'})

                        # Filter rows with desired labels
                        estimation_matrix = estimation_matrix[
                            estimation_matrix['Label'].isin(['aggregated', 'consumer', 'prosumer'])]
                        estimation_matrix_individual = estimation_matrix_individual[
                            estimation_matrix_individual['Label'].isin(['aggregated', 'consumer', 'prosumer'])]

                        if group == '5':
                            for measure in Measurements:
                                amount = estimation_matrix_individual[measure].median()
                                if "Costs" in measure:
                                    amount = -amount

                                group1 = '1'
                                df_row = [group1, measure, season, amount]

                                df.loc[len(df.index)] = df_row

                        for measure in Measurements:

                            amount = estimation_matrix[measure].median()

                            if "Costs" in measure:
                                amount = -amount

                            df_row = [group, measure, season, amount]

                            df.loc[len(df.index)] = df_row

        fig = px.bar(df, x="Group", y="Benefits/Costs in Euro", color="Benefits/Costs subcategories", barmode="relative",facet_col="Season",
                     category_orders={"Season": ["summer", "transition", "winter"]})
        #fig.layout.template = 'seaborn'
        col = 1
        for season in list_seasons:
            x3 = 0.5
            for group in list_group_size:

                for forecasting_method in list_forecasting_methods:

                    if forecasting_method == "naive":
                        path = "05_results/scenario_pv_60_hp_60_ev_60_" + scenario + "_" + group + "_" + season + "_" + forecasting_method
                        path_individual = "05_results/scenario_pv_60_hp_60_ev_60_individual_" + season + "_" + forecasting_method

                        estimation_matrix_individual = f.load_file(
                            path=os.path.join(two_levels_up, path_individual, 'general', 'estimation_matrix.csv'),
                            df='pandas')
                        estimation_matrix = f.load_file(
                            path=os.path.join(two_levels_up, path, 'general', 'estimation_matrix.csv'), df='pandas')

                        estimation_matrix_individual = estimation_matrix_individual.groupby('Agent').agg(
                            {'Timestamp': 'first',
                             'Label': 'first',
                             'Underestimation': 'sum',
                             'Overestimation': 'sum',
                             'Benefits': 'sum',
                             'Benefits_Market': 'sum',
                             'Benefits_Balancing': 'sum',
                             'Costs': 'sum',
                             'Costs_Market': 'sum',
                             'Costs_Balancing': 'sum',
                             'Costs_Fees': 'sum',
                             'Balance': 'sum'})

                        estimation_matrix = estimation_matrix.groupby('Agent').agg(
                            {'Timestamp': 'first',
                             'Label': 'first',
                             'Underestimation': 'sum',
                             'Overestimation': 'sum',
                             'Benefits': 'sum',
                             'Benefits_Market': 'sum',
                             'Benefits_Balancing': 'sum',
                             'Costs': 'sum',
                             'Costs_Market': 'sum',
                             'Costs_Balancing': 'sum',
                             'Costs_Fees': 'sum',
                             'Balance': 'sum'})

                        # Filter rows with desired labels
                        estimation_matrix = estimation_matrix[
                            estimation_matrix['Label'].isin(['aggregated', 'consumer', 'prosumer'])]
                        estimation_matrix_individual = estimation_matrix_individual[
                            estimation_matrix_individual['Label'].isin(['aggregated', 'consumer', 'prosumer'])]

                        amount = estimation_matrix['Balance'].median()

                        x4 = x3 + 1
                        fig.add_shape(
                            type='line', line={'color': 'rgb(50, 171, 96)', 'width': 4},
                            xref="x",
                            yref="y",
                            x0=x3,
                            x1=x4,
                            y0=amount,
                            y1=amount,
                            row=1,
                            col=col
                        )
                        x3 = x3 + 1

                        if group == '5':

                            amount = estimation_matrix_individual['Balance'].median()

                            x1 = -0.5
                            x2 = 0.5
                            if season == "summer":
                                fig.add_shape(
                                    name="Balance",
                                    showlegend=True,
                                    type='line', line={'color': 'rgb(50, 171, 96)','width': 4},
                                    xref="x",
                                    yref="y",
                                    x0=x1,
                                    x1=x2,
                                    y0=amount,
                                    y1=amount,
                                    row=1,
                                    col=col
                                )
                            else:
                                fig.add_shape(
                                    name="Balance",
                                    type='line', line={'color': 'rgb(50, 171, 96)', 'width': 4},
                                    xref="x",
                                    yref="y",
                                    x0=x1,
                                    x1=x2,
                                    y0=amount,
                                    y1=amount,
                                    row=1,
                                    col=col
                                )

            col = col + 1

        title = scenario.capitalize() + '-Group ' + 'Benefits/Costs per Agent over all Groups at Naive Forecasting'

        fig.update_layout(
            title=title, yaxis_title="Amount in Euro"
        )
        fig.show()


Measurements = ['Benefits_Market', 'Benefits_Balancing', 'Costs_Market', 'Costs_Balancing', 'Costs_Fees']
for scenario in list_scenarios:
    if not scenario == 'individual':
        df = pd.DataFrame(columns=['Group', 'Benefits/Costs subcategories', 'Season', 'Benefits/Costs in Euro'])
        for season in list_seasons:

            for group in list_group_size:

                for forecasting_method in list_forecasting_methods:

                    if forecasting_method == "naive":
                        path = "05_results/scenario_pv_60_hp_60_ev_60_" + scenario + "_" + group + "_" + season + "_" + forecasting_method
                        path_individual = "05_results/scenario_pv_60_hp_60_ev_60_individual_" + season + "_" + forecasting_method

                        estimation_matrix_individual = f.load_file(
                            path=os.path.join(two_levels_up, path_individual, 'general', 'estimation_matrix.csv'),
                            df='pandas')
                        estimation_matrix = f.load_file(
                            path=os.path.join(two_levels_up, path, 'general', 'estimation_matrix.csv'), df='pandas')

                        estimation_matrix_individual = estimation_matrix_individual.groupby('Agent').agg(
                            {'Timestamp': 'first',
                             'Label': 'first',
                             'Underestimation': 'sum',
                             'Overestimation': 'sum',
                             'Benefits': 'sum',
                             'Benefits_Market': 'sum',
                             'Benefits_Balancing': 'sum',
                             'Costs': 'sum',
                             'Costs_Market': 'sum',
                             'Costs_Balancing': 'sum',
                             'Costs_Fees': 'sum',
                             'Balance': 'sum'})

                        estimation_matrix = estimation_matrix.groupby('Agent').agg(
                            {'Timestamp': 'first',
                             'Label': 'first',
                             'Underestimation': 'sum',
                             'Overestimation': 'sum',
                             'Benefits': 'sum',
                             'Benefits_Market': 'sum',
                             'Benefits_Balancing': 'sum',
                             'Costs': 'sum',
                             'Costs_Market': 'sum',
                             'Costs_Balancing': 'sum',
                             'Costs_Fees': 'sum',
                             'Balance': 'sum'})

                        # Filter rows with desired labels
                        estimation_matrix = estimation_matrix[
                            estimation_matrix['Label'].isin(['aggregated', 'consumer', 'prosumer'])]
                        estimation_matrix_individual = estimation_matrix_individual[
                            estimation_matrix_individual['Label'].isin(['aggregated', 'consumer', 'prosumer'])]

                        if group =='5':
                            for measure in Measurements:
                                amount = estimation_matrix_individual[measure].mean()
                                if "Costs" in measure:
                                    amount = -amount
                                group1 = '1'
                                df_row = [group1, measure, season, amount]

                                df.loc[len(df.index)] = df_row


                        for measure in Measurements:

                            amount = estimation_matrix[measure].mean()
                            if "Costs" in measure:
                                amount = -amount

                            df_row = [group, measure, season, amount]

                            df.loc[len(df.index)] = df_row


        fig = px.bar(df, x="Group", y="Benefits/Costs in Euro", color="Benefits/Costs subcategories", barmode="relative", facet_col="Season",
                     category_orders={"Season": ["summer", "transition", "winter"]})
        #df = {'col1': ['DarkSlateGrey', 'green'], 'col2': ['red', 'blue']}
        fig.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')), opacity=0.7)
        #fig.layout.template = 'custom'
        col = 1
        for season in list_seasons:
            x3 = 0.5
            for group in list_group_size:

                for forecasting_method in list_forecasting_methods:

                    if forecasting_method == "perfect":
                        path = "05_results/scenario_pv_60_hp_60_ev_60_" + scenario + "_" + group + "_" + season + "_" + forecasting_method
                        path_individual = "05_results/scenario_pv_60_hp_60_ev_60_individual_" + season + "_" + forecasting_method

                        estimation_matrix_individual = f.load_file(
                            path=os.path.join(two_levels_up, path_individual, 'general', 'estimation_matrix.csv'),
                            df='pandas')
                        estimation_matrix = f.load_file(
                            path=os.path.join(two_levels_up, path, 'general', 'estimation_matrix.csv'), df='pandas')

                        estimation_matrix_individual = estimation_matrix_individual.groupby('Agent').agg(
                            {'Timestamp': 'first',
                             'Label': 'first',
                             'Underestimation': 'sum',
                             'Overestimation': 'sum',
                             'Benefits': 'sum',
                             'Benefits_Market': 'sum',
                             'Benefits_Balancing': 'sum',
                             'Costs': 'sum',
                             'Costs_Market': 'sum',
                             'Costs_Balancing': 'sum',
                             'Costs_Fees': 'sum',
                             'Balance': 'sum'})

                        estimation_matrix = estimation_matrix.groupby('Agent').agg(
                            {'Timestamp': 'first',
                             'Label': 'first',
                             'Underestimation': 'sum',
                             'Overestimation': 'sum',
                             'Benefits': 'sum',
                             'Benefits_Market': 'sum',
                             'Benefits_Balancing': 'sum',
                             'Costs': 'sum',
                             'Costs_Market': 'sum',
                             'Costs_Balancing': 'sum',
                             'Costs_Fees': 'sum',
                             'Balance': 'sum'})

                        # Filter rows with desired labels
                        estimation_matrix = estimation_matrix[
                            estimation_matrix['Label'].isin(['aggregated', 'consumer', 'prosumer'])]
                        estimation_matrix_individual = estimation_matrix_individual[
                            estimation_matrix_individual['Label'].isin(['aggregated', 'consumer', 'prosumer'])]

                        amount = estimation_matrix['Balance'].mean()

                        x4 = x3 + 1
                        fig.add_shape(
                            type='line', line={'color': 'DarkSlateGrey', 'width': 4},
                            xref="x",
                            yref="y",
                            x0=x3,
                            x1=x4,
                            y0=amount,
                            y1=amount,
                            row=1,
                            col=col,
                            opacity=0.7
                        )
                        x3 = x3 + 1

                        if group == '5':

                            amount = estimation_matrix_individual['Balance'].mean()

                            x1 = -0.5
                            x2 = 0.5
                            if season == "summer":
                                fig.add_shape(
                                    name="Balance",
                                    showlegend=True,
                                    type='line', line={'color': 'DarkSlateGrey', 'width': 4},
                                    xref="x",
                                    yref="y",
                                    x0=x1,
                                    x1=x2,
                                    y0=amount,
                                    y1=amount,
                                    row=1,
                                    col=col,
                                    opacity=0.7
                                )
                            else:
                                fig.add_shape(
                                    name="Balance",
                                    type='line', line={'color': 'DarkSlateGrey', 'width': 4},
                                    xref="x",
                                    yref="y",
                                    x0=x1,
                                    x1=x2,
                                    y0=amount,
                                    y1=amount,
                                    row=1,
                                    col=col,
                                    opacity=0.7
                                )
                            #x1 = x1 + 4
                            #x2 = x2 + 4
            col = col + 1

        title = scenario.capitalize() + '-Group ' + 'Benefits/Costs per Agent over all Groups at Perfect Forecasting'

        fig.update_layout(
            title=title, yaxis_title="Amount in Euro")
        fig.show()



## ERROR PLOTS ##



#error_list = ["MAAPE", "NRMSE", "MAE", "MASE", "RMSE", "MAPE"]
error_list = ["MAAPE2", "MAE", "RMSE"]

for error in error_list:

    for scenario in list_scenarios:
        if not scenario == 'individual':

            dict_season = ndict({'mean': ndict({'summer': {}, 'transition': {}, 'winter': {}}),
                                 'median': ndict({'summer': {}, 'transition': {}, 'winter': {}})})
            dict_group_aggregator = ndict({'mean': ndict({'summer': {}, 'transition': {}, 'winter': {}}),
                                           'median': ndict({'summer': {}, 'transition': {}, 'winter': {}})})
            dict_group_consumer = ndict({'mean': ndict({'summer': {}, 'transition': {}, 'winter': {}}),
                                         'median': ndict({'summer': {}, 'transition': {}, 'winter': {}})})
            dict_group_prosumer = ndict({'mean': ndict({'summer': {}, 'transition': {}, 'winter': {}}),
                                         'median': ndict({'summer': {}, 'transition': {}, 'winter': {}})})
            for season in list_seasons:
                group_list_mean = np.array([])
                aggregator_list_mean = np.array([])
                consumers_list_mean = np.array([])
                prosumers_list_mean = np.array([])

                group_list_median = np.array([])
                aggregator_list_median = np.array([])
                consumers_list_median = np.array([])
                prosumers_list_median = np.array([])

                for group in list_group_size:

                    for forecasting_method in list_forecasting_methods:
                        if forecasting_method == "naive":

                            path = "05_results/scenario_pv_60_hp_60_ev_60_" + scenario + "_" + group + "_" + season + "_" + forecasting_method
                            path_individual = "05_results/scenario_pv_60_hp_60_ev_60_individual_" + season + "_" + forecasting_method


                            error_matrix_individual = f.load_file(path=os.path.join(two_levels_up, path_individual, 'general', 'error_matrix.csv'), df='pandas')
                            error_matrix = f.load_file(path=os.path.join(two_levels_up, path, 'general', 'error_matrix.csv'), df='pandas')

                            error_matrix = error_matrix.fillna(0)
                            error_matrix[error] = error_matrix[error].abs()


                            if group == '5':
                                my_list = list(error_matrix_individual[error])
                                group_list_mean = np.append(group_list_mean, sum(my_list) / len(my_list))
                                group_list_median = np.append(group_list_median, statistics.median(my_list))


                            consumers_list_mean = np.append(consumers_list_mean,
                                                       error_matrix.groupby('Label').get_group('consumer')[error].mean())
                            consumers_list_median = np.append(consumers_list_median,
                                                            statistics.median(error_matrix.groupby('Label').get_group('consumer')[
                                                                error]))
                            prosumers_list_mean = np.append(prosumers_list_mean,
                                                       error_matrix.groupby('Label').get_group('prosumer')[error].mean())
                            prosumers_list_median = np.append(prosumers_list_median,
                                                            statistics.median(error_matrix.groupby('Label').get_group('prosumer')[
                                                                error]))

                            value_to_add = float(error_matrix.groupby('Label').get_group('Aggregator')[error].mean())
                            value_to_add_single = value_to_add/int(''.join(filter(str.isdigit, group)))
                            aggregator_list_mean = np.append(aggregator_list_mean, value_to_add_single)

                            n = int(''.join(filter(str.isdigit, group))) - 1
                            my_list = list(error_matrix[error])
                            my_list.extend([value_to_add_single] * n)
                            group_list_mean = np.append(group_list_mean, sum(my_list) / len(my_list))
                            value_to_add = float(error_matrix.groupby('Label').get_group('Aggregator')[error].median())
                            value_to_add_single = value_to_add / int(''.join(filter(str.isdigit, group)))
                            aggregator_list_median = np.append(aggregator_list_mean, value_to_add_single)

                            n = int(''.join(filter(str.isdigit, group))) - 1
                            my_list = list(error_matrix[error])
                            my_list.extend([value_to_add_single] * n)
                            group_list_median = np.append(group_list_median, statistics.median(my_list))

                dict_season['mean'][season] = group_list_mean
                dict_group_aggregator['mean'][season] = aggregator_list_mean
                dict_group_consumer['mean'][season] = consumers_list_mean
                dict_group_prosumer['mean'][season] = prosumers_list_mean

                dict_season['median'][season] = group_list_median
                dict_group_aggregator['median'][season] = aggregator_list_median
                dict_group_consumer['median'][season] = consumers_list_median
                dict_group_prosumer['median'][season] = prosumers_list_median

            x1 = np.array(['1', '5', '10', '15'])
            x = np.array(['5', '10', '15'])
            measures = ['mean', 'median']

            for measure in measures:
                if error in ["MAAPE", "MAPE", "NRMSE"]:
                    yaxis_title = measure.capitalize() + " " + error.capitalize() + ' [%]'
                else:
                    yaxis_title = measure.capitalize() + " " + error.capitalize()

                fig = go.Figure()

                # line_shape='spline'

                fig.add_trace(go.Scatter(x=x1, y=dict_season[measure]['summer'], name="summer", marker=dict(color='rgb(158,202,225)', size=7), line=dict(color='rgb(49,130,189)', width=6)))
                fig.add_trace(go.Scatter(x=x1, y=dict_season[measure]['winter'], name="winter", marker=dict(color='rgb(49,130,189)', size=7), line=dict(color='rgb(49,130,189)', width=6)))
                fig.add_trace(go.Scatter(x=x1, y=dict_season[measure]['transition'], name='transition', marker=dict(color='#64A0C8', size=15), line=dict(color='#64A0C8', width=6)))

                fig.update_traces(hoverinfo='text+name', mode='lines+markers')
                res = scenario[0].upper() + scenario[1:]
                title = res + '-Group Forecast Error'
                # Edit the layout
                fig.layout.template = 'custom'
                fig.update_layout(title=title,
                                  xaxis_title='Group Size of the Aggregator',
                                  yaxis_title=yaxis_title
                                  )
                fig.show()

                fig = go.Figure()

                fig.add_trace(go.Scatter(x=x, y=dict_group_aggregator[measure]['summer'], name="summer",
                                         text=["tweak line smoothness<br>with 'smoothing' in line object"],
                                         hoverinfo='text+name'))
                fig.add_trace(go.Scatter(x=x, y=dict_group_aggregator[measure]['winter'], name="winter",
                                         text=["tweak line smoothness<br>with 'smoothing' in line object"],
                                         hoverinfo='text+name'))
                fig.add_trace(go.Scatter(x=x, y=dict_group_aggregator[measure]['transition'], name="transition",
                                         text=["tweak line smoothness<br>with 'smoothing' in line object"],
                                         hoverinfo='text+name'))

                fig.update_traces(hoverinfo='text+name', mode='lines+markers')
                res = scenario[0].upper() + scenario[1:]
                title = res + '-Group Forecast Error for the Aggregator'
                # Edit the layout
                fig.layout.template = 'custom'
                fig.update_layout(title=title,
                                  xaxis_title='Group Size of the Aggregator',
                                  yaxis_title=yaxis_title
                                  )
                fig.show()

                fig = go.Figure()

                fig.add_trace(go.Scatter(x=x, y=dict_group_consumer[measure]['summer'], name="summer",
                                         text=["tweak line smoothness<br>with 'smoothing' in line object"],
                                         hoverinfo='text+name'))
                fig.add_trace(go.Scatter(x=x, y=dict_group_consumer[measure]['winter'], name="winter",
                                         text=["tweak line smoothness<br>with 'smoothing' in line object"],
                                         hoverinfo='text+name'))
                fig.add_trace(go.Scatter(x=x, y=dict_group_consumer[measure]['transition'], name="transition",
                                         text=["tweak line smoothness<br>with 'smoothing' in line object"],
                                         hoverinfo='text+name'))

                fig.update_traces(hoverinfo='text+name', mode='lines+markers')
                res = scenario[0].upper() + scenario[1:]
                title = res + '-Group Forcecast Error Plot for Not Aggregated Consumers'
                # Edit the layout
                fig.layout.template = 'custom'
                fig.update_layout(title=title,
                                  xaxis_title='Group Size of the Aggregator',
                                  yaxis_title=yaxis_title
                                  )

                fig.show()

                fig = go.Figure()

                fig.add_trace(go.Scatter(x=x, y=dict_group_prosumer[measure]['summer'], name="summer",
                                         text=["tweak line smoothness<br>with 'smoothing' in line object"],
                                         hoverinfo='text+name'))
                fig.add_trace(go.Scatter(x=x, y=dict_group_prosumer[measure]['winter'], name="winter",
                                         text=["tweak line smoothness<br>with 'smoothing' in line object"],
                                         hoverinfo='text+name'))
                fig.add_trace(go.Scatter(x=x, y=dict_group_prosumer[measure]['transition'], name="transition",
                                         text=["tweak line smoothness<br>with 'smoothing' in line object"],
                                         hoverinfo='text+name'))

                fig.update_traces(hoverinfo='text+name', mode='lines+markers')
                res = scenario[0].upper() + scenario[1:]
                title = res + '-Group Forecast Error for Not Aggregated Prosumers'
                # Edit the layout
                fig.layout.template = 'custom'
                fig.update_layout(title=title,
                                  xaxis_title='Group Size of the Aggregator',
                                  yaxis_title=yaxis_title
                                  )
                fig.show()