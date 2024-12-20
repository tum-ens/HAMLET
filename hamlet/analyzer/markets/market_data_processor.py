import os
import pandas as pd
import numpy as np
import hamlet.functions as f
import hamlet.constants as c
from hamlet.analyzer.data_processor_base import DataProcessorBase


class MarketDataProcessor(DataProcessorBase):
    def __init__(self, path: dict, config: dict):
        super().__init__(path=path, config=config, name_subdirectory='markets')

    def process_total_balancing(self):
        """
        Compute total balancing data over time for all scenarios.

        Returns:
            dict: A dictionary where keys are scenario names and values are the total balancing data for each scenario,
            represented as DataFrames.
        """
        results_summary = {}

        for scenario_name, results_path in self.path.items():
            scenario_data = self._get_market_transactions_for_scenario(path=results_path, scenario_data={})
            for market_name, market_transactions in scenario_data.items():
                # Extract unique transaction types
                transaction_types = market_transactions[c.TC_TYPE_TRANSACTION].unique().tolist()
                result_df = pd.DataFrame(index=transaction_types, columns=['cost', 'revenue'])

                # Compute cost and revenue for each transaction type
                for transaction_type in transaction_types:
                    transactions_of_type = market_transactions[market_transactions[c.TC_TYPE_TRANSACTION] ==
                                                               transaction_type]
                    revenue = transactions_of_type[c.TC_PRICE_OUT].sum().item()
                    cost = transactions_of_type[c.TC_PRICE_IN].sum().item()
                    result_df.loc[transaction_type, 'cost'] = -cost
                    result_df.loc[transaction_type, 'revenue'] = revenue

                # Scale results for easier interpretation
                result_df /= 1e7
                scenario_data[market_name] = result_df

            results_summary[scenario_name] = scenario_data

        return results_summary

    def process_agent_balancing(self):
        """
        Compute total balancing data for each agent for all scenarios.

        Returns:
            dict: A dictionary where keys are scenario names and values are the total balancing data for each agent for
            each scenario, represented as DataFrames.
        """
        results_summary = {}

        for scenario_name, results_path in self.path.items():   # iterate all scenarios
            agent_balancing = None
            scenario_data = self._get_market_transactions_for_scenario(path=results_path, scenario_data={})
            for market_name, market_transactions in scenario_data.items():  # iterate all markets
                # adjust market transactions and get balancing for each agent
                if agent_balancing is None:
                    agent_balancing = market_transactions[[c.TC_TYPE_TRANSACTION, c.TC_PRICE_IN, c.TC_PRICE_OUT,
                                                           c.TC_ID_AGENT]].groupby(by=[c.TC_ID_AGENT,
                                                                                       c.TC_TYPE_TRANSACTION]).sum()
                else:
                    agent_balancing += market_transactions[[c.TC_TYPE_TRANSACTION, c.TC_PRICE_IN, c.TC_PRICE_OUT,
                                                            c.TC_ID_AGENT]].groupby(by=[c.TC_ID_AGENT,
                                                                                        c.TC_TYPE_TRANSACTION]).sum()

            results_summary[scenario_name] = agent_balancing

        return results_summary

    def process_average_pricing(self):
        """
        Process average pricing data for all scenarios.

        Description:
            Calculates the average price per transaction type for each market across all scenarios.
            Results are scaled for easier interpretation.

        Returns:
            dict: A dictionary where keys are scenario names and values are dictionaries of average
                  pricing data per market.
        """
        results_summary = {}

        for scenario_name, results_path in self.path.items():
            # Retrieve market transactions for the scenario
            scenario_data = self._get_market_transactions_for_scenario(path=results_path, scenario_data={})

            for market_name, market_transactions in scenario_data.items():
                # Group data by timestep and transaction type, and calculate sums
                market_transactions = market_transactions[[
                    c.TC_TIMESTEP, c.TC_TYPE_TRANSACTION, c.TC_PRICE_IN, c.TC_ENERGY_IN
                ]].groupby(by=[c.TC_TIMESTEP, c.TC_TYPE_TRANSACTION]).sum()

                # Calculate average price and scale for easier interpretation
                market_transactions['average_price'] = (
                                                               market_transactions[c.TC_PRICE_IN] / market_transactions[
                                                           c.TC_ENERGY_IN]
                                                       ) / 1e5

                # Retain only the average price column
                scenario_data[market_name] = market_transactions['average_price']

            # Store processed data for the scenario
            results_summary[scenario_name] = scenario_data

        return results_summary

    def _get_market_transactions_for_scenario(self, path, scenario_data):
        """
        Recursively read market transactions data for a given scenario.

        Args:
            path (str): The file path of the current directory.
            scenario_data (dict): Dictionary to store computed results for the scenario.

        Returns:
            dict: Updated `scenario_data` with market transactions data for the scenario.
        """
        # Recursively traverse subdirectories
        subdirectories = f.get_all_subdirectories(path)

        if subdirectories:
            for sub_directory in subdirectories:
                scenario_data = self._get_market_transactions_for_scenario(os.path.join(path, sub_directory),
                                                                           scenario_data)

        # Process market transactions if available
        if 'market_transactions.ft' in os.listdir(path):
            market_transactions = pd.read_feather(os.path.join(path, 'market_transactions.ft'))

            # Filter out retailer transactions
            market_transactions = market_transactions[market_transactions[c.TC_ID_AGENT] != 'retailer']

            scenario_data[path] = market_transactions

        return scenario_data
