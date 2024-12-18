import os
import pandas as pd
import numpy as np
import hamlet.functions as f
import hamlet.constants as c
from hamlet.analyzer.data_processor_base import DataProcessorBase


class MarketDataProcessor(DataProcessorBase):
    def __init__(self, path: dict, config: dict):
        super().__init__(path=path, config=config)

    def process_total_balancing(self):
        """
        Compute total balancing data over time for all scenarios.

        Returns:
            dict: A dictionary where keys are scenario names and values are the total
                  balancing data for each scenario, represented as DataFrames.
        """
        results_summary = {
            scenario_name: self.get_total_balancing_for_scenario(
                path=results_path, scenario_data={}
            )
            for scenario_name, results_path in self.path.items()
        }
        return results_summary

    def get_total_balancing_for_scenario(self, path, scenario_data):
        """
        Recursively compute total balancing data for a given scenario.

        Args:
            path (str): The file path of the current directory.
            scenario_data (dict): Dictionary to store computed results for the scenario.

        Returns:
            dict: Updated `scenario_data` with total balancing data for the scenario.
        """
        # Recursively traverse subdirectories
        subdirectories = f.get_all_subdirectories(path)

        if subdirectories:
            for sub_directory in subdirectories:
                scenario_data = self.get_total_balancing_for_scenario(os.path.join(path, sub_directory), scenario_data)

        # Process market transactions if available
        if 'market_transactions.ft' in os.listdir(path):
            market_transactions = pd.read_feather(os.path.join(path, 'market_transactions.ft'))

            # Extract unique transaction types
            transaction_types = market_transactions[c.TC_TYPE_TRANSACTION].unique().tolist()
            result_df = pd.DataFrame(index=transaction_types, columns=['cost', 'revenue'])

            # Filter out retailer transactions
            transactions = market_transactions[market_transactions[c.TC_ID_AGENT] != 'retailer']

            # Compute cost and revenue for each transaction type
            for transaction_type in transaction_types:
                transactions_of_type = transactions[transactions[c.TC_TYPE_TRANSACTION] == transaction_type]
                revenue = transactions_of_type[c.TC_PRICE_OUT].sum().item()
                cost = transactions_of_type[c.TC_PRICE_IN].sum().item()
                result_df.loc[transaction_type, 'cost'] = -cost
                result_df.loc[transaction_type, 'revenue'] = revenue

            # Scale results for easier interpretation
            result_df /= 1e7
            scenario_data[path] = result_df

        return scenario_data
