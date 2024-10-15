import polars as pl

import hamlet.constants as c
from hamlet.executor.markets.market import Market
from hamlet.executor.utilities.tasks_execution.task_executioner import TaskExecutioner


class MarketTaskExecutioner(TaskExecutioner):
    """
    A class to manage market tasks execution

    Attributes:
        database (Database): database instance
        num_workers (int): number of workers
        pool (MarketsPool): None since multiprocessing is currently disabled for markets.
        results_path (str): results path
    """
    def __init__(self, database, num_workers):
        super().__init__(database, num_workers)

    def prepare_para_tasks(self, tasks):
        """Prepares parallel tasks"""
        markets_list = []
        for task in tasks.iter_rows(named=True):
            market = self.database.get_market_data(region=task[c.TC_REGION],
                                                   market_type=task[c.TC_MARKET],
                                                   market_name=task[c.TC_NAME])
            markets_list.append(Market(data=market, tasks=task, database=self.database))
        return markets_list

    def execute_serial(self, tasks):
        """Executes serial tasks"""
        results = []
        # Reuse parallel function to prepare tasks
        markets = self.prepare_para_tasks(tasks)
        for market in markets:
            results.append(market.execute())
        return results

    def postprocess_results(self, tasks, results):
        """Post-processes results"""
        region_name = tasks.select(pl.first(c.TC_REGION)).item()
        self.database.post_markets_to_region(region=region_name, markets=results)
