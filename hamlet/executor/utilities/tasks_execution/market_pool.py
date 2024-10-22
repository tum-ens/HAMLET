__author__ = "HodaHamdy"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"


from hamlet.executor.markets.market import Market
from hamlet.executor.utilities.tasks_execution.process_pool import ProcessPool


# Define the function to be executed in parallel
def task(market: Market):
    # Execute the given market
    return market.execute()


class MarketPool(ProcessPool):
    """
    A class to manage the markets multiprocessing pool

    Attributes:
        num_workers (int): the number of processes to spawn
        task: method to execute in parallel
    """
    def __init__(self, num_workers: int):
        super().__init__(num_workers, task)
