__author__ = "HodaHamdy"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

from multiprocessing import Pool, get_context


class ProcessPool:
    """
    A base class to manage multiprocessing pools

    Attributes:
        num_workers (int): the number of processes to spawn
        pool (multiprocessing.pool.Pool): the multiprocessing pool
        task: method to execute in parallel
    """
    # Maximum tasks per child process after which the pool will spawn a fresh process
    # Findings showed using the child indefinitely does not make the amount of memory explode
    MAX_TASKS_PER_CHILD = None

    def __init__(self, num_workers: int, task):
        self.num_workers = num_workers
        self.pool: Pool = None
        self.task = task

    def update_num_workers(self, num_workers: int):
        """Updates the number of processes pool"""
        self.num_workers = num_workers

    def execute(self, task_args):
        """Executes the parallel tasks in the pool"""
        # Initialize parallel pool if not already initialized
        if self.pool is None:
            self.pool = get_context("spawn").Pool(self.num_workers, maxtasksperchild=self.MAX_TASKS_PER_CHILD)

        # Compute the chunksize
        chunksize, extra = divmod(len(task_args), self.num_workers)
        if extra:
            chunksize += 1

        # TODO: strongly recommend remove the try - except here since otherwise this function will return a None and further debug will be annoying
        # try:
        #     # Submit the tasks for parallel execution
        results = self.pool.map(self.task, task_args, chunksize=chunksize)
        return results
        # except Exception as e:
        #     print(e)
        #     self.close()

    def close(self):
        """Closes the pool"""
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None
