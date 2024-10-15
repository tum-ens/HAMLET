import multiprocessing as mp
import os

import psutil


class TaskExecutioner:
    """
    A base class to manage tasks execution

    Attributes:
        database (Database): database instance
        num_workers (int): number of workers
        pool: None. Needs to be initialized by child classes if multiprocessing is enabled
        results_path (str): results path
    """
    MIN_GB_AVAILABLE = 35

    def __init__(self, database, num_workers):
        self.database = database
        self.num_workers = num_workers
        if not self.num_workers:
            self.num_workers = max(1, mp.cpu_count() - 1)  # physical processors
        self.pool = None
        self.results_path = None

    def execute(self, tasks):
        """Executes input tasks"""
        # Check if multiprocessing is enabled
        if self.num_workers == 1 or self.pool is None:
            results = self.execute_serial(tasks)
        else:
            # Prepare parallel tasks
            para_tasks = self.prepare_para_tasks(tasks)
            # Save database to file to allow loading inside each process
            self.database.save_database(os.path.dirname(self.results_path))
            # Update workers according to number of required parallel tasks
            self.update_num_workers(len(para_tasks))
            # Also update the pool's workers
            self.pool.update_num_workers(self.num_workers)
            # Execute multiprocessing pool
            results = self.pool.execute(para_tasks)
        # Postprocess results of tasks execution
        self.postprocess_results(tasks, results)

    def enough_memory(self):
        available_gigabytes = psutil.virtual_memory().available / (1024.0 ** 3)
        return available_gigabytes >= self.MIN_GB_AVAILABLE

    def close_pool(self):
        """Closes pool"""
        if self.pool is not None:
            self.pool.close()

    def prepare_para_tasks(self, tasks):
        """Prepares parallel tasks"""
        return tasks

    def update_num_workers(self, num_tasks):
        """Updates number of workers"""
        self.num_workers = max(1, min(num_tasks, self.num_workers))

    def execute_serial(self, tasks):
        """Executes serial tasks"""
        raise NotImplementedError

    def postprocess_results(self, tasks, results):
        """Post-processes results"""
        raise NotImplementedError

    def set_results_path(self, results_path):
        """Set results path"""
        self.results_path = results_path
