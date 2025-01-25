__author__ = "HodaHamdy"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

import os
import shutil

import polars as pl

import hamlet.constants as c
from hamlet.executor.agents.agent import Agent
from hamlet.executor.utilities.tasks_execution.agent_pool import AgentPool
from hamlet.executor.utilities.tasks_execution.agent_pool import init_agentdb_full
from hamlet.executor.utilities.tasks_execution.task_executioner import TaskExecutioner


class AgentTaskExecutioner(TaskExecutioner):
    """
    A class to manage agents tasks execution

    Attributes:
        database (Database): database instance
        num_workers (int): number of workers
        pool (AgentPool): agents pool instance if multiprocessing is enabled, None otherwise
        results_path (str): results path
    """
    def __init__(self, database, num_workers):
        super().__init__(database, num_workers)
        # Setup up the tasks_execution pool for parallelization
        if self.num_workers > 1:
            self.pool = AgentPool(self.num_workers)

    def prepare_para_tasks(self, tasks):
        """Prepares tasks for parallel execution"""
        region_name = str(tasks.select(c.TC_REGION).sample(n=1).item())
        agents_data = self.database.get_agent_data(region=region_name)
        all_type_agents = [(agent_type, agent_id, tasks, self.results_path) for agent_type, agents in
                           agents_data.items() for agent_id in agents]
        return all_type_agents

    def execute_serial(self, tasks):
        """Executes all agent tasks for all agents sequentially"""
        region_name = str(tasks.select(c.TC_REGION).sample(n=1).item())
        # Get the data of the agents that are part of the tasklist
        agents = self.database.get_agent_data(region=region_name)

        results = []

        # Get the data of the markets that are part of the tasklist
        markets = self.database.get_market_data(region=region_name)
        # Iterate over the agents and execute them sequentially
        for agent_type, agent in agents.items():
            for agent_id, agent_db in agent.items():
                # Update save path for agent
                agent_db.agent_save = os.path.join(self.results_path, 'agents', agent_type, agent_id)
                # Create an instance of the Agent class and execute its tasks
                results.append(Agent(agent_type=agent_type, data=agent_db, timetable=tasks,
                                     market=markets).execute())
        return results

    def postprocess_results(self, tasks, results):
        """Post-processes the results of all agent tasks"""
        region_name = tasks.select(pl.first(c.TC_REGION)).item()
        # Update agents data in database
        self.database.post_agents_to_region(region=region_name, agents=results)

    def load_results_from_para(self, result_dirs: list[str]):
        """Load results (e.g. from file) if needed"""
        results = list(map(lambda x: init_agentdb_full(*x), result_dirs)) # is list needed here? squash with "remove folders"
        for fn in result_dirs:
            shutil.rmtree(fn[4])
        return results
