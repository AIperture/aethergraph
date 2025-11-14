import asyncio
from typing import Literal

from aethergraph.core.graph.task_node import TaskNodeRuntime

# from aethergraph.logging_config import logger


ExecutionMode = Literal["forward", "backward"]


class BaseScheduler:
    def __init__(self, graph, mode: ExecutionMode):
        self.graph = graph
        self.mode = mode
        self._pause_event = asyncio.Event()
        self._pause_event.set()
        self._terminated = False
        self.running_tasks = {}  # Dictionary to track currently running tasks

        self._nodes_to_pause = []  # Nodes that are requested to be paused

    @property
    def status(self) -> str:
        if self._terminated:
            return "terminated"
        if not self._pause_event.is_set():
            return "paused"
        if self.get_running_task_node_ids():
            return "running"
        return "idle"

    def reset_status(self):
        """
        Reset the scheduler's status to idle.
        """
        self._terminated = False
        self._pause_event.set()
        self.running_tasks.clear()
        # logger.info(f"🔄 Scheduler status reset for graph `{self.graph.id}`")

    def set_mode(self, mode: ExecutionMode):
        """
        Set the execution mode for this scheduler.
        """
        if mode not in ["forward", "backward"]:
            raise ValueError(f"❌ Invalid execution mode: {mode}")
        self.mode = mode
        # logger.info(f"🔄 Execution mode set to {self.mode}")

    def get_running_task_node_ids(self) -> list[str]:
        """
        Get a list of currently running task node IDs.
        """
        return [nid for nid, task in self.running_tasks.items() if not task.done()]

    async def run(self):
        raise NotImplementedError

    async def run_from(self, node_ids: list[str]):
        raise NotImplementedError("run_from() must be implemented in subclass")

    async def pause(self):
        self._pause_event.clear()

    async def resume(self):
        self._pause_event.set()

    async def terminate(self):
        self._terminated = True
        raise NotImplementedError("terminate() must be implemented in subclass")

    async def run_node(self, node: TaskNodeRuntime):
        raise NotImplementedError("run_node() must be implemented in subclass")

    async def step_next(self):
        raise NotImplementedError("step_next() must be implemented in subclass")
