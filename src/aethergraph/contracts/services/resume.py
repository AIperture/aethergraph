from abc import abstractmethod
from dataclasses import dataclass
from typing import Protocol


class ResumeBus(Protocol):
    """
    Abstract transport for resuming a waiting node.
    Implementations may be:
      - InProcessResumeBus: directly calls the in-memory scheduler
      - HttpResumeBus: POSTs to a remote scheduler service (not shown here)
    """

    @abstractmethod
    async def enqueue_resume(self, *, run_id: str, node_id: str, token: str, payload: dict) -> None:
        """
        Verify the continuation/token (or let the backend do it), then
        trigger a resume for (run_id, node_id) with the given payload.
        Should be idempotent and safe to call multiple times.
        """
        raise NotImplementedError


@dataclass
class ResumeEvent:
    run_id: str
    node_id: str
    payload: dict
