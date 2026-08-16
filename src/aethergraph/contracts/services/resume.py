"""Runtime delivery contract for already-authorized continuations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from aethergraph.services.continuations.continuation import Continuation


class ResumeBus(Protocol):
    """Deliver one trusted tokenless continuation to its scheduler."""

    async def enqueue_resume(self, *, continuation: Continuation, payload: dict) -> None:
        """Dispatch an already-authorized continuation.

        Examples:
            Deliver an externally authorized wait:
            ```python
            await bus.enqueue_resume(continuation=wait, payload={"approved": True})
            ```
            Deliver a trusted timer candidate:
            ```python
            await bus.enqueue_resume(continuation=due, payload={"timer_fired": True})
            ```

        Args:
            continuation: Exact tokenless waiting record selected by the caller.
            payload: Resume data delivered to the waiting node.

        Returns:
            None: Scheduler delivery and terminal persistence are complete.

        Notes:
            Bearer authorization belongs exclusively to `ResumeRouter`.
        """
        ...


@dataclass
class ResumeEvent:
    """Scheduler event carrying one run/node resume payload."""

    run_id: str
    node_id: str
    payload: dict
