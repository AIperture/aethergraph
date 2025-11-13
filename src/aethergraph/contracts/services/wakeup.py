# core/contracts/wakeup.py
from dataclasses import dataclass
from typing import Protocol, Dict, Any, Optional

class WakeLease(Protocol):
    """Lease handle for a wakeup message."""
    id: str
    msg: Dict[str, Any]       # {run_id, node_id, token, payload, ...}
    visibility_deadline: float

class WakeupQueue(Protocol):
    """Protocol for a wakeup queue service."""
    async def enqueue(self, topic: str, msg: Dict[str, Any], delay_s: float = 0) -> str: ...
    async def lease(self, topic: str, max_items: int = 1, lease_s: int = 60) -> list[WakeLease]: ...
    async def extend(self, lease: WakeLease, lease_s: int) -> None: ...
    async def ack(self, lease: WakeLease) -> None: ...
    async def nack(self, lease: WakeLease, requeue_delay_s: float = 5) -> None: ... # re-enqueue the message


@dataclass
class WakeupEvent:
    node_id: str
