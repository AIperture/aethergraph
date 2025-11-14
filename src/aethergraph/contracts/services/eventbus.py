from collections.abc import Awaitable, Callable
from typing import Any, Protocol

Handler = Callable[[dict[str, Any]], Awaitable[None]]


class EventBus(Protocol):
    """Protocol for an event bus service."""

    async def publish(self, topic: str, event: dict[str, Any]) -> None: ...

    def subscribe(self, topic: str, handler: Handler) -> None: ...
