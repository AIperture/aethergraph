from typing import Protocol, Awaitable, Callable, Dict, Any

Handler = Callable[[Dict[str, Any]], Awaitable[None]]

class EventBus(Protocol):
    """Protocol for an event bus service."""
    async def publish(self, topic: str, event: Dict[str, Any]) -> None:
        ...
    
    def subscribe(self, topic: str, handler: Handler) -> None:
        ...