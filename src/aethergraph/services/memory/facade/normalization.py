from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aethergraph.contracts.services.memory import Event, MemoryFacadeProtocol


class EventNormalizationMixin:
    """
    Shared helpers for normalizing Event payloads across `recent_*` APIs.
    """

    def event_to_dict(
        self: MemoryFacadeProtocol,
        e: Event,
    ) -> dict[str, Any]:
        """
        Convert an Event into a normalized dictionary.

        Examples:
            Convert one event:
            ```python
            payload = context.memory().event_to_dict(evt)
            ```

        Args:
            e: Event object to normalize.

        Returns:
            dict[str, Any]: Canonical event dictionary representation.
        """
        return {
            "event_id": getattr(e, "event_id", None),
            "ts": getattr(e, "ts", None),
            "kind": getattr(e, "kind", None),
            "stage": getattr(e, "stage", None),
            "text": getattr(e, "text", None),
            "tags": list(getattr(e, "tags", None) or []),
            "data": getattr(e, "data", None),
            "metrics": getattr(e, "metrics", None),
            "tool": getattr(e, "tool", None),
            "topic": getattr(e, "topic", None),
            "severity": getattr(e, "severity", None),
            "signal": getattr(e, "signal", None),
            "inputs": getattr(e, "inputs", None),
            "outputs": getattr(e, "outputs", None),
            "run_id": getattr(e, "run_id", None),
            "scope_id": getattr(e, "scope_id", None),
            "session_id": getattr(e, "session_id", None),
            "graph_id": getattr(e, "graph_id", None),
            "node_id": getattr(e, "node_id", None),
            "user_id": getattr(e, "user_id", None),
            "org_id": getattr(e, "org_id", None),
            "client_id": getattr(e, "client_id", None),
        }

    def normalize_recent_output(
        self: MemoryFacadeProtocol,
        events: list[Event],
        *,
        return_event: bool = True,
    ) -> list[Any]:
        """
        Normalize output shape for `recent_*` methods.

        Examples:
            Return normalized dict payloads:
            ```python
            rows = context.memory().normalize_recent_output(events, return_event=False)
            ```

        Args:
            events: Event list to normalize.
            return_event: If True, return Event objects; otherwise dict payloads.

        Returns:
            list[Any]: Events or normalized dict payloads.
        """
        if return_event:
            return events
        return [self.event_to_dict(e) for e in events]
