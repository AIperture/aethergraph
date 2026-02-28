from __future__ import annotations

from collections.abc import Sequence
import dataclasses
from typing import Any

from aethergraph.contracts.services.memory import Event
from aethergraph.services.memory.facade.retrieval import EventSearchResult
from aethergraph.services.scope.scope import ScopeLevel


class StateMixin:
    async def record_state(
        self,
        key: str,
        value: Any,
        *,
        tags: list[str] | None = None,
        meta: dict[str, Any] | None = None,
        severity: int = 2,
        signal: float | None = None,
        kind: str = "state.snapshot",
        stage: str | None = None,
    ) -> Event:
        """
        Record a structured state snapshot event.

        This method normalizes the value into a serializable payload and
        appends a state event tagged with both `state` and `state:{key}`.

        Examples:
            Record a basic state snapshot:
            ```python
            await context.memory().record_state(
                key="planner",
                value={"step": "draft", "attempt": 1},
            )
            ```

            Record state with custom metadata:
            ```python
            await context.memory().record_state(
                key="session_config",
                value={"temperature": 0.2},
                tags=["runtime"],
                meta={"source": "bootstrap"},
                severity=1,
            )
            ```

        Args:
            key: Logical state key (for example, `"planner"`).
            value: Value to snapshot; converted to a serializable representation.
            tags: Optional additional tags appended to default state tags.
            meta: Optional metadata stored in the event payload.
            severity: Event severity to store with the snapshot.
            signal: Optional signal override for the event.
            kind: Event kind. Defaults to `"state.snapshot"`.
            stage: Optional event stage.

        Returns:
            Event: The persisted state snapshot event.

        """

        def _to_serializable(obj: Any) -> Any:
            # TODO: make this more sophisticated later (pydantic, etc.)
            if dataclasses.is_dataclass(obj):
                return dataclasses.asdict(obj)
            if hasattr(obj, "model_dump"):
                try:
                    return obj.model_dump()
                except Exception:
                    pass
            if isinstance(obj, (str, int, float, bool)) or obj is None:  # noqa: UP038
                return obj
            if isinstance(obj, (dict, list, tuple)):  # noqa: UP038
                return obj
            # Fallback: repr; or later, put into DocStore and store a URI.
            return {"__repr__": repr(obj)}

        extra_tags: list[str] = ["state", f"state:{key}"]
        if tags:
            extra_tags.extend(tags)

        payload = {
            "key": key,
            "value": _to_serializable(value),
            "meta": meta or {},
        }

        return await self.record(
            kind=kind,
            text="",  # optional; we keep state in data
            data=payload,
            tags=extra_tags,
            severity=severity,
            stage=stage,
            signal=signal,
        )

    async def latest_state(
        self,
        key: str,
        *,
        tags: Sequence[str] | None = None,
        level: ScopeLevel | None = None,
        user_persistence: bool = False,
        kind: str = "state.snapshot",
    ) -> Any | None:
        """
        Fetch the most recent state value for a key.

        This method finds the newest matching state snapshot and returns only
        its `value` field from the stored payload.

        Examples:
            Read latest planner state:
            ```python
            latest = await context.memory().latest_state("planner")
            ```

            Read from persisted user-level history:
            ```python
            latest = await context.memory().latest_state(
                "session_config",
                level="user",
                user_persistence=True,
            )
            ```

        Args:
            key: Logical state key to retrieve.
            tags: Optional additional required tags.
            level: Optional scope level filter.
            user_persistence: If True, query persistence; otherwise use hotlog.
            kind: Event kind used for state snapshots.

        Returns:
            Any | None: The latest stored state value, or None if unavailable.
        """
        base_tags = ["state", f"state:{key}"]
        if tags:
            base_tags.extend(tags)

        events = await self.recent_events(
            kinds=[kind],
            tags=base_tags,
            limit=1,
            overfetch=5,
            level=level,
            use_persistence=user_persistence,
        )
        if not events:
            return None

        e = events[-1]  # get the most recent one
        if not e.data:
            return None

        # By convention, we stored {"key": key, "value": ..., "meta": ...} in data
        return e.data.get("value")

    async def state_history(
        self,
        key: str,
        *,
        tags: Sequence[str] | None = None,
        limit: int = 50,
        level: ScopeLevel | None = None,
        kind: str = "state.snapshot",
        use_persistence: bool = False,
    ) -> list[Event]:
        """
        Fetch state snapshot history for a key.

        This method returns full `Event` rows so callers can inspect state
        values, metadata, timestamps, and tags together.

        Examples:
            Load the latest 20 snapshots:
            ```python
            events = await context.memory().state_history("planner", limit=20)
            ```

            Load persisted user-level snapshots:
            ```python
            events = await context.memory().state_history(
                "session_config",
                level="user",
                use_persistence=True,
            )
            ```

        Args:
            key: Logical state key to retrieve history for.
            tags: Optional additional required tags.
            limit: Maximum number of events to return.
            level: Optional scope level filter.
            kind: Event kind used for state snapshots.
            use_persistence: If True, query persistence; otherwise use hotlog.

        Returns:
            list[Event]: Matching state snapshot events in chronological order.
        """
        base_tags = ["state", f"state:{key}"]
        if tags:
            base_tags.extend(tags)

        events = await self.recent_events(
            kinds=[kind],
            tags=base_tags,
            limit=limit,
            overfetch=5,
            level=level,
            use_persistence=use_persistence,
        )
        return events

    async def search_state(
        self,
        query: str,
        *,
        key: str | None = None,
        tags: Sequence[str] | None = None,
        top_k: int = 10,
        time_window: str | None = None,
        created_at_min: float | None = None,
        created_at_max: float | None = None,
    ) -> list[EventSearchResult]:
        """
        Search indexed state snapshot events.

        This method applies state-specific filters and delegates search to the
        scoped index backend. If no backend exists, it returns an empty list.

        Examples:
            Search all state snapshots:
            ```python
            results = await context.memory().search_state(query="temperature", top_k=5)
            ```

            Search a specific state key in a time window:
            ```python
            results = await context.memory().search_state(
                query="planner",
                key="session_config",
                time_window="7d",
            )
            ```

        Args:
            query: Free-text query string.
            key: Optional logical state key filter.
            tags: Optional additional required tags.
            top_k: Maximum number of scored results to return.
            time_window: Optional relative time-window expression.
            created_at_min: Optional lower timestamp bound.
            created_at_max: Optional upper timestamp bound.

        Returns:
            list[EventSearchResult]: Scored search matches with resolved events.
        """

        # If we don't have indices, fall back gracefully.
        scoped = getattr(self, "scoped_indices", None)
        if scoped is None or scoped.backend is None:
            return []

        # Build filters
        filter_tags: list[str] = ["state"]
        if key:
            filter_tags.append(f"state:{key}")
        if tags:
            filter_tags.extend(tags)

        filters: dict[str, Any] = {
            "kind": "state.snapshot",
            "tags": filter_tags,  # will be treated as post-filter in search backend
        }

        scored = await scoped.search_events(
            query=query,
            filters=filters,
            top_k=top_k,
            time_window=time_window,
            created_at_min=created_at_min,
            created_at_max=created_at_max,
        )

        return await self.fetch_events_for_search_results(scored, corpus="event")
