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
        Record a structured state snapshot.

        - key: logical name for the state (e.g. "optimizer", "session_config").
        - value: arbitrary JSON-serializable structure (dataclass/dict/list/…).
        - tags: extra tags; "state" and f"state:{key}" are always added.
        - meta: additional metadata stored alongside the value.
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
        Fetch the most recent state snapshot for a given key.

        - level: which scope to search within (None/"scope", "session", "run", "user", "org").
        - use_persistence: whether to look into full history (True) or hotlog only (False).
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
            user_persistence=user_persistence,
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
        Fetch a history of state snapshots for a given key.
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
        Full-text + metadata search over state snapshots.

        - query: free-text query ("" for pure metadata/time search).
        - key: logical state key (adds tag "state:{key}").
        - tags: extra tags to require.
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
