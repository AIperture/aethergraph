from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
import dataclasses
from typing import Any, Generic, Literal, TypeVar, cast

from aethergraph.contracts.services.memory import ExternalResourceChangedEvent
from aethergraph.services.memory.facade import MemoryFacade
from aethergraph.services.scope.scope import ScopeLevel

AgentStateBackend = Literal["hybrid", "memory", "local"]
T = TypeVar("T")


class AgentStateConflictError(RuntimeError):
    """Report a stale enclosing Agent-state revision before snapshot commit."""

    def __init__(self, *, key: str, expected_revision: int, actual_revision: int) -> None:
        self.key = str(key)
        self.expected_revision = int(expected_revision)
        self.actual_revision = int(actual_revision)
        super().__init__(
            f"Agent state {self.key!r} revision changed: expected "
            f"{self.expected_revision}, actual {self.actual_revision}"
        )


def _to_serializable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


async def _call_memory_method(memory: Any, primary: str, fallback: str, /, *args, **kwargs):
    method = getattr(memory, primary, None)
    if callable(method):
        return await method(*args, **kwargs)
    fallback_method = getattr(memory, fallback, None)
    if callable(fallback_method):
        return await fallback_method(*args, **kwargs)
    raise AttributeError(f"Memory object has neither {primary!r} nor {fallback!r}")


class AgentStateHandle(Generic[T]):
    def __init__(
        self,
        *,
        memory: MemoryFacade,
        key: str,
        model: type[T] | None = None,
        default_factory: Callable[[], T] | None = None,
        level: ScopeLevel | None = None,
        backend: AgentStateBackend = "hybrid",
        tags: Sequence[str] | None = None,
        meta: dict[str, Any] | None = None,
        kind: str = "state.snapshot",
    ) -> None:
        if backend not in {"hybrid", "memory", "local"}:
            raise ValueError(f"Unsupported agent state backend: {backend!r}")
        self.memory = memory
        self.key = key
        self.model = model
        self.default_factory = default_factory
        self.level = level
        self.backend: AgentStateBackend = backend
        self.tags = list(tags or [])
        self.meta = dict(meta or {})
        self.kind = kind
        self._cached: T | None = None
        self._revision = 0
        self._lock = asyncio.Lock()

    @property
    def revision(self) -> int:
        """Return the loaded or most recently committed snapshot revision.

        The revision belongs to the enclosing Agent-state snapshot and is the
        equality token accepted by `commit(expected_revision=...)`.

        Examples:
            Read a new handle revision:
                ```python
                assert handle.revision == 0
                ```

            Capture a compare-and-swap token:
                ```python
                state = await handle.load()
                expected = handle.revision
                ```

        Args:
            None.

        Returns:
            int: Non-negative enclosing snapshot revision.

        Notes:
            The token is local to one logical state key and is not a content
            hash or Tool activation lease revision.
        """

        return self._revision

    def _default(self) -> T:
        if self.default_factory is not None:
            return self.default_factory()
        if self.model is not None:
            return self.model()  # type: ignore[misc,call-arg]
        return cast(T, {})

    def _hydrate(self, raw: Any) -> T:
        if raw is None:
            return self._default()
        if self.model is None:
            return cast(T, raw)
        from_dict = getattr(self.model, "from_dict", None)
        if callable(from_dict):
            return cast(T, from_dict(raw))
        if isinstance(raw, dict):
            return self.model(**raw)  # type: ignore[misc,call-arg]
        return cast(T, raw)

    async def load(self, *, force: bool = False, user_persistence: bool = True) -> T:
        """Load the current state value and hydrate its snapshot revision.

        Hybrid handles reuse the cached value unless `force` is true. Durable
        and memory handles read the latest revisioned snapshot record when the
        bound Memory facade exposes it, with a legacy value-only fallback.

        Examples:
            Load a cached hybrid state:
                ```python
                state = await handle.load()
                ```

            Force a durable refresh:
                ```python
                state = await handle.load(force=True, user_persistence=True)
                ```

        Args:
            force: Bypass a hybrid cached value and refresh the snapshot.
            user_persistence: Read user-scoped durable persistence when true.

        Returns:
            T: Hydrated state model or the configured default value.

        Notes:
            Legacy value-only Memory implementations hydrate revision zero.
        """

        async with self._lock:
            return await self._load_unlocked(
                force=force,
                user_persistence=user_persistence,
            )

    async def _load_unlocked(
        self,
        *,
        force: bool = False,
        user_persistence: bool = True,
    ) -> T:
        if self.backend == "local" and self._cached is not None:
            return self._cached
        if self.backend == "hybrid" and self._cached is not None and not force:
            return self._cached
        if self.backend == "local":
            self._cached = self._default()
            return self._cached
        record_method = getattr(self.memory, "get_latest_state_record", None)
        if callable(record_method):
            record = await record_method(
                self.key,
                level=self.level,
                use_persistence=user_persistence,
                kind=self.kind,
            )
            if record is None:
                raw = None
                self._revision = 0
            else:
                raw = record.get("value")
                self._revision = max(0, int(record.get("revision") or 0))
        else:
            raw = await _call_memory_method(
                self.memory,
                "get_latest_state",
                "latest_state",
                self.key,
                level=self.level,
                use_persistence=user_persistence,
                kind=self.kind,
            )
            self._revision = 0
        self._cached = self._hydrate(raw)
        return self._cached

    async def commit(
        self,
        state: T,
        *,
        reason: str = "",
        stage_id: str | None = None,
        tags: Sequence[str] | None = None,
        meta: dict[str, Any] | None = None,
        severity: int = 1,
        signal: float | None = None,
        expected_revision: int | None = None,
    ) -> Any | None:
        """Commit one snapshot, optionally comparing its enclosing revision.

        All commits on the canonical handle are serialized. When
        `expected_revision` is supplied, a stale value raises before snapshot
        metadata, cache state, or Memory events are changed.

        Examples:
            Commit without a comparison:
                ```python
                await handle.commit(state, reason="turn_completed")
                ```

            Commit with compare-and-swap protection:
                ```python
                expected = handle.revision
                await handle.commit(
                    state,
                    reason="tool_activation",
                    expected_revision=expected,
                )
                ```

        Args:
            state: Complete state value to persist.
            reason: Optional compact commit reason stored in metadata.
            stage_id: Optional current execution stage identity.
            tags: Additional snapshot tags.
            meta: Additional snapshot metadata.
            severity: Memory Event importance level.
            signal: Optional relevance signal.
            expected_revision: Optional exact enclosing snapshot revision.

        Returns:
            Any | None: Persisted snapshot Event, or `None` for local state.

        Notes:
            This compare-and-swap is atomic for writers sharing the canonical
            handle. Persistence backends do not yet expose cross-process
            conditional append.
        """

        async with self._lock:
            return await self._commit_unlocked(
                state,
                reason=reason,
                stage_id=stage_id,
                tags=tags,
                meta=meta,
                severity=severity,
                signal=signal,
                expected_revision=expected_revision,
            )

    async def _commit_unlocked(
        self,
        state: T,
        *,
        reason: str = "",
        stage_id: str | None = None,
        tags: Sequence[str] | None = None,
        meta: dict[str, Any] | None = None,
        severity: int = 1,
        signal: float | None = None,
        expected_revision: int | None = None,
    ) -> Any | None:
        if expected_revision is not None:
            if isinstance(expected_revision, bool) or int(expected_revision) < 0:
                raise ValueError("expected_revision must be a non-negative integer")
            if int(expected_revision) != self._revision:
                raise AgentStateConflictError(
                    key=self.key,
                    expected_revision=int(expected_revision),
                    actual_revision=self._revision,
                )
        next_revision = self._revision + 1
        if self.backend == "local":
            self._revision = next_revision
            self._cached = state
            return None
        merged_meta = {
            **self.meta,
            **dict(meta or {}),
            "revision": next_revision,
        }
        if reason:
            merged_meta["reason"] = reason
        if stage_id:
            merged_meta["stage_id"] = stage_id
        result = await _call_memory_method(
            self.memory,
            "append_state_snapshot",
            "record_state",
            key=self.key,
            value=_to_serializable(state),
            tags=[*self.tags, *list(tags or [])],
            meta=merged_meta,
            severity=severity,
            signal=signal,
            kind=self.kind,
            stage=stage_id,
        )
        self._revision = next_revision
        self._cached = state
        return result

    async def update(
        self,
        fn: Callable[[T], Any],
        *,
        reason: str = "",
        stage_id: str | None = None,
        tags: Sequence[str] | None = None,
        meta: dict[str, Any] | None = None,
        severity: int = 1,
        signal: float | None = None,
    ) -> T:
        async with self._lock:
            state = await self._load_unlocked()
            expected_revision = self._revision
            result = fn(state)
            if result is not None:
                state = cast(T, result)
            await self._commit_unlocked(
                state,
                reason=reason,
                stage_id=stage_id,
                tags=tags,
                meta=meta,
                severity=severity,
                signal=signal,
                expected_revision=expected_revision,
            )
            return state

    async def emit_change(
        self,
        *,
        event_id: str,
        source_sequence: int,
        revision: str,
        recorded_at: str,
        reason: str,
        patch: dict[str, Any] | None = None,
        summary: str = "",
        previous_revision: str = "",
        previous_content_hash: str = "",
        content_hash: str = "",
        effective_at: str = "",
    ) -> Any | None:
        """Publish a direct state mutation through the common external contract.

        Callers provide the committed mutation's outbox identity, monotonic
        sequence, and revision. The state value itself is not copied into the
        change event; only changed field paths and a compact summary are stored.

        Examples:
            Publish a committed state change:
            ```python
                await handle.emit_change(
                    event_id="state-outbox-7",
                    source_sequence=7,
                    revision="7",
                    recorded_at="2026-07-10T20:00:01Z",
                    reason="stage_started",
                    patch={"pipeline.active_stage_id": "stage-a"},
                )
            ```

            Include revision hashes without state content:
            ```python
                await handle.emit_change(
                    event_id="state-outbox-8",
                    source_sequence=8,
                    revision="8",
                    recorded_at="2026-07-10T20:00:02Z",
                    reason="settings_changed",
                    content_hash="sha256:new",
                )
            ```

        Args:
            event_id: Unique authoritative outbox event identifier.
            source_sequence: Monotonic sequence within this state source/scope.
            revision: Opaque equality-comparable state revision.
            recorded_at: Timezone-aware ISO timestamp of outbox insertion.
            reason: Compact producer reason used when no summary is supplied.
            patch: Changed field mapping; values are not persisted in the event.
            summary: Optional human-readable change summary.
            previous_revision: Previously committed opaque revision.
            previous_content_hash: Hash of the previous authoritative state.
            content_hash: Hash of the new authoritative state.
            effective_at: Optional timezone-aware mutation-effective timestamp.

        Returns:
            Any | None: Persisted external-resource memory event, or `None` for
            a deliberately local-only state backend.

        Notes:
            Mutation plus outbox insertion belongs to the authoritative state
            transaction. This method ingests its committed row and never starts
            an agent run.
        """

        if self.backend == "local":
            return None
        append = getattr(self.memory, "append_external_resource_change", None)
        if not callable(append):
            raise AttributeError("Memory object does not expose append_external_resource_change()")
        return await append(
            ExternalResourceChangedEvent(
                event_id=event_id,
                scope_id=str(getattr(self.memory, "memory_scope_id", "") or ""),
                session_id=str(getattr(self.memory, "session_id", "") or ""),
                source_sequence=source_sequence,
                resource_key=f"agent_state:{self.key}",
                resource_kind="agent_state",
                previous_revision=previous_revision,
                revision=revision,
                previous_content_hash=previous_content_hash,
                content_hash=content_hash,
                changed_fields=tuple(str(key) for key in dict(patch or {})),
                summary=summary or f"agent state changed: {self.key} {reason}".strip(),
                source="agent_state",
                effective_at=effective_at,
                recorded_at=recorded_at,
            )
        )

    async def history(
        self,
        *,
        tags: Sequence[str] | None = None,
        limit: int = 50,
        level: ScopeLevel | None = None,
        kind: str | None = None,
        use_persistence: bool = False,
    ) -> list[Any]:
        return await _call_memory_method(
            self.memory,
            "list_state_history",
            "state_history",
            self.key,
            tags=tags,
            limit=limit,
            level=level if level is not None else self.level,
            kind=kind or self.kind,
            use_persistence=use_persistence,
        )

    async def search(
        self,
        query: str,
        *,
        tags: Sequence[str] | None = None,
        top_k: int = 10,
        time_window: str | None = None,
        created_at_min: float | None = None,
        created_at_max: float | None = None,
    ) -> list[Any]:
        return await self.memory.search_state(
            query=query,
            key=self.key,
            tags=tags,
            top_k=top_k,
            time_window=time_window,
            created_at_min=created_at_min,
            created_at_max=created_at_max,
        )


class AgentStateFacade:
    def __init__(self, *, memory: MemoryFacade) -> None:
        self.memory = memory
        self._handles: dict[tuple[Any, ...], AgentStateHandle[Any]] = {}

    @staticmethod
    def _cache_key(
        *,
        key: str,
        model: type[Any] | None,
        default_factory: Callable[[], Any] | None,
        level: ScopeLevel | None,
        backend: AgentStateBackend,
        tags: Sequence[str] | None,
        meta: dict[str, Any] | None,
        kind: str,
    ) -> tuple[Any, ...]:
        return (
            key,
            backend,
            level,
            kind,
            model,
            default_factory,
            tuple(tags or ()),
            tuple(sorted((meta or {}).items())),
        )

    def bind(
        self,
        *,
        key: str,
        model: type[T] | None = None,
        default_factory: Callable[[], T] | None = None,
        level: ScopeLevel | None = None,
        backend: AgentStateBackend = "hybrid",
        tags: Sequence[str] | None = None,
        meta: dict[str, Any] | None = None,
        kind: str = "state.snapshot",
    ) -> AgentStateHandle[T]:
        cache_key = self._cache_key(
            key=key,
            model=model,
            default_factory=default_factory,
            level=level,
            backend=backend,
            tags=tags,
            meta=meta,
            kind=kind,
        )
        if cache_key not in self._handles:
            self._handles[cache_key] = AgentStateHandle(
                memory=self.memory,
                key=key,
                model=model,
                default_factory=default_factory,
                level=level,
                backend=backend,
                tags=tags,
                meta=meta,
                kind=kind,
            )
        return cast(AgentStateHandle[T], self._handles[cache_key])
