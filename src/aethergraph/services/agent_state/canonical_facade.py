"""Canonical Agent-state facade over the provider-owned current-state primitive."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
import dataclasses
import json
from typing import Any, Generic, TypeVar, cast

from aethergraph.services.agent_state.contracts import (
    AgentStateBackend,
    AgentStateConflictError,
)
from aethergraph.services.scope.scope import ScopeLevel
from aethergraph.storage.contracts import (
    Page,
    PageRequest,
    SortDirection,
    StateHistoryQuery,
    StateRecord,
    StateStore,
    StorageConflictError,
    StorageScope,
)

T = TypeVar("T")
_AGENT_STATE_NAMESPACE_PREFIX = "agent_state"


class CanonicalAgentStateHandle(Generic[T]):
    """One typed Agent-state key bound to exact provider scope."""

    def __init__(
        self,
        *,
        state_store: StateStore,
        scope: StorageScope,
        key: str,
        model: type[T] | None = None,
        default_factory: Callable[[], T] | None = None,
        backend: AgentStateBackend = "hybrid",
        tags: Sequence[str] | None = None,
        meta: Mapping[str, Any] | None = None,
        kind: str = "state.snapshot",
    ) -> None:
        """Bind one canonical Agent-state identity and cache policy.

        Retains the exact provider store and projected scope without selecting a
        backend, probing methods, or opening a second persistence path.

        Examples:
            Bind a typed state key:
                ```python
                handle = CanonicalAgentStateHandle(
                    state_store=state,
                    scope=scope,
                    key="planner",
                    model=PlannerState,
                )
                ```

            Bind a local-only key:
                ```python
                handle = CanonicalAgentStateHandle(
                    state_store=state,
                    scope=scope,
                    key="scratch",
                    backend="local",
                )
                ```

        Args:
            state_store: Canonical provider-owned current-state repository.
            scope: Exact projected Agent-state scope.
            key: Stable caller-owned state key.
            model: Optional model type used to hydrate stored mappings.
            default_factory: Optional callable producing missing state.
            backend: Exact cache policy: `hybrid`, `memory`, or `local`.
            tags: Optional immutable audit tags applied on commits.
            meta: Optional JSON-compatible audit metadata applied on commits.
            kind: Exact state family separating same-named keys.

        Returns:
            None: The handle is ready without performing I/O.

        Notes:
            `local` never touches the supplied store; other modes use only `StateStore`.
        """
        if backend not in {"hybrid", "memory", "local"}:
            raise ValueError(f"Unsupported agent state backend: {backend!r}")
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Agent state key must be a non-empty string")
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("Agent state kind must be a non-empty string")
        self._state = state_store
        self.scope = scope
        self.key = key
        self.model = model
        self.default_factory = default_factory
        self.backend = backend
        self.tags = tuple(tags or ())
        self.meta = dict(meta or {})
        self.kind = kind
        self._cached: T | None = None
        self._revision = 0
        self._loaded = False
        self._lock = asyncio.Lock()

    @property
    def revision(self) -> int:
        """Return the loaded or committed provider state revision.

        The revision is the exact optimistic-concurrency token for this scoped
        namespace/key identity.

        Examples:
            Read a new handle:
                ```python
                assert handle.revision == 0
                ```

            Capture a CAS token:
                ```python
                await handle.load()
                expected = handle.revision
                ```

        Args:
            None.

        Returns:
            int: Current non-negative provider state revision.

        Notes:
            This token is not a graph revision, content hash, or memory-event sequence.
        """
        return self._revision

    async def load(self, *, force: bool = False) -> T:
        """Load and hydrate the exact current Agent-state row.

        Hybrid mode reuses a loaded value unless forced; memory mode reads every call;
        local mode initializes and retains only process-local state.

        Examples:
            Load current state:
                ```python
                state = await handle.load()
                ```

            Refresh hybrid state:
                ```python
                state = await handle.load(force=True)
                ```

        Args:
            force: Bypass a hybrid cache and read the provider current row.

        Returns:
            T: Hydrated stored value or configured default.

        Notes:
            Missing state is revision zero and does not create a provider row.
        """
        async with self._lock:
            return await self._load_unlocked(force=force)

    async def commit(
        self,
        state: T,
        *,
        reason: str = "",
        stage_id: str | None = None,
        tags: Sequence[str] | None = None,
        meta: Mapping[str, Any] | None = None,
        severity: int = 1,
        signal: float | None = None,
        expected_revision: int | None = None,
    ) -> StateRecord | None:
        """Commit complete Agent state through provider CAS.

        Loads an uninitialized durable handle before writing, validates any explicit
        expectation, and commits exactly the next provider revision.

        Examples:
            Commit current state:
                ```python
                record = await handle.commit(state, reason="turn_completed")
                ```

            Commit with explicit CAS:
                ```python
                record = await handle.commit(state, expected_revision=handle.revision)
                ```

        Args:
            state: Complete next Agent-state value.
            reason: Optional compact audit reason.
            stage_id: Optional execution-stage identity.
            tags: Additional audit tags.
            meta: Additional JSON-compatible audit metadata.
            severity: Integer audit importance retained as state metadata.
            signal: Optional numeric audit relevance retained as state metadata.
            expected_revision: Optional exact provider revision expected by the caller.

        Returns:
            StateRecord | None: Committed provider row, or `None` for local mode.

        Notes:
            Provider conflicts become `AgentStateConflictError`; no retry or fallback occurs.
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

    async def update(
        self,
        fn: Callable[[T], Any],
        *,
        reason: str = "",
        stage_id: str | None = None,
        tags: Sequence[str] | None = None,
        meta: Mapping[str, Any] | None = None,
        severity: int = 1,
        signal: float | None = None,
    ) -> T:
        """Load, mutate, and CAS one Agent-state value under the handle lock.

        Uses the loaded provider revision as the exact expectation and commits either
        the callback return value or the mutated original when the callback returns none.

        Examples:
            Mutate in place:
                ```python
                state = await handle.update(lambda value: setattr(value, "count", 2))
                ```

            Return replacement state:
                ```python
                state = await handle.update(lambda _value: AgentState(count=2))
                ```

        Args:
            fn: Synchronous mutation or replacement callback.
            reason: Optional compact audit reason.
            stage_id: Optional execution-stage identity.
            tags: Additional audit tags.
            meta: Additional JSON-compatible audit metadata.
            severity: Integer audit importance retained as state metadata.
            signal: Optional numeric audit relevance retained as state metadata.

        Returns:
            T: The committed in-memory state value.

        Notes:
            Cross-process conflicts fail directly and do not rerun the callback.
        """
        async with self._lock:
            state = await self._load_unlocked(force=self.backend == "memory")
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
                expected_revision=self._revision,
            )
            return state

    async def history(
        self,
        *,
        limit: int = 50,
        cursor: str | None = None,
        order: SortDirection = SortDirection.DESCENDING,
    ) -> Page[StateRecord]:
        """Read one bounded cursor page of exact Agent-state history.

        Delegates directly to provider state history using the same projected scope,
        namespace, and key as current-state reads.

        Examples:
            Read recent revisions:
                ```python
                page = await handle.history(limit=20)
                ```

            Continue history:
                ```python
                page = await handle.history(cursor=previous.next_cursor)
                ```

        Args:
            limit: Bounded page size between one and 1000.
            cursor: Optional opaque provider continuation cursor.
            order: Stable revision ordering direction.

        Returns:
            Page[StateRecord]: Historical state revisions and continuation cursor.

        Notes:
            Local-only handles return an empty page and never query the provider.
        """
        if self.backend == "local":
            return Page(items=())
        return await self._state.history(
            StateHistoryQuery(
                scope=self.scope,
                namespace=self._namespace,
                key=self.key,
                page=PageRequest(limit=limit, cursor=cursor),
                order=order,
            )
        )

    @property
    def _namespace(self) -> str:
        return f"{_AGENT_STATE_NAMESPACE_PREFIX}:{self.kind}"

    def _default(self) -> T:
        if self.default_factory is not None:
            return self.default_factory()
        if self.model is not None:
            return self.model()  # type: ignore[misc,call-arg]
        return cast(T, {})

    def _hydrate(self, raw: object) -> T:
        value = _thaw_json(raw)
        if value is None:
            return self._default()
        if self.model is None:
            return cast(T, value)
        from_dict = getattr(self.model, "from_dict", None)
        if callable(from_dict):
            return cast(T, from_dict(value))
        if isinstance(value, dict):
            return self.model(**value)  # type: ignore[misc,call-arg]
        return cast(T, value)

    async def _load_unlocked(self, *, force: bool) -> T:
        if self.backend == "local":
            if self._cached is None:
                self._cached = self._default()
            self._loaded = True
            return self._cached
        if self.backend == "hybrid" and self._loaded and not force:
            return cast(T, self._cached)
        record = await self._state.get(self.scope, self._namespace, self.key)
        self._revision = record.revision if record is not None else 0
        self._cached = self._hydrate(record.value if record is not None else None)
        self._loaded = True
        return self._cached

    async def _commit_unlocked(
        self,
        state: T,
        *,
        reason: str,
        stage_id: str | None,
        tags: Sequence[str] | None,
        meta: Mapping[str, Any] | None,
        severity: int,
        signal: float | None,
        expected_revision: int | None,
    ) -> StateRecord | None:
        _validate_expected_revision(expected_revision)
        if self.backend == "local":
            if expected_revision is not None and expected_revision != self._revision:
                raise AgentStateConflictError(
                    key=self.key,
                    expected_revision=expected_revision,
                    actual_revision=self._revision,
                )
            self._revision += 1
            self._cached = state
            self._loaded = True
            return None
        if not self._loaded:
            await self._load_unlocked(force=True)
        if expected_revision is not None and expected_revision != self._revision:
            raise AgentStateConflictError(
                key=self.key,
                expected_revision=expected_revision,
                actual_revision=self._revision,
            )
        expectation = self._revision
        metadata = {
            **self.meta,
            **dict(meta or {}),
            "key": self.key,
            "kind": self.kind,
            "tags": [*self.tags, *tuple(tags or ())],
            "severity": severity,
        }
        if reason:
            metadata["reason"] = reason
        if stage_id:
            metadata["stage_id"] = stage_id
        if signal is not None:
            metadata["signal"] = signal
        try:
            record = await self._state.compare_and_set(
                self.scope,
                self._namespace,
                self.key,
                expectation,
                _to_serializable(state),
                metadata,
            )
        except StorageConflictError as exc:
            current = await self._state.get(self.scope, self._namespace, self.key)
            actual = current.revision if current is not None else 0
            raise AgentStateConflictError(
                key=self.key,
                expected_revision=expectation,
                actual_revision=actual,
            ) from exc
        self._revision = record.revision
        self._cached = state
        self._loaded = True
        return record


class CanonicalAgentStateFacade:
    """Bind typed Agent-state handles to one canonical provider store and base scope."""

    def __init__(self, *, state_store: StateStore, scope: StorageScope) -> None:
        """Compose the provider-backed canonical Agent-state service.

        Retains one provider store and canonical base scope without switching the active
        runtime, adapting a legacy store, or selecting storage dynamically.

        Examples:
            Build from one storage bundle:
                ```python
                facade = CanonicalAgentStateFacade(state_store=bundle.state, scope=node_scope)
                ```

            Bind a runtime-owned service:
                ```python
                services.agent_state = CanonicalAgentStateFacade(
                    state_store=bundle.state,
                    scope=node_scope,
                )
                ```

        Args:
            state_store: Canonical provider-owned current-state repository.
            scope: Canonical node/runtime base scope projected per bound level.

        Returns:
            None: The facade is ready without performing I/O.

        Notes:
            S9 performs the one-cut runtime activation; this class has no legacy mode.
        """
        self._state = state_store
        self.scope = scope
        self._handles: dict[tuple[Any, ...], CanonicalAgentStateHandle[Any]] = {}

    def bind(
        self,
        *,
        key: str,
        model: type[T] | None = None,
        default_factory: Callable[[], T] | None = None,
        level: ScopeLevel | None = None,
        backend: AgentStateBackend = "hybrid",
        tags: Sequence[str] | None = None,
        meta: Mapping[str, Any] | None = None,
        kind: str = "state.snapshot",
    ) -> CanonicalAgentStateHandle[T]:
        """Bind or reuse one typed canonical Agent-state handle.

        Projects the base scope to the requested logical level and caches only exact
        equivalent handle configurations.

        Examples:
            Bind session state:
                ```python
                handle = facade.bind(key="planner", model=PlannerState, level="session")
                ```

            Bind run-local state:
                ```python
                handle = facade.bind(key="turn", backend="memory", level="run")
                ```

        Args:
            key: Stable caller-owned state key.
            model: Optional model type used to hydrate stored mappings.
            default_factory: Optional callable producing missing state.
            level: Canonical logical scope projection; defaults deterministically.
            backend: Exact cache policy: `hybrid`, `memory`, or `local`.
            tags: Optional commit audit tags.
            meta: Optional JSON-compatible commit audit metadata.
            kind: Exact state family separating same-named keys.

        Returns:
            CanonicalAgentStateHandle[T]: Cached exact handle configuration.

        Notes:
            Deprecated App/client identity and legacy memory bucket aliases are absent.
        """
        projected_scope = project_agent_state_scope(self.scope, level=level)
        cache_key = (
            projected_scope,
            key,
            model,
            default_factory,
            level,
            backend,
            tuple(tags or ()),
            _stable_json_identity(dict(meta or {})),
            kind,
        )
        if cache_key not in self._handles:
            self._handles[cache_key] = CanonicalAgentStateHandle(
                state_store=self._state,
                scope=projected_scope,
                key=key,
                model=model,
                default_factory=default_factory,
                backend=backend,
                tags=tags,
                meta=meta,
                kind=kind,
            )
        return cast(CanonicalAgentStateHandle[T], self._handles[cache_key])


def project_agent_state_scope(
    scope: StorageScope,
    *,
    level: ScopeLevel | None,
) -> StorageScope:
    """Project canonical runtime scope to stable Agent-state identity.

    Uses only provider-neutral owner dimensions plus Agent identity and the exact
    dimension selected by the logical level.

    Examples:
        Project session state:
            ```python
            session_scope = project_agent_state_scope(scope, level="session")
            ```

        Project user state:
            ```python
            user_scope = project_agent_state_scope(scope, level="user")
            ```

    Args:
        scope: Canonical base runtime scope.
        level: Requested logical level or `None` for deterministic inference.

    Returns:
        StorageScope: Exact immutable Agent-state scope.

    Notes:
        Node, deprecated App/client, and legacy string bucket identities are omitted.
    """
    selected = level or _default_level(scope)
    if selected not in {"scope", "org", "user", "session", "run"}:
        raise ValueError(f"Unsupported Agent-state scope level: {selected!r}")
    values: dict[str, str | None] = {
        "tenant_id": scope.tenant_id,
        "project_id": scope.project_id,
        "agent_id": scope.agent_id,
        "scope_key": scope.scope_key,
    }
    if selected in {"org", "user", "session", "run"}:
        values["org_id"] = scope.org_id
    if selected in {"user", "session", "run"}:
        values["user_id"] = scope.user_id
    if selected == "session":
        values["session_id"] = scope.session_id
    if selected == "run":
        values["run_id"] = scope.run_id
        values["graph_id"] = scope.graph_id
    return StorageScope(**values)


def _default_level(scope: StorageScope) -> ScopeLevel:
    if scope.session_id is not None:
        return "session"
    if scope.run_id is not None:
        return "run"
    if scope.user_id is not None:
        return "user"
    if scope.org_id is not None:
        return "org"
    return "scope"


def _validate_expected_revision(value: int | None) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
        raise ValueError("expected_revision must be a non-negative integer")


def _to_serializable(value: Any) -> Any:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return value


def _thaw_json(value: object) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _stable_json_identity(value: object) -> str:
    return json.dumps(
        _to_serializable(value),
        sort_keys=True,
        separators=(",", ":"),
    )
