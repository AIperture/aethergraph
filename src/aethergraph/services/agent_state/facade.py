from __future__ import annotations

from collections.abc import Callable, Sequence
import dataclasses
from typing import Any, Generic, Literal, TypeVar, cast

from aethergraph.services.memory.facade import MemoryFacade
from aethergraph.services.scope.scope import ScopeLevel

AgentStateBackend = Literal["hybrid", "memory", "local"]
T = TypeVar("T")


def _to_serializable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


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
        if self.backend == "local" and self._cached is not None:
            return self._cached
        if self.backend == "hybrid" and self._cached is not None and not force:
            return self._cached
        if self.backend == "local":
            self._cached = self._default()
            return self._cached
        raw = await self.memory.latest_state(
            self.key,
            level=self.level,
            user_persistence=user_persistence,
            kind=self.kind,
        )
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
    ) -> Any | None:
        self._revision += 1
        self._cached = state
        if self.backend == "local":
            return None
        merged_meta = {
            **self.meta,
            **dict(meta or {}),
            "revision": self._revision,
        }
        if reason:
            merged_meta["reason"] = reason
        if stage_id:
            merged_meta["stage_id"] = stage_id
        return await self.memory.record_state(
            key=self.key,
            value=_to_serializable(state),
            tags=[*self.tags, *list(tags or [])],
            meta=merged_meta,
            severity=severity,
            signal=signal,
            kind=self.kind,
            stage=stage_id,
        )

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
        state = await self.load()
        result = fn(state)
        if result is not None:
            state = cast(T, result)
        await self.commit(
            state,
            reason=reason,
            stage_id=stage_id,
            tags=tags,
            meta=meta,
            severity=severity,
            signal=signal,
        )
        return state

    async def emit_change(
        self,
        *,
        reason: str,
        stage_id: str | None = None,
        patch: dict[str, Any] | None = None,
        summary: str = "",
        tags: Sequence[str] | None = None,
        severity: int = 1,
        signal: float | None = None,
    ) -> Any | None:
        if self.backend == "local":
            return None
        data = {
            "key": self.key,
            "revision": self._revision,
            "reason": reason,
            "stage_id": stage_id,
            "summary": summary,
            "patch": dict(patch or {}),
        }
        text = summary or f"agent state changed: {self.key} {reason}".strip()
        return await self.memory.record(
            kind="agent.state.change",
            text=text,
            data=data,
            tags=[
                "state",
                f"state:{self.key}",
                "agent.state.change",
                *self.tags,
                *list(tags or []),
            ],
            severity=severity,
            stage=stage_id,
            signal=signal,
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
        return await self.memory.state_history(
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
        self._handles: dict[tuple[str, AgentStateBackend], AgentStateHandle[Any]] = {}

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
        cache_key = (key, backend)
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
