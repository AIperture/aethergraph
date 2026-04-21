from __future__ import annotations

from typing import TYPE_CHECKING, Any
import warnings

if TYPE_CHECKING:
    from aethergraph.contracts.services.memory import Event, MemoryFacadeProtocol


class DeprecatedMixin:
    async def record(
        self: MemoryFacadeProtocol,
        kind: str,
        data: Any,
        tags: list[str] | None = None,
        severity: int = 2,
        stage: str | None = None,
        inputs_ref=None,
        outputs_ref=None,
        metrics: dict[str, float] | None = None,
        signal: float | None = None,
        text: str | None = None,
    ) -> Event:
        warnings.warn(
            "record() is deprecated; use append_event().", DeprecationWarning, stacklevel=2
        )
        return await self.append_event(
            kind=kind,
            data=data,
            tags=tags,
            severity=severity,
            stage=stage,
            inputs=inputs_ref,
            outputs=outputs_ref,
            metrics=metrics,
            signal=signal,
            text=text,
        )

    async def recent(
        self: MemoryFacadeProtocol,
        *,
        kinds: list[str] | None = None,
        limit: int = 50,
        level: str | None = None,
        return_event: bool = True,
    ) -> list[Any]:
        return await self.query_events(
            kinds=kinds,
            limit=limit,
            level=level,
            use_persistence=False,
            return_event=return_event,
        )

    async def recent_events(
        self: MemoryFacadeProtocol,
        *,
        kinds: list[str] | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
        overfetch: int = 5,
        level: str | None = None,
        use_persistence: bool = False,
        return_event: bool = True,
    ) -> list[Any]:
        fetch_n = limit if not tags else max(limit * overfetch, 100)
        events = await self.query_events(
            kinds=kinds,
            tags=tags,
            limit=fetch_n,
            level=level,
            use_persistence=use_persistence,
            return_event=True,
        )
        events = events[-limit:] if limit is not None else events
        return self.normalize_recent_output(events, return_event=return_event)

    async def record_chat(self: MemoryFacadeProtocol, role, text: str, **kwargs) -> Event:
        return await self.append_chat_turn(role, text, **kwargs)

    async def record_chat_user(self: MemoryFacadeProtocol, text: str, **kwargs) -> Event:
        return await self.append_chat_turn("user", text, **kwargs)

    async def record_chat_assistant(self: MemoryFacadeProtocol, text: str, **kwargs) -> Event:
        return await self.append_chat_turn("assistant", text, **kwargs)

    async def record_chat_system(self: MemoryFacadeProtocol, text: str, **kwargs) -> Event:
        return await self.append_chat_turn("system", text, **kwargs)

    async def record_tool_result(self: MemoryFacadeProtocol, **kwargs) -> Event:
        return await self.append_tool_result(**kwargs)

    async def recent_tool_results(
        self: MemoryFacadeProtocol,
        *,
        tool: str,
        limit: int = 10,
        return_event: bool = True,
    ) -> list[Any]:
        events = await self.query_events(
            kinds=["tool_result"],
            limit=limit,
            use_persistence=True,
            return_event=True,
            tool=tool,
        )
        return self.normalize_recent_output(events[-limit:], return_event=return_event)

    async def record_state(self: MemoryFacadeProtocol, key: str, value: Any, **kwargs) -> Event:
        return await self.append_state_snapshot(key, value, **kwargs)

    async def latest_state(self: MemoryFacadeProtocol, key: str, **kwargs) -> Any | None:
        if "user_persistence" in kwargs:
            kwargs["use_persistence"] = kwargs.pop("user_persistence")
        return await self.get_latest_state(key, **kwargs)

    async def state_history(self: MemoryFacadeProtocol, key: str, **kwargs) -> list[Event]:
        return await self.list_state_history(key, **kwargs)

    async def search(self: MemoryFacadeProtocol, **kwargs):
        return await self.search_events(**kwargs)

    async def distill_long_term(self: MemoryFacadeProtocol, **kwargs):
        return await self.distill_summary(**kwargs)

    async def load_recent_summaries(self: MemoryFacadeProtocol, **kwargs):
        return await self.list_summaries(**kwargs)

    async def load_last_summary(self: MemoryFacadeProtocol, *args, **kwargs):
        return await self.get_latest_summary(*args, **kwargs)
