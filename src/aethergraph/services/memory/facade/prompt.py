from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aethergraph.contracts.services.memory import Event, MemoryFacadeProtocol


class PromptMixin:
    async def recent_chat(
        self: MemoryFacadeProtocol,
        *,
        limit: int = 50,
        roles: Sequence[str] | None = None,
        tags: Sequence[str] | None = None,
        level: str | None = None,
        use_persistence: bool = False,
        return_event: bool = False,
        include_tags: bool = True,
        include_ts: bool = True,
    ) -> list[Any]:
        events = await self.query_events(
            kinds=["chat.turn"],
            tags=list(tags) if tags else None,
            limit=limit,
            level=level,
            use_persistence=use_persistence,
            return_event=True,
        )
        if return_event:
            if roles is None:
                return events[-limit:] if limit else []
            filtered: list[Event] = []
            for event in events:
                role = (
                    getattr(event, "stage", None)
                    or ((event.data or {}).get("role") if event.data else None)
                    or "user"
                )
                if role in roles:
                    filtered.append(event)
            return filtered[-limit:] if limit else []
        out: list[dict[str, Any]] = []
        for event in events:
            role = (
                getattr(event, "stage", None)
                or ((event.data or {}).get("role") if event.data else None)
                or "user"
            )
            if roles is not None and role not in roles:
                continue
            raw_text = getattr(event, "text", "") or ""
            if not raw_text and getattr(event, "data", None):
                raw_text = (event.data or {}).get("text", "") or ""
            item = {"role": role, "text": raw_text}
            if include_ts:
                item["ts"] = getattr(event, "ts", None)
            if include_tags:
                item["tags"] = list(event.tags or [])
            out.append(item)
        return out[-limit:] if limit else []

    async def chat_history_for_llm(
        self: MemoryFacadeProtocol,
        *,
        limit: int = 20,
        include_system_summary: bool = True,
        summary_tag: str = "session",
        summary_scope_id: str | None = None,
        summary_kind: str = "long_term_summary",
        max_summaries: int = 3,
        level=None,
        use_persistence: bool = False,
    ) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        summary_text = ""
        if include_system_summary:
            try:
                summaries = await self.list_summaries(
                    summary_tag=summary_tag,
                    summary_kind=summary_kind,
                    limit=max_summaries,
                    scope_id=summary_scope_id,
                    level=level or "scope",
                )
            except Exception:
                summaries = []
            parts = []
            for summary in summaries:
                text = (
                    summary.get("summary")
                    or summary.get("text")
                    or summary.get("body")
                    or summary.get("value")
                    or ""
                )
                if text:
                    parts.append(text)
            if parts:
                summary_text = "\n\n".join(parts)
                messages.append(
                    {"role": "system", "content": f"Summary of previous context:\n{summary_text}"}
                )
        for item in await self.recent_chat(
            limit=limit,
            level=level,
            use_persistence=use_persistence,
            include_tags=False,
            include_ts=False,
        ):
            role = item["role"]
            mapped_role = role if role in {"user", "assistant", "system"} else "assistant"
            messages.append({"role": mapped_role, "content": item["text"]})
        return {"summary": summary_text, "messages": messages}

    async def build_prompt_segments(
        self: MemoryFacadeProtocol,
        *,
        recent_chat_limit: int = 12,
        include_long_term: bool = True,
        summary_tag: str = "session",
        summary_scope_id: str | None = None,
        summary_kind: str = "long_term_summary",
        max_summaries: int = 3,
        include_recent_tools: bool = False,
        tool: str | None = None,
        tool_limit: int = 10,
        recent_chat_tags: list[str] | None = None,
        recent_tool_tags: list[str] | None = None,
        recent_chat_include_tags: bool = True,
        recent_chat_include_ts: bool = True,
        level=None,
        use_persistence: bool = False,
    ) -> dict[str, Any]:
        span = await self._start_trace(
            operation="build_prompt_segments",
            request={
                "recent_chat_limit": recent_chat_limit,
                "include_long_term": include_long_term,
                "summary_tag": summary_tag,
                "summary_scope_id": summary_scope_id,
                "summary_kind": summary_kind,
                "max_summaries": max_summaries,
                "include_recent_tools": include_recent_tools,
                "tool": tool,
                "tool_limit": tool_limit,
                "level": level,
                "use_persistence": use_persistence,
            },
            tags=["memory", "prompt_context"],
        )
        try:
            long_term_text = ""
            if include_long_term:
                try:
                    summaries = await self.list_summaries(
                        summary_tag=summary_tag,
                        summary_kind=summary_kind,
                        limit=max_summaries,
                        scope_id=summary_scope_id,
                        level=level or "scope",
                    )
                except Exception:
                    summaries = []
                parts = []
                for summary in summaries:
                    text = (
                        summary.get("summary")
                        or summary.get("text")
                        or summary.get("body")
                        or summary.get("value")
                        or ""
                    )
                    if text:
                        parts.append(text)
                if parts:
                    long_term_text = "\n\n".join(parts)
            recent_chat = await self.recent_chat(
                limit=recent_chat_limit,
                tags=recent_chat_tags,
                include_tags=recent_chat_include_tags,
                include_ts=recent_chat_include_ts,
                level=level,
                use_persistence=use_persistence,
            )
            recent_tools: list[dict[str, Any]] = []
            if include_recent_tools:
                events = await self.query_events(
                    kinds=["tool_result"],
                    tags=recent_tool_tags,
                    limit=max(tool_limit * 5, 50) if (recent_tool_tags or tool) else tool_limit,
                    level=level,
                    use_persistence=use_persistence,
                    return_event=True,
                    tool=tool,
                    topic=tool,
                )
                events = events[-tool_limit:] if tool_limit else []
                for event in events:
                    recent_tools.append(
                        {
                            "ts": getattr(event, "ts", None),
                            "tool": getattr(event, "tool", None),
                            "message": getattr(event, "text", None),
                            "inputs": getattr(event, "inputs", None),
                            "outputs": getattr(event, "outputs", None),
                            "tags": list(event.tags or []),
                        }
                    )
            result = {
                "long_term": long_term_text,
                "recent_chat": recent_chat,
                "recent_tools": recent_tools,
            }
            await span.finish(
                response={
                    "long_term_length": len(long_term_text),
                    "recent_chat_count": len(recent_chat),
                    "recent_tools_count": len(recent_tools),
                },
                metadata=self._trace_meta(),
            )
            return result
        except Exception as exc:
            await span.fail(exc, metadata=self._trace_meta())
            raise
