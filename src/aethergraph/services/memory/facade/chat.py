from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from aethergraph.contracts.services.memory import Event

if TYPE_CHECKING:
    from .types import MemoryFacadeInterface


class ChatMixin:
    """
    Mixin adding chat-related memory functionality to MemoryFacade.

    Include methods:
    - record_chat
    - record_chat_user
    - record_chat_assistant
    - record_chat_system
    - record_chat_tool
    - recent_chat
    - chat_history_for_llm
    """

    async def record_chat(
        self: MemoryFacadeInterface,
        role: Literal["user", "assistant", "system", "tool"],
        text: str,
        *,
        tags: list[str] | None = None,
        data: dict[str, Any] | None = None,
        severity: int = 2,
        signal: float | None = None,
    ) -> Event:
        """
        Record a single chat turn in a normalized way.

        - role: "user" | "assistant" | "system" | "tool"
        - text: primary message text
        - tags: optional extra tags (we always add "chat")
        - data: extra JSON payload merged into {"role", "text"}
        """
        extra_tags = ["chat"]
        if tags:
            extra_tags.extend(tags)
        payload: dict[str, Any] = {"role": role, "text": text}
        if data:
            payload.update(data)

        return await self.record(
            kind="chat.turn",
            text=text,
            data=payload,
            tags=extra_tags,
            severity=severity,
            stage=role,
            signal=signal,
        )

    async def record_chat_user(
        self: MemoryFacadeInterface,
        text: str,
        *,
        tags: list[str] | None = None,
        data: dict[str, Any] | None = None,
        severity: int = 2,
        signal: float | None = None,
    ) -> Event:
        """DX sugar: record a user chat turn."""
        return await self.record_chat(
            "user",
            text,
            tags=tags,
            data=data,
            severity=severity,
            signal=signal,
        )

    async def record_chat_assistant(
        self: MemoryFacadeInterface,
        text: str,
        *,
        tags: list[str] | None = None,
        data: dict[str, Any] | None = None,
        severity: int = 2,
        signal: float | None = None,
    ) -> Event:
        """DX sugar: record an assistant chat turn."""
        return await self.record_chat(
            "assistant",
            text,
            tags=tags,
            data=data,
            severity=severity,
            signal=signal,
        )

    async def record_chat_system(
        self: MemoryFacadeInterface,
        text: str,
        *,
        tags: list[str] | None = None,
        data: dict[str, Any] | None = None,
        severity: int = 1,
        signal: float | None = None,
    ) -> Event:
        """DX sugar: record a system message."""
        return await self.record_chat(
            "system",
            text,
            tags=tags,
            data=data,
            severity=severity,
            signal=signal,
        )

    async def record_chat_tool(
        self: MemoryFacadeInterface,
        tool_name: str,
        text: str,
        *,
        tags: list[str] | None = None,
        data: dict[str, Any] | None = None,
        severity: int = 2,
        signal: float | None = None,
    ) -> Event:
        """
        DX sugar: record a tool-related message as a chat turn.

        Adds tag "tool:<tool_name>" and records tool_name in data.
        """
        tool_tags = list(tags or [])
        tool_tags.append(f"tool:{tool_name}")
        payload: dict[str, Any] = {"tool_name": tool_name}
        if data:
            payload.update(data)

        return await self.record_chat(
            "tool",
            text,
            tags=tool_tags,
            data=payload,
            severity=severity,
            signal=signal,
        )

    async def recent_chat(
        self: MemoryFacadeInterface,
        *,
        limit: int = 50,
        roles: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return the last `limit` chat.turns as a normalized list.

        Each item: {"ts", "role", "text", "tags"}.

        - roles: optional filter on role (e.g. {"user", "assistant"}).
        """
        events = await self.recent(kinds=["chat.turn"], limit=limit)
        out: list[dict[str, Any]] = []

        for e in events:
            # 1) Resolve role (from stage or data)
            role = (
                getattr(e, "stage", None)
                or ((e.data or {}).get("role") if getattr(e, "data", None) else None)
                or "user"
            )

            if roles is not None and role not in roles:
                continue

            # 2) Resolve text:
            #    - prefer Event.text
            #    - fall back to data["text"]
            raw_text = getattr(e, "text", "") or ""
            if not raw_text and getattr(e, "data", None):
                raw_text = (e.data or {}).get("text", "") or ""

            out.append(
                {
                    "ts": getattr(e, "ts", None),
                    "role": role,
                    "text": raw_text,
                    "tags": list(e.tags or []),
                }
            )

        return out

    async def chat_history_for_llm(
        self: MemoryFacadeInterface,
        *,
        limit: int = 20,
        include_system_summary: bool = True,
        summary_tag: str = "session",
        summary_scope_id: str | None = None,
        max_summaries: int = 3,
    ) -> dict[str, Any]:
        """
        Build a ready-to-send OpenAI-style chat message list.

        Returns:
          {
            "summary": "<combined long-term summary or ''>",
            "messages": [
               {"role": "system", "content": "..."},
               {"role": "user", "content": "..."},
               ...
            ]
          }

        Long-term summary handling:
          - We load up to `max_summaries` recent summaries for the tag,
            oldest → newest, and join their text with blank lines.
        """
        messages: list[dict[str, str]] = []
        summary_text = ""

        if include_system_summary:
            try:
                summaries = await self.load_recent_summaries(
                    scope_id=summary_scope_id,
                    summary_tag=summary_tag,
                    limit=max_summaries,
                )
            except Exception:
                summaries = []

            parts: list[str] = []
            for s in summaries:
                st = s.get("summary") or s.get("text") or s.get("body") or s.get("value") or ""
                if st:
                    parts.append(st)

            if parts:
                summary_text = "\n\n".join(parts)
                messages.append(
                    {
                        "role": "system",
                        "content": f"Summary of previous context:\n{summary_text}",
                    }
                )

        # Append recent chat turns
        for item in await self.recent_chat(limit=limit):
            role = item["role"]
            # Map unknown roles (e.g. "tool") to "assistant" by default
            mapped_role = role if role in {"user", "assistant", "system"} else "assistant"
            messages.append({"role": mapped_role, "content": item["text"]})

        return {"summary": summary_text, "messages": messages}
