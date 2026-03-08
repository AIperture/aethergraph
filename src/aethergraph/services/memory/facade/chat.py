from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal
import warnings

from aethergraph.contracts.services.memory import Event
from aethergraph.services.scope.scope import ScopeLevel

if TYPE_CHECKING:
    from aethergraph.contracts.services.memory import MemoryFacadeProtocol


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
        self: MemoryFacadeProtocol,
        role: Literal["user", "assistant", "system", "tool"],
        text: str,
        *,
        tags: list[str] | None = None,
        data: dict[str, Any] | None = None,
        severity: int = 2,
        signal: float | None = None,
    ) -> Event:
        """
        Record a single chat turn in a normalized format.

        This method automatically handles timestamping, standardizes the `role`,
        and dispatches the event to the configured persistence layer.

        Examples:
            Basic usage for a user message:
            ```python
            await context.memory().record_chat("user", "Hello graph!")
            ```

            Recording a tool output with extra metadata:
            ```python
            await context.memory().record_chat(
                "tool",
                "Search results found.",
                data={"query": "weather", "hits": 5}
            )
            ```

        Args:
            role: The semantic role of the speaker. Must be one of:
                `"user"`, `"assistant"`, `"system"`, or `"tool"`.
            text: The primary text content of the message.
            tags: A list of string labels for categorization. The tag `"chat"`
                is automatically appended to this list.
            data: Arbitrary JSON-serializable dictionary containing extra
                context (e.g., token counts, model names).
            severity: An integer (1-3) indicating importance.
                (1=Low, 2=Normal, 3=High).
            signal: Manual override for the signal strength (0.0 to 1.0).
                If None, it is calculated heuristically.

        Returns:
            Event: The fully persisted `Event` object containing the generated ID and timestamp.
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
        self: MemoryFacadeProtocol,
        text: str,
        *,
        tags: list[str] | None = None,
        data: dict[str, Any] | None = None,
        severity: int = 2,
        signal: float | None = None,
    ) -> Event:
        """
        Record a user chat turn in a normalized format.

        This method automatically handles timestamping, standardizes the `role`,
        and dispatches the event to the configured persistence layer.

        Examples:
            Basic usage for a user message:
            ```python
            await context.memory().record_chat_user("Hello, how are you doing?")
            ```

            Recording a user message with extra metadata:
            ```python
            await context.memory().record_chat_user(
                "I need help with my account.",
                tags=["support", "account"],
                data={"issue": "login failure"}
            )
            ```

        Args:
            text: The primary text content of the user's message.
            tags: A list of string labels for categorization. The tag `"chat"`
                is automatically appended to this list.
            data: Arbitrary JSON-serializable dictionary containing extra
                context (e.g., user metadata, session details).
            severity: An integer (1-3) indicating importance.
                (1=Low, 2=Normal, 3=High). Defaults to 2.
            signal: Manual override for the signal strength (0.0 to 1.0).
                If None, it is calculated heuristically.

        Returns:
            Event: The fully persisted `Event` object containing the generated ID and timestamp.
        """

        return await self.record_chat(
            "user",
            text,
            tags=tags,
            data=data,
            severity=severity,
            signal=signal,
        )

    async def record_chat_assistant(
        self: MemoryFacadeProtocol,
        text: str,
        *,
        tags: list[str] | None = None,
        data: dict[str, Any] | None = None,
        severity: int = 2,
        signal: float | None = None,
    ) -> Event:
        """
        Record an assistant chat turn in a normalized format.

        This method automatically handles timestamping, standardizes the `role`,
        and dispatches the event to the configured persistence layer.

        Examples:
            Basic usage for an assistant message:
            ```python
            await context.memory().record_chat_assistant("How can I assist you?")
            ```

            Recording an assistant message with extra metadata:
            ```python
            await context.memory().record_chat_assistant(
                "Here are the search results.",
                tags=["search", "response"],
                data={"query": "latest news", "results_count": 10}
            )
            ```

        Args:
            text: The primary text content of the assistant's message.
            tags: A list of string labels for categorization. The tag `"chat"`
                is automatically appended to this list.
            data: Arbitrary JSON-serializable dictionary containing extra
                context (e.g., token counts, model names).
            severity: An integer (1-3) indicating importance.
                (1=Low, 2=Normal, 3=High).
            signal: Manual override for the signal strength (0.0 to 1.0).
                If None, it is calculated heuristically.

        Returns:
            Event: The fully persisted `Event` object containing the generated ID and timestamp.
        """
        return await self.record_chat(
            "assistant",
            text,
            tags=tags,
            data=data,
            severity=severity,
            signal=signal,
        )

    async def record_chat_system(
        self: MemoryFacadeProtocol,
        text: str,
        *,
        tags: list[str] | None = None,
        data: dict[str, Any] | None = None,
        severity: int = 1,
        signal: float | None = None,
    ) -> Event:
        """
        Record a system message in a normalized format.

        This method automatically handles timestamping, standardizes the `role`,
        and dispatches the event to the configured persistence layer.

        Examples:
            Basic usage for a system message:
            ```python
            await context.memory().record_chat_system("System initialized.")
            ```

            Recording a system message with extra metadata:
            ```python
            await context.memory().record_chat_system(
                "Configuration updated.",
                tags=["config", "update"],
                data={"version": "1.2.3"}
            )
            ```

        Args:
            text: The primary text content of the system message.
            tags: A list of string labels for categorization. The tag `"chat"`
                is automatically appended to this list.
            data: Arbitrary JSON-serializable dictionary containing extra
                context (e.g., configuration details, system state).
            severity: An integer (1-3) indicating importance.
                (1=Low, 2=Normal, 3=High). Defaults to 1.
            signal: Manual override for the signal strength (0.0 to 1.0).
                If None, it is calculated heuristically.

        Returns:
            Event: The fully persisted `Event` object containing the generated ID and timestamp.
        """
        return await self.record_chat(
            "system",
            text,
            tags=tags,
            data=data,
            severity=severity,
            signal=signal,
        )

    async def recent_chat(
        self: MemoryFacadeProtocol,
        *,
        limit: int = 50,
        roles: Sequence[str] | None = None,
        tags: Sequence[str] | None = None,
        level: str | None = None,
        use_persistence: bool = False,
        return_event: bool = False,
    ) -> list[Any]:
        """
        Retrieve the most recent chat turns as a normalized list.

        This method fetches the last `limit` chat events of type `chat.turn`
        and returns them in a standardized format. Each item in the returned
        list contains the timestamp, role, text, and tags associated with the
        chat event. If `tags` is provided, over-fetch and filter because HotLog doesn't filter by tags.
        Returned messages are chronological, with the most recent last.


        Examples:
            Fetch the last 10 chat turns:
            ```python
            recent_chats = await context.memory().recent_chat(limit=10)
            ```

            Fetch the last 20 chat turns for specific roles:
            ```python
            recent_chats = await context.memory().recent_chat(
                limit=20, roles=["user", "assistant"]
            )
            ```

        Args:
            limit: The maximum number of chat events to retrieve. Defaults to 50.
            roles: An optional sequence of roles to filter by (e.g., `["user", "assistant"]`).
            tags: An optional sequence of tags to filter by. If provided, the method will over-fetch and filter results to include only those that have at least one of the specified tags.
            level: Optional scope level to filter events by (e.g., "session", "run", "user", "org"). If provided, the search will be constrained to events associated with the specified scope level.
            use_persistence: Whether to include events from the full persistence layer (True) or just the hotlog (False). Defaults to False.
            return_event: If True, return `Event` objects; otherwise normalized chat dicts.

        Returns:
            list[Any]: Event list when `return_event=True`, else dictionaries
            with the following keys:
                - "ts": The timestamp of the event.
                - "role": The role of the speaker (e.g., "user", "assistant").
                - "text": The text content of the chat message.
                - "tags": A list of tags associated with the event.
        """
        events = await self.recent_events(
            kinds=["chat.turn"],
            tags=list(tags) if tags else None,
            limit=limit,
            overfetch=5,
            level=level,
            use_persistence=use_persistence,
            return_event=True,
        )

        if return_event:
            out_events = events
            if roles is not None:
                filtered: list[Event] = []
                for e in out_events:
                    role = (
                        getattr(e, "stage", None)
                        or ((e.data or {}).get("role") if getattr(e, "data", None) else None)
                        or "user"
                    )
                    if role in roles:
                        filtered.append(e)
                out_events = filtered
            return out_events[-limit:] if limit else []

        out: list[dict[str, Any]] = []
        for e in events:
            role = (
                getattr(e, "stage", None)
                or ((e.data or {}).get("role") if getattr(e, "data", None) else None)
                or "user"
            )
            if roles is not None and role not in roles:
                continue

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

        # events from recent_events should already be <= limit and chronological;
        # this slice is safe but technically redundant
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
        level: ScopeLevel | None = "scope",
        use_persistence: bool = False,
    ) -> dict[str, Any]:
        """
        Build a ready-to-send OpenAI-style chat message list.

        This method constructs a dictionary containing a summary of previous
        context and a list of chat messages formatted for use with OpenAI-style
        chat models. It includes options to limit the number of messages and
        incorporate long-term summaries.

        Examples:
            Basic usage with default parameters:
            ```python
            history = await context.memory().chat_history_for_llm()
            ```

            Including a system summary and limiting messages:
            ```python
            history = await context.memory().chat_history_for_llm(
                limit=10, include_system_summary=True
            )
            ```

        Args:
            limit: The maximum number of recent chat messages to include. Defaults to 20.
            include_system_summary: Whether to include a system summary of previous
                context. Defaults to True.
            summary_tag: The tag used to filter summaries. Defaults to "session".
            summary_scope_id: An optional scope ID for filtering summaries. Defaults to None.
            max_summaries: The maximum number of summaries to load. Defaults to 3.

        Returns:
            dict[str, Any]: A dictionary with the following structure:
                - "summary": A combined long-term summary or an empty string.
                - "messages": A list of chat messages, each represented as a dictionary
                  with "role" and "content" keys.

        Example of returned structure:
        ```python
            {
                "summary": "Summary of previous context...",
                "messages": [
                    {"role": "system", "content": "Summary of previous context..."},
                    {"role": "user", "content": "Hello!"},
                    {"role": "assistant", "content": "Hi there! How can I help?"}
                ]
            }
        ```
        """
        messages: list[dict[str, str]] = []
        summary_text = ""

        if include_system_summary:
            try:
                summaries = await self.load_recent_summaries(
                    summary_tag=summary_tag,
                    summary_kind=summary_kind,
                    limit=max_summaries,
                    scope_id=summary_scope_id,
                    level=level,
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
        for item in await self.recent_chat(
            limit=limit,
            level=level,
            use_persistence=use_persistence,
        ):
            role = item["role"]
            # Map unknown roles (e.g. "tool") to "assistant" by default
            mapped_role = role if role in {"user", "assistant", "system"} else "assistant"
            messages.append({"role": mapped_role, "content": item["text"]})

        return {"summary": summary_text, "messages": messages}

    async def record_chat_tool(
        self: MemoryFacadeProtocol,
        tool_name: str,
        text: str,
        *,
        tags: list[str] | None = None,
        data: dict[str, Any] | None = None,
        severity: int = 2,
        signal: float | None = None,
    ) -> Event:
        """
        Deprecated helper to record a tool-role chat turn.

        Prefer `record_chat(role="tool", ...)` directly.

        Examples:
            Legacy usage:
            ```python
            await context.memory().record_chat_tool(
                tool_name="search",
                text="Found 3 results.",
            )
            ```

        Args:
            tool_name: Tool identifier to attach as tag/payload.
            text: Tool-role message text.
            tags: Optional extra tags.
            data: Optional additional payload fields.
            severity: Event severity.
            signal: Optional explicit signal override.

        Returns:
            Event: Persisted chat-turn event.
        """
        warnings.warn(
            "record_chat_tool() is deprecated and will be removed in a future version. "
            "Use record_chat(role='tool', ...).",
            DeprecationWarning,
            stacklevel=2,
        )
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
