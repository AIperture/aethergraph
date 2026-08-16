"""Prompt and summary behavior for canonical public Memory."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import json
import math
from typing import TYPE_CHECKING, Any

from aethergraph.contracts.services.memory import Event

if TYPE_CHECKING:
    from .canonical_public import CanonicalPublicMemoryFacade

_MAX_PUBLIC_SCAN = 10_000


class CanonicalPromptMemoryMixin:
    """Project prompt-oriented Memory behavior onto canonical events."""

    async def recent_chat(
        self: CanonicalPublicMemoryFacade,
        *,
        limit: int = 50,
        roles: Sequence[str] | None = None,
        tags: Sequence[str] | None = None,
        level: str | None = None,
        use_persistence: bool = False,
        return_event: bool = False,
        include_tags: bool = False,
        include_ts: bool = False,
    ) -> list[Any]:
        """Return recent chat turns in chronological prompt order.

        The caller explicitly chooses the hot or durable canonical event source. Role
        filtering is applied after bounded retrieval without consulting another tier.

        Examples:
            Read compact prompt messages:
                ```python
                messages = await memory.recent_chat(limit=20)
                ```

            Read durable user Event DTOs:
                ```python
                events = await memory.recent_chat(
                    roles=["user"],
                    use_persistence=True,
                    return_event=True,
                )
                ```

        Args:
            limit: Non-negative maximum number of returned chat turns.
            roles: Optional exact roles retained after retrieval.
            tags: Optional tags every chat event must contain.
            level: Recognized compatibility scope level for the already-bound facade.
            use_persistence: Select durable canonical events instead of hot cache.
            return_event: Return Event DTOs instead of prompt mappings when true.
            include_tags: Include event tags in returned prompt mappings.
            include_ts: Include event timestamps in returned prompt mappings.

        Returns:
            list[Any]: Oldest-to-newest Event DTOs or chat prompt mappings.

        Notes:
            A zero limit returns an empty list. Cache misses and provider failures do
            not trigger a durable, legacy, or alternate-provider fallback.
        """
        _limit("limit", limit)
        if limit == 0:
            return []
        role_filter = _roles(roles)
        if isinstance(tags, str):
            raise TypeError("tags must be a sequence of exact strings, not a string")
        fetch_limit = _MAX_PUBLIC_SCAN if role_filter is not None else limit
        events: list[Event] = await self.query_events(
            kinds=["chat.turn"],
            tags=list(tags) if tags is not None else None,
            limit=fetch_limit,
            level=level,
            use_persistence=use_persistence,
            return_event=True,
            order_dir="desc",
        )
        selected = (
            [event for event in reversed(events) if _chat_role(event) in role_filter]
            if role_filter is not None
            else list(reversed(events))
        )
        selected = selected[-limit:]
        if return_event:
            return selected
        output: list[dict[str, Any]] = []
        for event in selected:
            item: dict[str, Any] = {"role": _chat_role(event), "text": _event_text(event)}
            if include_tags:
                item["tags"] = list(event.tags or ())
            if include_ts:
                item["ts"] = event.ts
            output.append(item)
        return output

    async def list_summaries(
        self: CanonicalPublicMemoryFacade,
        *,
        summary_tag: str = "session",
        limit: int = 3,
        summary_kind: str = "long_term_summary",
        scope_id: str | None = None,
        level: str | None = None,
    ) -> list[dict[str, Any]]:
        """List durable canonical summaries in chronological order.

        Summary kind and tags are provider-side filters. Optional public `scope_id`
        compatibility filtering is applied within a fixed ten-thousand-record window.

        Examples:
            Load recent session summaries:
                ```python
                summaries = await memory.list_summaries(limit=3)
                ```

            Restrict a named public summary scope:
                ```python
                summaries = await memory.list_summaries(
                    summary_tag="design",
                    scope_id="session:session-1",
                )
                ```

        Args:
            summary_tag: Exact tag identifying the summary stream.
            limit: Non-negative maximum number of returned summaries.
            summary_kind: Exact canonical Event kind for summaries.
            scope_id: Optional public payload scope label, never provider scope.
            level: Recognized compatibility scope level for the already-bound facade.

        Returns:
            list[dict[str, Any]]: Oldest-to-newest detached summary payloads.

        Notes:
            Summaries always use durable canonical events. Provider failures propagate;
            an empty list means the successful bounded query found no matching records.
        """
        _limit("limit", limit)
        _exact("summary_tag", summary_tag)
        _exact("summary_kind", summary_kind)
        _optional_exact("scope_id", scope_id)
        if limit == 0:
            return []
        fetch_limit = _MAX_PUBLIC_SCAN if scope_id is not None else limit
        events: list[Event] = await self.query_events(
            kinds=[summary_kind],
            tags=["summary", summary_tag],
            limit=fetch_limit,
            level=level if level is not None else "scope",
            use_persistence=True,
            return_event=True,
            order_dir="desc",
        )
        if scope_id is not None:
            events = [event for event in events if _summary_scope(event) == scope_id]
        selected = list(reversed(events[:limit]))
        return [
            _summary_mapping(
                event,
                summary_kind=summary_kind,
                summary_tag=summary_tag,
            )
            for event in selected
        ]

    async def get_latest_summary(
        self: CanonicalPublicMemoryFacade,
        scope_id: str | None = None,
        *,
        summary_tag: str = "session",
        summary_kind: str = "long_term_summary",
        level: str | None = None,
    ) -> dict[str, Any] | None:
        """Return the latest durable canonical summary.

        The method delegates to the same bounded durable summary query and preserves
        its direct error behavior.

        Examples:
            Load the latest session summary:
                ```python
                summary = await memory.get_latest_summary()
                ```

            Load the latest design summary:
                ```python
                summary = await memory.get_latest_summary(
                    summary_tag="design",
                    summary_kind="design_summary",
                )
                ```

        Args:
            scope_id: Optional public payload scope label, never provider scope.
            summary_tag: Exact tag identifying the summary stream.
            summary_kind: Exact canonical Event kind for summaries.
            level: Recognized compatibility scope level for the already-bound facade.

        Returns:
            dict[str, Any] | None: Latest detached summary payload or `None` after a
            successful query with no matching summary.

        Notes:
            Provider failures are not converted to `None` or an empty payload.
        """
        summaries = await self.list_summaries(
            summary_tag=summary_tag,
            limit=1,
            summary_kind=summary_kind,
            scope_id=scope_id,
            level=level,
        )
        return summaries[-1] if summaries else None

    async def distill_summary(
        self: CanonicalPublicMemoryFacade,
        *,
        level: str | None = None,
        summary_tag: str = "session",
        summary_kind: str = "long_term_summary",
        include_kinds: list[str] | None = None,
        include_tags: list[str] | None = None,
        max_events: int = 200,
        min_signal: float | None = None,
        use_llm: bool = False,
        scope_id: str | None = None,
        extra_data: dict[str, Any] | None = None,
        extra_tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Distill durable canonical events and append one summary Event.

        Deterministic distillation renders a bounded chronological transcript. LLM
        distillation requires an explicitly injected client and strict JSON output.

        Examples:
            Distill recent chat deterministically:
                ```python
                summary = await memory.distill_summary(
                    include_kinds=["chat.turn"],
                    use_llm=False,
                )
                ```

            Distill with an injected LLM:
                ```python
                summary = await memory.distill_summary(
                    summary_tag="design",
                    use_llm=True,
                )
                ```

        Args:
            level: Recognized compatibility scope level for the already-bound facade.
            summary_tag: Exact tag identifying the produced summary stream.
            summary_kind: Exact canonical Event kind for the produced summary.
            include_kinds: Optional exact source Event kinds.
            include_tags: Optional tags every source Event must contain.
            max_events: Positive maximum number of selected source events.
            min_signal: Optional finite minimum Event signal; uses facade default when absent.
            use_llm: Require the explicitly injected LLM summarizer when true.
            scope_id: Optional public label stored in the summary payload.
            extra_data: Optional non-reserved fields added to the summary payload.
            extra_tags: Optional additional tags on the summary Event.

        Returns:
            dict[str, Any]: Stored summary payload with authoritative Event identity,
            or an empty mapping after a successful query with no qualifying events.

        Notes:
            Missing LLM configuration, malformed LLM JSON, provider errors, and
            reserved-field conflicts fail directly without heuristic or backend fallback.
        """
        _positive_limit("max_events", max_events)
        _exact("summary_tag", summary_tag)
        _exact("summary_kind", summary_kind)
        _optional_exact("scope_id", scope_id)
        if isinstance(extra_tags, str):
            raise TypeError("extra_tags must be a sequence of exact strings, not a string")
        threshold = self.default_signal_threshold if min_signal is None else min_signal
        _finite("min_signal", threshold)
        events: list[Event] = await self.query_events(
            kinds=include_kinds,
            tags=include_tags,
            limit=max_events,
            level=level if level is not None else "scope",
            use_persistence=True,
            return_event=True,
            order_dir="desc",
        )
        selected = [event for event in reversed(events) if float(event.signal or 0.0) >= threshold]
        selected = selected[-max_events:]
        if not selected:
            return {}
        payload = await _distilled_payload(
            self,
            events=selected,
            summary_kind=summary_kind,
            summary_tag=summary_tag,
            include_kinds=include_kinds,
            include_tags=include_tags,
            max_events=max_events,
            min_signal=threshold,
            use_llm=use_llm,
        )
        if scope_id is not None:
            payload["scope_id"] = scope_id
        _merge_extra(payload, extra_data)
        preview = _summary_text(payload)
        event = await self.append_event(
            kind=summary_kind,
            data=payload,
            tags=["summary", summary_tag, *(extra_tags or ()), *(["llm"] if use_llm else ())],
            severity=2,
            stage="summary_llm" if use_llm else "summary",
            signal=0.7 if use_llm else None,
            text=preview[:2000] + (" ...[truncated]" if len(preview) > 2000 else ""),
            metrics={"num_events": float(len(selected))},
        )
        return {
            **payload,
            "event_id": event.event_id,
            "summary_kind": summary_kind,
            "summary_tag": summary_tag,
        }

    async def build_prompt_segments(
        self: CanonicalPublicMemoryFacade,
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
        recent_chat_include_tags: bool = False,
        recent_chat_include_ts: bool = False,
        level: str | None = None,
        use_persistence: bool = False,
    ) -> dict[str, Any]:
        """Assemble long-term, recent-chat, and recent-Tool prompt segments.

        Each segment uses one explicit canonical query. Summary failures propagate
        instead of being rewritten as a successful empty long-term segment.

        Examples:
            Build the default prompt context:
                ```python
                segments = await memory.build_prompt_segments(recent_chat_limit=20)
                ```

            Add durable Tool history:
                ```python
                segments = await memory.build_prompt_segments(
                    include_recent_tools=True,
                    tool="search",
                    use_persistence=True,
                )
                ```

        Args:
            recent_chat_limit: Non-negative recent chat result bound.
            include_long_term: Include durable summary text when true.
            summary_tag: Exact tag identifying the summary stream.
            summary_scope_id: Optional public payload scope label for summaries.
            summary_kind: Exact canonical Event kind for summaries.
            max_summaries: Non-negative summary result bound.
            include_recent_tools: Include Tool result history when true.
            tool: Optional exact Tool topic filter.
            tool_limit: Non-negative Tool result bound.
            recent_chat_tags: Optional tags every chat Event must contain.
            recent_tool_tags: Optional tags every Tool Event must contain.
            recent_chat_include_tags: Include tags in chat prompt mappings.
            recent_chat_include_ts: Include timestamps in chat prompt mappings.
            level: Recognized compatibility scope level for the already-bound facade.
            use_persistence: Select durable events for chat and Tool segments.

        Returns:
            dict[str, Any]: Mapping with `long_term`, `recent_chat`, and
            `recent_tools` prompt segments.

        Notes:
            This method performs no provider selection, retry against another tier,
            catch-and-empty substitution, or legacy observer probing.
        """
        _limit("recent_chat_limit", recent_chat_limit)
        _limit("max_summaries", max_summaries)
        _limit("tool_limit", tool_limit)
        long_term = ""
        if include_long_term and max_summaries:
            summaries = await self.list_summaries(
                summary_tag=summary_tag,
                limit=max_summaries,
                summary_kind=summary_kind,
                scope_id=summary_scope_id,
                level=level if level is not None else "scope",
            )
            long_term = "\n\n".join(
                text for summary in summaries if (text := _summary_text(summary))
            )
        recent_chat = await self.recent_chat(
            limit=recent_chat_limit,
            tags=recent_chat_tags,
            level=level,
            use_persistence=use_persistence,
            include_tags=recent_chat_include_tags,
            include_ts=recent_chat_include_ts,
        )
        recent_tools: list[dict[str, Any]] = []
        if include_recent_tools and tool_limit:
            tool_events: list[Event] = await self.query_events(
                kinds=["tool_result"],
                tags=recent_tool_tags,
                limit=tool_limit,
                level=level,
                use_persistence=use_persistence,
                return_event=True,
                tool=tool,
                order_dir="desc",
            )
            recent_tools = [_tool_mapping(event) for event in reversed(tool_events)]
        return {
            "long_term": long_term,
            "recent_chat": recent_chat,
            "recent_tools": recent_tools,
        }

    async def record_state(
        self: CanonicalPublicMemoryFacade,
        key: str,
        value: Any,
        **kwargs: Any,
    ) -> Event:
        """Append state through the deprecated public method name.

        This alias delegates exactly once to canonical `append_state_snapshot`.

        Examples:
            Record ordinary state:
                ```python
                event = await memory.record_state("counter", {"value": 1})
                ```

            Record state with revision control:
                ```python
                event = await memory.record_state(
                    "counter", {"value": 2}, expected_revision=1
                )
                ```

        Args:
            key: Exact canonical state key.
            value: JSON-compatible state value.
            **kwargs: Arguments accepted by `append_state_snapshot`.

        Returns:
            Event: Public state Event reconstructed from canonical `StateStore`.

        Notes:
            Deprecated compatibility alias; use `append_state_snapshot`. No EventStore
            duplicate, method probe, or persistence fallback is performed.
        """
        return await self.append_state_snapshot(key, value, **kwargs)

    async def record_chat_user(
        self: CanonicalPublicMemoryFacade,
        text: str,
        **kwargs: Any,
    ) -> Event:
        """Append a user chat turn through the deprecated public method name.

        This alias delegates exactly once to canonical `append_chat_turn`.

        Examples:
            Record user text:
                ```python
                event = await memory.record_chat_user("Hello")
                ```

            Record annotated user text:
                ```python
                event = await memory.record_chat_user("Hello", tags=["session.chat"])
                ```

        Args:
            text: Exact user-authored chat text.
            **kwargs: Arguments accepted by `append_chat_turn`.

        Returns:
            Event: Persisted public user chat Event.

        Notes:
            Deprecated compatibility alias; use `append_chat_turn("user", ...)`.
        """
        return await self.append_chat_turn("user", text, **kwargs)

    async def distill_long_term(
        self: CanonicalPublicMemoryFacade,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Distill a summary through the deprecated public method name.

        This alias delegates exactly once to canonical `distill_summary`.

        Examples:
            Distill session chat:
                ```python
                summary = await memory.distill_long_term(include_kinds=["chat.turn"])
                ```

            Require explicit LLM distillation:
                ```python
                summary = await memory.distill_long_term(use_llm=True)
                ```

        Args:
            **kwargs: Arguments accepted by `distill_summary`.

        Returns:
            dict[str, Any]: Stored summary payload or an empty mapping when no source
            Events qualify.

        Notes:
            Deprecated compatibility alias; use `distill_summary`. Errors propagate
            unchanged and never select heuristic behavior after an LLM failure.
        """
        return await self.distill_summary(**kwargs)


async def _distilled_payload(
    memory: CanonicalPublicMemoryFacade,
    *,
    events: list[Event],
    summary_kind: str,
    summary_tag: str,
    include_kinds: list[str] | None,
    include_tags: list[str] | None,
    max_events: int,
    min_signal: float,
    use_llm: bool,
) -> dict[str, Any]:
    common = {
        "type": summary_kind,
        "version": 1,
        "summary_tag": summary_tag,
        "time_window": {"from": events[0].ts, "to": events[-1].ts},
        "num_events": len(events),
        "source_event_ids": [event.event_id for event in events],
        "include_kinds": include_kinds,
        "include_tags": include_tags,
        "min_signal": min_signal,
        "max_events": max_events,
    }
    transcript = _transcript(events)
    if not use_llm:
        return {
            **common,
            "ts": _utc_timestamp(memory._clock()),
            "text": transcript,
        }
    if memory.llm is None:
        raise RuntimeError("LLM client not configured for canonical Memory distillation")
    raw, usage = await memory.llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Summarize the event transcript as strict JSON with keys summary, "
                    "key_facts, and open_loops."
                ),
            },
            {"role": "user", "content": transcript},
        ],
        output_format="json",
    )
    if not isinstance(raw, str):
        raise TypeError("canonical Memory distillation requires textual JSON output")
    decoded = json.loads(raw)
    if not isinstance(decoded, dict):
        raise TypeError("canonical Memory distillation JSON must be an object")
    summary = decoded.get("summary")
    key_facts = decoded.get("key_facts")
    open_loops = decoded.get("open_loops")
    if not isinstance(summary, str):
        raise ValueError("canonical Memory distillation summary must be a string")
    _string_list("key_facts", key_facts)
    _string_list("open_loops", open_loops)
    return {
        **common,
        "summary": summary,
        "key_facts": key_facts,
        "open_loops": open_loops,
        "llm_usage": usage,
    }


def _transcript(events: list[Event]) -> str:
    lines: list[str] = []
    for event in events:
        content = _event_text(event)
        if len(content) > 500:
            content = content[:500] + "…"
        if content:
            lines.append(f"[{event.stage or event.kind or 'event'}] {content}")
    return "\n".join(lines)


def _event_text(event: Event) -> str:
    return event.text or ""


def _chat_role(event: Event) -> str:
    if event.stage not in {"user", "assistant", "system", "tool"}:
        raise ValueError("canonical chat.turn Event must contain an exact public role stage")
    return event.stage


def _roles(values: Sequence[str] | None) -> frozenset[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        raise TypeError("roles must be a sequence of exact strings, not a string")
    result = frozenset(values)
    if any(
        not isinstance(role, str) or not role.strip() or role != role.strip() for role in result
    ):
        raise ValueError("roles must contain exact non-empty strings")
    if not result.issubset({"user", "assistant", "system", "tool"}):
        raise ValueError("roles must contain recognized public chat roles")
    return result


def _summary_scope(event: Event) -> str:
    if isinstance(event.data, Mapping) and isinstance(event.data.get("scope_id"), str):
        return event.data["scope_id"]
    return event.scope_id


def _summary_mapping(
    event: Event,
    *,
    summary_kind: str,
    summary_tag: str,
) -> dict[str, Any]:
    if not isinstance(event.data, Mapping):
        raise ValueError("canonical summary Event must contain an object payload")
    payload = dict(event.data)
    payload.update(
        {
            "event_id": event.event_id,
            "summary_kind": summary_kind,
            "summary_tag": summary_tag,
            "ts": payload.get("ts") or event.ts,
        }
    )
    return payload


def _summary_text(summary: Mapping[str, Any]) -> str:
    for key in ("summary", "text"):
        value = summary.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _tool_mapping(event: Event) -> dict[str, Any]:
    return {
        "ts": event.ts,
        "tool": event.tool,
        "message": event.text,
        "inputs": event.inputs,
        "outputs": event.outputs,
        "tags": list(event.tags or ()),
    }


def _merge_extra(payload: dict[str, Any], extra: dict[str, Any] | None) -> None:
    if not extra:
        return
    conflicts = sorted((set(payload) | {"event_id", "summary_kind"}).intersection(extra))
    if conflicts:
        raise ValueError("extra_data conflicts with summary fields: " + ", ".join(conflicts))
    payload.update(extra)


def _limit(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _MAX_PUBLIC_SCAN:
        raise ValueError(f"{name} must be between 0 and {_MAX_PUBLIC_SCAN}")


def _positive_limit(name: str, value: int) -> None:
    _limit(name, value)
    if value == 0:
        raise ValueError(f"{name} must be positive")


def _finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


def _string_list(name: str, value: object) -> None:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"canonical Memory distillation {name} must be a string list")


def _exact(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be an exact non-empty string")


def _optional_exact(name: str, value: str | None) -> None:
    if value is not None:
        _exact(name, value)


def _utc_timestamp(value: datetime) -> str:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != UTC.utcoffset(value)
    ):
        raise ValueError("Memory clock must return a timezone-aware UTC datetime")
    return value.isoformat()
