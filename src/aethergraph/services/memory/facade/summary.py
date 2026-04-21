from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aethergraph.contracts.services.memory import Event

if TYPE_CHECKING:
    from aethergraph.contracts.services.memory import MemoryFacadeProtocol


class SummaryMixin:
    def _summary_matches_scope_id(
        self: MemoryFacadeProtocol,
        evt: Event,
        *,
        scope_id: str | None,
    ) -> bool:
        if scope_id is None:
            return True
        if getattr(evt, "scope_id", None) == scope_id:
            return True
        data = getattr(evt, "data", None)
        return isinstance(data, dict) and data.get("scope_id") == scope_id

    async def distill_summary(
        self: MemoryFacadeProtocol,
        *,
        level=None,
        summary_tag: str = "session",
        summary_kind: str = "long_term_summary",
        include_kinds: list[str] | None = None,
        include_tags: list[str] | None = None,
        max_events: int = 200,
        min_signal: float | None = None,
        use_llm: bool = False,
    ) -> dict[str, Any]:
        eff_level = level or "scope"
        min_signal = min_signal if min_signal is not None else self.default_signal_threshold
        events = await self.query_events(
            kinds=include_kinds,
            tags=include_tags,
            limit=max(max_events * (8 if include_tags else 2), 200),
            level=eff_level,
            use_persistence=True,
            return_event=True,
        )
        filtered = [
            event
            for event in events[-max_events:]
            if (getattr(event, "signal", None) or 0.0) >= min_signal
        ]
        if not filtered:
            return {}
        if use_llm:
            if not self.llm:
                raise RuntimeError("LLM client not configured")
            from aethergraph.services.memory.distillers.llm_long_term import LLMLongTermSummarizer

            summarizer = LLMLongTermSummarizer(
                llm=self.llm,
                summary_kind=summary_kind,
                summary_tag=summary_tag,
                include_kinds=include_kinds,
                include_tags=include_tags,
                max_events=max_events,
                min_signal=min_signal,
            )
        else:
            from aethergraph.services.memory.distillers.long_term import LongTermSummarizer

            summarizer = LongTermSummarizer(
                summary_kind=summary_kind,
                summary_tag=summary_tag,
                include_kinds=include_kinds,
                include_tags=include_tags,
                max_events=max_events,
                min_signal=min_signal,
            )
        summary = await summarizer.distill(events=filtered)
        if not summary:
            return {}
        text = summary.get("summary", "") or summary.get("text", "")
        preview = text[:2000] + (" ...[truncated]" if len(text) > 2000 else "")
        evt = await self.append_event(
            kind=summary_kind,
            data=summary,
            tags=["summary", summary_tag, *(["llm"] if use_llm else [])],
            severity=2,
            stage="summary_llm" if use_llm else "summary",
            signal=0.7 if use_llm else None,
            text=preview,
            metrics={"num_events": summary.get("num_events", len(filtered))},
        )
        summary["event_id"] = evt.event_id
        summary["summary_kind"] = summary_kind
        summary["summary_tag"] = summary_tag
        return summary

    async def list_summaries(
        self: MemoryFacadeProtocol,
        *,
        summary_tag: str = "session",
        limit: int = 3,
        summary_kind: str = "long_term_summary",
        scope_id: str | None = None,
        level=None,
    ) -> list[dict[str, Any]]:
        fetch_limit = max(limit * 5, 20) if scope_id is not None else limit
        events: list[Event] = await self.query_events(
            kinds=[summary_kind],
            tags=["summary", summary_tag],
            limit=fetch_limit,
            level=level or "scope",
            use_persistence=True,
            return_event=True,
        )
        if scope_id is not None:
            events = [
                event
                for event in events
                if self._summary_matches_scope_id(event, scope_id=scope_id)
            ]
        events = sorted(events, key=lambda event: event.ts)[-limit:]
        out: list[dict[str, Any]] = []
        for evt in events:
            if evt.data:
                out.append(evt.data)
            else:
                out.append(
                    {
                        "summary": evt.text or "",
                        "summary_kind": summary_kind,
                        "summary_tag": summary_tag,
                        "event_id": evt.event_id,
                        "ts": evt.ts,
                    }
                )
        return out

    async def get_latest_summary(
        self: MemoryFacadeProtocol,
        scope_id: str | None = None,
        *,
        summary_tag: str = "session",
        summary_kind: str = "long_term_summary",
        level=None,
    ) -> dict[str, Any] | None:
        summaries = await self.list_summaries(
            summary_tag=summary_tag,
            summary_kind=summary_kind,
            limit=1,
            scope_id=scope_id,
            level=level or "scope",
        )
        return summaries[-1] if summaries else None

    async def soft_hydrate_last_summary(
        self: MemoryFacadeProtocol,
        *,
        summary_tag: str = "session",
        summary_kind: str = "long_term_summary",
        scope_id: str | None = None,
        level=None,
    ) -> dict[str, Any] | None:
        summary = await self.get_latest_summary(
            scope_id=scope_id,
            summary_tag=summary_tag,
            summary_kind=summary_kind,
            level=level or "scope",
        )
        if not summary:
            return None
        text = summary.get("summary") or summary.get("text") or ""
        preview = text[:2000] + (" ...[truncated]" if len(text) > 2000 else "")
        await self.append_event(
            kind=f"{summary_kind}_hydrate",
            data={"summary": summary},
            tags=["summary", "hydrate", summary_tag],
            severity=1,
            stage="hydrate",
            signal=0.4,
            text=preview,
            metrics={"num_events": summary.get("num_events", 0)},
        )
        return summary
