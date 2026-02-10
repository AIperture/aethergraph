from __future__ import annotations

from collections.abc import Iterable
import json
import time
from typing import Any

from aethergraph.contracts.services.memory import Distiller, Event


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class LongTermSummarizer(Distiller):
    """
    Non-LLM long-term summarizer.

    v2: does NOT talk to HotLog or DocStore. It just takes a list of Event
    objects and produces a structured summary dict. Storage is handled by
    MemoryFacade.record_raw().
    """

    def __init__(
        self,
        *,
        summary_kind: str = "long_term_summary",
        summary_tag: str = "session",
        include_kinds: list[str] | None = None,
        include_tags: list[str] | None = None,
        max_events: int = 200,
        min_signal: float = 0.0,
    ):
        self.summary_kind = summary_kind
        self.summary_tag = summary_tag
        self.include_kinds = include_kinds
        self.include_tags = include_tags
        self.max_events = max_events
        self.min_signal = min_signal

    def _filter_events(self, events: Iterable[Event]) -> list[Event]:
        out: list[Event] = []
        kinds = set(self.include_kinds) if self.include_kinds else None
        tags = set(self.include_tags) if self.include_tags else None

        for e in events:
            if kinds is not None and e.kind not in kinds:
                continue
            if tags is not None:
                et = set(e.tags or [])
                if not tags.issubset(et):  # AND semantics
                    continue
            if (e.signal or 0.0) < self.min_signal:
                continue
            out.append(e)
        return out

    async def distill(
        self,
        *,
        events: list[Event],
    ) -> dict[str, Any]:
        """
        Produce a long-term summary from a list of events.

        The caller (MemoryFacade) is responsible for:
        - choosing which events to pass (max_events / overfetch / level)
        - recording the resulting summary as a memory event.
        """
        if not events:
            return {}

        # Filter + cap
        kept = self._filter_events(events)
        if not kept:
            return {}

        kept = kept[-self.max_events :]

        first_ts = kept[0].ts
        last_ts = kept[-1].ts

        # Build digest text (simple transcript-like format) + source ids
        lines: list[str] = []
        src_ids: list[str] = []

        for e in kept:
            src_ids.append(e.event_id)

            role = e.stage or e.kind or "event"

            content = (e.text or "").strip()
            if not content and getattr(e, "data", None) is not None:
                # fall back to a compact JSON line
                try:
                    content = json.dumps(e.data, ensure_ascii=False)
                except Exception:
                    content = str(e.data)

            if content:
                if len(content) > 500:
                    content = content[:500] + "…"
                lines.append(f"[{role}] {content}")

        digest_text = "\n".join(lines)
        ts = _now_iso()

        # This object will be stored into evt.data by MemoryFacade.record_raw()
        summary_obj: dict[str, Any] = {
            "type": self.summary_kind,
            "version": 1,
            "summary_tag": self.summary_tag,
            "ts": ts,
            "time_window": {"from": first_ts, "to": last_ts},
            "num_events": len(kept),
            "source_event_ids": src_ids,
            "text": digest_text,
            "include_kinds": self.include_kinds,
            "include_tags": self.include_tags,
            "min_signal": self.min_signal,
            # kept for introspection; this used to be the overfetch limit
            "max_events": self.max_events,
        }

        # MemoryFacade will:
        #   - generate preview text
        #   - record as kind=summary_kind, tags=["summary", summary_tag], data=summary_obj
        return summary_obj
