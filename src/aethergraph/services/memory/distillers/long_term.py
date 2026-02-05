from __future__ import annotations

from collections.abc import Iterable
import json
import time
from typing import Any

from aethergraph.contracts.services.memory import Distiller, Event, HotLog
from aethergraph.contracts.storage.doc_store import DocStore
from aethergraph.services.memory.utils import _summary_doc_id


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class LongTermSummarizer(Distiller):
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
        run_id: str,
        timeline_id: str,
        scope_id: str | None = None,
        *,
        hotlog: HotLog,
        docs: DocStore,
        **kw: Any,
    ) -> dict[str, Any]:
        # Over-fetch strategy:
        # Tag filtering can be very selective (thread/session tags), so fetch more.
        base_mult = 2
        if self.include_tags:
            base_mult = 8

        fetch_limit = max(self.max_events * base_mult, 200)

        # Narrow by kinds early when possible (less noise => more chance to fill max_events)
        raw = await hotlog.recent(
            timeline_id,
            kinds=self.include_kinds,
            limit=fetch_limit,
        )

        kept = self._filter_events(raw)
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

        scope = scope_id or run_id
        summary = {
            "type": self.summary_kind,
            "version": 1,
            "run_id": run_id,
            "scope_id": scope,
            "summary_tag": self.summary_tag,
            "ts": ts,
            "time_window": {"from": first_ts, "to": last_ts},
            "num_events": len(kept),
            "source_event_ids": src_ids,
            "text": digest_text,
            "include_kinds": self.include_kinds,
            "include_tags": self.include_tags,
            "min_signal": self.min_signal,
            "fetch_limit": fetch_limit,
        }

        doc_id = _summary_doc_id(scope, self.summary_tag, ts)
        await docs.put(doc_id, summary)

        preview = digest_text[:2000] + (" …[truncated]" if len(digest_text) > 2000 else "")

        return {
            "summary_doc_id": doc_id,
            "summary_kind": self.summary_kind,
            "summary_tag": self.summary_tag,
            "time_window": summary["time_window"],
            "num_events": len(kept),
            "preview": preview,
            "ts": ts,
        }
