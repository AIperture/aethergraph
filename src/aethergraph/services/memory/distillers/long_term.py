from __future__ import annotations

from collections.abc import Iterable
import time
from typing import Any

from aethergraph.contracts.services.memory import Distiller, Event, HotLog, Indices, Persistence

# re-use stable_event_id from the MemoryFacade module
from aethergraph.contracts.storage.doc_store import DocStore
from aethergraph.core.runtime.runtime_metering import current_meter_context, current_metering
from aethergraph.services.memory.facade import stable_event_id
from aethergraph.services.memory.utils import _summary_doc_id


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ar_summary_uri_by_run_id(run_id: str, tag: str, ts: str) -> str:
    """
    NOTE: To deprecate this function in favor of ar_summary_uri below.

    Save summaries under the same base "mem/<run_id>/..." tree as append_event,
    but using a file:// URI so FSPersistence can handle it.
    """
    safe_ts = ts.replace(":", "-")
    return f"file://mem/{run_id}/summaries/{tag}/{safe_ts}.json"


def ar_summary_uri(scope_id: str, tag: str, ts: str) -> str:
    """
    Scope summaries by a logical memory scope, not by run_id.
    In simple setups, scope_id == run_id. For long-lived companions, scope_id
    might be something like "user:zcliu:persona:companion_v1".
    """
    safe_ts = ts.replace(":", "-")
    return f"file://mem/{scope_id}/summaries/{tag}/{safe_ts}.json"


class LongTermSummarizer(Distiller):
    """
    Generic long-term summarizer.

    Goal:
      - Take a slice of recent events (by kind and/or tag).
      - Build a compact textual digest plus small structured metadata.
      - Persist the summary as JSON via Persistence.save_json(...).
      - Emit a summary Event with kind=summary_kind and data["summary_uri"].

    This does NOT call an LLM by itself; it's a structural/logical summarizer.
    An LLM-based distiller can be layered on top later (using the same URI scheme).

    Typical usage:
      - Kinds: ["chat_user", "chat_assistant"] or app-specific kinds.
      - Tag:   "session", "daily", "episode:<id>", etc.
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
                if not e.tags:
                    continue
                if not tags.issubset(set(e.tags)):
                    continue
            if (e.signal or 0.0) < self.min_signal:
                continue
            out.append(e)
        return out

    async def distill(
        self,
        run_id: str,
        timeline_id: str,
        scope_id: str = None,
        *,
        hotlog: HotLog,
        persistence: Persistence,
        indices: Indices,
        docs: DocStore,
        **kw: Any,
    ) -> dict[str, Any]:
        """
        Steps:
          1) Grab recent events from HotLog for this run.
          2) Filter by kinds/tags/min_signal.
          3) Build a digest:
             - simple text transcript (role: text)
             - metadata: ts range, num events
          4) Save JSON summary via Persistence.save_json(file://...).
          5) Log a summary Event to hotlog + persistence, with data.summary_uri.
        """
        # 1) fetch more than we might keep to give filter some slack
        raw = await hotlog.recent(timeline_id, kinds=None, limit=self.max_events * 2)
        kept = self._filter_events(raw)
        if not kept:
            return {}

        # keep only max_events most recent
        kept = kept[-self.max_events :]

        # 2) Build digest text (simple transcript-like format)
        lines: list[str] = []
        src_ids: list[str] = []
        first_ts = kept[0].ts
        last_ts = kept[-1].ts

        for e in kept:
            role = e.stage or e.kind or "event"
            if e.text:
                lines.append(f"[{role}] {e.text}")
                src_ids.append(e.event_id)

        digest_text = "\n".join(lines)
        ts = _now_iso()

        # 3) Summary JSON shape
        summary = {
            "type": self.summary_kind,
            "version": 1,
            "run_id": run_id,
            "scope_id": scope_id or run_id,
            "summary_tag": self.summary_tag,
            "ts": ts,
            "time_window": {
                "from": first_ts,
                "to": last_ts,
            },
            "num_events": len(kept),
            "source_event_ids": src_ids,
            "text": digest_text,
        }

        # 4) Persist JSON summary via DocStore
        scope = scope_id or run_id
        doc_id = _summary_doc_id(scope, self.summary_tag, ts)
        await docs.put(doc_id, summary)

        # 5) Emit summary Event
        # NOTE: we only store a preview in text and full summary in data["summary_uri"]
        preview = digest_text[:2000] + (" …[truncated]" if len(digest_text) > 2000 else "")

        evt = Event(
            event_id="",  # fill below
            ts=ts,
            run_id=run_id,
            scope_id=scope,
            kind=self.summary_kind,
            stage="summary",
            text=preview,
            tags=["summary", self.summary_tag],
            data={
                "summary_doc_id": doc_id,
                "summary_tag": self.summary_tag,
                "time_window": summary["time_window"],
                "num_events": len(kept),
            },
            metrics={"num_events": len(kept)},
            severity=1,
            signal=0.5,
        )

        evt.event_id = stable_event_id(
            {
                "ts": ts,
                "run_id": run_id,
                "kind": self.summary_kind,
                "summary_tag": self.summary_tag,
                "text": preview[:200],
            }
        )

        await hotlog.append(timeline_id, evt, ttl_s=7 * 24 * 3600, limit=1000)
        await persistence.append_event(timeline_id, evt)

        # Metering: record summary event
        try:
            meter = current_metering()
            ctx = current_meter_context.get()
            user_id = ctx.get("user_id")
            org_id = ctx.get("org_id")

            await meter.record_event(
                user_id=user_id,
                org_id=org_id,
                run_id=run_id,
                scope_id=scope,
                kind=f"memory.{self.summary_kind}",  # e.g. "memory.long_term_summary"
            )
        except Exception:
            import logging

            logger = logging.getLogger("aethergraph.services.memory.distillers.long_term")
            logger.error("Failed to record metering event for long_term_summary")

        return {
            "summary_doc_id": doc_id,
            "summary_kind": self.summary_kind,
            "summary_tag": self.summary_tag,
            "time_window": summary["time_window"],
            "num_events": len(kept),
        }
