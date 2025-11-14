from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from aethergraph.contracts.services.memory import Distiller, Event, HotLog, Indices, Persistence


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _stable_event_id(parts: dict[str, Any]) -> str:
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


def _episode_uri(sessiorun_idn_id: str, tool: str, run_id: str) -> str:
    safe = tool.replace("/", "_")
    return f"file://mem/{run_id}/episodes/{safe}/{run_id}.json"


class EpisodeSummarizer(Distiller):
    """
    Aggregate all events for (tool, run_id) into a compact episode summary:
      - sources: event_ids
      - merged metrics (last-write-wins)
      - notes: last N textual notes
    Writes a JSON summary artifact and emits a lightweight run_summary event.
    """

    def __init__(
        self, *, include_metrics: bool = True, note_limit: int = 8, note_chars: int = 2000
    ):
        self.include_metrics = include_metrics
        self.note_limit = note_limit
        self.note_chars = note_chars

    async def distill(
        self,
        run_id: str,
        *,
        hotlog: HotLog,
        persistence: Persistence,
        indices: Indices,
        tool: str,
        **kw,
    ) -> dict[str, Any]:
        # Pull a reasonable window from hot memory; filter in-process.
        # (If needed later, add a Persistence scan by day.)
        events = await hotlog.recent(
            run_id, kinds=["tool_start", "tool_result", "error", "run_summary"], limit=400
        )

        eps = [e for e in events if e.run_id == run_id and (e.tool or "") == tool]
        if not eps:
            return {}

        srcs: list[str] = []
        notes: list[str] = []
        metrics: dict[str, float] = {}

        for e in eps:
            if e.event_id:
                srcs.append(e.event_id)
            if self.include_metrics and e.metrics:
                metrics.update(e.metrics)  # simple merge; last-write-wins
            if e.text:
                notes.append(e.text)

        ts = _now_iso()
        summary = {
            "kind": "episode_summary",
            "run_id": run_id,
            "tool": tool,
            "ts": ts,
            "sources": srcs,
            "metrics": metrics,
            "notes": notes[-self.note_limit :],
        }

        uri = _episode_uri(run_id, tool, run_id)
        await persistence.save_json(uri, summary)

        # Emit a compact run_summary event
        compact_text = "\n".join(summary["notes"][-self.note_limit :])[: self.note_chars]
        evt_base = {
            "run_id": run_id,
            "tool": tool,
            "kind": "run_summary",
            "severity": 1,
            "tags": ["summary", "episode"],
        }
        eid = _stable_event_id(
            {
                "ts": ts,
                "run_id": run_id,
                "tool": tool,
                "kind": "run_summary",
                "text": compact_text[:200],
            }
        )
        evt = Event(
            event_id=eid,
            ts=ts,
            text=compact_text,
            metrics={"notes": len(notes)},
            signal=0.5,
            **evt_base,
        )
        await hotlog.append(run_id, evt, ttl_s=7 * 24 * 3600, limit=1000)
        await persistence.append_event(run_id, evt)

        return {"uri": uri, "sources": srcs, "metrics": metrics}
