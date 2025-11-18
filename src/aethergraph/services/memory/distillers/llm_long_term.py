from __future__ import annotations

from collections.abc import Iterable
import json
from typing import Any

from aethergraph.contracts.services.llm import LLMClientProtocol
from aethergraph.contracts.services.memory import Distiller, Event, HotLog, Indices, Persistence
from aethergraph.services.memory.distillers.long_term import ar_summary_uri
from aethergraph.services.memory.facade import now_iso, stable_event_id


class LLMLongTermSummarizer(Distiller):
    """
    LLM-based long-term summarizer.

    Flow:
      1) Pull recent events from HotLog.
      2) Filter by kind/tag/signal.
      3) Build a prompt that shows the most important events as a transcript.
      4) Call LLM to generate a structured summary.
      5) Save summary JSON via Persistence.save_json(uri).
      6) Emit a long_term_summary Event pointing to summary_uri.

    This is complementary to RAG:
      - LLM distiller compresses sequences into a digest.
      - RAG uses many such digests + raw docs for retrieval.
    """

    def __init__(
        self,
        *,
        llm: LLMClientProtocol,
        summary_kind: str = "long_term_summary",
        summary_tag: str = "session",
        include_kinds: list[str] | None = None,
        include_tags: list[str] | None = None,
        max_events: int = 200,
        min_signal: float = 0.0,
        model: str | None = None,
    ):
        self.llm = llm
        self.summary_kind = summary_kind
        self.summary_tag = summary_tag
        self.include_kinds = include_kinds
        self.include_tags = include_tags
        self.max_events = max_events
        self.min_signal = min_signal
        self.model = model  # optional model override

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

    def _build_prompt(self, events: list[Event]) -> list[dict[str, str]]:
        """
        Convert events into a chat-style context for summarization.

        We keep it model-agnostic: a list of {role, content} messages.
        """
        lines: list[str] = []

        for e in events:
            role = e.stage or e.kind or "event"
            if e.text:
                lines.append(f"[{role}] {e.text}")

        transcript = "\n".join(lines)

        system = (
            "You are a log summarizer for an agent's memory. "
            "Given a chronological transcript of events, produce a concise summary "
            "of what happened, key themes, important user facts, and open TODOs."
        )

        user = (
            "Here is the recent event transcript:\n\n"
            f"{transcript}\n\n"
            "Return a JSON object with keys: "
            "`summary` (string), "
            "`key_facts` (list of strings), "
            "`open_loops` (list of strings)."
        )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    async def distill(
        self,
        run_id: str,
        *,
        hotlog: HotLog,
        persistence: Persistence,
        indices: Indices,
        **kw: Any,
    ) -> dict[str, Any]:
        # 1) fetch more events than needed, then filter
        raw = await hotlog.recent(run_id, kinds=None, limit=self.max_events * 2)
        kept = self._filter_events(raw)
        if not kept:
            return {}

        kept = kept[-self.max_events :]
        first_ts = kept[0].ts
        last_ts = kept[-1].ts

        # 2) Build prompt and call LLM
        messages = self._build_prompt(kept)
        model = self.model  # could be None if llm has a default

        # LLMClientProtocol: assume chat(...) returns (text, usage)
        summary_json_str, usage = await self.llm.chat(
            messages,
            model=model,
        )

        # 3) Parse LLM JSON response
        try:
            payload = json.loads(summary_json_str)
        except Exception:
            payload = {
                "summary": summary_json_str,
                "key_facts": [],
                "open_loops": [],
            }

        ts = now_iso()

        summary_obj = {
            "type": self.summary_kind,
            "version": 1,
            "run_id": run_id,
            "summary_tag": self.summary_tag,
            "ts": ts,
            "time_window": {"from": first_ts, "to": last_ts},
            "num_events": len(kept),
            "source_event_ids": [e.event_id for e in kept],
            "summary": payload.get("summary", ""),
            "key_facts": payload.get("key_facts", []),
            "open_loops": payload.get("open_loops", []),
            "llm_usage": usage,
            "llm_model": model,
        }

        uri = ar_summary_uri(run_id, self.summary_tag, ts)
        await persistence.save_json(uri=uri, obj=summary_obj)

        # 4) Emit summary Event with preview + uri in data
        text = summary_obj["summary"] or ""
        preview = text[:2000] + (" …[truncated]" if len(text) > 2000 else "")

        evt = Event(
            event_id="",
            ts=ts,
            run_id=run_id,
            kind=self.summary_kind,
            stage="summary_llm",
            text=preview,
            tags=["summary", "llm", self.summary_tag],
            data={
                "summary_uri": uri,
                "summary_tag": self.summary_tag,
                "time_window": summary_obj["time_window"],
                "num_events": len(kept),
            },
            metrics={"num_events": len(kept)},
            severity=2,
            signal=0.7,
        )

        evt.event_id = stable_event_id(
            {
                "ts": ts,
                "run_id": run_id,
                "kind": self.summary_kind,
                "summary_tag": self.summary_tag,
                "preview": preview[:200],
            }
        )

        await hotlog.append(run_id, evt, ttl_s=7 * 24 * 3600, limit=1000)
        await persistence.append_event(run_id, evt)

        return {
            "uri": uri,
            "summary_kind": self.summary_kind,
            "summary_tag": self.summary_tag,
            "time_window": summary_obj["time_window"],
            "num_events": len(kept),
        }
