from __future__ import annotations

from collections.abc import Iterable
import json
from typing import Any

from aethergraph.contracts.services.llm import LLMClientProtocol
from aethergraph.contracts.services.memory import Distiller, Event, HotLog, Indices, Persistence
from aethergraph.contracts.storage.doc_store import DocStore
from aethergraph.services.memory.distillers.long_term import ar_summary_uri
from aethergraph.services.memory.facade import now_iso, stable_event_id
from aethergraph.services.memory.utils import _summary_doc_id, _summary_prefix

"""
Meta-summary pipeline (multi-scale memory):

  1) Raw events (chat_user / chat_assistant) are recorded via `mem.record(...)`.
  2) `mem.distill_long_term(...)` compresses recent events into JSON summaries under:
       mem/<scope_id>/summaries/<summary_tag>/...
     e.g. summary_tag="session" → session-level long-term summaries.
  3) `mem.distill_meta_summary(...)` loads those saved summaries from disk and asks the LLM
     to produce a higher-level "summary of summaries" (meta summary), written under:
       mem/<scope_id>/summaries/<meta_tag>/...

ASCII view:

    [events in HotLog + Persistence]
              │
              ▼
     distill_long_term(...)
              │
              ▼
    file://mem/<scope>/summaries/session/*.json   (long_term_summary)
              │
              ▼
     distill_meta_summary(...)
              │
              ▼
    file://mem/<scope>/summaries/meta/*.json      (meta_summary: summary of summaries)

You control time scales via `summary_tag` (e.g. "session", "weekly", "meta") and
`scope_id` (e.g. user+persona).
"""


class LLMMetaSummaryDistiller(Distiller):
    """
    LLM-based "summary of summaries" distiller.

    Intended use:
      - Input: previously generated summary Events (e.g. kind="long_term_summary").
      - Output: higher-level meta summary (e.g. kind="meta_summary") for a broader time scale.

    Example:
      - Source: summary_tag="session"  (daily/session summaries)
      - Target: summary_tag="meta"     (multi-session / weekly/monthly view)
    """

    def __init__(
        self,
        *,
        llm: LLMClientProtocol,
        # Source summaries (what we are compressing)
        source_kind: str = "long_term_summary",
        source_tag: str = "session",
        # Target summary (what we produce)
        summary_kind: str = "meta_summary",
        summary_tag: str = "meta",
        max_summaries: int = 20,
        min_signal: float = 0.0,
        model: str | None = None,
    ):
        self.llm = llm
        self.source_kind = source_kind
        self.source_tag = source_tag
        self.summary_kind = summary_kind
        self.summary_tag = summary_tag
        self.max_summaries = max_summaries
        self.min_signal = min_signal
        self.model = model  # optional model override

    def _filter_source_summaries(self, events: Iterable[Event]) -> list[Event]:
        """
        Keep only summary Events matching:
          - kind == source_kind
          - tags include source_tag (and ideally 'summary')
          - signal >= min_signal
        """
        out: list[Event] = []
        for e in events:
            if e.kind != self.source_kind:
                continue
            if (e.signal or 0.0) < self.min_signal:
                continue
            tags = set(e.tags or [])
            if self.source_tag and self.source_tag not in tags:
                continue
            # Optional, but helps avoid mixing random summaries:
            # require generic "summary" tag if present in your existing pipeline.
            # if "summary" not in tags:
            #     continue
            out.append(e)
        return out

    def _build_prompt(self, summaries: list[Event]) -> list[dict[str, str]]:
        """
        Convert summary Events into a chat prompt for the LLM.

        We use:
          - e.text as the main human-readable summary preview.
          - e.data.get("time_window") if present.
        """

        lines: list[str] = []

        for idx, e in enumerate(summaries, start=1):
            tw = (e.data or {}).get("time_window") if e.data else None
            tw_from = (tw or {}).get("from", e.ts)
            tw_to = (tw or {}).get("to", e.ts)
            body = e.text or ""
            lines.append(f"Summary {idx} [{tw_from} → {tw_to}]:\n{body}\n")

        transcript = "\n\n".join(lines)

        system = (
            "You are a higher-level summarizer over an agent's existing summaries. "
            "Given multiple prior summaries (each covering a period of time), you "
            "should produce a concise, higher-level meta-summary capturing: "
            "  - long-term themes and patterns, "
            "  - important user facts that remain true, "
            "  - long-running goals or open loops."
        )

        user = (
            "Here are several previous summaries, each describing a time window:"
            "\n\n"
            f"{transcript}\n\n"
            "Return a JSON object with keys: "
            "`summary` (string), "
            "`key_facts` (list of strings), "
            "`open_loops` (list of strings). "
            "Do not use markdown or include explanations outside the JSON."
        )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _build_prompt_from_saved(self, summaries: list[dict[str, Any]]) -> list[dict[str, str]]:
        """
        Build an LLM prompt from persisted summary JSONs.

        Each summary dict is the JSON you showed:
          {
            "type": "long_term_summary",
            "summary_tag": "session",
            "summary": "...",
            "time_window": {...},
            ...
          }
        """
        lines: list[str] = []

        for idx, s in enumerate(summaries, start=1):
            tw = s.get("time_window") or {}
            tw_from = tw.get("from", s.get("ts"))
            tw_to = tw.get("to", s.get("ts"))
            body = s.get("summary", "") or ""

            # (Optional) strip ```json fences if present
            stripped = body.strip()
            if stripped.startswith("```"):
                # very minimal fence strip; you can refine later
                stripped = stripped.strip("`")
                # fall back to original if this gets too messy
                body_for_prompt = stripped or body
            else:
                body_for_prompt = body

            lines.append(f"Summary {idx} [{tw_from} → {tw_to}]:\n{body_for_prompt}\n")

        transcript = "\n\n".join(lines)

        system = (
            "You are a higher-level summarizer over an agent's existing long-term summaries. "
            "Given multiple prior summaries (each describing a period), produce a meta-summary "
            "that captures long-term themes, stable user facts, and persistent open loops."
        )

        user = (
            "Here are several previous summaries:\n\n"
            f"{transcript}\n\n"
            "Return a JSON object with keys: "
            "`summary` (string), "
            "`key_facts` (list of strings), "
            "`open_loops` (list of strings). "
            "Do not include any extra explanation outside the JSON."
        )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    async def distill(
        self,
        run_id: str,
        scope_id: str = None,
        *,
        hotlog: HotLog,
        persistence: Persistence,
        indices: Indices,
        docs: DocStore,
        **kw: Any,
    ) -> dict[str, Any]:
        """
        Distill method following the Distiller protocol.

        IMPORTANT:
          - This implementation is optimized for FSPersistence and reads
            previously saved summary JSONs from:
              mem/<scope_id>/summaries/<source_tag>/*.json
          - If a different Persistence is used, we currently bail out.
        """
        scope = scope_id or run_id
        prefix = _summary_prefix(scope, self.source_tag)

        # 1) Load existing long-term summary JSONs from DocStore
        try:
            all_ids = await docs.list()
        except Exception:
            all_ids = []

        candidates = sorted(d for d in all_ids if d.startswith(prefix))
        if not candidates:
            return {}

        chosen_ids = candidates[-self.max_summaries :]
        summaries: list[dict[str, Any]] = []
        for doc_id in chosen_ids:
            try:
                doc = await docs.get(doc_id)
                if doc is not None:
                    summaries.append(doc)  # type: ignore[arg-type]
            except Exception:
                continue

        if not summaries:
            return {}

        # Optional: filter by min_signal if present in saved JSON
        filtered: list[dict[str, Any]] = []
        for s in summaries:
            sig = (
                float(s.get("signal", 0.0)) if isinstance(s.get("signal"), int | float) else 1.0
            )  # default 1.0
            if sig < self.min_signal:
                continue
            # Also enforce type/tag consistency:
            if s.get("type") != self.source_kind:
                continue
            if s.get("summary_tag") != self.source_tag:
                continue
            filtered.append(s)

        if not filtered:
            return {}

        # Keep order as loaded (already sorted by filename)
        kept = filtered

        # 2) Derive aggregated time window
        first_from = None
        last_to = None
        for s in kept:
            tw = s.get("time_window") or {}
            start = tw.get("from") or s.get("ts")
            end = tw.get("to") or s.get("ts")
            if start:
                first_from = start if first_from is None else min(first_from, start)
            if end:
                last_to = end if last_to is None else max(last_to, end)
        if first_from is None:
            first_from = kept[0].get("ts")
        if last_to is None:
            last_to = kept[-1].get("ts")

        # 3) Build prompt and call LLM
        messages = self._build_prompt_from_saved(kept)
        summary_json_str, usage = await self.llm.chat(messages)

        # 4) Parse LLM JSON response
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
            "scope_id": scope,
            "summary_tag": self.summary_tag,
            "source_summary_kind": self.source_kind,
            "source_summary_tag": self.source_tag,
            "ts": ts,
            "time_window": {"from": first_from, "to": last_to},
            "num_source_summaries": len(kept),
            "source_summary_uris": [
                # reconstruct the URI pattern we originally use
                # (this assumes summaries were written under ar_summary_uri)
                ar_summary_uri(scope, self.source_tag, s.get("ts", ts))
                for s in kept
            ],
            "summary": payload.get("summary", ""),
            "key_facts": payload.get("key_facts", []),
            "open_loops": payload.get("open_loops", []),
            "llm_usage": usage,
            "llm_model": getattr(self.llm, "model", None),
        }

        doc_id = _summary_doc_id(scope, self.summary_tag, ts)
        await docs.put(doc_id, summary_obj)

        # 5) Emit meta_summary Event
        text = summary_obj["summary"] or ""
        preview = text[:2000] + (" …[truncated]" if len(text) > 2000 else "")

        evt = Event(
            event_id="",
            ts=ts,
            run_id=run_id,
            kind=self.summary_kind,
            stage="summary_llm_meta",
            text=preview,
            tags=["summary", "llm", self.summary_tag],
            data={
                "summary_doc_id": doc_id,
                "summary_tag": self.summary_tag,
                "time_window": summary_obj["time_window"],
                "num_source_summaries": len(kept),
                "source_summary_kind": self.source_kind,
                "source_summary_tag": self.source_tag,
            },
            metrics={"num_source_summaries": len(kept)},
            severity=2,
            signal=0.8,
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
            "summary_doc_id": doc_id,
            "summary_kind": self.summary_kind,
            "summary_tag": self.summary_tag,
            "time_window": summary_obj["time_window"],
            "num_source_summaries": len(kept),
        }
