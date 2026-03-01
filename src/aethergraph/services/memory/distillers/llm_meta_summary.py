from __future__ import annotations

import json
from typing import Any

from aethergraph.contracts.services.llm import LLMClientProtocol
from aethergraph.contracts.services.memory import Distiller, Event
from aethergraph.services.memory.facade.utils import now_iso


class LLMMetaSummaryDistiller(Distiller):
    """
    LLM-based meta-summary distiller.

    v2: no HotLog / DocStore. It receives a list of summary Events, each with
    a summary payload in evt.data, and produces a higher-level summary dict.
    """

    def __init__(
        self,
        *,
        llm: LLMClientProtocol,
        source_kind: str = "long_term_summary",
        source_tag: str = "session",
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
        self.model = model

    def _build_prompt_from_saved(
        self,
        summaries: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        lines: list[str] = []
        for idx, s in enumerate(summaries, start=1):
            tw = s.get("time_window") or {}
            tw_from = tw.get("from") or s.get("ts")
            tw_to = tw.get("to") or s.get("ts")

            # Support both "summary" (LLM distiller) and "text" (non-LLM distiller)
            body = (s.get("summary") or s.get("text") or "").strip()

            # Minimal fence stripping if someone stored fenced content
            if body.startswith("```"):
                body = body.strip().strip("`").strip()

            if len(body) > 2000:
                body = body[:2000] + "…"

            lines.append(f"Summary {idx} [{tw_from} → {tw_to}]:\n{body}\n")

        transcript = "\n\n".join(lines)

        system = (
            "You are a higher-level summarizer over an agent's existing long-term summaries. "
            "Given multiple prior summaries (each describing a time window), produce a meta-summary "
            "capturing long-term themes, stable user facts, and persistent open loops."
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

        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    async def distill(
        self,
        *,
        events: list[Event],
    ) -> dict[str, Any]:
        """
        Produce a meta-summary from a list of *summary* events.

        Caller responsibilities:
        - Pass only relevant events (e.g. kind=long_term_summary, correct tags).
        - Respect max_summaries / level outside, if desired.
        """
        if not events:
            return {}

        # Enforce min_signal on the events themselves
        kept_events: list[Event] = []
        for e in events:
            sig = getattr(e, "signal", None)
            if isinstance(sig, (int, float)) and float(sig) < self.min_signal:  # noqa: UP038
                continue
            kept_events.append(e)

        if not kept_events:
            return {}

        # Cap at max_summaries, keep most recent
        kept_events = kept_events[-self.max_summaries :]

        # Convert events -> summary dicts (what we will actually feed to LLM)
        loaded: list[dict[str, Any]] = []
        for e in kept_events:
            # Prefer the payload written by the long-term summarizer
            if isinstance(e.data, dict):  # noqa: SIM108
                s = dict(e.data)
            else:
                s = {}

            # Ensure some basic fields exist
            s.setdefault("ts", e.ts)
            s.setdefault("summary", e.text or s.get("summary") or "")
            if "time_window" not in s:
                # fall back to a degenerate window around ts
                s["time_window"] = {"from": e.ts, "to": e.ts}

            # Keep the event id for traceability
            s.setdefault("source_event_id", e.event_id)

            # Type / tag sanity (doesn't strictly need to match, but nice to keep)
            s.setdefault("type", self.source_kind)
            s.setdefault("summary_tag", self.source_tag)

            loaded.append(s)

        if not loaded:
            return {}

        # Compute aggregated time window
        def _pick_time(s: dict[str, Any], key: str) -> str | None:
            tw = s.get("time_window") or {}
            return tw.get(key) or s.get("ts")

        times_from = [t for t in (_pick_time(s, "from") for s in loaded) if t]
        times_to = [t for t in (_pick_time(s, "to") for s in loaded) if t]

        first_from = min(times_from) if times_from else (loaded[0].get("ts") or now_iso())
        last_to = max(times_to) if times_to else (loaded[-1].get("ts") or now_iso())

        # Build prompt and call LLM
        messages = self._build_prompt_from_saved(loaded)
        try:
            if self.model:
                summary_json_str, usage = await self.llm.chat(messages, model=self.model)  # type: ignore[arg-type]
            else:
                summary_json_str, usage = await self.llm.chat(messages)
        except TypeError:
            # client doesn't accept model=...
            summary_json_str, usage = await self.llm.chat(messages)

        try:
            payload = json.loads(summary_json_str)
        except Exception:
            payload = {"summary": summary_json_str, "key_facts": [], "open_loops": []}

        ts = now_iso()

        summary_obj: dict[str, Any] = {
            "type": self.summary_kind,
            "version": 1,
            "summary_tag": self.summary_tag,
            "source_summary_kind": self.source_kind,
            "source_summary_tag": self.source_tag,
            "ts": ts,
            "time_window": {"from": first_from, "to": last_to},
            "num_source_summaries": len(loaded),
            "source_event_ids": [e.event_id for e in kept_events],
            "summary": payload.get("summary", ""),
            "key_facts": payload.get("key_facts", []),
            "open_loops": payload.get("open_loops", []),
            "llm_usage": usage,
            "llm_model": getattr(self.llm, "model", None),
            "llm_model_override": self.model,
            "min_signal": self.min_signal,
        }

        return summary_obj
