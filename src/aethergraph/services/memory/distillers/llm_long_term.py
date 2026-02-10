from __future__ import annotations

from collections.abc import Iterable
import json
from typing import Any

from aethergraph.contracts.services.llm import LLMClientProtocol
from aethergraph.contracts.services.memory import Distiller, Event


class LLMLongTermSummarizer(Distiller):
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
        self.model = model

    def _filter_events(self, events: Iterable[Event]) -> list[Event]:
        out: list[Event] = []
        kinds = set(self.include_kinds) if self.include_kinds else None
        tags = set(self.include_tags) if self.include_tags else None

        for e in events:
            if kinds is not None and e.kind not in kinds:
                continue
            if tags is not None:
                et = set(e.tags or [])
                if not tags.issubset(et):
                    continue
            if (e.signal or 0.0) < self.min_signal:
                continue
            out.append(e)
        return out

    def _build_prompt(self, events: list[Event]) -> list[dict[str, str]]:
        lines: list[str] = []

        for e in events:
            role = e.stage or e.kind or "event"
            content = (e.text or "").strip()
            if not content and getattr(e, "data", None) is not None:
                try:
                    content = json.dumps(e.data, ensure_ascii=False)
                except Exception:
                    content = str(e.data)

            if content:
                if len(content) > 500:
                    content = content[:500] + "…"
                lines.append(f"[{role}] {content}")

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
            "`open_loops` (list of strings). "
            "Do not use markdown or include explanations outside the JSON."
        )

        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    async def distill(
        self,
        *,
        events: list[Event],
    ) -> dict[str, Any]:
        kept = self._filter_events(events)
        if not kept:
            return {}

        kept = kept[-self.max_events :]

        first_ts = kept[0].ts
        last_ts = kept[-1].ts

        messages = self._build_prompt(kept)

        try:
            if self.model:
                summary_json_str, usage = await self.llm.chat(messages, model=self.model)  # type: ignore[arg-type]
            else:
                summary_json_str, usage = await self.llm.chat(messages)
        except TypeError:
            summary_json_str, usage = await self.llm.chat(messages)

        try:
            payload = json.loads(summary_json_str)
        except Exception:
            payload = {"summary": summary_json_str, "key_facts": [], "open_loops": []}

        return {
            "type": self.summary_kind,
            "version": 1,
            "summary_tag": self.summary_tag,
            "time_window": {"from": first_ts, "to": last_ts},
            "num_events": len(kept),
            "source_event_ids": [e.event_id for e in kept],
            "summary": payload.get("summary", ""),
            "key_facts": payload.get("key_facts", []),
            "open_loops": payload.get("open_loops", []),
            "llm_usage": usage,
            "llm_model": getattr(self.llm, "model", None),
            "llm_model_override": self.model,
            "include_kinds": self.include_kinds,
            "include_tags": self.include_tags,
            "min_signal": self.min_signal,
        }
