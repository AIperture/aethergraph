from __future__ import annotations

import json
from typing import Any

from aethergraph.contracts.services.llm import LLMClientProtocol
from aethergraph.contracts.services.memory import Distiller, HotLog
from aethergraph.contracts.storage.doc_store import DocStore
from aethergraph.services.memory.facade.utils import now_iso
from aethergraph.services.memory.utils import _summary_doc_id, _summary_prefix


class LLMMetaSummaryDistiller(Distiller):
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

    def _build_prompt_from_saved(self, summaries: list[dict[str, Any]]) -> list[dict[str, str]]:
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
        run_id: str,
        timeline_id: str,
        scope_id: str | None = None,
        *,
        hotlog: HotLog,
        docs: DocStore,
        **kw: Any,
    ) -> dict[str, Any]:
        scope = scope_id or run_id
        prefix = _summary_prefix(scope, self.source_tag)

        # Load persisted long-term summaries from DocStore
        try:
            all_ids = await docs.list()
        except Exception:
            all_ids = []

        candidates = sorted(d for d in all_ids if d.startswith(prefix))
        if not candidates:
            return {}

        chosen_ids = candidates[-self.max_summaries :]
        loaded: list[dict[str, Any]] = []
        for doc_id in chosen_ids:
            try:
                doc = await docs.get(doc_id)
                if doc is not None:
                    loaded.append(doc)  # type: ignore[arg-type]
            except Exception:
                continue

        if not loaded:
            return {}

        # Enforce consistency + min_signal if present
        kept: list[dict[str, Any]] = []
        for s in loaded:
            if s.get("type") != self.source_kind:
                continue
            if s.get("summary_tag") != self.source_tag:
                continue

            sig_val = s.get("signal", None)
            if isinstance(sig_val, (int, float)) and float(sig_val) < self.min_signal:  # noqa: UP038
                continue
            kept.append(s)

        if not kept:
            return {}

        # Derive aggregated time window safely
        def _pick_time(s: dict[str, Any], key: str) -> str | None:
            tw = s.get("time_window") or {}
            return tw.get(key) or s.get("ts")

        times_from = [t for t in (_pick_time(s, "from") for s in kept) if t]
        times_to = [t for t in (_pick_time(s, "to") for s in kept) if t]

        first_from = min(times_from) if times_from else (kept[0].get("ts") or now_iso())
        last_to = max(times_to) if times_to else (kept[-1].get("ts") or now_iso())

        # Build prompt and call LLM (respect model override)
        messages = self._build_prompt_from_saved(kept)
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
            # ✅ store doc_ids you actually read (truth)
            "source_summary_doc_ids": chosen_ids[-len(kept) :],
            "summary": payload.get("summary", ""),
            "key_facts": payload.get("key_facts", []),
            "open_loops": payload.get("open_loops", []),
            "llm_usage": usage,
            "llm_model": getattr(self.llm, "model", None),
            "llm_model_override": self.model,
            "min_signal": self.min_signal,
        }

        doc_id = _summary_doc_id(scope, self.summary_tag, ts)
        await docs.put(doc_id, summary_obj)

        text = summary_obj["summary"] or ""
        preview = text[:2000] + (" …[truncated]" if len(text) > 2000 else "")

        return {
            "summary_doc_id": doc_id,
            "summary_kind": self.summary_kind,
            "summary_tag": self.summary_tag,
            "time_window": summary_obj["time_window"],
            "num_source_summaries": summary_obj["num_source_summaries"],
            "preview": preview,
            "ts": ts,
        }
