from __future__ import annotations

import json
import logging
from typing import Any

from aethergraph.contracts.services.llm import LLMClientProtocol
from aethergraph.contracts.services.memory import Event, HotLog, Indices, Persistence
from aethergraph.contracts.storage.artifact_store import AsyncArtifactStore
from aethergraph.contracts.storage.doc_store import DocStore
from aethergraph.core.runtime.runtime_metering import current_metering
from aethergraph.services.memory.utils import _summary_prefix
from aethergraph.services.rag.facade import RAGFacade
from aethergraph.services.scope.scope import Scope

from .chat import ChatMixin
from .distillation import DistillationMixin
from .rag import RAGMixin
from .results import ResultMixin
from .retrieval import RetrievalMixin
from .utils import now_iso, stable_event_id


class MemoryFacade(ChatMixin, ResultMixin, RetrievalMixin, DistillationMixin, RAGMixin):
    """
    MemoryFacade coordinates core memory services for a specific run/session.
    Functionality is split across mixins in the `facade/` directory.
    """

    def __init__(
        self,
        *,
        run_id: str,
        session_id: str | None,
        graph_id: str | None,
        node_id: str | None,
        scope: Scope | None = None,
        hotlog: HotLog,
        persistence: Persistence,
        indices: Indices,
        docs: DocStore,
        artifact_store: AsyncArtifactStore,
        hot_limit: int = 1000,
        hot_ttl_s: int = 7 * 24 * 3600,
        default_signal_threshold: float = 0.0,
        logger=None,
        rag: RAGFacade | None = None,
        llm: LLMClientProtocol | None = None,
    ):
        self.run_id = run_id
        self.session_id = session_id
        self.graph_id = graph_id
        self.node_id = node_id
        self.scope = scope
        self.hotlog = hotlog
        self.persistence = persistence
        self.indices = indices
        self.docs = docs
        self.artifacts = artifact_store
        self.hot_limit = hot_limit
        self.hot_ttl_s = hot_ttl_s
        self.default_signal_threshold = default_signal_threshold
        self.logger = logger or logging.getLogger(__name__)
        self.rag = rag
        self.llm = llm

        self.memory_scope_id = (
            self.scope.memory_scope_id() if self.scope else self.session_id or self.run_id
        )
        self.timeline_id = self.memory_scope_id or self.run_id

    async def record_raw(
        self,
        *,
        base: dict[str, Any],
        text: str | None = None,
        metrics: dict[str, float] | None = None,
    ) -> Event:
        ts = now_iso()

        # Merge Scope dimensions
        dims: dict[str, str] = {}
        if self.scope is not None:
            dims = self.scope.metering_dimensions()

        run_id = base.get("run_id") or dims.get("run_id") or self.run_id
        session_id = base.get("session_id") or dims.get("session_id") or self.session_id
        scope_id = base.get("scope_id") or self.memory_scope_id or session_id or run_id

        base.setdefault("run_id", run_id)
        base.setdefault("scope_id", scope_id)
        base.setdefault("session_id", session_id)
        # ... (populate other fields from dims if needed) ...

        severity = int(base.get("severity", 2))
        signal = base.get("signal")
        if signal is None:
            signal = self._estimate_signal(text=text, metrics=metrics, severity=severity)

        kind = base.get("kind") or "misc"

        eid = stable_event_id(
            {
                "ts": ts,
                "run_id": base["run_id"],
                "kind": kind,
                "text": (text or "")[:6000],
                "tool": base.get("tool"),
            }
        )

        evt = Event(
            event_id=eid,
            ts=ts,
            run_id=run_id,
            scope_id=scope_id,
            kind=kind,
            text=text,
            data=base.get("data"),
            tags=base.get("tags"),
            metrics=metrics,
            tool=base.get("tool"),
            severity=severity,
            signal=signal,
            inputs=base.get("inputs"),
            outputs=base.get("outputs"),
            # ... pass other fields ...
            version=2,
        )

        await self.hotlog.append(self.timeline_id, evt, ttl_s=self.hot_ttl_s, limit=self.hot_limit)
        await self.persistence.append_event(self.timeline_id, evt)

        # Metering hook
        try:
            meter = current_metering()
            await meter.record_event(scope=self.scope, scope_id=scope_id, kind=f"memory.{kind}")
        except Exception:
            if self.logger:
                self.logger.exception("Error recording metering event")

        return evt

    async def record(
        self,
        kind: str,
        data: Any,
        tags: list[str] | None = None,
        severity: int = 2,
        stage: str | None = None,
        inputs_ref=None,
        outputs_ref=None,
        metrics: dict[str, float] | None = None,
        signal: float | None = None,
        text: str | None = None,  # optional override
    ) -> Event:
        """
        Convenience wrapper around record_raw() with common fields.

        - kind     : logical kind (e.g. "user_msg", "tool_call", "chat_turn")
        - data     : JSON-serializable content, or string
        - tags     : optional list of labels
        - severity : 1=low, 2=medium, 3=high
        - stage    : optional stage (user/assistant/system/etc.)
        - inputs_ref / outputs_ref : optional Value[] references
        - metrics  : numeric map (latency, tokens, etc.)
        - signal   : optional override for signal strength
        - text     : optional preview text override (if None, derived from data)
        """

        # 1) derive short preview text
        if text is None and data is not None:
            if isinstance(data, str):
                text = data
            else:
                try:
                    raw = json.dumps(data, ensure_ascii=False)
                    text = raw
                except Exception as e:
                    text = f"<unserializable data: {e!s}>"
                    if self.logger:
                        self.logger.warning(text)

        # 2) optionally truncate preview text (enforce token discipline)
        if text and len(text) > 2000:
            text = text[:2000] + " …[truncated]"

        # 3) full structured payload in Event.data when possible
        data_field: dict[str, Any] | None = None
        if isinstance(data, dict):
            data_field = data
        elif data is not None and not isinstance(data, str):
            # store under "value" if it's JSON-serializable
            try:
                json.dumps(data, ensure_ascii=False)
                data_field = {"value": data}
            except Exception:
                data_field = {"repr": repr(data)}

        base: dict[str, Any] = dict(
            kind=kind,
            stage=stage,
            severity=severity,
            tags=tags or [],
            data=data_field,
            inputs=inputs_ref,
            outputs=outputs_ref,
        )
        if signal is not None:
            base["signal"] = signal

        return await self.record_raw(base=base, text=text, metrics=metrics)

    def _estimate_signal(
        self, *, text: str | None, metrics: dict[str, Any] | None, severity: int
    ) -> float:
        score = 0.15 + 0.1 * severity
        if text:
            score += min(len(text) / 400.0, 0.4)
        if metrics:
            score += 0.2
        return max(0.0, min(1.0, score))

    async def load_last_summary(
        self,
        scope_id: str | None = None,
        *,
        summary_tag: str = "session",
    ) -> dict[str, Any] | None:
        """
        Load the most recent JSON summary for this memory scope and tag.

        Uses DocStore IDs:
        mem/{scope_id}/summaries/{summary_tag}/{ts}
        so it works regardless of persistence backend.
        """
        scope_id = scope_id or self.memory_scope_id
        prefix = _summary_prefix(scope_id, summary_tag)

        try:
            ids = await self.docs.list()
        except Exception as e:
            self.logger and self.logger.warning("load_last_summary: doc_store.list() failed: %s", e)
            return None

        # Filter and take the latest
        candidates = [d for d in ids if d.startswith(prefix)]
        if not candidates:
            return None

        latest_id = sorted(candidates)[-1]
        try:
            return await self.docs.get(latest_id)  # type: ignore[return-value]
        except Exception as e:
            self.logger and self.logger.warning(
                "load_last_summary: failed to load %s: %s", latest_id, e
            )
            return None

    async def load_recent_summaries(
        self,
        scope_id: str | None = None,
        *,
        summary_tag: str = "session",
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Load up to `limit` most recent JSON summaries for this scope+tag.

        Ordered oldest→newest (so the last item is the most recent).
        """
        scope_id = scope_id or self.memory_scope_id
        prefix = _summary_prefix(scope_id, summary_tag)

        try:
            ids = await self.docs.list()
        except Exception as e:
            self.logger and self.logger.warning(
                "load_recent_summaries: doc_store.list() failed: %s", e
            )
            return []

        candidates = sorted(d for d in ids if d.startswith(prefix))
        if not candidates:
            return []

        chosen = candidates[-limit:]
        out: list[dict[str, Any]] = []
        for doc_id in chosen:
            try:
                doc = await self.docs.get(doc_id)
                if doc is not None:
                    out.append(doc)  # type: ignore[arg-type]
            except Exception:
                continue
        return out

    async def soft_hydrate_last_summary(
        self,
        scope_id: str | None = None,
        *,
        summary_tag: str = "session",
        summary_kind: str = "long_term_summary",
    ) -> dict[str, Any] | None:
        """
        Load the last summary JSON for this tag (if any) and log a small hydrate Event
        into the current run's HotLog. Returns the loaded summary dict, or None.
        """
        scope_id = scope_id or self.memory_scope_id
        summary = await self.load_last_summary(scope_id=scope_id, summary_tag=summary_tag)
        if not summary:
            return None

        text = summary.get("text") or ""
        preview = text[:2000] + (" …[truncated]" if len(text) > 2000 else "")

        evt = Event(
            scope_id=self.memory_scope_id or self.run_id,
            event_id=stable_event_id(
                {
                    "ts": now_iso(),
                    "run_id": self.run_id,
                    "kind": f"{summary_kind}_hydrate",
                    "summary_tag": summary_tag,
                    "preview": preview[:200],
                }
            ),
            ts=now_iso(),
            run_id=self.run_id,
            kind=f"{summary_kind}_hydrate",
            stage="hydrate",
            text=preview,
            tags=["summary", "hydrate", summary_tag],
            data={"summary": summary},
            metrics={"num_events": summary.get("num_events", 0)},
            severity=1,
            signal=0.4,
        )

        await self.hotlog.append(self.timeline_id, evt, ttl_s=self.hot_ttl_s, limit=self.hot_limit)
        await self.persistence.append_event(self.timeline_id, evt)
        return summary

    # ----- Stubs for future memory facade features -----
    async def mark_event_important(
        self,
        event_id: str,
        *,
        reason: str | None = None,
        topic: str | None = None,
    ) -> None:
        """
        Stub / placeholder:

        Mark a given event as "important" / "core_fact" for future policies.

        Intended future behavior (not implemented yet):
          - Look up the Event by event_id (via Persistence).
          - Re-emit an updated Event with an added tag (e.g. "core_fact" or "pinned").
          - Optionally promote to a fact artifact or RAG doc.

        For now, this is a no-op / NotImplementedError to avoid surprise behavior.
        """
        raise NotImplementedError("mark_event_important is reserved for future memory policy")

    async def save_core_fact_artifact(
        self,
        *,
        scope_id: str,
        topic: str,
        fact_id: str,
        content: dict[str, Any],
    ):
        """
        Stub / placeholder:

        Save a canonical, long-lived fact as a pinned artifact.
        Intended future behavior:
          - Use artifacts.save_json(...) to write the fact payload under a
            stable path like file://mem/<scope_id>/facts/<topic>/<fact_id>.json
          - Mark the artifact pinned in the index.
          - Optionally write a tool_result Event referencing this artifact.

        Not implemented yet; provided as an explicit extension hook.
        """
        raise NotImplementedError("save_core_fact_artifact is reserved for future memory policy")

    async def build_prompt_segments(
        self,
        *,
        recent_chat_limit: int = 12,
        include_long_term: bool = True,
        summary_tag: str = "session",
        max_summaries: int = 3,
        include_recent_tools: bool = False,
        tool: str | None = None,
        tool_limit: int = 10,
    ) -> dict[str, Any]:
        """
        High-level helper to assemble memory context for prompts.

        Returns:
          {
            "long_term": "<combined summary text or ''>",
            "recent_chat": [ {ts, role, text, tags}, ... ],
            "recent_tools": [ {ts, tool, message, inputs, outputs, tags}, ... ]
          }
        """
        long_term_text = ""
        if include_long_term:
            try:
                summaries = await self.load_recent_summaries(
                    summary_tag=summary_tag,
                    limit=max_summaries,
                )
            except Exception:
                summaries = []

            parts: list[str] = []
            for s in summaries:
                st = s.get("summary") or s.get("text") or s.get("body") or s.get("value") or ""
                if st:
                    parts.append(st)

            if parts:
                # multiple long-term summaries → concatenate oldest→newest
                long_term_text = "\n\n".join(parts)

        recent_chat = await self.recent_chat(limit=recent_chat_limit)

        recent_tools: list[dict[str, Any]] = []
        if include_recent_tools:
            events = await self.recent_tool_results(
                tool=tool,
                limit=tool_limit,
            )
            for e in events:
                recent_tools.append(
                    {
                        "ts": getattr(e, "ts", None),
                        "tool": e.tool,
                        "message": e.text,
                        "inputs": getattr(e, "inputs", None),
                        "outputs": getattr(e, "outputs", None),
                        "tags": list(e.tags or []),
                    }
                )

        return {
            "long_term": long_term_text,
            "recent_chat": recent_chat,
            "recent_tools": recent_tools,
        }
