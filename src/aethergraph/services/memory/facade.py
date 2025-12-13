from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import os
import re
import time
from typing import Any, Literal
import unicodedata

from aethergraph.contracts.services.llm import LLMClientProtocol
from aethergraph.contracts.services.memory import Event, HotLog, Indices, Persistence
from aethergraph.contracts.storage.artifact_store import AsyncArtifactStore
from aethergraph.contracts.storage.doc_store import DocStore
from aethergraph.core.runtime.runtime_metering import current_metering
from aethergraph.services.rag.facade import RAGFacade
from aethergraph.services.scope.scope import Scope

from .utils import _summary_prefix

"""
MemoryFacade coordinates core memory services for a specific run/session.

          ┌───────────────────────────┐
          │      Agent / Graph        │
          │  (tools, flows, chat)     │
          └───────────┬───────────────┘
                      │  emits Event
                      ▼
               ┌─────────────────┐
               │   MemoryFacade  │
               │ (per run_id)    │
               └───────┬─────────┘
        record_raw/record/write_result
                      │
       ┌──────────────┼─────────────────┐
       ▼              ▼                 ▼
┌────────────┐  ┌─────────────┐   ┌──────────────┐
│   HotLog   │  │ FSPersistence│   │   Indices    │
│  (KV ring) │  │ (JSONL, FS) │   │ (name/topic) │
└────┬───────┘  └──────┬──────┘   └──────┬───────┘
     │                │                 │
     │                │ distillers read │
     │                ▼                 │
     │       ┌───────────────────┐      │
     │       │  Distillers       │      │
     │       │  (LongTerm, LLM)  │      │
     │       └─────────┬─────────┘      │
     │                 │                │
     │     save_json() │                │ update()
     │                 ▼                │
     │        ┌─────────────────────┐   │
     │        │ Summary JSON (FS)   │   │
     │        └─────────────────────┘   │
     │                 │                │
     │                 │ (optional)     │
     │                 ▼                │
     │         ┌────────────────────┐   │
     └────────▶│ Summary Event      │◀──┘
               │ (kind=long_term_*) │
               └────────────────────┘
"""

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stable_event_id(parts: dict[str, Any]) -> str:
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


def _short_hash(s: str, n: int = 8) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]


def _slug(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).strip()
    s = s.replace(" ", "-")
    s = _SAFE.sub("-", s)
    return s.strip("-") or "default"


def _load_sticky(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_sticky(path: str, m: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)


class MemoryFacade:
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
        self.logger = logger
        self.rag = rag
        self.llm = llm  # optional LLM service for RAG answering, etc.

        # order of precedence for memory scope ID:
        self.memory_scope_id = (
            self.scope.memory_scope_id() if self.scope else self.session_id or self.run_id
        )

    # ---------- recording ----------
    async def record_raw(
        self,
        *,
        base: dict[str, Any],
        text: str | None = None,
        metrics: dict[str, float] | None = None,
    ) -> Event:
        ts = now_iso()

        # 1) Derive identity/execution dimentions from Scope
        dims: dict[str, str] = {}
        if self.scope is not None:
            dims = self.scope.metering_dimensions()

        run_id = base.get("run_id") or dims.get("run_id") or self.run_id
        graph_id = base.get("graph_id") or dims.get("graph_id") or self.graph_id
        node_id = base.get("node_id") or dims.get("node_id") or self.node_id
        session_id = base.get("session_id") or dims.get("session_id") or self.session_id

        user_id = base.get("user_id") or dims.get("user_id")
        org_id = base.get("org_id") or dims.get("org_id")
        client_id = base.get("client_id") or dims.get("client_id")
        app_id = base.get("app_id") or dims.get("app_id")

        # Memory scope key (for multi-tenant memory within a run)
        scope_id = base.get("scope_id") or self.memory_scope_id or session_id or run_id

        base.setdefault("run_id", run_id)
        base.setdefault("graph_id", graph_id)
        base.setdefault("node_id", node_id)
        base.setdefault("scope_id", scope_id)
        base.setdefault("user_id", user_id)
        base.setdefault("org_id", org_id)
        base.setdefault("client_id", client_id)
        base.setdefault("app_id", app_id)
        base.setdefault("session_id", session_id)

        severity = int(base.get("severity", 2))
        signal = base.get("signal")
        if signal is None:
            signal = self._estimate_signal(text=text, metrics=metrics, severity=severity)

        # ensure kind is always present
        kind = base.get("kind") or "misc"

        eid = stable_event_id(
            {
                "ts": ts,
                "run_id": base["run_id"],
                "graph_id": base.get("graph_id"),
                "node_id": base.get("node_id"),
                "tool": base.get("tool"),
                "kind": kind,
                "stage": base.get("stage"),
                "severity": severity,
                "text": (text or "")[:6000],
                "metrics_present": bool(metrics),
            }
        )

        evt = Event(
            event_id=eid,
            ts=ts,
            run_id=run_id,
            scope_id=scope_id,
            user_id=user_id,
            org_id=org_id,
            client_id=client_id,
            app_id=app_id,
            session_id=session_id,
            kind=kind,
            stage=base.get("stage"),
            text=text,
            tags=base.get("tags"),
            data=base.get("data"),
            metrics=metrics,
            graph_id=graph_id,
            node_id=node_id,
            tool=base.get("tool"),
            topic=base.get("topic"),
            severity=severity,
            signal=signal,
            inputs=base.get("inputs"),
            outputs=base.get("outputs"),
            embedding=base.get("embedding"),
            pii_flags=base.get("pii_flags"),
            version=2,
        )

        # 2) persist to HotLog + Persistence
        await self.hotlog.append(self.run_id, evt, ttl_s=self.hot_ttl_s, limit=self.hot_limit)
        await self.persistence.append_event(self.run_id, evt)

        # Metering hook
        try:
            meter = current_metering()
            await meter.record_event(
                scope=self.scope,
                scope_id=scope_id,
                kind=f"memory.{kind}",
            )
        except Exception:
            if self.logger:
                self.logger.exception("Error recording metering event in MemoryFacade.record_raw")
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
        """
        # 1) derive short preview text
        text: str | None = None
        if data is not None:
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

    async def write_result(
        self,
        *,
        tool: str | None = None,  # back compatibility with 'topic'
        inputs: list[dict[str, Any]] | None = None,
        outputs: list[dict[str, Any]] | None = None,
        tags: list[str] | None = None,
        metrics: dict[str, float] | None = None,
        message: str | None = None,
        severity: int = 3,
        topic: str | None = None,  # alias for tool, backwards compatibility
    ) -> Event:
        """
        Convenience for recording a “tool/agent/flow result” with typed I/O.

        `tool`    : tool/agent/flow identifier (also used by KVIndices.last_outputs_by_topic)
        `inputs`  : List[Value]-like dicts
        `outputs` : List[Value]-like dicts
        `tags`    : labels like ["rag","qa"] for filtering/search
        """
        if tool is None and topic is not None:
            tool = topic
        if tool is None:
            raise ValueError("write_result requires a 'tool' (or legacy 'topic') name")

        inputs = inputs or []
        outputs = outputs or []

        evt = await self.record_raw(
            base=dict(
                tool=tool,
                kind="tool_result",
                severity=severity,
                tags=tags or [],
                inputs=inputs,
                outputs=outputs,
            ),
            text=message,
            metrics=metrics,
        )
        await self.indices.update(self.run_id, evt)
        return evt

    async def write_tool_result(
        self,
        *,
        tool: str,
        inputs: list[dict[str, Any]] | None = None,
        outputs: list[dict[str, Any]] | None = None,
        tags: list[str] | None = None,
        metrics: dict[str, float] | None = None,
        message: str | None = None,
        severity: int = 3,
    ) -> Event:
        """
        Convenience wrapper around write_result() for tool results.
        """
        return await self.write_result(
            tool=tool,
            inputs=inputs,
            outputs=outputs,
            tags=tags,
            metrics=metrics,
            message=message,
            severity=severity,
        )

    # ---------- retrieval ----------
    async def recent(self, *, kinds: list[str] | None = None, limit: int = 50) -> list[Event]:
        """Return recent events from HotLog (most recent last), optionally filtered by kind."""
        return await self.hotlog.recent(self.run_id, kinds=kinds, limit=limit)

    async def recent_data(
        self,
        *,
        kinds: list[str] | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
    ) -> list[Any]:
        evts = await self.recent(kinds=kinds, limit=limit)
        if tags:
            want = set(tags)
            evts = [e for e in evts if want.issubset(set(e.tags or []))]

        out: list[Any] = []
        for e in evts:
            if e.data is not None:
                out.append(e.data)
            elif e.text:
                # last-resort: treat text as JSON if it looks like it, else raw string
                t = e.text.strip()
                if (t.startswith("{") and t.endswith("}")) or (
                    t.startswith("[") and t.endswith("]")
                ):
                    try:
                        out.append(json.loads(t))
                        continue
                    except Exception:
                        pass
                out.append(e.text)
        return out

    async def last_by_name(self, name: str):
        """Return the last output value by `name` from Indices (fast path)."""
        return await self.indices.last_by_name(self.run_id, name)

    async def last_output_by_name(self, name: str):
        """Return the last output value (Value.value) by `name` from Indices (fast path)."""
        out = await self.indices.last_by_name(self.run_id, name)
        if out is None:
            return None
        return out.get("value")  # type: ignore

    async def last_outputs_by_topic(self, topic: str):
        """Return the last output map for a given topic (tool/flow/agent) from Indices."""
        return await self.indices.last_outputs_by_topic(self.run_id, topic)

    # replace last_tool_result_outputs
    async def last_tool_result_outputs(self, tool: str) -> dict[str, Any] | None:
        """
        Convenience wrapper around KVIndices.last_outputs_by_topic for this run.
        Returns the last outputs map for a given tool, or None.
        """
        return await self.indices.last_outputs_by_topic(self.run_id, tool)

    async def recent_tool_results(
        self,
        *,
        tool: str | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
    ) -> list[Event]:
        """
        Return recent tool_result events from HotLog, optionally filtered by tool name and tags.
        """
        events = await self.recent(kinds=["tool_result"], limit=limit)
        if tool is not None:
            events = [e for e in events if e.tool == tool]
        if tags:
            want = set(tags)
            events = [e for e in events if want.issubset(set(e.tags or []))]
        return events

    async def latest_refs_by_kind(self, kind: str, *, limit: int = 50):
        """Return latest ref outputs by ref.kind (fast path, KV-backed)."""
        return await self.indices.latest_refs_by_kind(self.run_id, kind, limit=limit)

    async def search(
        self,
        *,
        query: str,
        kinds: list[str] | None = None,
        tags: list[str] | None = None,
        limit: int = 100,
        use_embedding: bool = True,
    ) -> list[Event]:
        """
        Search recent events by lexical matching and optional embedding similarity.
        - kinds: optional filter by event kinds
        - tags: optional filter by tags (AND semantics)
        - limit: max number of results to return
        - use_embedding: whether to use embedding-based ranking (requires LLM client)

        NOTE: This is an in-memory scan of recent events. No indexing is done yet.
        """
        events = await self.recent(kinds=kinds, limit=limit)
        if tags:
            want = set(tags)
            events = [e for e in events if want.issubset(set(e.tags or []))]

        query_l = query.lower()

        # 1) simple fallback: lexical
        lexical_hits = [e for e in events if (e.text or "").lower().find(query_l) >= 0]

        if not use_embedding:
            return lexical_hits or events

        raise NotImplementedError("Embedding-based search not implemented yet")

        # 2) optional: embedding-based ranking (if you embed query + have e.embedding) [stub]
        if not (self.llm and any(e.embedding for e in events)):
            return lexical_hits or events

        q_emb = await self.llm.embed(query)  # TODO: adapt to LLMClientProtocol

        # compute cosine similarity in Python for now
        def sim(e: Event) -> float:
            if not e.embedding:
                return -1.0
            # naive dot product
            return sum(a * b for a, b in zip(q_emb, e.embedding, strict=False))

        scored = sorted(events, key=sim, reverse=True)
        return scored[:limit]

    # ---------- distillation (plug strategies) ----------

    # ---------- distillation helpers ----------
    async def distill_long_term(
        self,
        scope_id: str | None = None,
        *,
        summary_tag: str = "session",
        summary_kind: str = "long_term_summary",
        include_kinds: list[str] | None = None,
        include_tags: list[str] | None = None,
        max_events: int = 200,
        min_signal: float | None = None,
        use_llm: bool = False,
    ) -> dict[str, Any]:
        """
        Run the generic LongTermSummarizer over this run's memory and persist a summary.

        Returns a descriptor like:
          {
            "uri": "file://mem/<run_id>/summaries/<tag>/<ts>.json",
            "summary_kind": "...",
            "summary_tag": "...",
            "time_window": {...},
            "num_events": N,
          }

        This is suitable for:
          - soft re-hydration (load summary into a new run),
          - RAG promotion,
          - or analytics.
        """
        scope_id = scope_id or self.memory_scope_id  # order of precedence
        if use_llm:
            if not self.llm:
                raise RuntimeError("LLM client not configured in MemoryFacade for LLM distillation")
            from aethergraph.services.memory.distillers.llm_long_term import LLMLongTermSummarizer

            d = LLMLongTermSummarizer(
                llm=self.llm,
                summary_kind=summary_kind,
                summary_tag=summary_tag,
                include_kinds=include_kinds,
                include_tags=include_tags,
                max_events=max_events,
                min_signal=min_signal if min_signal is not None else self.default_signal_threshold,
            )
            return await d.distill(
                self.run_id,
                scope_id=scope_id or self.memory_scope_id,
                hotlog=self.hotlog,
                persistence=self.persistence,
                indices=self.indices,
                docs=self.docs,
            )

        from aethergraph.services.memory.distillers.long_term import LongTermSummarizer

        # non-LLM path -- structured digest
        d = LongTermSummarizer(
            summary_kind=summary_kind,
            summary_tag=summary_tag,
            include_kinds=include_kinds,
            include_tags=include_tags,
            max_events=max_events,
            min_signal=min_signal if min_signal is not None else self.default_signal_threshold,
        )
        return await d.distill(
            self.run_id,
            scope_id=scope_id or self.memory_scope_id,
            hotlog=self.hotlog,
            persistence=self.persistence,
            indices=self.indices,
            docs=self.docs,
        )

    async def distill_meta_summary(
        self,
        scope_id: str | None = None,
        *,
        source_kind: str = "long_term_summary",
        source_tag: str = "session",
        summary_kind: str = "meta_summary",
        summary_tag: str = "meta",
        max_summaries: int = 20,
        min_signal: float | None = None,
        use_llm: bool = True,
    ) -> dict[str, Any]:
        """
        Run an LLM-based meta summarizer over existing summary events.

        Typical usage:
          - source_kind="long_term_summary", source_tag="session"
          - summary_kind="meta_summary",    summary_tag="weekly" or "meta"

        Returns a descriptor like:
          {
            "uri": "file://mem/<scope_id>/summaries/<summary_tag>/<ts>.json",
            "summary_kind": "...",
            "summary_tag": "...",
            "time_window": {...},
            "num_source_summaries": N,
          }
        """
        scope_id = scope_id or self.memory_scope_id  # order of precedence

        if not use_llm:
            # Placeholder for a future non-LLM meta summarizer if desired.
            raise NotImplementedError("Non-LLM meta summarization is not implemented yet")

        if not self.llm:
            raise RuntimeError("LLM client not configured in MemoryFacade for meta distillation")

        from aethergraph.services.memory.distillers.llm_meta_summary import (
            LLMMetaSummaryDistiller,
        )

        d = LLMMetaSummaryDistiller(
            llm=self.llm,
            source_kind=source_kind,
            source_tag=source_tag,
            summary_kind=summary_kind,
            summary_tag=summary_tag,
            max_summaries=max_summaries,
            min_signal=min_signal if min_signal is not None else self.default_signal_threshold,
        )
        return await d.distill(
            self.run_id,
            scope_id=scope_id or self.memory_scope_id,
            hotlog=self.hotlog,
            persistence=self.persistence,
            indices=self.indices,
            docs=self.docs,
        )

    # ---------- RAG facade ----------
    async def rag_upsert(
        self, *, corpus_id: str, docs: Sequence[dict[str, Any]], topic: str | None = None
    ) -> dict[str, Any]:
        """Upsert documents into RAG corpus via RAG facade, if configured."""
        if not self.rag:
            raise RuntimeError("RAG facade not configured in MemoryFacade")
        stats = await self.rag.upsert_docs(corpus_id=corpus_id, docs=list(docs))
        # Optional write result -- disable for now
        # self.write_result(
        #     topic=topic or f"rag.upsert.{corpus_id}",
        #     outputs=[{"name": "stats", "kind": "json", "value": stats}],
        #     tags=["rag", "ingest"],
        #     message=f"Upserted {stats.get('chunks',0)}  chunks into {corpus_id}"
        # )
        return stats

    # ---------- helpers ----------
    def _estimate_signal(
        self, *, text: str | None, metrics: dict[str, Any] | None, severity: int
    ) -> float:
        """
        Cheap heuristic to gauge “signal” of an event (0.0–1.0).
        - Rewards presence/length of text and presence of metrics.
        - Used as a noise gate in rolling summaries; can be overridden by caller.
        """
        score = 0.15 + 0.1 * severity
        if text:
            score += min(len(text) / 400.0, 0.4)
        if metrics:
            score += 0.2
        return max(0.0, min(1.0, score))

    def resolve(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Synchronous version of parameter resolution (for sync contexts).
        See `aethergraph.services.memory.resolver.resolve_params` for details.
        """
        from aethergraph.services.memory.resolver import ResolverContext, resolve_params

        rctx = ResolverContext(mem=self)
        return resolve_params(params, rctx)

    # ----------- RAG: corpus binding & status -----------
    async def rag_bind(
        self,
        *,
        corpus_id: str | None = None,
        key: str | None = None,
        create_if_missing: bool = True,
        labels: dict | None = None,
    ) -> str:
        if not self.rag:
            raise RuntimeError("RAG facade not configured")

        mem_scope = self.memory_scope_id  # derived from Scope
        # dims = self.scope.metering_dimensions() if self.scope else {}

        if corpus_id:
            cid = corpus_id
        else:
            logical_key = key or "default"
            base = f"{mem_scope}:{logical_key}"
            cid = f"mem:{_slug(mem_scope)}:{_slug(logical_key)}-{_short_hash(base, 8)}"

        scope_labels = {}
        if self.scope:
            scope_labels = self.scope.rag_labels(scope_id=mem_scope)

        meta = {"scope": scope_labels, **(labels or {})}
        if create_if_missing:
            await self.rag.add_corpus(cid, meta=meta, scope_labels=scope_labels)
        return cid

    async def rag_status(self, *, corpus_id: str) -> dict:
        """Quick stats about a corpus."""
        if not self.rag:
            raise RuntimeError("RAG facade not configured in MemoryFacade")
        # lightweight: count docs/chunks by scanning the jsonl (fast enough for now)
        return await self.rag.stats(corpus_id)

    async def rag_snapshot(self, *, corpus_id: str, title: str, labels: dict | None = None) -> dict:
        """Export corpus into an artifact bundle and return its URI."""
        if not self.rag:
            raise RuntimeError("RAG facade not configured in MemoryFacade")
        bundle = await self.rag.export(corpus_id)
        # Optionally log a tool_result
        await self.write_result(
            tool=f"rag.snapshot.{corpus_id}",
            outputs=[{"name": "bundle_uri", "kind": "uri", "value": bundle.get("uri")}],
            tags=["rag", "snapshot"],
            message=title,
            severity=2,
        )
        return bundle

    async def rag_compact(self, *, corpus_id: str, policy: dict | None = None) -> dict:
        """
        Simple compaction policy:
        - Optionally drop docs by label or min_score
        - Optional re-embed with a new model
        For now we just expose reembed() plumbing and a placeholder for pruning.

        NOTE: this function is a placeholder for future compaction strategies.
        """
        if not self.rag:
            raise RuntimeError("RAG facade not configured in MemoryFacade")
        policy = policy or {}
        model = policy.get("reembed_model")
        pruned = 0  # placeholder
        if model:
            await self.rag.reembed(corpus_id, model=model)
        return {"pruned_docs": pruned, "reembedded": bool(model)}

    # ----------- RAG: event → doc promotion -----------
    async def rag_promote_events(
        self,
        *,
        corpus_id: str,
        events: list[Event] | None = None,
        where: dict | None = None,
        policy: dict | None = None,
    ) -> dict:
        """
        Convert events to documents and upsert.
        where: optional filter like {"kinds": ["tool_result"], "min_signal": 0.25, "limit": 200}
        policy: {"min_signal": float} In the future may support more (chunksize, overlap, etc.)
        """
        if not self.rag:
            raise RuntimeError("RAG facade not configured in MemoryFacade")
        policy = policy or {}
        min_signal = policy.get("min_signal", self.default_signal_threshold)

        # Select events if not provided
        if events is None:
            kinds = (where or {}).get("kinds")
            limit = int((where or {}).get("limit", 200))
            recent = await self.recent(kinds=kinds, limit=limit)
            events = [e for e in recent if (getattr(e, "signal", 0.0) or 0.0) >= float(min_signal)]

        docs: list[dict] = []
        for e in events:
            title = f"{e.kind}:{(e.tool or e.stage or 'n/a')}:{e.ts}"
            scope_labels = (
                self.scope.rag_labels(scope_id=self.memory_scope_id) if self.scope else {}
            )
            labels = {
                **scope_labels,
                "kind": e.kind,
                "tool": e.tool,
                "stage": e.stage,
                "severity": e.severity,
                "run_id": e.run_id,
                "graph_id": e.graph_id,
                "node_id": e.node_id,
                "scope_id": e.scope_id,
                "tags": list(e.tags or []),
            }
            body = e.text
            if not body:
                # Fallback to compact JSON of I/O + metrics
                body = json.dumps(
                    {"inputs": e.inputs, "outputs": e.outputs, "metrics": e.metrics},
                    ensure_ascii=False,
                )
            docs.append({"text": body, "title": title, "labels": labels})

        if not docs:
            return {
                "added": 0,
                "chunks": 0,
                "index": getattr(self.rag.index, "__class__", type("X", (object,), {})).__name__,
            }

        stats = await self.rag.upsert_docs(corpus_id=corpus_id, docs=docs)
        # (Optional) write a result for traceability
        await self.write_result(
            tool=f"rag.promote.{corpus_id}",
            outputs=[
                {"name": "added_docs", "kind": "number", "value": stats.get("added", 0)},
                {"name": "chunks", "kind": "number", "value": stats.get("chunks", 0)},
            ],
            tags=["rag", "ingest"],
            message=f"Promoted {stats.get('added', 0)} events into {corpus_id}",
            severity=2,
        )
        return stats

    # ----------- RAG: search & answer -----------
    async def rag_search(
        self,
        *,
        corpus_id: str,
        query: str,
        k: int = 8,
        filters: dict | None = None,
        mode: Literal["hybrid", "dense"] = "hybrid",
    ) -> list[dict]:
        """Thin pass-through, but returns serializable dicts."""
        if not self.rag:
            raise RuntimeError("RAG facade not configured in MemoryFacade")

        scope = self.scope
        s_filters = scope.rag_filter(scope_id=self.memory_scope_id) if scope else {}
        if filters:
            s_filters.update(filters)
        hits = await self.rag.search(corpus_id, query, k=k, filters=s_filters, mode=mode)
        return [
            dict(
                chunk_id=h.chunk_id,
                doc_id=h.doc_id,
                corpus_id=h.corpus_id,
                score=h.score,
                text=h.text,
                meta=h.meta,
            )
            for h in hits
        ]

    async def rag_answer(
        self,
        *,
        corpus_id: str,
        question: str,
        style: Literal["concise", "detailed"] = "concise",
        with_citations: bool = True,
        k: int = 6,
    ) -> dict:
        """Answer with citations, then log as a tool_result."""
        if not self.rag:
            raise RuntimeError("RAG facade not configured in MemoryFacade")
        ans = await self.rag.answer(
            corpus_id=corpus_id,
            question=question,
            llm=self.llm,
            style=style,
            with_citations=with_citations,
            k=k,
        )
        # Flatten citations into outputs for indices
        outs = [{"name": "answer", "kind": "text", "value": ans.get("answer", "")}]
        for i, rc in enumerate(ans.get("resolved_citations", []), start=1):
            outs.append({"name": f"cite_{i}", "kind": "json", "value": rc})
        await self.write_result(
            tool=f"rag.answer.{corpus_id}",
            outputs=outs,
            tags=["rag", "qa"],
            message=f"Q: {question}",
            metrics=ans.get("usage", {}),
            severity=2,
        )
        return ans

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

        await self.hotlog.append(self.run_id, evt, ttl_s=self.hot_ttl_s, limit=self.hot_limit)
        await self.persistence.append_event(self.run_id, evt)
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
