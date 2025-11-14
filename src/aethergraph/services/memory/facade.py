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
from aethergraph.services.artifacts.fs_store import FileArtifactStoreSync
from aethergraph.services.rag.facade import RAGFacade

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
    """
    MemoryFacade = “session memory” front-door bound to a specific runtime scope.

    What it is:
    -----------
    A small, async façade that coordinates three *core* memory components:
      • HotLog        — fast, transient ring-buffer of recent events (in KV/Redis/etc.)
      • Persistence   — durable append/replay of events & JSON blobs (e.g., FS JSONL, S3, DB)
      • Indices       — small KV-based derived views for fast lookups (latest by name/ref kind/topic)

    Optionally:
      • artifact_store — for content-addressed, immutable artifacts (large files/dirs, bundles).
                         Not required for core memory; used by distillers/tools when you want CAS URIs.

    Why this split:
    ---------------
      • HotLog:   low latency read/write for “what just happened?”, used by routers/LLM context builders.
      • Persistence: durable, append-only event log + JSON blobs (summaries, episodes) for replay/analytics.
      • Indices:  derived KV tables to avoid scanning logs for common “last X” queries.
      • Artifacts: big assets (images, datasets, reports) that benefit from CAS, pinning, and reuse.

    Binding / Scope:
    ----------------
    A MemoryFacade instance is bound to a scope via:
      run_id, graph_id, node_id, agent_id
    Typically constructed by a MemoryFactory at run/node creation, so tools/agents can just call:
      await ctx.services.memory.record_raw(...)
      await ctx.services.memory.write_result(...)

    Concurrency:
    ------------
    All public methods are async; implementations of HotLog/Persistence/Indices SHOULD be non-blocking
    (use asyncio primitives or run blocking IO via asyncio.to_thread).

    Configuration knobs:
    --------------------
      • hot_limit:   max events kept in HotLog per session (ring buffer).
      • hot_ttl_s:   TTL for HotLog entries (e.g., 7 days).
      • default_signal_threshold: heuristic floor for “signal” scoring in rolling summaries, etc.

    Typical flow:
    -------------
      1) `record_raw(...)` appends an Event to HotLog (fast) and to Persistence JSONL (durable).
      2) `write_result(...)` is a typed helper for tool/agent outputs; also updates Indices.
      3) `recent(...)`, `last_by_name(...)`, `latest_refs_by_kind(...)` read from HotLog/Indices.
      4) Distillers (rolling / episode) pull from HotLog & Persistence to synthesize summaries,
         then write back via Persistence (JSON) and/or ArtifactStore (CAS) if configured.

    Extensibility:
    --------------
      • Add more distillers (RAG digests, long-term memory compaction).
      • Add helpers to save content-addressed artifacts (e.g., `save_summary_as_artifact`).
      • Swap backends by providing different implementations of the protocols.
    """

    def __init__(
        self,
        *,
        run_id: str,
        graph_id: str | None,
        node_id: str | None,
        agent_id: str | None,
        hotlog: HotLog,
        persistence: Persistence,
        indices: Indices,
        artifact_store: FileArtifactStoreSync,
        hot_limit: int = 1000,
        hot_ttl_s: int = 7 * 24 * 3600,
        default_signal_threshold: float = 0.25,
        logger=None,
        rag: RAGFacade | None = None,
        llm: LLMClientProtocol | None = None,
    ):
        self.run_id = run_id
        self.graph_id = graph_id
        self.node_id = node_id
        self.agent_id = agent_id
        self.hotlog = hotlog
        self.persistence = persistence
        self.indices = indices
        self.artifacts = artifact_store
        self.hot_limit = hot_limit
        self.hot_ttl_s = hot_ttl_s
        self.default_signal_threshold = default_signal_threshold
        self.logger = logger
        self.rag = rag
        self.llm = llm  # optional LLM service for RAG answering, etc.

    # ---------- recording ----------
    async def record_raw(
        self,
        *,
        base: dict[str, Any],
        text: str | None = None,
        metrics: dict[str, Any] | None = None,
        sources: list[str] | None = None,
    ) -> Event:
        """
        Append a normalized event to HotLog (fast) and Persistence (durable).

        - `base` carries identity + classification:
            { kind, stage, severity, tool, tags, entities, inputs, outputs, ... }
          The façade stamps missing scope with  (run_id, graph_id, node_id, agent_id).
        - `text`  : optional human-readable note/message
        - `metrics`: optional numeric map (latency, token counts, costs, etc.)
        - `sources`: optional list of event_ids this event summarizes/derives from

        Returns the Event (with stable event_id and computed `signal`).

        Notes:
        - We compute a lightweight “signal” score if caller didn’t set one.
        - We DO NOT update `indices` here automatically; only `write_result(...)` does that,
          because indices are tuned for typed outputs (Value[]). You can call `indices.update`
          yourself if you need to index from a raw event.
        """
        ts = now_iso()
        base.setdefault("run_id", self.run_id)
        base.setdefault("graph_id", self.graph_id)
        base.setdefault("node_id", self.node_id)
        base.setdefault("agent_id", self.agent_id)
        severity = int(base.get("severity", 2))
        signal = base.get("signal")
        if signal is None:
            signal = self._estimate_signal(text=text, metrics=metrics, severity=severity)

        eid = stable_event_id(
            {
                "ts": ts,
                "run_id": base["run_id"],
                "graph_id": base.get("graph_id"),
                "node_id": base.get("node_id"),
                "agent_id": base.get("agent_id"),
                "tool": base.get("tool"),
                "kind": base.get("kind"),
                "stage": base.get("stage"),
                "severity": severity,
                "text": (text or "")[:6000],
                "metrics_present": bool(metrics),
                "sources": sources or [],
            }
        )

        evt = Event(event_id=eid, ts=ts, text=text, metrics=metrics, signal=signal, **base)
        await self.hotlog.append(self.run_id, evt, ttl_s=self.hot_ttl_s, limit=self.hot_limit)
        await self.persistence.append_event(self.run_id, evt)

        # cheap per-kind index (kv_index_key) – optional to keep:
        # await kv.list_append_unique(f"mem:{self.run_id}:idx:{base.get('kind','misc')}", [{"id": eid}], id_key="id", ttl_s=self.hot_ttl_s)

        return evt

    async def record(
        self,
        kind,
        data,
        tags=None,
        entities=None,
        severity=2,
        stage=None,
        inputs_ref=None,
        outputs_ref=None,
        metrics=None,
        sources=None,
        signal=None,
    ) -> Event:
        """
        Convenience wrapper around record_raw() with common fields.

        Parameters:
        - kind       : event kind (e.g., "user_msg", "tool_call", etc.)
        - data       : json-serializable text content (will be stringified)
        - tags       : optional list of string tags
        - entities   : optional list of entity IDs
        - severity   : integer severity (1=low ... 5=high)
        - stage      : optional stage label (e.g., "observe", "act", etc.)
        - inputs_ref : optional typed input references (e.g., List[Value] dicts)
        - outputs_ref: optional typed output references (e.g., List[Value] dicts)
        - metrics    : optional numeric map (latency, token counts, costs, etc.)
        - sources    : optional list of event_ids this event summarizes/derives from
        - signal     : optional float signal score (0.0–1.0); if None, computed heuristically

        Returns the Event.
        """
        # if data is not a json-serializable string, log warning and log as json string
        text = None
        if data is not None:
            if isinstance(data, str):
                text = data
            else:
                try:
                    text = json.dumps(data, ensure_ascii=False)
                except Exception as e:
                    text = f"<unserializable data: {e!s}>"
                    if self.logger:
                        self.logger.warning(text)
        base = dict(
            kind=kind,
            stage=stage,
            severity=severity,
            tags=tags or [],
            entities=entities or [],
            inputs=inputs_ref,
            outputs=outputs_ref,
        )
        return await self.record_raw(base=base, text=text, metrics=metrics, sources=sources)

    async def write_result(
        self,
        *,
        topic: str,
        inputs: list[dict[str, Any]] | None = None,
        outputs: list[dict[str, Any]] | None = None,
        tags: list[str] | None = None,
        metrics: dict[str, float] | None = None,
        message: str | None = None,
        severity: int = 3,
    ) -> Event:
        """
        Convenience for recording a “tool/agent/flow result” with typed I/O.

        Why this exists:
        - Creates a normalized `tool_result` event.
        - Updates `indices` with latest-by-name, latest-ref-by-kind, and last outputs-by-topic.
        - Keeps raw event appends (HotLog + Persistence) consistent.

        `topic`   : tool/agent/flow identifier (used by indices.last_outputs_by_topic)
        `inputs`  : List[Value]
        `outputs` : List[Value]  <-- indices derive from these
        """
        inputs = inputs or []
        outputs = outputs or []
        evt = await self.record_raw(
            base=dict(
                tool=topic,
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

    # ---------- retrieval ----------
    async def recent(self, *, kinds: list[str] | None = None, limit: int = 50) -> list[Event]:
        """Return recent events from HotLog (most recent last), optionally filtered by kind."""
        return await self.hotlog.recent(self.run_id, kinds=kinds, limit=limit)

    async def recent_data(self, *, kinds: list[str], limit: int = 50) -> list[Any]:
        """
        Convenience wrapper around `recent()` that returns decoded `data`
        instead of raw Event objects.

        Works with the same JSON-in-text convention as `record()`.
        """
        evts = await self.recent(kinds=kinds, limit=limit)
        out = []
        for evt in evts:
            if not evt.text:
                continue
            try:
                out.append(json.loads(evt.text))
            except Exception:
                out.append(evt.text)
        return out

    async def last_by_name(self, name: str):
        """Return the last output value by `name` from Indices (fast path)."""
        return await self.indices.last_by_name(self.run_id, name)

    async def latest_refs_by_kind(self, kind: str, *, limit: int = 50):
        """Return latest ref outputs by ref.kind (fast path, KV-backed)."""
        return await self.indices.latest_refs_by_kind(self.run_id, kind, limit=limit)

    async def last_outputs_by_topic(self, topic: str):
        """Return the last output map for a given topic (tool/flow/agent) from Indices."""
        return await self.indices.last_outputs_by_topic(self.run_id, topic)

    # alias for easy readability for users
    async def get_last_value(self, name: str):
        """Alias for last_by_name()."""
        return await self.last_by_name(name)

    async def get_latest_values_by_kind(self, kind: str, *, limit: int = 50):
        """Alias for latest_refs_by_kind()."""
        return await self.latest_refs_by_kind(kind, limit=limit)

    async def get_last_outputs_for_topic(self, topic: str):
        """Alias for last_outputs_by_topic()."""
        return await self.last_outputs_by_topic(topic)

    # ---------- distillation (plug strategies) ----------
    async def distill_rolling_chat(
        self,
        *,
        max_turns: int = 20,
        min_signal: float | None = None,
        turn_kinds: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Build a rolling chat summary from recent user/assistant turns.
        - Reads from HotLog; may emit a JSON summary via Persistence.
        - Uses `default_signal_threshold` unless overridden.
        - Returns a small descriptor (e.g., { "uri": ..., "sources": [...], ... }).

        For turn_kinds, default to ["user_msg","assistant_msg"] if not provided.
        """
        from aethergraph.services.memory.distillers.rolling import RollingSummarizer

        d = RollingSummarizer(
            max_turns=max_turns,
            min_signal=min_signal or self.default_signal_threshold,
            turn_kinds=turn_kinds,
        )
        return await d.distill(
            self.run_id, hotlog=self.hotlog, persistence=self.persistence, indices=self.indices
        )

    async def distill_episode(
        self, *, tool: str, run_id: str, include_metrics: bool = True
    ) -> dict[str, Any]:
        """
        Summarize a tool/agent episode (all events for a given run_id+tool).
        - Reads from HotLog/Persistence, writes back a summary JSON (and optionally CAS bundle).
        - Returns descriptor (e.g., { "uri": ..., "sources": [...], "metrics": {...} }).
        """
        from aethergraph.services.memory.distillers.episode import EpisodeSummarizer

        d = EpisodeSummarizer(
            include_metrics=include_metrics,
        )
        return await d.distill(
            self.run_id,
            hotlog=self.hotlog,
            persistence=self.persistence,
            indices=self.indices,
            tool=tool,
            run_id=run_id,
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

        if corpus_id:
            if create_if_missing:
                await self.rag.add_corpus(corpus_id, meta=labels or {})
            return corpus_id

        # prefer explicit key; else stable from run_id
        chosen = key or self.run_id
        cid = f"run:{_slug(chosen)}-{_short_hash(chosen, 12)}"
        if create_if_missing:
            await self.rag.add_corpus(cid, meta=labels or {})
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
            topic=f"rag.snapshot.{corpus_id}",
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
            labels = {
                "kind": e.kind,
                "tool": e.tool,
                "stage": e.stage,
                "severity": e.severity,
                "run_id": e.run_id,
                "graph_id": e.graph_id,
                "node_id": e.node_id,
                "agent_id": e.agent_id,
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
            topic=f"rag.promote.{corpus_id}",
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
        hits = await self.rag.search(corpus_id, query, k=k, filters=filters, mode=mode)
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
            topic=f"rag.answer.{corpus_id}",
            outputs=outs,
            tags=["rag", "qa"],
            message=f"Q: {question}",
            metrics=ans.get("usage", {}),
            severity=2,
        )
        return ans
