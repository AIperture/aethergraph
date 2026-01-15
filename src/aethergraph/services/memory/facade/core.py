from __future__ import annotations

import json
import logging
import time
from typing import Any

from aethergraph.contracts.services.llm import LLMClientProtocol
from aethergraph.contracts.services.memory import Event, HotLog, Indices, Persistence
from aethergraph.contracts.storage.artifact_store import AsyncArtifactStore
from aethergraph.contracts.storage.doc_store import DocStore
from aethergraph.core.runtime.runtime_metering import current_metering
from aethergraph.services.indices.scoped_indices import ScopedIndices
from aethergraph.services.rag.facade import RAGFacade
from aethergraph.services.scope.scope import Scope
from aethergraph.storage.vector_index.utils import build_index_meta_from_scope

from .chat import ChatMixin
from .distillation import DistillationMixin
from .rag import RAGMixin
from .results import ResultMixin
from .retrieval import RetrievalMixin
from .utils import now_iso, stable_event_id


def derive_timeline_id(
    *,
    memory_scope_id: str | None,
    run_id: str,
    org_id: str | None = None,
    sep: str = "|",
) -> str:
    """
    Derive the storage partition key for timeline events.

    - If org_id is present, prefix with `org:{org_id}|...` to prevent cross-org mixing.
    - Keep fallback behavior: if memory_scope_id is missing, fall back to run_id.
    - Avoid redundant `org:{org_id}|org:{org_id}` when the bucket is already org-scoped.
    """
    bucket = (memory_scope_id or "").strip()
    if not bucket:
        bucket = run_id

    if org_id:
        org_prefix = f"org:{org_id}"

        # If the bucket is already exactly org-scoped, don't double-prefix
        if bucket == org_prefix:
            return org_prefix

        return f"{org_prefix}{sep}{bucket}"

    # No org context -> behave like current local mode
    return bucket


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
        mem_indices: Indices,
        scoped_indices: ScopedIndices | None = None,
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
        self.indices = mem_indices
        self.scoped_indices = scoped_indices
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
        self.timeline_id = derive_timeline_id(
            memory_scope_id=self.memory_scope_id,
            run_id=self.run_id,
            org_id=self.scope.org_id if self.scope else None,
        )

    async def record_raw(
        self,
        *,
        base: dict[str, Any],
        text: str | None = None,
        metrics: dict[str, float] | None = None,
    ) -> Event:
        """
        Record an unstructured event with optional preview text and metrics.

        This method generates a stable event ID, populates standard fields
        (e.g., `run_id`, `scope_id`, `severity`, `signal`), and appends the
        event to both the HotLog and Persistence layers. Additionally, it
        records a metering event for tracking purposes.

        Examples:
            Basic usage with minimal fields:
            ```python
            await context.memory().record_raw(
                base={"kind": "user_action", "severity": 2},
                text="User clicked a button."
            )
            ```

            Including metrics and additional fields:
            ```python
            await context.memory().record_raw(
                base={"kind": "tool_call", "stage": "execution", "severity": 3},
                text="Tool executed successfully.",
                metrics={"latency": 0.123, "tokens_used": 45}
            )
            ```

        Args:
            base: A dictionary containing event fields such as `kind`, `stage`,
                `data`, `tags`, `severity`, etc.
            text: Optional preview text for the event. If None, it is derived
                from the `data` field in `base`.
            metrics: Optional dictionary of numeric metrics (e.g., latency,
                token usage) to include in the event.

        Returns:
            Event: The fully constructed and persisted `Event` object.
        """
        ts_iso = now_iso()
        ts_num = time.time()  # numeric timestamp for created_at_ts

        # Merge Scope dimensions
        dims: dict[str, str] = {}
        if self.scope is not None:
            dims = self.scope.metering_dimensions()

        run_id = base.get("run_id") or dims.get("run_id") or self.run_id
        session_id = base.get("session_id") or dims.get("session_id") or self.session_id
        scope_id = base.get("scope_id") or self.memory_scope_id or session_id or run_id

        user_id = base.get("user_id") or dims.get("user_id")
        org_id = base.get("org_id") or dims.get("org_id")
        client_id = base.get("client_id") or dims.get("client_id")
        graph_id = base.get("graph_id") or dims.get("graph_id") or self.graph_id
        node_id = base.get("node_id") or dims.get("node_id") or self.node_id

        base.setdefault("run_id", run_id)
        base.setdefault("scope_id", scope_id)
        base.setdefault("session_id", session_id)
        base.setdefault("run_id", run_id)
        base.setdefault("graph_id", graph_id)
        base.setdefault("node_id", node_id)
        base.setdefault("scope_id", scope_id)
        base.setdefault("user_id", user_id)
        base.setdefault("org_id", org_id)
        base.setdefault("client_id", client_id)
        base.setdefault("session_id", session_id)
        severity = int(base.get("severity", 2))

        signal = base.get("signal")
        if signal is None:
            signal = self._estimate_signal(text=text, metrics=metrics, severity=severity)

        kind = base.get("kind") or "misc"

        eid = stable_event_id(
            {
                "ts": ts_iso,
                "run_id": base["run_id"],
                "kind": kind,
                "text": (text or "")[:6000],
                "tool": base.get("tool"),
            }
        )

        evt = Event(
            event_id=eid,
            ts=ts_iso,
            run_id=run_id,
            scope_id=scope_id,
            user_id=user_id,
            org_id=org_id,
            client_id=client_id,
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

        await self.hotlog.append(self.timeline_id, evt, ttl_s=self.hot_ttl_s, limit=self.hot_limit)
        await self.persistence.append_event(self.timeline_id, evt)

        # wire memory event text into ScopedIndices for searchability
        if self.scoped_indices is not None and self.scoped_indices.backend is not None:
            try:
                kind_val = getattr(evt.kind, "value", str(evt.kind))
                preview = (text or "")[:500] if text else ""

                extra_meta = {
                    "run_id": evt.run_id,
                    "scope_id": evt.scope_id,
                    "session_id": evt.session_id,
                    "graph_id": evt.graph_id,
                    "node_id": evt.node_id,
                    "stage": evt.stage,
                    "tags": evt.tags or [],
                    "severity": evt.severity,
                    "signal": evt.signal,
                    "tool": evt.tool,
                    "topic": evt.topic,
                    "preview": preview,  # short text preview
                    "timeline_id": self.timeline_id,
                }

                meta = build_index_meta_from_scope(
                    kind=kind_val,
                    source="memory",
                    ts=ts_iso,
                    created_at_ts=ts_num,
                    extra=extra_meta,
                )

                await self.scoped_indices.upsert(
                    corpus="event",
                    item_id=evt.event_id,
                    text=evt.text or "",
                    metadata=meta,
                )

            except Exception:
                if self.logger:
                    self.logger.exception("Error indexing memory event %s", evt.event_id)

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
        Record an event with common fields.

        This method standardizes event creation by populating fields such as
        `kind`, `severity`, `tags`, and `metrics`. It also supports optional
        references for inputs and outputs, and allows for signal strength
        overrides.

        Examples:
            Basic usage for a user action:
            ```python
            await context.memory().record(
                kind="user_action",
                data={"action": "clicked_button"},
                tags=["ui", "interaction"]
            )
            ```

            Recording a tool execution with metrics:
            ```python
            await context.memory().record(
                kind="tool_call",
                data={"tool": "search", "query": "weather"},
                metrics={"latency": 0.123, "tokens_used": 45},
                severity=3
            )
            ```

        Args:
            kind: Logical kind of event (e.g., `"user_msg"`, `"tool_call"`, `"chat_turn"`).
            data: JSON-serializable content or string providing event details.
            tags: A list of string labels for categorization. Defaults to None.
            severity: An integer (1-3) indicating importance. Defaults to 2.
            stage: Optional stage of the event (e.g., `"user"`, `"assistant"`, `"system"`). Defaults to None.
            inputs_ref: Optional references for input values. Defaults to None.
            outputs_ref: Optional references for output values. Defaults to None.
            metrics: A dictionary of numeric metrics (e.g., latency, token usage). Defaults to None.
            signal: Manual override for the signal strength (0.0 to 1.0). If None, it is calculated heuristically.
            text: Optional preview text override. If None, it is derived from `data`.

        Returns:
            Event: The fully constructed and persisted `Event` object.

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
        Assemble memory context for prompts, including long-term summaries,
        recent chat history, and recent tool usage.

        Examples:
            Build prompt segments with default settings:
            ```python
            segments = await context.memory().build_prompt_segments()
            ```

            Include recent tool usage and filter by a specific tool:
            ```python
            segments = await context.memory().build_prompt_segments(
                include_recent_tools=True,
                tool="search",
                tool_limit=5
            )
            ```

        Args:
            recent_chat_limit: The maximum number of recent chat messages to include.
                Defaults to 12.
            include_long_term: Whether to include long-term memory summaries.
                Defaults to True.
            summary_tag: The tag used to filter long-term summaries.
                Defaults to "session".
            max_summaries: The maximum number of long-term summaries to include.
                Defaults to 3.
            include_recent_tools: Whether to include recent tool usage.
                Defaults to False.
            tool: The specific tool to filter recent tool usage.
                Defaults to None.
            tool_limit: The maximum number of recent tool events to include.
                Defaults to 10.

        Returns:
            dict[str, Any]: A dictionary containing the following keys:

                - "long_term" (str): Combined long-term summary text or an empty
                  string if not included.

                - "recent_chat" (list[dict[str, Any]]): A list of recent chat
                  messages, each represented as a dictionary with the following keys:
                    - "ts" (str): Timestamp of the message.
                    - "role" (str): Role of the sender (e.g., "user", "assistant").
                    - "text" (str): The content of the message.
                    - "tags" (list[str]): Tags associated with the message.

                - "recent_tools" (list[dict[str, Any]]): A list of recent tool
                  usage events, each represented as a dictionary with the following keys:
                    - "ts" (str): Timestamp of the tool event.
                    - "tool" (str): Name of the tool used.
                    - "message" (str): Message or description of the tool event.
                    - "inputs" (Any): Inputs provided to the tool.
                    - "outputs" (Any): Outputs generated by the tool.
                    - "tags" (list[str]): Tags associated with the tool event.
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
