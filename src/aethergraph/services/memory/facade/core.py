from __future__ import annotations

import logging
from typing import Any

from aethergraph.contracts.services.llm import LLMClientProtocol
from aethergraph.contracts.services.memory import HotLog, Persistence
from aethergraph.contracts.storage.artifact_store import AsyncArtifactStore
from aethergraph.services.indices.scoped_indices import ScopedIndices
from aethergraph.services.memory.facade.deprecated import DeprecatedMixin
from aethergraph.services.memory.facade.introspection import IntrospectionMixin
from aethergraph.services.memory.facade.normalization import EventNormalizationMixin
from aethergraph.services.memory.facade.prompt import PromptMixin
from aethergraph.services.memory.facade.read import ReadMixin
from aethergraph.services.memory.facade.summary import SummaryMixin
from aethergraph.services.memory.facade.write import WriteMixin
from aethergraph.services.scope.scope import Scope
from aethergraph.services.tracing import resolve_tracer


def derive_timeline_id(
    *,
    memory_scope_id: str | None,
    run_id: str,
    org_id: str | None = None,
    sep: str = "|",
) -> str:
    bucket = (memory_scope_id or "").strip()
    if not bucket:
        bucket = run_id
    if org_id:
        org_prefix = f"org:{org_id}"
        if bucket == org_prefix:
            return org_prefix
        return f"{org_prefix}{sep}{bucket}"
    return bucket


class MemoryFacade(
    EventNormalizationMixin,
    WriteMixin,
    ReadMixin,
    SummaryMixin,
    PromptMixin,
    DeprecatedMixin,
    IntrospectionMixin,
):
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
        scoped_indices: ScopedIndices | None = None,
        artifact_store: AsyncArtifactStore,
        hot_limit: int = 1000,
        hot_ttl_s: int = 7 * 24 * 3600,
        default_signal_threshold: float = 0.0,
        logger=None,
        llm: LLMClientProtocol | None = None,
    ):
        self.run_id = run_id
        self.session_id = session_id
        self.graph_id = graph_id
        self.node_id = node_id
        self.scope = scope
        self.hotlog = hotlog
        self.persistence = persistence
        self.scoped_indices = scoped_indices
        self.artifacts = artifact_store
        self.hot_limit = hot_limit
        self.hot_ttl_s = hot_ttl_s
        self.default_signal_threshold = default_signal_threshold
        self.logger = logger or logging.getLogger(__name__)
        self.llm = llm
        self.memory_scope_id = (
            self.scope.memory_scope_id() if self.scope else self.session_id or self.run_id
        )
        self.memory_tenant = (
            self.scope.memory_tenant_filter()
            if self.scope and hasattr(self.scope, "memory_tenant_filter")
            else {}
        )
        self.timeline_id = derive_timeline_id(
            memory_scope_id=self.memory_scope_id,
            run_id=self.run_id,
            org_id=self.scope.org_id if self.scope else None,
        )

    def _trace_meta(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "graph_id": self.graph_id,
            "node_id": self.node_id,
            "scope_id": self.memory_scope_id,
            "timeline_id": self.timeline_id,
        }
        if self.scope is not None:
            meta["app_id"] = getattr(self.scope, "app_id", None)
            meta["agent_id"] = getattr(self.scope, "agent_id", None)
            meta["user_id"] = getattr(self.scope, "user_id", None)
            meta["org_id"] = getattr(self.scope, "org_id", None)
        if extra:
            meta.update({k: v for k, v in extra.items() if v is not None})
        return meta

    async def _start_trace(
        self,
        *,
        operation: str,
        request: Any | None = None,
        tags: list[str] | None = None,
        metrics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        tracer = resolve_tracer()
        return await tracer.start_span(
            service="memory",
            operation=operation,
            request=request,
            tags=tags or ["memory"],
            metrics=metrics,
            metadata=self._trace_meta(metadata),
        )

    def _estimate_signal(
        self, *, text: str | None, metrics: dict[str, Any] | None, severity: int
    ) -> float:
        score = 0.15 + 0.1 * severity
        if text:
            score += min(len(text) / 400.0, 0.4)
        if metrics:
            score += 0.2
        return max(0.0, min(1.0, score))
