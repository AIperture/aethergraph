from __future__ import annotations

from typing import Any, Protocol

from aethergraph.contracts.services.llm import LLMClientProtocol
from aethergraph.contracts.services.memory import Event, HotLog, Indices, Persistence
from aethergraph.contracts.storage.artifact_store import AsyncArtifactStore
from aethergraph.contracts.storage.doc_store import DocStore
from aethergraph.services.rag.facade import RAGFacade
from aethergraph.services.scope.scope import Scope


class MemoryFacadeInterface(Protocol):
    """
    Protocol defining the state and core methods available on the MemoryFacade.
    Mixins use this to type-hint 'self'.
    """

    run_id: str
    timeline_id: str
    memory_scope_id: str

    hotlog: HotLog
    persistence: Persistence
    indices: Indices
    docs: DocStore
    artifacts: AsyncArtifactStore
    scope: Scope | None

    rag: RAGFacade | None
    llm: LLMClientProtocol | None
    logger: Any

    default_signal_threshold: float
    hot_limit: int
    hot_ttl_s: int

    async def record_raw(
        self,
        *,
        base: dict[str, Any],
        text: str | None = None,
        metrics: dict[str, float] | None = None,
    ) -> Event: ...

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
    ) -> Event: ...

    async def write_result(
        self,
        *,
        tool: str,
        inputs: list[dict[str, Any]] | None = None,
        outputs: list[dict[str, Any]] | None = None,
        tags: list[str] | None = None,
        metrics: dict[str, float] | None = None,
        message: str | None = None,
        severity: int = 3,
    ) -> Event: ...

    # Required for RetrievalMixin to expose 'recent' to other mixins
    async def recent(self, *, kinds: list[str] | None = None, limit: int = 50) -> list[Event]: ...

    # Required for RAGMixin to expose 'rag_bind'
    async def rag_bind(
        self, *, key: str = "default", create_if_missing: bool = True, labels: dict | None = None
    ) -> str: ...
