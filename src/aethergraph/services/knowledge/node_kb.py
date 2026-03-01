from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from aethergraph.contracts.services.knowledge import (
    KBAnswer,
    KBSearchHit,
    KnowledgeBackend,
)
from aethergraph.contracts.storage.search_backend import SearchMode
from aethergraph.services.scope.scope import Scope, ScopeLevel


@dataclass
class NodeKB:
    """
    Provide a scope-bound facade over `KnowledgeBackend`.

    `NodeKB` is the node-facing KB entrypoint (typically from `context.kb()`).
    Each method forwards to `backend` while injecting the bound `scope` so callers
    do not pass tenancy/filter scope on every call.

    Examples:
        Construct via factory-bound identity:
        ```python
        kb = kb_factory.for_identity(request_identity)
        # Uses kb.scope internally when forwarding to kb.backend.
        hits = await context.kb().search(
            corpus_id="product_docs",
            query="refund policy",
            top_k=5,
            kb_namespace="support",
        )
        ```

        Direct construction for tests with explicit attrs:
        ```python
        kb = NodeKB(backend=knowledge_backend, scope=test_scope)
        await context.kb().upsert_docs(
            corpus_id="runbook",
            docs=[{"text": "Restart by rotating token", "labels": {"source": "ops"}}],
            kb_namespace="infra",
        )
        answer = await context.kb().answer(
            corpus_id="runbook",
            question="How do I restart safely?",
            style="concise",
            kb_namespace="infra",
        )
        ```

    Args:
        backend: Concrete `KnowledgeBackend` implementation that performs ingest,
            retrieval, and QA work.
        scope: Bound `Scope` used for every forwarded backend call (for example via
            `scope.kb_filter()` / `scope.kb_index_labels()` in backend logic).

    Returns:
        NodeKB: Dataclass instance that proxies KB operations through `backend`
            with this instance's `scope`.

    Notes:
        `NodeKB` does not implement storage/index logic itself; behavior such as
        dedupe, filtering, search mode semantics, and citation resolution is owned
        by the configured backend.
    """

    backend: KnowledgeBackend
    scope: Scope

    async def upsert_docs(
        self,
        corpus_id: str,
        docs: list[dict[str, Any]],
        *,
        kb_namespace: str | None = None,
    ) -> dict[str, Any]:
        """
        Ingest or update documents in a corpus under the bound KB scope.

        This forwards to `backend.upsert_docs(...)` and always injects
        `scope=self.scope` so dedupe and metadata scoping are applied by the
        configured backend implementation.

        Examples:
            Ingest inline text:
            ```python
            result = await context.kb().upsert_docs(
                corpus_id="product_docs",
                docs=[{"text": "Returns accepted for 30 days.", "labels": {"topic": "policy"}}],
                kb_namespace="support",
            )
            ```

            Ingest from file path with title:
            ```python
            result = await context.kb().upsert_docs(
                corpus_id="product_docs",
                docs=[{"path": "C:/docs/refund.md", "title": "Refund Policy"}],
            )
            ```

        Args:
            corpus_id: Logical corpus identifier to write into.
            docs: Documents to ingest. Each item is typically path-based
                (`{"path": ...}`) or inline text (`{"text": ...}`), with
                optional `title` and `labels`.
            kb_namespace: Optional namespace partition inside the corpus.

        Returns:
            dict[str, Any]: Backend-defined ingestion summary such as added
                document and chunk counts.

        Notes:
            Scope and index-level labels are derived from `self.scope` by the
            backend; callers should pass only document-level inputs.
        """
        return await self.backend.upsert_docs(
            scope=self.scope,
            corpus_id=corpus_id,
            docs=docs,
            kb_namespace=kb_namespace,
        )

    async def search(
        self,
        *,
        corpus_id: str,
        query: str,
        top_k: int = 10,
        kb_namespace: str | None = None,
        filters: Mapping[str, Any] | None = None,
        level: ScopeLevel | None = None,
        time_window: str | None = None,
        created_at_min: float | None = None,
        created_at_max: float | None = None,
        mode: SearchMode | None = None,
        lexical_rerank: bool = True,
    ) -> list[KBSearchHit]:
        """
        Retrieve relevant KB chunks for a query from the scoped corpus.

        This method forwards all search controls to `backend.search(...)` and
        injects `scope=self.scope` so tenant and KB scope filters are applied
        consistently by the backend.

        Examples:
            Basic semantic retrieval:
            ```python
            hits = await context.kb().search(
                corpus_id="product_docs",
                query="refund timeline",
                top_k=5,
                kb_namespace="support",
            )
            ```

            Retrieval with filters and explicit mode:
            ```python
            hits = await context.kb().search(
                corpus_id="engineering_runbook",
                query="how to rotate credentials",
                filters={"labels.env": "prod"},
                mode="hybrid",
                lexical_rerank=True,
                created_at_min=1735707600.0,
            )
            ```

        Args:
            corpus_id: Logical corpus identifier to query.
            query: User query text to retrieve against.
            top_k: Maximum number of hits to return.
            kb_namespace: Optional namespace partition inside the corpus.
            filters: Optional metadata filters merged with scope-derived filters.
            level: Optional scope level hint for backend-specific behavior.
            time_window: Optional relative time filter (backend-defined format).
            created_at_min: Optional inclusive lower bound (epoch seconds).
            created_at_max: Optional inclusive upper bound (epoch seconds).
            mode: Optional backend search mode (`semantic`, `lexical`, `hybrid`,
                etc., depending on backend support).
            lexical_rerank: Whether backend should apply lexical reranking when
                supported.

        Returns:
            list[KBSearchHit]: Normalized chunk hits ranked by backend scoring
                policy.

        Notes:
            `NodeKB` does not merge filters itself; it forwards arguments and
            relies on backend semantics.
        """
        return await self.backend.search(
            scope=self.scope,
            corpus_id=corpus_id,
            query=query,
            top_k=top_k,
            kb_namespace=kb_namespace,
            filters=filters,
            level=level,
            time_window=time_window,
            created_at_min=created_at_min,
            created_at_max=created_at_max,
            mode=mode,
            lexical_rerank=lexical_rerank,
        )

    async def answer(
        self,
        *,
        corpus_id: str,
        question: str,
        style: str = "concise",
        kb_namespace: str | None = None,
        filters: Mapping[str, Any] | None = None,
        k: int = 10,
        level: ScopeLevel | None = None,
        time_window: str | None = None,
        created_at_min: float | None = None,
        created_at_max: float | None = None,
        mode: SearchMode | None = None,
        lexical_rerank: bool = True,
    ) -> KBAnswer:
        """
        Generate an answer using corpus retrieval plus backend QA logic.

        This forwards to `backend.answer(...)` with `scope=self.scope`. The
        backend owns retrieval, prompting, and citation shaping, while this
        facade keeps caller code scope-safe and concise.

        Examples:
            Concise QA response:
            ```python
            result = await context.kb().answer(
                corpus_id="product_docs",
                question="What is the refund window?",
                style="concise",
                kb_namespace="support",
            )
            ```

            Detailed QA with metadata filtering:
            ```python
            result = await context.kb().answer(
                corpus_id="engineering_runbook",
                question="How should I recover from token leak?",
                style="detailed",
                filters={"labels.team": "platform"},
                k=8,
                mode="semantic",
            )
            ```

        Args:
            corpus_id: Logical corpus identifier to query.
            question: Question to answer from indexed corpus content.
            style: Backend prompt style hint (for example `concise` or
                `detailed`).
            kb_namespace: Optional namespace partition inside the corpus.
            filters: Optional metadata filters merged with scoped filters by the
                backend.
            k: Retrieval depth used by backend QA before synthesis.
            level: Optional scope level hint for backend-specific behavior.
            time_window: Optional relative time filter.
            created_at_min: Optional inclusive lower bound (epoch seconds).
            created_at_max: Optional inclusive upper bound (epoch seconds).
            mode: Optional backend search mode used by retrieval.
            lexical_rerank: Whether retrieval should apply lexical reranking when
                supported.

        Returns:
            KBAnswer: Answer payload with text and citation metadata.

        Notes:
            Empty retrieval handling (for example returning blank answer/citations)
            is backend-defined.
        """
        return await self.backend.answer(
            scope=self.scope,
            corpus_id=corpus_id,
            question=question,
            style=style,
            kb_namespace=kb_namespace,
            filters=filters,
            k=k,
            level=level,
            time_window=time_window,
            created_at_min=created_at_min,
            created_at_max=created_at_max,
            mode=mode,
            lexical_rerank=lexical_rerank,
        )

    async def list_corpora(self) -> list[dict[str, Any]]:
        """
        List corpora visible to the bound scope.

        This delegates to `backend.list_corpora(scope=self.scope)` so callers can
        enumerate available corpora without manually threading scope.

        Examples:
            List corpora for the current identity:
            ```python
            corpora = await context.kb().list_corpora()
            ```

            Extract corpus ids for UI options:
            ```python
            corpus_ids = [row["corpus_id"] for row in await context.kb().list_corpora()]
            ```

        Args:
            None: This method accepts no caller parameters.

        Returns:
            list[dict[str, Any]]: Backend-provided corpus records.

        Notes:
            Record shape is backend-defined, commonly including `corpus_id` and
            metadata.
        """
        return await self.backend.list_corpora(scope=self.scope)

    async def list_docs(
        self,
        *,
        corpus_id: str,
        limit: int = 200,
        after: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        List documents for a corpus under the bound scope.

        This forwards pagination parameters to `backend.list_docs(...)` and
        injects `scope=self.scope` automatically.

        Examples:
            First page of docs:
            ```python
            docs = await context.kb().list_docs(corpus_id="product_docs", limit=50)
            ```

            Continue after a known document id:
            ```python
            next_docs = await context.kb().list_docs(
                corpus_id="product_docs",
                limit=50,
                after="doc_abc123",
            )
            ```

        Args:
            corpus_id: Logical corpus identifier to inspect.
            limit: Maximum number of docs to return.
            after: Optional pagination cursor (backend-specific doc id semantics).

        Returns:
            list[dict[str, Any]]: Backend-provided document metadata records.

        Notes:
            Ordering and cursor semantics are backend-defined.
        """
        return await self.backend.list_docs(
            scope=self.scope,
            corpus_id=corpus_id,
            limit=limit,
            after=after,
        )

    async def delete_docs(
        self,
        *,
        corpus_id: str,
        doc_ids: list[str],
    ) -> dict[str, Any]:
        """
        Delete one or more documents from a corpus.

        This delegates to `backend.delete_docs(...)` with the bound scope and
        returns backend deletion counters/status information.

        Examples:
            Delete a single document:
            ```python
            result = await context.kb().delete_docs(
                corpus_id="product_docs",
                doc_ids=["doc_abc123"],
            )
            ```

            Delete a batch:
            ```python
            result = await context.kb().delete_docs(
                corpus_id="engineering_runbook",
                doc_ids=["doc_1", "doc_2", "doc_3"],
            )
            ```

        Args:
            corpus_id: Logical corpus identifier to mutate.
            doc_ids: Document ids to remove.

        Returns:
            dict[str, Any]: Backend-defined deletion summary (for example removed
                docs/chunks).

        Notes:
            Partial deletion behavior and error reporting are backend-defined.
        """
        return await self.backend.delete_docs(
            scope=self.scope,
            corpus_id=corpus_id,
            doc_ids=doc_ids,
        )

    async def reembed(
        self,
        *,
        corpus_id: str,
        doc_ids: list[str] | None = None,
        batch: int = 64,
    ) -> dict[str, Any]:
        """
        Recompute embeddings for documents in a corpus.

        This forwards to `backend.reembed(...)` with `scope=self.scope`.
        Backends typically re-upsert chunk vectors in batches.

        Examples:
            Re-embed all docs in a corpus:
            ```python
            result = await context.kb().reembed(corpus_id="product_docs")
            ```

            Re-embed selected docs with smaller batch size:
            ```python
            result = await context.kb().reembed(
                corpus_id="engineering_runbook",
                doc_ids=["doc_abc123", "doc_def456"],
                batch=16,
            )
            ```

        Args:
            corpus_id: Logical corpus identifier to process.
            doc_ids: Optional subset of document ids. `None` means all docs.
            batch: Batch size hint for embedding/upsert loops.

        Returns:
            dict[str, Any]: Backend-defined re-embedding summary.

        Notes:
            Embedding model name and exact counters are backend-defined.
        """
        return await self.backend.reembed(
            scope=self.scope,
            corpus_id=corpus_id,
            doc_ids=doc_ids,
            batch=batch,
        )

    async def stats(self, *, corpus_id: str) -> dict[str, Any]:
        """
        Return corpus-level statistics for the bound scope.

        This method forwards to `backend.stats(...)` with `scope=self.scope` and
        returns backend-provided counters/metadata.

        Examples:
            Fetch high-level stats:
            ```python
            stats = await context.kb().stats(corpus_id="product_docs")
            ```

            Read document and chunk counts:
            ```python
            stats = await context.kb().stats(corpus_id="engineering_runbook")
            docs = stats.get("docs", 0)
            chunks = stats.get("chunks", 0)
            ```

        Args:
            corpus_id: Logical corpus identifier to inspect.

        Returns:
            dict[str, Any]: Backend-defined corpus statistics payload.

        Notes:
            Metric names and additional fields depend on backend implementation.
        """
        return await self.backend.stats(scope=self.scope, corpus_id=corpus_id)
