from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from aethergraph.contracts.services.llm import EmbeddingClientProtocol, LLMClientProtocol
from aethergraph.services.scope.scope import Scope


@dataclass
class KBSearchHit:
    """
    Normalized search hit from the knowledge base.

    This is a logical view that is independent of any particular storage/index
    implementation (local fs, remote vector DB, GraphRAG, etc.).
    """

    chunk_id: str
    doc_id: str
    corpus_id: str | None
    score: float
    text: str
    meta: dict[str, Any]


@dataclass
class KBAnswer:
    """
    Answer produced by the KB QA pipeline.

    `citations` is a raw list of references to chunks/docs.
    `resolved_citations` is an optional richer structure with snippets/URIs/etc.
    """

    answer: str
    citations: list[dict[str, Any]]
    usage: dict[str, Any] | None = None
    resolved_citations: list[KBSearchHit] | None = None


class KnowledgeBackend(Protocol):
    """
    Contract for knowledge-base backends.

    Implementations can be:
    - Local (fs + vector index)
    - Remote (GraphRAG, hosted vector DB)
    - Hybrid aggregators

    They MUST be scope-aware via the `scope` argument, but retain freedom
    about how they map scope into their own metadata / tenancy model.
    """

    embed_client: EmbeddingClientProtocol
    llm_client: LLMClientProtocol

    async def upsert_docs(
        self,
        *,
        scope: Scope | None,
        corpus_id: str | None,
        docs: list[dict[str, Any]],
        kb_namespace: str | None = None,
    ) -> dict[str, Any]:
        """
        Ingest (or update) a list of documents into the given corpus.

        Docs can be:
          - {"path": "/abs/path/to/file", "labels": {...}, "title": "..."}
          - {"text": "inline text...", "labels": {...}, "title": "..."}

        Implementations:
          - Own parsing/chunking and vector upsert semantics.
        """
        ...

    async def search(
        self,
        *,
        scope: Scope | None,
        corpus_id: str | None,
        query: str,
        top_k: int = 10,
        kb_namespace: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[KBSearchHit]:
        """
        Search for relevant chunks/docs given the query and optional filters.

        Implementations:
          - Own search semantics, but MUST return results in the normalized KBSearchHit format.
        """
        ...

    async def answer(
        self,
        *,
        scope: Scope | None,
        corpus_id: str | None,
        question: str,
        style: str = "concise",
        kb_namespace: str | None = None,
        filters: dict[str, Any] | None = None,
        k: int = 10,
    ) -> KBAnswer:
        """
        Produce an answer to the question using the knowledge base.

        Implementations:
          - Own retrieval and generation semantics, but MUST return results in the normalized KBAnswer format.
        """
        ...

    async def list_corpora(
        self,
        *,
        scope: Scope | None,
    ) -> list[dict[str, Any]]:
        """
        List corpora available in the given scope.

        Implementations:
          - Own listing semantics, but MUST return a list of dicts with corpus metadata (e.g. corpus_id, name, doc_count).
        """
        ...

    async def list_docs(
        self,
        *,
        scope: Scope | None,
        corpus_id: str | None,
        limit: int = 100,
        after: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        List documents in the given corpus.

        Implementations:
          - Own listing semantics, but MUST return a list of dicts with doc metadata (e.g. doc_id, title, labels).
        """
        ...

    async def delete_docs(
        self,
        *,
        scope: Scope | None,
        corpus_id: str | None,
        doc_ids: Sequence[str],
    ) -> dict[str, Any]:
        """
        Delete documents from the given corpus.

        Implementations:
          - Own deletion semantics, but MUST return a dict with deletion results (e.g. deleted_count, errors).
        """
        ...

    async def reembed(
        self,
        *,
        scope: Scope | None,
        corpus_id: str | None,
        doc_ids: Sequence[str] | None = None,
        batch: int = 64,
    ) -> dict[str, Any]:
        """
        Re-embed documents in the given corpus.

        If doc_ids is None, re-embed all documents in the corpus.

        Implementations:
          - Own re-embedding semantics, but MUST return a dict with re-embedding results (e.g. reembedded_count, errors).
        """
        ...

    async def stats(
        self,
        *,
        scope: Scope | None,
        corpus_id: str | None,
    ) -> dict[str, Any]:
        """
        Get statistics about the knowledge base in the given scope.

        Implementations:
          - Own stats semantics, but MUST return a dict with relevant stats (e.g. corpus_count, doc_count, avg_chunks_per_doc).
        """
        ...
