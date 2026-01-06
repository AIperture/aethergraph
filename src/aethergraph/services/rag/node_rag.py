from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aethergraph.services.rag.facade import RAGFacade, SearchHit
from aethergraph.services.scope.scope import Scope


@dataclass
class NodeRAG:
    """
    Node-scoped RAG helper.

    - Wraps a global RAGFacade.
    - Injects Scope into the common node-facing calls.
    - Delegates everything else via __getattr__.
    """

    rag: RAGFacade
    scope: Scope
    default_scope_id: str | None = None

    # -------- internals --------

    def _scope_id(self, scope_id: str | None) -> str | None:
        if scope_id is not None:
            return scope_id
        if self.default_scope_id is not None:
            return self.default_scope_id
        return self.scope.memory_scope_id()

    # -------- scope-aware helpers --------

    async def bind_corpus(
        self,
        *,
        corpus_id: str | None = None,
        key: str | None = None,
        create_if_missing: bool = True,
        labels: dict[str, Any] | None = None,
        scope_id: str | None = None,
    ) -> str:
        """
        Bind or create a RAG corpus for the current node scope.

        This method ensures a corpus exists for the given scope and key, creating it if necessary.
        It automatically injects scope labels and metadata, and returns the resolved corpus ID.

        Examples:
            Bind a default corpus for the current node:
            ```python
            corpus_id = await context.rag().bind_corpus()
            ```

            Bind or create a corpus with a custom key and extra labels:
            ```python
            corpus_id = await context.rag().bind_corpus(
                key="my-data",
                labels={"source": "user-upload"}
            )
            ```

        Args:
            corpus_id: Optional explicit corpus identifier. If not provided, one is generated from the scope and key.
            key: Optional string to distinguish corpora within the same scope (e.g., "default", "my-data").
            create_if_missing: If True (default), create the corpus if it does not exist.
            labels: Optional dictionary of additional metadata to attach to the corpus.
            scope_id: Optional override for the scope identifier. Defaults to the current node's scope.

        Returns:
            str: The resolved corpus ID, guaranteed to exist if `create_if_missing` is True.

        Notes:
            - The corpus ID is derived from the scope and key if not explicitly provided.
            - Scope labels are automatically merged into the corpus metadata.
        """
        sid = self._scope_id(scope_id)
        scope_labels = self.scope.rag_scope_labels(scope_id=sid)

        if corpus_id:
            cid = corpus_id
        else:
            # e.g. mem:<scope>:<key>
            cid = self.scope.rag_corpus_id(scope_id=sid, key=key or "default")

        meta = {"scope": scope_labels, **(labels or {})}

        if create_if_missing:
            await self.rag.add_corpus(
                corpus_id=cid,
                meta=meta,
                scope_labels=scope_labels,
            )
        return cid

    async def upsert_docs(
        self,
        corpus_id: str,
        docs: list[dict[str, Any]],
        *,
        scope_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Ingest and index a list of documents into the specified corpus for the current node scope.

        This method ensures the corpus exists for the given scope, merges scope labels into each document,
        and handles both file-based and inline text documents. Documents are chunked, embedded, and indexed
        for retrieval.

        Examples:
            Ingest a list of inline documents:
            ```python
            await context.rag().upsert_docs(
                corpus_id="my-corpus",
                docs=[
                    {"text": "Document content...", "title": "Doc Title"},
                    {"text": "Another doc", "labels": {"source": "user-upload"}}
                ]
            )
            ```

            Ingest a PDF file with custom labels:
            ```python
            await context.rag().upsert_docs(
                corpus_id="my-corpus",
                docs=[{"path": "/path/to/file.pdf", "labels": {"type": "pdf"}}]
            )
            ```

        Args:
            corpus_id: The target corpus identifier.
            docs: A list of document specifications. Each document can be:
                - File-based: {"path": "/path/to/doc.pdf", "labels": {...}}
                - Inline text: {"text": "Document content...", "title": "Doc Title", "labels": {...}}
            scope_id: Optional override for the scope identifier. Defaults to the current node's scope.

        Returns:
            dict[str, Any]: Summary of the ingestion, including number of documents and chunks added.

        Notes:
            - Scope labels are merged into each document's labels.
            - File-based documents are read and chunked automatically.
            - Inline text documents are chunked based on configured chunk size.
        """
        sid = self._scope_id(scope_id)
        return await self.rag.upsert_docs(
            corpus_id=corpus_id,
            docs=docs,
            scope=self.scope,
            scope_id=sid,
        )

    async def search(
        self,
        corpus_id: str,
        query: str,
        *,
        k: int = 8,
        filters: dict[str, Any] | None = None,
        scope_id: str | None = None,
        mode: str = "hybrid",
    ) -> list[SearchHit]:
        """
        Search the specified RAG corpus for relevant chunks matching a query.

        This method performs a dense or hybrid (dense + lexical) search over the corpus,
        automatically injecting node scope filters. It returns the top-k most relevant
        results as `SearchHit` objects, including chunk text, metadata, and scores.

        Examples:
            Basic usage to search a corpus:
            ```python
            hits = await context.rag().search(
                corpus_id="my-corpus",
                query="What is the capital of France?"
            )
            ```

            Search with custom filters and top-3 results:
            ```python
            hits = await context.rag().search(
                corpus_id="my-corpus",
                query="project roadmap",
                k=3,
                filters={"type": "meeting-notes"}
            )
            ```

        Args:
            corpus_id: The target corpus identifier to search within.
            query: The search query string.
            k: The number of top results to return (default: 8).
            filters: Optional dictionary of metadata filters to apply (merged with scope filters).
            scope_id: Optional override for the scope identifier. Defaults to the current node's scope.
            mode: Search mode, either `"dense"` or `"hybrid"` (default: "hybrid").

        Returns:
            list[SearchHit]: A list of matching `SearchHit` objects, each containing chunk text,
            metadata, score, and identifiers.

        Notes:
            - Scope filters are automatically merged with any provided filters.
            - Hybrid mode fuses dense and lexical search for improved relevance.
            - Results are sorted by descending relevance score.
        """
        sid = self._scope_id(scope_id)
        scoped_filters = self.scope.rag_filter(scope_id=sid)
        if filters:
            scoped_filters.update(filters)
        return await self.rag.search(
            corpus_id=corpus_id,
            query=query,
            k=k,
            filters=scoped_filters,
            mode=mode,
        )

    async def answer(
        self,
        corpus_id: str,
        question: str,
        *,
        llm: str | None = None,
        style: str = "concise",
        with_citations: bool = True,
        k: int = 6,
        scope_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Answer a question using retrieved context from a specified corpus.

        This method retrieves relevant context chunks from the target corpus, constructs a prompt for the language model, and generates an answer. Citations to the retrieved chunks are included if requested. The function is accessed via `context.rag().answer(...)`.

        Examples:
            Basic usage to answer a question:
            ```python
            result = await context.rag().answer(
                corpus_id="my-corpus",
                question="What is the capital of France?"
            print(result["answer"])
            ```

            Requesting a detailed answer with citations:
            ```python
            result = await context.rag().answer(
                corpus_id="my-corpus",
                question="Explain the process of photosynthesis.",
                style="detailed",
                with_citations=True,
                k=8
            )
            print("Answer:", result["answer"])
            for cite in result["citations"]:
                print(f"Citation: {cite['text']} (Score: {cite['score']})")
            ```

        Args:
            corpus_id: Identifier of the target corpus to search for context.
            question: The question to be answered.
            llm: Optional language model client to use for answer generation. If None, the default LLM is used.
            style: The style of the answer, either "concise" (default) or "detailed".
            with_citations: Whether to include citations to the retrieved context chunks in the answer (default: True).
            k: Number of context chunks to retrieve for answering (default: 6).
            scope_id: Optional identifier to restrict retrieval to a specific scope.

        Returns:
            dict[str, Any]: A dictionary containing the generated answer, citations, usage statistics, and optionally resolved citation metadata.

        Notes:
            - the generated dictionary includes:

                - `answer`: The generated answer text.
                - `citations`: List of retrieved context chunks used as citations.
                - `usage`: LLM usage statistics (tokens, time, etc.).
                - `resolved_citations`: Optional metadata for citations if available.

            - Example response:
            ```python
            {
                "answer": "The capital of France is Paris.",
                "citations": [
                    {"text": "Paris is the capital city of France...", "score": 0.95, ...},
                    ...
                ],
                "usage": {"prompt_tokens": 150, "completion_tokens": 50, "total_tokens": 200, ...},
                "resolved_citations": [
                    {"doc_id": "doc123", "title": "Geography of France", ...},
                    ...
                ]
            }
            ```
        """
        sid = self._scope_id(scope_id)
        return await self.rag.answer(
            corpus_id=corpus_id,
            question=question,
            llm=llm,
            style=style,
            with_citations=with_citations,
            k=k,
            scope=self.scope,
            scope_id=sid,
        )

    # -------- delegation: everything else --------

    def __getattr__(self, name: str) -> Any:
        """
        Fallback: expose the underlying RAGFacade API for advanced users.

        Node code can still call low-level stuff if needed:
            ctx.rag.stats(...)
            ctx.rag.list_corpora()
        etc.
        """
        return getattr(self.rag, name)
