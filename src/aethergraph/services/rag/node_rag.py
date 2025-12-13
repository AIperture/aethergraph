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
