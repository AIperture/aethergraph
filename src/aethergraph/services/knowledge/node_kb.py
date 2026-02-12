from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from aethergraph.contracts.services.knowledge import (
    KBAnswer,
    KBSearchHit,
    KnowledgeBackend,
)
from aethergraph.services.scope.scope import Scope, ScopeLevel


@dataclass
class NodeKB:
    """
    Scope-bound facade for the KnowledgeBackend.

    This is what tools / nodes should use via `context.kb()`.
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
    ) -> list[KBSearchHit]:
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
    ) -> KBAnswer:
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
        )

    async def list_corpora(self) -> list[dict[str, Any]]:
        return await self.backend.list_corpora(scope=self.scope)

    async def list_docs(
        self,
        *,
        corpus_id: str,
        limit: int = 200,
        after: str | None = None,
    ) -> list[dict[str, Any]]:
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
        return await self.backend.reembed(
            scope=self.scope,
            corpus_id=corpus_id,
            doc_ids=doc_ids,
            batch=batch,
        )

    async def stats(self, *, corpus_id: str) -> dict[str, Any]:
        return await self.backend.stats(scope=self.scope, corpus_id=corpus_id)
