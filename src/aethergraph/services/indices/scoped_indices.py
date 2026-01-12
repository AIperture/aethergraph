# aethergraph/indices.py
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from aethergraph.contracts.storage.search_backend import ScoredItem, SearchBackend
from aethergraph.services.scope.scope import Scope


@dataclass
class ScopedIndices:
    """
    Scope-aware wrapper around the global SearchBackend.

    - scope: Scope defining org/user/app/run/session/node
    - scope_id: usually a memory_scope_id for memory-tied corpora,
      but can be anything logical (or None).
    """

    backend: SearchBackend
    scope: Scope
    scope_id: str | None = None

    # --- internals --------------------------------------------------------

    def _base_metadata(self) -> dict[str, Any]:
        """
        Default metadata to attach on *writes*.

        For memory-ish corpora, this matches RAG docs:
        - user_id, org_id, client_id, app_id, session_id, run_id, graph_id, node_id
        - scope_id (usually memory_scope_id)
        """
        return self.scope.rag_labels(scope_id=self.scope_id)

    def _base_filters(self) -> dict[str, Any]:
        """
        Default filters for *reads*.

        For memory-ish corpora, this matches RAG search:
        - user_id, org_id, (and scope_id if provided)
        """
        return self.scope.rag_filter(scope_id=self.scope_id)

    # --- public APIs ------------------------------------------------------

    async def upsert(
        self,
        *,
        corpus: str,
        item_id: str,
        text: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        base = self._base_metadata()
        merged: dict[str, Any] = {**base, **(metadata or {})}
        # strip None so backends can treat them as wildcards
        merged = {k: v for k, v in merged.items() if v is not None}

        await self.backend.upsert(
            corpus=corpus,
            item_id=item_id,
            text=text,
            metadata=merged,
        )

    async def search(
        self,
        *,
        corpus: str,
        query: str,
        top_k: int = 10,
        filters: Mapping[str, Any] | None = None,
        time_window: str | None = None,
        created_at_min: float | None = None,
        created_at_max: float | None = None,
    ) -> list[ScoredItem]:
        """
        time_window: human-friendly duration like "7d", "24h", "30m"
            - interpreted as [now - window, now] in created_at_ts.
            - ignored if created_at_min is explicitly given.

        created_at_min / created_at_max: UNIX timestamps (float).
        """
        base = self._base_filters()
        merged: dict[str, Any] = {**base, **(filters or {})}
        merged = {k: v for k, v in merged.items() if v is not None}

        return await self.backend.search(
            corpus=corpus,
            query=query,
            top_k=top_k,
            filters=merged,
            time_window=time_window,
            created_at_min=created_at_min,
            created_at_max=created_at_max,
        )

    # ergonomic helpers (optional but nice)

    async def search_events(
        self,
        query: str,
        *,
        top_k: int = 20,
        filters: Mapping[str, Any] | None = None,
        time_window: str | None = None,
        created_at_min: float | None = None,
        created_at_max: float | None = None,
    ) -> list[ScoredItem]:
        return await self.search(
            corpus="event",
            query=query,
            top_k=top_k,
            filters=filters,
            time_window=time_window,
            created_at_min=created_at_min,
            created_at_max=created_at_max,
        )

    async def search_artifacts(
        self,
        query: str,
        *,
        top_k: int = 20,
        filters: Mapping[str, Any] | None = None,
        time_window: str | None = None,
        created_at_min: float | None = None,
        created_at_max: float | None = None,
    ) -> list[ScoredItem]:
        return await self.search(
            corpus="artifact",
            query=query,
            top_k=top_k,
            filters=filters,
            time_window=time_window,
            created_at_min=created_at_min,
            created_at_max=created_at_max,
        )
