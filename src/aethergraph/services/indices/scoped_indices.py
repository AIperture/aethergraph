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
        """
        Upsert (insert or update) a text item with associated metadata into the backend index.
        This method merges base metadata with any provided metadata, strips out keys with None values,
        and delegates the upsert operation to the backend. This ensures that only meaningful metadata
        is stored and that None values are treated as wildcards by the backend.

        Examples:
            Basic usage to upsert a text item:
            ```python
            await service.upsert(
                corpus="my_corpus",
                item_id="item123",
                text="Sample document text."
            ```

            Upserting with additional metadata:
            ```python
            await service.upsert(
                corpus="my_corpus",
                item_id="item123",
                text="Sample document text.",
                metadata={"author": "Alice", "category": "news"}

            ```
        Args:
            corpus: The name of the corpus or collection to upsert the item into.
            item_id: The unique identifier for the item within the corpus.
            text: The text content to be indexed or updated.
            metadata: Optional mapping of additional metadata to associate with the item.

        Returns:
            None

        Notes:
            Metadata keys with None values are omitted before upserting to the backend.
        """

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
        Perform a search operation on the specified corpus.
        This method executes a search query against the backend, applying optional filters,
        time constraints, and other parameters to refine the results.

        Examples:
            Basic usage to search a corpus:
            ```python
            results = await search(corpus="documents", query="machine learning")
            ```

            Searching with additional filters and time constraints:
            ```python
            results = await search(
            corpus="articles",
            query="AI advancements",
            top_k=5,
            filters={"author": "John Doe"},
            time_window="7d"
            ```

        Args:
            corpus: The name of the corpus to search within.
            query: The search query string.
            top_k: The maximum number of results to return (default: 10).
            filters: Optional dictionary of additional filters to apply to the search.
            time_window: Optional human-friendly duration (e.g., "7d", "24h", "30m")
            interpreted as [now - window, now] in created_at_ts. Ignored if `created_at_min` is provided.
            created_at_min: Optional minimum UNIX timestamp (float) for filtering results by creation time.
            created_at_max: Optional maximum UNIX timestamp (float) for filtering results by creation time.

        Returns:
            A list of `ScoredItem` objects representing the search results.

        Notes:
            - If `time_window` is provided, it is used to calculate the time range unless `created_at_min` is explicitly set.
            - Filters with `None` values are automatically excluded from the search.
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
        """
        Perform a search for events based on the given query and optional filters.

        This method queries the "event" corpus using the specified parameters to retrieve
        a list of scored items matching the search criteria.

        Examples:
            Basic usage to search for events:
            ```python
            results = await search_events("error logs")
            ```

            Searching with additional filters and a time window:
            ```python
            results = await search_events(
            "user activity",
            top_k=10,
            filters={"status": "active"},
            time_window="last_24_hours",
            created_at_min=1672531200.0,
            created_at_max=1672617600.0
            ```

        Args:
            query: The search query string.
            top_k: The maximum number of results to return (default: 20).
            filters: Optional dictionary of filters to apply to the search.
            time_window: Optional time window for the search (e.g., "last_24_hours").
            created_at_min: Optional minimum creation timestamp for filtering results.
            created_at_max: Optional maximum creation timestamp for filtering results.

        Returns:
            A list of `ScoredItem` objects representing the search results.

        Notes:
        - The `filters` parameter allows you to specify key-value pairs to narrow down the search results.
        - The `time_window` parameter can be used to specify a predefined time range for the search.
        - The `created_at_min` and `created_at_max` parameters allow for fine-grained control over the
          creation time range of the events being searched.
        """

        items = await self.search(
            corpus="event",
            query=query,
            top_k=top_k,
            filters=filters,
            time_window=time_window,
            created_at_min=created_at_min,
            created_at_max=created_at_max,
        )

        return items

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
        """
        Perform a search for artifacts based on the provided query and optional filters.

        This method queries the "artifact" corpus using the specified parameters and returns
        a list of scored items matching the search criteria.

        Examples:
            Basic usage to search for artifacts:
            ```python
            results = await search_artifacts("example query")
            ```

            Searching with additional filters and a time window:
            ```python
            results = await search_artifacts(
            "example query",
            top_k=10,
            filters={"type": "document"},
            time_window="last_7_days",
            created_at_min=1672531200.0,
            created_at_max=1672617600.0,
            ```

        Args:
            query: The search query string.
            top_k: The maximum number of results to return (default: 20).
            filters: Optional dictionary of filters to apply to the search.
            time_window: Optional time window for the search (e.g., "last_7_days").
            created_at_min: Optional minimum creation timestamp for filtering results.
            created_at_max: Optional maximum creation timestamp for filtering results.

        Returns:
            A list of `ScoredItem` objects representing the search results.

        Notes:
            - The `filters` parameter allows specifying additional constraints for the search.
            - The `time_window` parameter can be used to limit results to a specific time range.
            - The `created_at_min` and `created_at_max` parameters allow filtering by creation time.
        """

        return await self.search(
            corpus="artifact",
            query=query,
            top_k=top_k,
            filters=filters,
            time_window=time_window,
            created_at_min=created_at_min,
            created_at_max=created_at_max,
        )
