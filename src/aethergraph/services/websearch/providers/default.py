from __future__ import annotations

from aethergraph.services.websearch.types import (
    WebSearchEngine,
    WebSearchHit,
    WebSearchOptions,
    WebSearchProvider,
)


class DefaultWebSearchProvider(WebSearchProvider):
    """
    Provider-agnostic default provider.

    This intentionally performs a no-op search and returns no hits.
    It allows WebSearchService to remain available so page fetching still works
    via the configured WebPageFetcher.
    """

    engine_name: WebSearchEngine = WebSearchEngine.custom

    async def search(
        self,
        query: str,
        options: WebSearchOptions,
    ) -> list[WebSearchHit]:
        _ = (query, options)
        return []
