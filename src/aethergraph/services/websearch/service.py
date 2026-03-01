from collections.abc import Mapping

from aethergraph.services.websearch.types import (
    WebPageContent,
    WebPageFetcher,
    WebPageFetchOptions,
    WebSearchEngine,
    WebSearchHit,
    WebSearchOptions,
    WebSearchProvider,
)


class WebSearchService:
    """
    High-level web search + page fetch orchestrator.

    - Wraps multiple WebSearchProvider implementations.
    - Delegates fetches to a WebPageFetcher.
    """

    def __init__(
        self,
        providers: Mapping[WebSearchEngine, WebSearchProvider],
        *,
        default_engine: WebSearchEngine,
        page_fetcher: WebPageFetcher,
    ) -> None:
        if default_engine == WebSearchEngine.default:
            raise ValueError("default_engine must be a concrete engine, not 'default'.")
        if default_engine not in providers:
            raise ValueError(f"default_engine {default_engine!r} is not present in providers dict.")

        self._providers: dict[WebSearchEngine, WebSearchProvider] = dict(providers)
        self._default_engine = default_engine
        self._page_fetcher = page_fetcher

    # ---------------- public API ----------------
    async def search(
        self,
        query: str,
        *,
        engine: WebSearchEngine | None = None,
        options: WebSearchOptions | None = None,
    ) -> list[WebSearchHit]:
        if options is None:
            options = WebSearchOptions()

        provider = self._resolve_provider(engine)
        hits = await provider.search(query, options)

        # Place for normalization / dedup / rerank if needed
        # For now we just return as-is.
        return hits

    async def fetch_page(
        self,
        url: str,
        *,
        options: WebPageFetchOptions | None = None,
    ) -> WebPageContent:
        return await self._page_fetcher.fetch(url, options)

    async def search_and_fetch(
        self,
        query: str,
        *,
        engine: WebSearchEngine | None = None,
        search_options: WebSearchOptions | None = None,
        fetch_options: WebPageFetchOptions | None = None,
    ) -> list[tuple[WebSearchHit, WebPageContent]]:
        hits = await self.search(query, engine=engine, options=search_options)
        pages: list[tuple[WebSearchHit, WebPageContent]] = []
        for h in hits:
            page = await self.fetch_page(h.url, options=fetch_options)
            pages.append((h, page))
        return pages

    # ---------------------- helpers ---------------------- #

    def _resolve_provider(self, engine: WebSearchEngine | None) -> WebSearchProvider:
        eng = engine or WebSearchEngine.default

        if eng == WebSearchEngine.default:
            eng = self._default_engine

        provider = self._providers.get(eng)
        if not provider:
            available = ", ".join(sorted(e.value for e in self._providers))
            raise ValueError(
                f"No WebSearchProvider configured for engine={eng!r}. " f"Available: {available}"
            )
        return provider
