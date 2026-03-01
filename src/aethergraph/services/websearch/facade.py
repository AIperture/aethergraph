from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from aethergraph.services.websearch.service import WebSearchService
from aethergraph.services.websearch.types import (
    WebPageContent,
    WebPageFetchOptions,
    WebSearchEngine,
    WebSearchHit,
    WebSearchOptions,
)


@dataclass
class WebSearchFacade:
    """
    Provide node-friendly web search and page fetch APIs over `WebSearchService`.

    This facade is intended for `context.web` usage and converts simple method
    arguments into `WebSearchOptions` and `WebPageFetchOptions` before delegating
    to `svc`.

    Examples:
        Construct and perform a search:
        ```python
        web = WebSearchFacade(svc=web_search_service)
        hits = await web.search(
            "latest sqlite release notes",
            top_k=5,
            engine=WebSearchEngine.brave,
            freshness_days=30,
        )
        ```

        Search and fetch page content:
        ```python
        web = WebSearchFacade(svc=web_search_service)
        rows = await web.search_and_fetch(
            "site:docs.python.org dataclasses",
            top_k=3,
            timeout_seconds=8.0,
        )
        ```

    Args:
        svc: Shared `WebSearchService` instance used for provider routing and
            page fetch orchestration.

    Returns:
        WebSearchFacade: Dataclass wrapper that exposes ergonomic async methods
            over `svc`.

    Notes:
        Scope is not currently part of this facade; if scoped caching/KB writes
        are added later, they should be wired above this layer.
    """

    svc: WebSearchService

    # Optionally accept a `scope` here later if needed
    # scope: Scope | None = None

    # -------- high-level search API -------- #

    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        engine: WebSearchEngine | None = None,
        site: str | None = None,
        freshness_days: int | None = None,
        language: str | None = None,
        safe_search: bool = False,
        extra: Mapping[str, Any] | None = None,
    ) -> list[WebSearchHit]:
        """
        Run a web search and return normalized hits.

        This method builds `WebSearchOptions` from the provided keyword arguments
        and delegates to `svc.search(...)`.

        Examples:
            Basic search with defaults:
            ```python
            hits = await web.search("python async context manager")
            ```

            Search with engine, site filter, and recency:
            ```python
            hits = await web.search(
                "migration guide",
                top_k=8,
                engine=WebSearchEngine.tavily,
                site="docs.aethergraph.dev",
                freshness_days=14,
                safe_search=True,
            )
            ```

        Args:
            query: Search query string.
            top_k: Maximum number of hits requested from the provider.
            engine: Optional engine override. `None` uses the service default.
            site: Optional domain constraint mapped to `site_filter`.
            freshness_days: Optional recency window in days.
            language: Optional language hint passed to providers.
            safe_search: Whether to request safe-search filtering when supported.
            extra: Provider-specific passthrough options.

        Returns:
            list[WebSearchHit]: Ordered search hits returned by the resolved
                provider.

        Notes:
            `region` is currently fixed to `None` in this facade.
        """
        options = WebSearchOptions(
            top_k=top_k,
            language=language,
            region=None,
            freshness_days=freshness_days,
            site_filter=site,
            safe_search=safe_search,
            extra=extra,
        )
        return await self.svc.search(query, engine=engine, options=options)

    async def fetch(
        self,
        url: str,
        *,
        timeout_seconds: float = 10.0,
        max_bytes: int = 2_000_000,
        allow_non_html: bool = False,
        allowed_content_types: Sequence[str] | None = None,
        user_agent: str | None = None,
    ) -> WebPageContent:
        """
        Fetch and parse page content for a single URL.

        This method constructs `WebPageFetchOptions` and delegates to
        `svc.fetch_page(...)`.

        Examples:
            Fetch a standard HTML page:
            ```python
            page = await web.fetch("https://example.com/docs")
            ```

            Fetch with stricter content controls:
            ```python
            page = await web.fetch(
                "https://example.com/data.json",
                allow_non_html=True,
                allowed_content_types=["application/json", "text/html"],
                timeout_seconds=5.0,
                max_bytes=500_000,
            )
            ```

        Args:
            url: Absolute URL to retrieve.
            timeout_seconds: Request timeout budget for the fetcher.
            max_bytes: Maximum response bytes to read.
            allow_non_html: Whether non-HTML content types are allowed.
            allowed_content_types: Optional allowlist enforced by the fetcher.
            user_agent: Optional custom user-agent string.

        Returns:
            WebPageContent: Parsed page payload including extracted text/metadata
                as defined by the fetcher implementation.

        Notes:
            Content-type enforcement is handled by the underlying fetcher.
        """
        options = WebPageFetchOptions(
            timeout_seconds=timeout_seconds,
            max_bytes=max_bytes,
            allow_non_html=allow_non_html,
            allowed_content_types=allowed_content_types,
            user_agent=user_agent,
        )
        return await self.svc.fetch_page(url, options=options)

    async def search_and_fetch(
        self,
        query: str,
        *,
        top_k: int = 5,
        engine: WebSearchEngine | None = None,
        site: str | None = None,
        freshness_days: int | None = None,
        language: str | None = None,
        safe_search: bool = False,
        extra: Mapping[str, Any] | None = None,
        timeout_seconds: float = 10.0,
        max_bytes: int = 2_000_000,
        allow_non_html: bool = False,
        allowed_content_types: Sequence[str] | None = None,
        user_agent: str | None = None,
    ) -> list[tuple[WebSearchHit, WebPageContent]]:
        """
        Search the web and fetch each returned hit sequentially.

        This method builds both option objects, runs `svc.search(...)`, then
        fetches each hit URL through `svc.fetch_page(...)`, returning paired
        `(hit, page)` tuples in hit order.

        Examples:
            End-to-end retrieval for top results:
            ```python
            pairs = await web.search_and_fetch(
                "zero downtime deploy checklist",
                top_k=3,
            )
            ```

            Domain-constrained search with fetch controls:
            ```python
            pairs = await web.search_and_fetch(
                "release notes",
                site="github.com",
                freshness_days=30,
                timeout_seconds=6.0,
                max_bytes=1_000_000,
            )
            ```

        Args:
            query: Search query string.
            top_k: Maximum number of hits to search before fetching.
            engine: Optional engine override.
            site: Optional domain constraint mapped to search options.
            freshness_days: Optional recency window in days.
            language: Optional language hint for search.
            safe_search: Whether to request safe-search filtering.
            extra: Provider-specific search passthrough options.
            timeout_seconds: Fetch timeout for each hit URL.
            max_bytes: Maximum bytes read per fetched response.
            allow_non_html: Whether non-HTML content is allowed when fetching.
            allowed_content_types: Optional fetch allowlist for content types.
            user_agent: Optional custom user-agent for fetching.

        Returns:
            list[tuple[WebSearchHit, WebPageContent]]: Ordered `(hit, page)`
                pairs for each successfully fetched search result.

        Notes:
            Fetches are currently sequential, not parallel.
        """
        search_opts = WebSearchOptions(
            top_k=top_k,
            language=language,
            region=None,
            freshness_days=freshness_days,
            site_filter=site,
            safe_search=safe_search,
            extra=extra,
        )
        fetch_opts = WebPageFetchOptions(
            timeout_seconds=timeout_seconds,
            max_bytes=max_bytes,
            allow_non_html=allow_non_html,
            allowed_content_types=allowed_content_types,
            user_agent=user_agent,
        )
        hits = await self.svc.search(query, engine=engine, options=search_opts)
        pages: list[tuple[WebSearchHit, WebPageContent]] = []
        for h in hits:
            page = await self.svc.fetch_page(h.url, options=fetch_opts)
            pages.append((h, page))
        return pages
