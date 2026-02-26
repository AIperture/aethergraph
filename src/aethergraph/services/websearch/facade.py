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
    Thin facade to be exposed as `context.web`.

    You will construct this in NodeContext wiring, e.g.:

        context.web = WebSearchFacade(web_search_service, scope)

    where `scope` is optional if you later want scoped caching / KB writes.
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
