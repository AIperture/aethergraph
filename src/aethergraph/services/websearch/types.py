from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Protocol


class WebSearchEngine(str, Enum):
    default = "default"  # indirect; resolves to actual engine at runtime
    tavily = "tavily"
    brave = "brave"
    serpapi = "serpapi"
    bing = "bing"
    google = "google"
    custom = "custom"  # e.g. local search engine, or a custom API endpoint


@dataclass
class WebSearchOptions:
    top_k: int = 5
    language: str | None = None
    region: str | None = None
    freshness_days: int | None = None
    site_filter: str | None = None
    safe_search: bool = False

    # Agent hints
    prefer_explanatory: bool = True  # prefer results that include explanations or summaries
    prefer_primary_sources: bool = (
        True  # prefer results from primary sources (e.g. official websites, research papers)
    )

    # Provider-specific extras
    extra: Mapping[str, Any] | None = (
        None  # provider-specific options, e.g. for SerpAPI: { "serpapi_api_key": "my_api_key" }
    )


@dataclass
class WebSearchHit:
    engine: WebSearchEngine
    query: str
    rank: int
    score: float | None

    url: str
    title: str | None
    snippet: str | None

    published_at: datetime | None = None
    fetched_at: datetime | None = (
        None  # when the search hit was fetched by the search engine (not necessarily when it was published on the web)
    )

    metadata: Mapping[str, Any] | None = None


@dataclass
class WebPageFetchOptions:
    timeout_seconds: float = 10.0
    max_bytes: int = 10 * 1024 * 1024  # 10 MB
    follow_redirects: bool = True
    allow_non_html: bool = False  # whether to allow fetching non-HTML content (e.g. PDFs, images)
    user_agent: str | None = None  # custom user agent string to use when fetching the page
    allowed_content_types: Sequence[str] | None = (
        None  # if specified, only allow fetching pages with these content types (e.g. ["text/html", "application/pdf"])
    )


@dataclass
class WebPageContent:
    url: str
    final_url: str
    status_code: int
    content_type: str | None
    encoding: str | None

    raw_bytes: bytes | None
    text: str | None
    title: str | None
    main_content: str | None
    html: str | None

    fetched_at: datetime
    headers: Mapping[str, str] | None = None

    metadata: Mapping[str, Any] | None = None


class WebSearchProvider(Protocol):
    engine_name: WebSearchEngine

    async def search(self, query: str, options: WebSearchOptions) -> list[WebSearchHit]: ...


class WebPageFetcher(Protocol):
    async def fetch(self, url: str, options: WebPageFetchOptions) -> WebPageContent: ...
