from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any

import httpx

from aethergraph.services.websearch.types import (
    WebSearchEngine,
    WebSearchHit,
    WebSearchOptions,
    WebSearchProvider,
)


class TaviliyWebSearchProvider(WebSearchProvider):
    """
    Minimal Tavily search provider.

    Requires TAVILY_API_KEY in env. You can extend `extra` to pass Tavily-specific
    options like search depth, topic, etc.
    """

    engine_name: WebSearchEngine = WebSearchEngine.tavily

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self._api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "TAVILY_API_KEY environment variable is required for TavilyWebSearchProvider"
            )
        self._base_url = base_url or "https://api.tavily.com/search"

    async def search(
        self,
        query: str,
        options: WebSearchOptions,
    ) -> list[WebSearchHit]:
        payload: dict[str, Any] = {
            "api_key": self._api_key,
            "query": query,
            "max_results": options.top_k,
            # Tavily-specific defaults; tweak as you like:
            "search_depth": "basic",
            "include_answer": False,
            "include_images": False,
            "include_raw_content": False,
        }

        if options.freshness_days is not None:
            # Tavily supports "days" under some plans; if not, drop it.
            payload["days"] = options.freshness_days

        if options.site_filter:
            payload["topic"] = "general"
            payload["include_domains"] = [options.site_filter]

        # Merge provider-specific extras
        if options.extra:
            payload.update(dict(options.extra))

        async with httpx.AsyncClient(
            timeout=options.extra.get("timeout", 15) if options.extra else 15
        ) as client:
            resp = await client.post(self._base_url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        results_raw = data.get("results", [])
        now = datetime.now(timezone.utc)
        hits: list[WebSearchHit] = []

        for rank, r in enumerate(results_raw, start=1):
            url = r.get("url") or ""
            title = r.get("title")
            snippet = r.get("content") or r.get("snippet")
            # Tavily may provide "published_date" or similar; parse if present.
            published_at: datetime | None = None
            published_raw = r.get("published_date") or r.get("date")
            if isinstance(published_raw, str):
                try:
                    published_at = datetime.fromisoformat(published_raw.replace("Z", "+00:00"))
                except Exception:
                    published_at = None

            hits.append(
                WebSearchHit(
                    engine=self.engine_name,
                    query=query,
                    rank=rank,
                    score=None,  # Tavily doesn't expose a direct numeric score
                    url=url,
                    title=title,
                    snippet=snippet,
                    published_at=published_at,
                    fetched_at=now,
                    metadata=r,
                )
            )

        return hits
