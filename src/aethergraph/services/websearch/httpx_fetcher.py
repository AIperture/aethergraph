from __future__ import annotations

from datetime import datetime, timezone

import httpx

from aethergraph.services.websearch.types import WebPageContent, WebPageFetcher, WebPageFetchOptions


class HttpxWebPageFetcher(WebPageFetcher):
    """
    Simple HTTP-based page fetcher.

    - Uses httpx.AsyncClient
    - Decodes text using httpx's encoding detection
    - Does NOT do JS execution (no Playwright)
    - `main_content` currently just equals `text` for HTML/text.
    """

    def __init__(self, default_user_agent: str | None = None) -> None:
        self._default_user_agent = (
            default_user_agent or "AetherGraph-WebFetcher/0.1 (+https://example.com/)"
        )

    async def fetch(
        self,
        url: str,
        options: WebPageFetchOptions | None = None,
    ) -> WebPageContent:
        opts = options or WebPageFetchOptions()
        headers = {
            "User-Agent": opts.user_agent or self._default_user_agent,
            "Accept": "*/*",
        }

        async with httpx.AsyncClient(
            timeout=opts.timeout_seconds,
            follow_redirects=opts.follow_redirects,
            headers=headers,
        ) as client:
            resp = await client.get(url)

        fetched_at = datetime.now(timezone.utc)
        final_url = str(resp.url)
        status_code = resp.status_code
        content_type = resp.headers.get("Content-Type")
        encoding = resp.encoding

        raw = resp.content or b""
        if len(raw) > opts.max_bytes:
            raw = raw[: opts.max_bytes]

        # Filter by allowed content types if configured
        if opts.allowed_content_types is not None and content_type is not None:
            # "text/html; charset=utf-8" -> "text/html"
            ct_main = content_type.split(";", 1)[0].strip().lower()
            allowed = {c.split(";", 1)[0].strip().lower() for c in opts.allowed_content_types}
            if ct_main not in allowed:
                # Not allowed; we still return limited info
                return WebPageContent(
                    url=url,
                    final_url=final_url,
                    status_code=status_code,
                    content_type=content_type,
                    encoding=encoding,
                    raw_bytes=None,
                    text=None,
                    title=None,
                    main_content=None,
                    html=None,
                    fetched_at=fetched_at,
                    headers=dict(resp.headers),
                    metadata={"reason": "content_type_not_allowed"},
                )

        # Decide how to handle content
        text: str | None = None
        html: str | None = None
        main_content: str | None = None
        title: str | None = None

        ct_main = content_type.split(";", 1)[0].strip().lower() if content_type else ""

        if ct_main in ("text/html", "text/xhtml", "application/xhtml+xml"):
            try:
                text = raw.decode(encoding or "utf-8", errors="replace")
            except Exception:
                text = raw.decode("utf-8", errors="replace")

            html = text
            main_content = text  # TODO: integrate readability / trafilatura later

            # naive title extraction
            title = _extract_html_title(text)

        elif ct_main.startswith("text/"):
            try:
                text = raw.decode(encoding or "utf-8", errors="replace")
            except Exception:
                text = raw.decode("utf-8", errors="replace")
            main_content = text
            html = None
            title = None
        else:
            # Non-text content (PDF, image, etc.)
            if opts.allow_non_html:
                # We just keep raw bytes; caller can decide how to handle.
                text = None
                main_content = None
                html = None
            else:
                # Strip content if non-html is not allowed
                raw = b""
                text = None
                main_content = None
                html = None

        return WebPageContent(
            url=url,
            final_url=final_url,
            status_code=status_code,
            content_type=content_type,
            encoding=encoding,
            raw_bytes=raw if raw else None,
            text=text,
            title=title,
            main_content=main_content,
            html=html,
            fetched_at=fetched_at,
            headers=dict(resp.headers),
            metadata={},
        )


def _extract_html_title(html: str) -> str | None:
    """
    Very naive <title> extractor; avoids adding BeautifulSoup dependency.
    """
    lowered = html.lower()
    start = lowered.find("<title>")
    if start == -1:
        return None
    end = lowered.find("</title>", start)
    if end == -1:
        return None
    # start+7 to skip "<title>"
    raw_title = html[start + len("<title>") : end]
    return raw_title.strip() or None
