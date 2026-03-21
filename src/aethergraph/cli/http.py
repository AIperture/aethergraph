from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from aethergraph.server.server_state import get_running_url_if_any


def resolve_server_base_url(*, workspace: str, server_url: str | None = None) -> str:
    return server_url or get_running_url_if_any(workspace) or "http://127.0.0.1:8745"


def post_json(
    url: str, payload: dict, headers: dict[str, str] | None = None, timeout: int = 20
) -> dict:
    req = Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def get_json(url: str, timeout: int = 10) -> dict:
    req = Request(url=url, method="GET")
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def format_http_error(exc: HTTPError | URLError) -> str:
    if isinstance(exc, HTTPError):
        return exc.read().decode("utf-8")
    return str(exc)
