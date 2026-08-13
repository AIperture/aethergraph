"""Shared asynchronous provider HTTP-client lifecycle helpers."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx


def _ensure_loop_http_client(
    client: Any | None,
    bound_loop: Any | None,
    *,
    timeout: float,
) -> tuple[Any, Any, Any | None]:
    """Bind one HTTP client to the current event loop without cross-loop close."""

    loop = asyncio.get_running_loop()
    if client is None:
        return httpx.AsyncClient(timeout=timeout), loop, None
    if bound_loop is None:
        return client, loop, None
    if bound_loop is not loop:
        return httpx.AsyncClient(timeout=timeout), loop, client
    return client, bound_loop, None


async def _close_http_clients(
    clients: list[Any | None],
    *,
    logger: logging.Logger,
    warning_key: str,
) -> None:
    """Close each distinct reachable HTTP client on a best-effort basis."""

    seen: set[int] = set()
    for client in clients:
        if client is None or id(client) in seen:
            continue
        seen.add(id(client))
        try:
            await client.aclose()
        except RuntimeError as exc:
            logger.warning("%s: %s", warning_key, exc)
