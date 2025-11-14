# aethergraph/net/http.py
from contextlib import asynccontextmanager

import httpx


@asynccontextmanager
async def get_async_client(timeout_s: float = 10.0, headers: dict | None = None):
    async with httpx.AsyncClient(timeout=timeout_s, headers=headers) as client:
        yield client
