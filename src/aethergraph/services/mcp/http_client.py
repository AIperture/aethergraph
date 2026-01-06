from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from aethergraph.contracts.services.mcp import MCPClientProtocol, MCPResource, MCPTool


class HttpMCPClient(MCPClientProtocol):
    """
    Initialize the HTTP client service with base URL, headers, and timeout.

    This constructor sets up the base URL for all requests, applies default and custom headers,
    and configures the request timeout. It also initializes internal state for the asynchronous
    HTTP client and concurrency control.

    Examples:
        Basic usage with default headers:
        ```python
        from aethergraph.services.mcp import HttpMCPClient
        client = HttpMCPClient("https://api.example.com")
        ```

        Custom headers and timeout:
        ```python
        from aethergraph.services.mcp import HttpMCPClient
        client = HttpMCPClient(
            "https://api.example.com",
            headers={"Authorization": "Bearer <token>"},
            timeout=30.0
        )
        ```

    Args:
        base_url: The root URL for all HTTP requests (e.g., "https://api.example.com").
        headers: Optional dictionary of additional HTTP headers to include with each request.
        The "Content-Type: application/json" header is always set by default.
        timeout: The maximum time (in seconds) to wait for a response before timing out.

    Returns:
        None: Initializes the HttpMCPClient instance.

    Notes:
        - Ensure that the base_url does not have a trailing slash; it will be added automatically.
        - The client uses asynchronous HTTP requests for non-blocking operations.
    """

    def __init__(
        self,
        base_url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        if headers:
            self.headers.update(headers)
        self.timeout = timeout

        self._client: httpx.AsyncClient | None = None
        self._id = 0
        self._lock = asyncio.Lock()

    async def open(self):
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)

    async def close(self):
        if self._client is not None:
            try:
                await self._client.aclose()
            finally:
                self._client = None

    async def _ensure(self):
        if self._client is None:
            await self.open()

    async def _rpc(self, method: str, params: dict[str, Any] | None = None) -> Any:
        await self._ensure()
        async with self._lock:
            self._id += 1
            req = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}}
            assert self._client is not None
            r = await self._client.post(
                f"{self.base_url}/rpc", headers=self.headers, content=json.dumps(req)
            )
            r.raise_for_status()
            resp = r.json()
            if "error" in resp:
                raise RuntimeError(str(resp["error"]))
            return resp.get("result")

    async def list_tools(self) -> list[MCPTool]:
        return await self._rpc("tools/list")

    async def call(self, tool: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self._rpc("tools/call", {"name": tool, "arguments": params or {}})

    async def list_resources(self) -> list[MCPResource]:
        return await self._rpc("resources/list")

    async def read_resource(self, uri: str) -> dict[str, Any]:
        return await self._rpc("resources/read", {"uri": uri})
