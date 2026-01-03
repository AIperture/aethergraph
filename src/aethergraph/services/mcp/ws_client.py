from __future__ import annotations

import asyncio
import contextlib
import json
from typing import TYPE_CHECKING, Any

import websockets

if TYPE_CHECKING:
    # only imported for static type checkers; no runtime import, no warning
    from websockets.client import WebSocketClientProtocol

from aethergraph.contracts.services.mcp import MCPClientProtocol, MCPResource, MCPTool


class WsMCPClient(MCPClientProtocol):
    """
    Initialize the WebSocket MCP client with URL, headers, and connection parameters.
    This class manages a WebSocket connection to an MCP (Modular Control Protocol) server,
    handling connection lifecycle, pinging for keepalive, and concurrency for sending JSON-RPC
    requests. It provides methods to list tools, call tools, list resources, and read resources
    via the MCP protocol.

    Examples:
        Basic usage with default headers:
        ```python
        from aethergraph.services.mcp import WsMCPClient
        client = WsMCPClient("wss://mcp.example.com/ws")
        await client.open()
        tools = await client.list_tools()
        await client.close()
        ```

        Custom headers and ping interval:
        ```python
        from aethergraph.services.mcp import WsMCPClient
        client = WsMCPClient(
            "wss://mcp.example.com/ws",
            headers={"Authorization": "Bearer <token>"},
            ping_interval=10.0,
            timeout=30.0
        await client.open()
        ```

    Args:
        url: The WebSocket URL of the MCP server (e.g., "wss://mcp.example.com/ws").
        headers: Optional dictionary of additional HTTP headers to include in the WebSocket handshake.
        timeout: Maximum time (in seconds) to wait for connection and responses.
        ping_interval: Interval (in seconds) between WebSocket ping frames for keepalive.
        ping_timeout: Maximum time (in seconds) to wait for a ping response before considering the connection dead.

    Returns:
        None: Initializes the WsMCPClient instance and prepares internal state.
    """

    def __init__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
        ping_interval: float = 20.0,
        ping_timeout: float = 10.0,
    ):
        self.url = url
        self.headers = headers or {}
        self.timeout = timeout
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout

        self._ws: WebSocketClientProtocol | None = None
        self._id = 0
        self._lock = asyncio.Lock()
        self._ping_task: asyncio.Task | None = None

    async def open(self):
        if self._ws and not self._ws.closed:
            return
        try:
            # websockets >=14
            self._ws = await websockets.connect(
                self.url, additional_headers=self.headers, open_timeout=self.timeout
            )
        except TypeError:
            # likely on websockets <=13
            self._ws = await websockets.connect(
                self.url, extra_headers=self.headers, open_timeout=self.timeout
            )

        self._start_ping()

    async def close(self):
        if self._ping_task:
            self._ping_task.cancel()
            with contextlib.suppress(Exception):
                await self._ping_task
            self._ping_task = None
        if self._ws and not self._ws.closed:
            await self._ws.close()
        self._ws = None

    def _start_ping(self):
        if self.ping_interval <= 0:
            return

        async def _pinger():
            try:
                while self._ws and not self._ws.closed:
                    await asyncio.sleep(self.ping_interval)
                    if not self._ws or self._ws.closed:
                        break
                    try:
                        await asyncio.wait_for(self._ws.ping(), timeout=self.ping_timeout)  # type: ignore
                    except Exception:
                        break
            except asyncio.CancelledError:
                pass

        self._ping_task = asyncio.create_task(_pinger())

    async def _ensure(self):
        if self._ws is None or self._ws.closed:
            await self.open()

    async def _rpc(self, method: str, params: dict[str, Any] | None = None) -> Any:
        await self._ensure()
        async with self._lock:
            self._id += 1
            req = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}}
            data = json.dumps(req)
            try:
                assert self._ws is not None
                await asyncio.wait_for(self._ws.send(data), timeout=self.timeout)
                raw = await asyncio.wait_for(self._ws.recv(), timeout=self.timeout)
            except Exception:
                await self.close()
                await self.open()
                assert self._ws is not None
                await asyncio.wait_for(self._ws.send(data), timeout=self.timeout)
                raw = await asyncio.wait_for(self._ws.recv(), timeout=self.timeout)
            resp = json.loads(raw)
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
