from __future__ import annotations
import asyncio, json
from typing import Any, Dict, List, Optional
import httpx
from aethergraph.contracts.services.mcp import MCPClientProtocol, MCPTool, MCPResource

class HttpMCPClient(MCPClientProtocol):
    def __init__(
        self,
        base_url: str,
        *,
        headers: Dict[str, str] | None = None,
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

    async def _rpc(self, method: str, params: Dict[str, Any] | None = None) -> Any:
        await self._ensure()
        async with self._lock:
            self._id += 1
            req = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}}
            assert self._client is not None
            r = await self._client.post(f"{self.base_url}/rpc", headers=self.headers, content=json.dumps(req))
            r.raise_for_status()
            resp = r.json()
            if "error" in resp:
                raise RuntimeError(str(resp["error"]))
            return resp.get("result")

    async def list_tools(self) -> List[MCPTool]:
        return await self._rpc("tools/list")

    async def call(self, tool: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return await self._rpc("tools/call", {"name": tool, "arguments": params or {}})

    async def list_resources(self) -> List[MCPResource]:
        return await self._rpc("resources/list")

    async def read_resource(self, uri: str) -> Dict[str, Any]:
        return await self._rpc("resources/read", {"uri": uri})
