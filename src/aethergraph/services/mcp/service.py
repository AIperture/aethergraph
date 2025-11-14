from __future__ import annotations

import logging
from typing import Any

from aethergraph.contracts.services.mcp import MCPClientProtocol, MCPResource, MCPTool

logger = logging.getLogger("aethergraph.services.mcp")


class MCPService:
    """
    Holds many MCP clients (stdio/ws) under names, manages lifecycle, and
    provides thin convenience helpers (open/close/call/list_tools).
    """

    def __init__(self, clients: dict[str, MCPClientProtocol] | None = None, *, secrets=None):
        self._clients: dict[str, MCPClientProtocol] = clients or {}
        self._secrets = secrets  # optional (Secrets provider) Not implemented here

    # ---- registration ----
    def register(self, name: str, client: MCPClientProtocol) -> None:
        self._clients[name] = client

    def remove(self, name: str) -> None:
        self._clients.pop(name, None)

    def has(self, name: str) -> bool:
        return name in self._clients

    def names(self) -> list[str]:
        return list(self._clients.keys())

    def list_clients(self) -> list[str]:
        return list(self._clients.keys())

    def get(self, name: str = "default") -> MCPClientProtocol:
        if name not in self._clients:
            raise KeyError(f"Unknown MCP server '{name}'")
        return self._clients[name]

    # ---- lifecycle ----
    async def open(self, name: str) -> None:
        await self.get(name).open()

    async def close(self, name: str) -> None:
        try:
            await self.get(name).close()
        except Exception:
            logger.warning(f"Failed to close MCP client '{name}'", exc_info=True)

    async def open_all(self) -> None:
        for n in self._clients:
            await self._clients[n].open()

    async def close_all(self) -> None:
        for n in self._clients:
            try:
                await self._clients[n].close()
            except Exception:
                logger.warning(f"Failed to close MCP client '{n}'", exc_info=True)

    # ---- call helpers (optional, keeps call sites tiny) ----
    async def call(
        self, name: str, tool: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        # lazy-open on first use; clients themselves also lazy-reconnect
        c = self.get(name)
        await c.open()
        return await c.call(tool, params or {})

    async def list_tools(self, name: str) -> list[MCPTool]:
        c = self.get(name)
        await c.open()
        return await c.list_tools()

    async def list_resources(self, name: str) -> list[MCPResource]:
        c = self.get(name)
        await c.open()
        return await c.list_resources()

    async def read_resource(self, name: str, uri: str) -> dict[str, Any]:
        c = self.get(name)
        await c.open()
        return await c.read_resource(uri)

    # ---- optional secrets helpers  ----
    def set_header(self, name: str, key: str, value: str) -> None:
        """For ws clients: set/override a header at runtime (demo/notebook UX)."""
        c = self.get(name)
        # duck-typing for ws client
        if hasattr(c, "headers") and isinstance(c.headers, dict):  # type: ignore[attr-defined]
            c.headers[key] = value  # type: ignore[attr-defined]
        else:
            raise RuntimeError(f"MCP '{name}' does not support headers")

    def persist_secret(self, secret_name: str, value: str) -> None:
        if not self._secrets or not hasattr(self._secrets, "set"):
            raise RuntimeError("Secrets provider is not writable")
        self._secrets.set(secret_name, value)  # type: ignore
