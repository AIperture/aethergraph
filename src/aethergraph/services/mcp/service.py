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
        """
        Initialize the MCPService with optional clients and secrets provider.

        Examples:
            Basic usage with no clients:
            ```python
            service = MCPService()
            ```

            With pre-registered clients:
            ```python
            service = MCPService(clients={"default": my_client})
            ```

        Args:
            clients: Optional dictionary mapping names to MCPClientProtocol instances.
            secrets: Optional secrets provider (not implemented here).
        """
        self._clients: dict[str, MCPClientProtocol] = clients or {}
        self._secrets = secrets  # optional (Secrets provider) Not implemented here

    # ---- registration ----
    def register(self, name: str, client: MCPClientProtocol) -> None:
        """
        Register a new MCP client under a given name.

        Examples:
            ```python
            context.mcp().register("myserver", my_client)
            ```

        Args:
            name: The name to register the client under.
            client: The MCPClientProtocol instance to register.
        """
        self._clients[name] = client

    def remove(self, name: str) -> None:
        """
        Remove a registered MCP client by name.

        Examples:
            ```python
            context.mcp().remove("myserver")
            ```

        Args:
            name: The name of the client to remove.
        """
        self._clients.pop(name, None)

    def has(self, name: str) -> bool:
        """
        Check if a client with the given name is registered.

        Examples:
            ```python
            if context.mcp().has("default"):
                print("Client exists")
            ```

        Args:
            name: The name to check.

        Returns:
            bool: True if the client exists, False otherwise.
        """
        return name in self._clients

    def names(self) -> list[str]:
        """
        Get a list of all registered client names.

        Examples:
            ```python
            names = context.mcp().names()
            ```

        Returns:
            list[str]: List of registered client names.
        """
        return list(self._clients.keys())

    def list_clients(self) -> list[str]:
        """
        List all registered client names.

        Examples:
            ```python
            clients = context.mcp().list_clients()
            ```

        Returns:
            list[str]: List of registered client names.
        """
        return list(self._clients.keys())

    def get(self, name: str = "default") -> MCPClientProtocol:
        """
        Retrieve a registered MCP client by name.

        Examples:
            ```python
            client = context.mcp().get("default")
            ```

        Args:
            name: The name of the client to retrieve.

        Returns:
            MCPClientProtocol: The registered client.

        Raises:
            KeyError: If the client is not found.
        """
        if name not in self._clients:
            raise KeyError(f"Unknown MCP server '{name}'")
        return self._clients[name]

    # ---- lifecycle ----
    async def open(self, name: str) -> None:
        """
        Open the connection for a specific MCP client.

        Examples:
            ```python
            await context.mcp().open("default")
            ```

        Args:
            name: The name of the client to open.
        """
        await self.get(name).open()

    async def close(self, name: str) -> None:
        """
        Close the connection for a specific MCP client.

        Examples:
            ```python
            await context.mcp().close("default")
            ```

        Args:
            name: The name of the client to close.
        """
        try:
            await self.get(name).close()
        except Exception:
            logger.warning(f"Failed to close MCP client '{name}'")

    async def open_all(self) -> None:
        """
        Open all registered MCP client connections.

        Examples:
            ```python
            await context.mcp().open_all()
            ```

        """
        for n in self._clients:
            await self._clients[n].open()

    async def close_all(self) -> None:
        """
        Close all registered MCP client connections.

        Examples:
            ```python
            await context.mcp().close_all()
            ```

        """
        for n in self._clients:
            try:
                await self._clients[n].close()
            except Exception:
                logger.warning(f"Failed to close MCP client '{n}'")

    # ---- call helpers (optional, keeps call sites tiny) ----
    async def call(
        self, name: str, tool: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Call a tool on a specific MCP client, opening the connection if needed.

        Examples:
            ```python
            result = await context.mcp().call("default", "sum", {"a": 1, "b": 2})
            ```

        Args:
            name: The name of the client to use.
            tool: The tool name to call.
            params: Optional dictionary of parameters for the tool.

        Returns:
            dict[str, Any]: The result from the tool call.
        """
        # lazy-open on first use; clients themselves also lazy-reconnect
        c = self.get(name)
        await c.open()
        return await c.call(tool, params or {})

    async def list_tools(self, name: str) -> list[MCPTool]:
        """
        List all tools available on a specific MCP client.

        Examples:
            ```python
            tools = await context.mcp().list_tools("default")
            ```

        Args:
            name: The name of the client.

        Returns:
            list[MCPTool]: List of available tools.
        """
        c = self.get(name)
        await c.open()
        return await c.list_tools()

    async def list_resources(self, name: str) -> list[MCPResource]:
        """
        List all resources available on a specific MCP client.

        Examples:
            ```python
            resources = await context.mcp().list_resources("default")
            ```

        Args:
            name: The name of the client.

        Returns:
            list[MCPResource]: List of available resources.
        """
        c = self.get(name)
        await c.open()
        return await c.list_resources()

    async def read_resource(self, name: str, uri: str) -> dict[str, Any]:
        """
        Read a resource from a specific MCP client.

        Examples:
            ```python
            data = await context.mcp().read_resource("default", "resource://foo/bar")
            ```

        Args:
            name: The name of the client.
            uri: The URI of the resource to read.

        Returns:
            dict[str, Any]: The resource data.
        """
        c = self.get(name)
        await c.open()
        return await c.read_resource(uri)

    # ---- optional secrets helpers  ----
    def set_header(self, name: str, key: str, value: str) -> None:
        """
        Set or override a header for a websocket client at runtime.

        Examples:
            ```python
            context.mcp().set_header("default", "Authorization", "Bearer token")
            ```

        Args:
            name: The name of the client.
            key: The header key.
            value: The header value.

        Raises:
            RuntimeError: If the client does not support headers.
        """
        c = self.get(name)
        # duck-typing for ws client
        if hasattr(c, "headers") and isinstance(c.headers, dict):  # type: ignore[attr-defined]
            c.headers[key] = value  # type: ignore[attr-defined]
        else:
            raise RuntimeError(f"MCP '{name}' does not support headers")

    def persist_secret(self, secret_name: str, value: str) -> None:
        """
        Persist a secret using the configured secrets provider.

        Examples:
            ```python
            context.mcp().persist_secret("API_KEY", "my-secret-value")
            ```

        Args:
            secret_name: The name of the secret.
            value: The value to persist.

        Raises:
            RuntimeError: If the secrets provider is not writable.
        """
        if not self._secrets or not hasattr(self._secrets, "set"):
            raise RuntimeError("Secrets provider is not writable")
        self._secrets.set(secret_name, value)  # type: ignore
