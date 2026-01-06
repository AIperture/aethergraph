import asyncio
import json
import os
from typing import Any

from aethergraph.contracts.services.mcp import MCPClientProtocol, MCPResource, MCPTool


class StdioMCPClient(MCPClientProtocol):
    """
    Initialize the MCP client service to communicate with a subprocess over stdio using JSON-RPC 2.0.

    This class launches a subprocess (typically an MCP server), manages its lifecycle, and provides
    asynchronous methods to interact with it using JSON-RPC 2.0 over standard input/output streams.
    It handles command execution, environment setup, request/response serialization, and concurrency
    control for safe multi-call usage.

    Examples:
        Basic usage with default environment:
        ```python
        from aethergraph.services.mcp import StdioMCPClient
        client = StdioMCPClient(["python", "mcp_server.py"])
        await client.open()
        tools = await client.list_tools()
        await client.close()
        ```

        Custom environment and timeout:
        ```python
        from aethergraph.services.mcp import StdioMCPClient
        client = StdioMCPClient(
            ["python", "mcp_server.py"],
            env={"MY_ENV_VAR": "value"},
            timeout=30.0
        )
        ```

    Args:
        cmd: Command to start the MCP server subprocess (list of str).
        env: Optional dictionary of environment variables for the subprocess.
        timeout: Timeout in seconds for each RPC call.

    Returns:
        None: Initializes the StdioMCPClient instance.

    Notes:
        - The subprocess should adhere to the JSON-RPC 2.0 specification over stdio.
        - Ensure proper error handling in the subprocess to avoid deadlocks.
    """

    def __init__(self, cmd: list[str], env: dict[str, str] | None = None, timeout: float = 60.0):
        self.cmd, self.env, self.timeout = cmd, env or {}, timeout
        self.proc = None
        self._id = 0
        self._lock = asyncio.Lock()

    async def open(self):
        if self.proc:
            return
        self.proc = await asyncio.create_subprocess_exec(
            *self.cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, **self.env},
        )

    async def close(self):
        if not self.proc:
            return
        try:
            self.proc.terminate()
        except Exception:
            self.logger.warning("mcp_stdio_terminate_failed")

        self.proc = None

    async def _rpc(self, method: str, params: dict[str, Any] | None = None) -> Any:
        await self.open()
        async with self._lock:
            self._id += 1
            req = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}}
            data = (json.dumps(req) + "\n").encode("utf-8")
            assert self.proc and self.proc.stdin and self.proc.stdout
            self.proc.stdin.write(data)
            await self.proc.stdin.drain()
            line = await asyncio.wait_for(self.proc.stdout.readline(), timeout=self.timeout)
            if not line:
                raise RuntimeError("MCP server closed")
            resp = json.loads(line.decode("utf-8"))
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
