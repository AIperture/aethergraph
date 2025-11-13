import asyncio, json, sys, os
from typing import Any, Dict, List, Optional
from aethergraph.contracts.services.mcp import MCPClientProtocol, MCPTool, MCPResource

class StdioMCPClient(MCPClientProtocol):
    def __init__(self, cmd: list[str], env: dict[str, str] | None = None, timeout: float = 60.0):
        """MCP client that talks to a subprocess over stdio using JSON-RPC 2.0.
         Args:
             cmd: Command to start the MCP server subprocess (list of str).
             env: Optional environment variables to set for the subprocess.
             timeout: Timeout in seconds for each RPC call.
         """
        self.cmd, self.env, self.timeout = cmd, env or {}, timeout
        self.proc = None
        self._id = 0
        self._lock = asyncio.Lock()
    
    async def open(self):
        if self.proc: return
        self.proc = await asyncio.create_subprocess_exec(
            *self.cmd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, env={**os.environ, **self.env}
        )

    async def close(self):
        if not self.proc: return
        try:
            self.proc.terminate()
        except: pass
        self.proc = None


    async def _rpc(self, method: str, params: Dict[str, Any] | None = None) -> Any:
        await self.open()
        async with self._lock:
            self._id += 1
            req = {"jsonrpc":"2.0","id":self._id,"method":method,"params":params or {}}
            data = (json.dumps(req) + "\n").encode("utf-8")
            assert self.proc and self.proc.stdin and self.proc.stdout
            self.proc.stdin.write(data); await self.proc.stdin.drain()
            line = await asyncio.wait_for(self.proc.stdout.readline(), timeout=self.timeout)
            if not line: raise RuntimeError("MCP server closed")
            resp = json.loads(line.decode("utf-8"))
            if "error" in resp: raise RuntimeError(str(resp["error"]))
            return resp.get("result")
        
    async def list_tools(self) -> List[MCPTool]:
        return await self._rpc("tools/list")

    async def call(self, tool: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return await self._rpc("tools/call", {"name": tool, "arguments": params or {}})

    async def list_resources(self) -> List[MCPResource]:
        return await self._rpc("resources/list")

    async def read_resource(self, uri: str) -> Dict[str, Any]:
        return await self._rpc("resources/read", {"uri": uri})