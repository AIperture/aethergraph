from typing import Dict
from contracts.services.mcp import MCPClientProtocol

class MCPRegistry:
    def __init__(self): self._clients: Dict[str, MCPClientProtocol] = {}
    def register(self, name: str, client: MCPClientProtocol): self._clients[name] = client
    def get(self, name: str) -> MCPClientProtocol:
        if name not in self._clients: raise KeyError(f"MCP server '{name}' not registered")
        return self._clients[name]


