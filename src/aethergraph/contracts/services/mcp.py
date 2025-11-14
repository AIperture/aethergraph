from typing import Any, TypedDict


class MCPTool(TypedDict):
    name: str
    description: str | None
    input_schema: dict[str, Any] | None


class MCPResource(TypedDict):
    uri: str
    mime: str | None
    description: str | None


class MCPClientProtocol:
    async def open(self): ...
    async def close(self): ...
    async def list_tools(self) -> list[MCPTool]: ...
    async def call(self, tool: str, params: dict[str, Any] | None = None) -> dict[str, Any]: ...
    async def list_resources(self) -> list[MCPResource]: ...
    async def read_resource(self, uri: str) -> dict[str, Any]: ...
