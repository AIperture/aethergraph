from __future__ import annotations
from aethergraph.core.runtime.node_context import NodeContext

async def mcp_call_logged(context: NodeContext, server: str, tool: str, params: dict | None = None):
    client = context.mcp(server)
    res = await client.call(tool, params or {})
    try:
        await context.mem().write_result(
            topic=f"mcp.{server}.{tool}",
            inputs=[{"name":"args","kind":"json","value":params or {}}],
            outputs=[{"name":"result","kind":"json","value":res}],
            tags=["mcp","tool_call"],
            message=f"MCP {server}:{tool}"
        )
    except Exception:
        pass
    return res
