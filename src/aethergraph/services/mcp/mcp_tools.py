from aethergraph import tool, NodeContext
from services.mcp.helpers import mcp_call_logged

@tool(outputs=["result"], name="mcp.call", version="0.1.0")
async def mcp_call(server: str, tool_name: str, args: dict | None = None, *, context: NodeContext):
    out = await mcp_call_logged(context, server, tool_name, args or {})
    await context.channel().send_text(f"🔌 MCP {server}:{tool_name} ✓")
    return {"result": out}

@tool(outputs=["tools"], name="mcp.list_tools", version="0.1.0")
async def mcp_list_tools(server: str, *, context: NodeContext):
    tools = await context.mcp(server).list_tools()
    await context.mem().write_result(topic=f"mcp.{server}.list_tools",
                                     outputs=[{"name":"tools","kind":"json","value":tools}],
                                     tags=["mcp"])
    return {"tools": tools}
