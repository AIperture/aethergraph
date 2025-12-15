# aethergraph/examples/agents/default_chat_agent.py (or similar)

from __future__ import annotations

from typing import Any

from aethergraph import NodeContext, graph_fn


@graph_fn(
    name="default_chat_agent",
    inputs=["message", "files", "session_id", "user_meta"],
    outputs=["reply"],
    as_agent={
        "id": "chat_agent",  # <- canonical agent_id
        "title": "Chat",
        "description": "Built-in chat agent that uses the configured LLM.",
        "icon": "message-circle",
        "color": "sky",
        "session_kind": "chat",
        "mode": "chat_v1",  # <- your chat schema {message, files}
        # optional: "tool_graphs": [...],
    },
)
async def default_chat_agent(
    message: str,
    files: list[Any] | None = None,
    session_id: str | None = None,
    user_meta: dict[str, Any] | None = None,
    *,
    context: NodeContext,
):
    """
    Simple built-in chat agent:

    - Takes {message, files}
    - Calls the configured LLM
    - Returns a single `reply` string

    Later we can make this agent smart enough to spawn runs, look up memory, etc.
    """

    # Adjust this depending on how your ctx exposes the LLM client.
    llm = context.llm()
    prompt = (
        "You are AetherGraph's built-in helper. "
        "Answer concisely and be explicit when you don't know something.\n\n"
        f"User message:\n{message}\n"
    )

    # Adapt this call to your actual GenericLLMClient API
    resp, _ = await llm.chat(
        messages=[{"role": "user", "content": prompt}],
        # model=None -> let your client pick default, or pass a default model name
    )

    channel_key = "ui:session/" + (session_id or "unknown")
    chan = context.channel(channel_key=channel_key)
    await chan.send_text(resp)
    return {
        "kind": "reply_only",
        "reply": resp,
    }
