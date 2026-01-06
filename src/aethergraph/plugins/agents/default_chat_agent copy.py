# aethergraph/examples/agents/default_chat_agent.py (or similar)

from __future__ import annotations

from typing import Any

from aethergraph import NodeContext, graph_fn
from aethergraph.plugins.agents.shared import build_session_memory_prompt_segments


@graph_fn(
    name="default_chat_agent",
    inputs=["message", "files", "session_id", "user_meta"],
    outputs=["reply"],
    as_agent={
        "id": "chat_agent",
        "title": "Chat",
        "description": "Built-in chat agent that uses the configured LLM.",
        "icon": "message-circle",
        "color": "sky",
        "session_kind": "chat",
        "mode": "chat_v1",
        "memory_level": "session",
        "memory_scope": "session.global",
    },
)
async def default_chat_agent(
    message: str,
    files: list[Any] | None = None,
    session_id: str | None = None,
    user_meta: dict[str, Any] | None = None,
    context_refs: list[dict[str, Any]] | None = None,
    *,
    context: NodeContext,
):
    """
    Simple built-in chat agent:

    - Takes {message, files}
    - Calls the configured LLM
    - Uses shared session memory (summary + recent events) in the prompt.
    """

    llm = context.llm()
    chan = context.ui_session_channel()

    # 1) Build memory segments for this session
    session_summary, recent_events = await build_session_memory_prompt_segments(
        context,
        summary_tag="session",
        recent_limit=12,
    )

    # print("Session summary:", session_summary)
    # print("Recent events:", recent_events)

    # 2) System + user messages (you can move this into PromptStore later)
    system_prompt = (
        "You are AetherGraph's built-in session helper.\n\n"
        "You can see a short summary of the session and a few recent events from all agents.\n"
        "Use them to answer questions about previous steps or runs, but do not invent details.\n"
        "If you are unsure, say that clearly.\n"
        "When return math or code snippets, use markdown formatting.\n"
        "Math formatting rules:\n"
        "- Use LaTeX math delimiters:\n"
        "  - Inline: \\( ... \\)  (no extra spaces right after \\( or before \\))\n"
        "  - Display: $$ ... $$  (for standalone equations)\n"
    )

    memory_context = ""
    if session_summary:
        memory_context += f"Session summary:\n{session_summary}\n\n"
    if recent_events:
        memory_context += f"Recent events:\n{recent_events}\n\n"

    user_prompt = f"{memory_context}" "User message:\n" f"{message}\n"

    # 3) Call LLM with chat-style API
    resp, _usage = await llm.chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    await chan.send_text(resp)

    return {
        "reply": resp,
    }
