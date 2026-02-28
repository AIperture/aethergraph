from __future__ import annotations

from typing import Any

from aethergraph import graph_fn
from aethergraph.core.runtime.node_context import NodeContext
from aethergraph.plugins.agents.agent_functions.builder_chat import builder_chat_handler

# ---------------------------------------------------------------------------
# Agent Builder agent graph
# ---------------------------------------------------------------------------


@graph_fn(
    name="agent_builder_agent",
    inputs=["message", "files", "context_refs", "session_id", "user_meta"],
    outputs=["reply"],
    as_agent={
        "id": "agent_builder",
        "title": "Agent & App Builder",
        "short_description": "Turns your intent or scripts into AetherGraph agents/apps.",
        "description": (
            "Given a high-level description or existing Python scripts, "
            "this agent proposes AetherGraph @graph_fn/@graphify graphs and "
            "registerable agents/apps with as_agent/as_app metadata."
        ),
        "icon_key": "hammer",
        "color": "orange",
        "mode": "chat_v1",
        "memory_level": "user",
    },
)
async def agent_builder_agent(
    message: str,
    files: list[Any] | None = None,
    context_refs: list[dict[str, Any]] | None = None,
    session_id: str | None = None,
    user_meta: dict[str, Any] | None = None,
    *,
    context: NodeContext,
) -> dict:
    """
    Agent & App Builder.

    Entry points:
      1. User explicitly selects this agent in the UI and describes what they want.
      2. builtin_agent decides to route here for agent/app-building queries.

    For now this agent:
      - Uses builder_chat_handler() to talk to the LLM with a builder-specific skill.
      - Streams the response to the UI (code + explanation).
      - Records the turn in memory for future context.
    """
    logger = context.logger()
    mem = context.memory()

    raw_message = (message or "").strip()
    logger.debug(
        "agent_builder_agent received message: %r (session_id=%s)",
        raw_message,
        session_id,
    )

    # Simple greeting when nothing is provided
    if not raw_message and not (files or []):
        reply = (
            "Hi! I'm the Agent & App Builder.\n\n"
            "- Tell me what kind of AetherGraph agent or app you want (e.g. "
            '"a chat agent that summarizes CSVs"), **or**\n'
            "- Attach a Python script and say how you want it wrapped "
            '(e.g. "wrap this as a no_input_v1 app").'
        )
        await mem.record_chat_assistant(
            text=reply,
            tags=["ag.agent_reply", "ag.agent_builder.reply"],
        )
        return {"reply": reply}

    # Main builder path
    reply = await builder_chat_handler(
        message=raw_message,
        files=files or [],
        context_refs=context_refs or [],
        session_id=session_id,
        user_meta=user_meta or {},
        context=context,
    )

    # Record user turn AFTER retrieval/LLM so it doesn't pollute retrieval for this step
    if raw_message:
        await mem.record_chat_user(
            text=raw_message,
            tags=["ag.user_message", "ag.agent_builder.user"],
        )

    # Record assistant reply for later reference
    await mem.record_chat_assistant(
        text=reply,
        tags=["ag.agent_reply", "ag.agent_builder.reply"],
    )

    return {"reply": reply}
