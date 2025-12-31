# aethergraph/examples/agents/default_chat_agent.py (or similar)

from __future__ import annotations

from typing import Any

from aethergraph import NodeContext, graph_fn


@graph_fn(
    name="default_chat_agent",
    inputs=["message", "files", "session_id", "user_meta"],
    outputs=["reply"],
    as_agent={
        "id": "chat_agent",
        "title": "Chat",
        "short_description": "General-purpose chat agent.",
        "description": "Built-in chat agent that uses the configured LLM and memory across sessions.",
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
    Built-in chat agent with session memory:

    - Hydrates long-term + recent chat memory into the prompt.
    - Records user and assistant messages as chat.turn events.
    - Periodically distills chat history into long-term summaries.
    """

    llm = context.llm()
    chan = context.ui_session_channel()

    mem = context.memory()

    # 1) Build memory segments for this session
    long_term_summary: str = ""
    recent_chat: list[dict[str, Any]] = []

    """
    Build prompt segments:    
    {
        "long_term": "<combined summary text or ''>",
        "recent_chat": [ {ts, role, text, tags}, ... ],
        "recent_tools": [ {ts, tool, message, inputs, outputs, tags}, ... ]
    }
    """
    segments = await mem.build_prompt_segments(
        recent_chat_limit=20,
        include_long_term=True,
        summary_tag="session",
        max_summaries=3,
        include_recent_tools=False,
    )
    long_term_summary = segments.get("long_term") or ""
    recent_chat = segments.get("recent_chat") or []

    # 2) System prompt
    system_prompt = (
        "You are AetherGraph's built-in session helper.\n\n"
        "You can see a summary of the session and some recent messages.\n"
        "Use them to answer questions about previous steps or runs, but do not invent details.\n"
        "If you are unsure, say that clearly.\n"
        # "When returning math or code snippets, use markdown formatting.\n"
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
    ]

    # Inject long-term summary as a system message (if present)
    if long_term_summary:
        messages.append(
            {
                "role": "system",
                "content": "Summary of previous context:\n" + long_term_summary,
            }
        )

    # Inject recent chat as prior turns
    for item in recent_chat:
        role = item.get("role") or "user"
        text = item.get("text") or ""
        # Map non-standard roles (e.g. "tool") to "assistant" for chat APIs
        mapped_role = role if role in {"user", "assistant", "system"} else "assistant"
        if text:
            messages.append({"role": mapped_role, "content": text})

    # Add some lightweight metadata about files / context refs into the user message
    meta_lines: list[str] = []
    if files:
        meta_lines.append(f"(User attached {len(files)} file(s).)")
    if context_refs:
        meta_lines.append(f"(User attached {len(context_refs)} context reference(s).)")
    meta_block = ""
    if meta_lines:
        meta_block = "\n\n" + "\n".join(meta_lines)

    user_content = f"{message}{meta_block}"

    # 3) Record the user message into memory
    user_data: dict[str, Any] = {}
    if files:
        # Store only lightweight file metadata; avoid huge payloads
        user_data["files"] = [
            {k: v for k, v in (f or {}).items() if k in {"name", "url", "mimetype", "size"}}
            for f in files
        ]
    if context_refs:
        user_data["context_refs"] = context_refs

    await mem.record_chat_user(
        message,
        data=user_data,
        tags=["session.chat"],
    )

    # Append current user message to LLM prompt
    messages.append({"role": "user", "content": user_content})
    # 4) Call LLM with chat-style API
    resp, _usage = await llm.chat(
        messages=messages,
    )

    # 5) Record assistant reply into memory and run simple distillation policy
    try:
        await mem.record_chat_assistant(
            resp,
            tags=["session.chat"],
        )

        # Simple distillation policy:
        # If we have "enough" chat turns in recent history, run a long-term summary.
        recent_for_distill = await mem.recent_chat(limit=120)
        if len(recent_for_distill) >= 80:
            # Non-LLM summarizer by default; flip use_llm=True later.
            await mem.distill_long_term(
                summary_tag="session",
                summary_kind="long_term_summary",
                include_kinds=["chat.turn"],
                include_tags=["chat"],
                max_events=200,
                use_llm=False,
            )
    except Exception:
        # Memory issues should never break the chat agent
        import traceback

        trace = traceback.format_exc()
        logger = context.logger()
        logger.warning("Chat agent memory record/distill error:\n" + trace)

    # 6) Send reply to UI channel
    await chan.send_text(resp)

    return {
        "reply": resp,
    }
