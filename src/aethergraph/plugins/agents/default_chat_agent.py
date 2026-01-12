# aethergraph/examples/agents/default_chat_agent.py

from __future__ import annotations

from typing import Any

from aethergraph import NodeContext, graph_fn


async def _maybe_distill_session(mem) -> None:
    """
    Simple distillation policy:

    - If we have "enough" chat turns, run a long-term summary.
    - Uses non-LLM summarizer by default (use_llm=False).
    """
    recent_for_distill = await mem.recent_chat(limit=120)
    if len(recent_for_distill) < 80:
        return

    await mem.distill_long_term(
        summary_tag="session",
        summary_kind="long_term_summary",
        include_kinds=["chat.turn"],
        include_tags=["chat"],
        max_events=200,
        use_llm=False,
    )


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
    - Records user messages as chat.turn events.
    - Assistant replies are auto-logged via ChannelSession.
    - Periodically distills chat history into long-term summaries.
    """

    logger = context.logger()
    llm = context.llm()
    chan = context.ui_session_channel()
    mem = context.memory()

    # ------------------------------------------------------------------
    # 1) Build memory segments for this session
    # ------------------------------------------------------------------
    segments = await mem.build_prompt_segments(
        recent_chat_limit=20,
        include_long_term=True,
        summary_tag="session",
        max_summaries=3,
        include_recent_tools=False,
    )

    long_term_summary: str = segments.get("long_term") or ""
    recent_chat: list[dict[str, Any]] = segments.get("recent_chat") or []

    # ------------------------------------------------------------------
    # 2) System prompt + memory-conditioned chat history
    # ------------------------------------------------------------------
    system_prompt = (
        "You are AetherGraph's built-in session helper.\n\n"
        "You can see a summary of the session and some recent messages.\n"
        "Use them to answer questions about previous steps or runs, "
        "but do not invent details.\n"
        "If you are unsure, say that clearly.\n"
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    if long_term_summary:
        messages.append(
            {
                "role": "system",
                "content": "Summary of previous context:\n" + long_term_summary,
            }
        )

    for item in recent_chat:
        role = item.get("role") or "user"
        text = item.get("text") or ""
        # Map non-standard roles (e.g. "tool") to "assistant" for chat APIs
        mapped_role = role if role in {"user", "assistant", "system"} else "assistant"
        if text:
            messages.append({"role": mapped_role, "content": text})

    # ------------------------------------------------------------------
    # 3) Build user message (with lightweight metadata hints for LLM)
    # ------------------------------------------------------------------
    meta_lines: list[str] = []
    if files:
        meta_lines.append(f"(User attached {len(files)} file(s).)")
    if context_refs:
        meta_lines.append(f"(User attached {len(context_refs)} context reference(s).)")

    meta_block = ""
    if meta_lines:
        meta_block = "\n\n" + "\n".join(meta_lines)

    user_content = f"{message}{meta_block}"

    # Record user message into memory (including light file/context metadata)
    user_data: dict[str, Any] = {}
    if files:
        user_data["files"] = [
            {k: v for k, v in (f or {}).items() if k in {"name", "url", "mimetype", "size"}}
            for f in files
        ]
    if context_refs:
        user_data["context_refs"] = context_refs

    try:
        await mem.record_chat_user(
            message,
            data=user_data,
            tags=["session.chat"],
        )
    except Exception:
        logger.warning("Failed to record user chat message to memory", exc_info=True)

    # Append current user turn to LLM prompt
    messages.append({"role": "user", "content": user_content})

    # ------------------------------------------------------------------
    # 4) Call LLM with chat-style API
    # ------------------------------------------------------------------
    resp, usage = await llm.chat(messages=messages)

    # ------------------------------------------------------------------
    # 5) Send + auto-log assistant reply via channel
    # ------------------------------------------------------------------
    # ChannelSession will:
    #   - send the reply to the UI
    #   - best-effort log it to memory as role="assistant" with tags ["chat", "session.chat", ...]
    try:
        memory_data = {"usage": usage} if usage else None
        await chan.send_text(
            resp,
            memory_tags=["session.chat"],
            memory_data=memory_data,
        )
    except Exception:
        # Even if memory/channel logging fails, don't break the agent.
        logger.warning("Failed to send/log assistant reply via channel", exc_info=True)

    # ------------------------------------------------------------------
    # 6) Periodic long-term distillation (best-effort)
    # ------------------------------------------------------------------
    try:
        await _maybe_distill_session(mem)
    except Exception:
        logger.warning("Chat agent memory distill error", exc_info=True)

    return {"reply": resp}
