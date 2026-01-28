# aethergraph/examples/agents/default_chat_agent.py

from __future__ import annotations

import asyncio
import time
from typing import Any

from aethergraph import NodeContext, graph_fn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _maybe_distill_session(mem) -> None:
    """
    Simple distillation policy (Layer 2 maintenance):

    - If we have "enough" chat turns, run a long-term summary.
    - Uses non-LLM summarizer by default (use_llm=False).
      The summary is stored in DocStore and also recorded as a memory event
      via `record_raw`, so it becomes searchable by indices.
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


def _should_search_artifacts(
    message: str,
    files: list[Any] | None,
    context_refs: list[dict[str, Any]] | None,
) -> bool:
    """
    Heuristic: when do we bother searching artifacts (files, reports, logs)?

    - Always, if user attached files or context refs.
    - Otherwise, only if the message looks artifact-oriented.
    """
    if files or context_refs:
        return True

    msg = (message or "").lower()
    artifact_keywords = [
        "file",
        "document",
        "doc",
        "pdf",
        "report",
        "notebook",
        "log",
        "logs",
        "plot",
        "graph",
        "artifact",
    ]
    return any(k in msg for k in artifact_keywords)


def _format_search_snippets(event_results, artifact_results, max_total: int = 8) -> str:
    """
    Convert search hits (Layer 3) into a compact textual block
    that the LLM can consume.

    We don't try to be fancy; just short bullet lines with a bit of context.
    """
    lines: list[str] = []

    # Events first
    for r in event_results:
        meta = getattr(r, "metadata", None) or {}
        kind = meta.get("kind", "event")
        tags = meta.get("tags") or []
        text = meta.get("preview") or ""
        print("🍏 Search event preview:", text)

        if not text:
            continue

        tag_str = f" tags={','.join(tags[:3])}" if tags else ""
        lines.append(f"- [event:{kind}]{tag_str} {text[:220]}")
        if len(lines) >= max_total:
            break

    # Then artifacts (if we still have budget)
    if len(lines) < max_total:
        remaining = max_total - len(lines)
        for r in artifact_results[:remaining]:
            meta = getattr(r, "metadata", None) or {}
            kind = meta.get("kind", "artifact")
            name = (
                meta.get("filename")
                or meta.get("name")
                or meta.get("path")
                or meta.get("uri")
                or r.item_id
            )
            desc = meta.get("description") or meta.get("summary") or ""
            snippet = f"{name}: {desc[:160]}" if desc else name
            lines.append(f"- [artifact:{kind}] {snippet}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Default chat agent with 3-layer memory
# ---------------------------------------------------------------------------


@graph_fn(
    name="default_chat_agent",
    inputs=["message", "files", "context_refs", "session_id", "user_meta"],
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
    Built-in chat agent with 3-layer session memory: Recency, Long-term summaries, Semantic search.
    """

    logger = context.logger()
    llm = context.llm()
    chan = context.ui_session_channel()
    mem = context.memory()
    indices = context.indices()  # ScopedIndices

    # ------------------------------------------------------------------
    # 1) Layer 1 + 2: recency + long-term summaries
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
    # 2) Base system prompt + memory-conditioned history
    # ------------------------------------------------------------------
    system_prompt = (
        "You are AetherGraph's built-in session helper.\n\n"
        "You can see:\n"
        "- A long-term summary of the session (distilled from prior turns).\n"
        "- A short window of recent chat messages.\n"
        "- Optionally, semantically retrieved snippets from past events "
        "  and artifacts.\n\n"
        "Use them to answer questions about previous steps or runs, "
        "but do not invent details.\n"
        "If you are unsure, say that clearly.\n"
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    # Long-term summary (Layer 2 as plain text)
    if long_term_summary:
        messages.append(
            {
                "role": "system",
                "content": "Summary of previous context:\n" + long_term_summary,
            }
        )

    # Recent chat turns (Layer 1)
    for item in recent_chat:
        role = item.get("role") or "user"
        text = item.get("text") or ""
        mapped_role = role if role in {"user", "assistant", "system"} else "assistant"
        if text:
            messages.append({"role": mapped_role, "content": text})

    # ------------------------------------------------------------------
    # 3) Layer 3: semantic search over events + artifacts
    # ------------------------------------------------------------------
    search_snippet_block = ""
    try:
        # Scope-aware filtering: prefer this memory scope if present
        scope_id = getattr(mem, "memory_scope_id", None) or None
        filters: dict[str, Any] = {}
        if scope_id:
            filters["scope_id"] = scope_id

        now_ts = time.time()
        # Example: look back up to ~90 days. You can adjust this.
        created_at_min = now_ts - 90 * 24 * 3600
        created_at_max = now_ts

        # Always search events with the user's message as query (cheap, high value).
        event_results = await indices.search_events(
            query=message,
            top_k=5,
            filters=filters or None,
            created_at_min=created_at_min,
            created_at_max=created_at_max,
        )
        print("🍏 Event search results:", event_results)

        # Search artifacts only when the message/files/context suggests it.
        artifact_results = []
        if _should_search_artifacts(message, files, context_refs):
            artifact_results = await indices.search_artifacts(
                query=message,
                top_k=5,
                filters=filters or None,
                created_at_min=created_at_min,
                created_at_max=created_at_max,
            )

        search_snippet_block = _format_search_snippets(event_results, artifact_results)

    except Exception:
        # If search backend is misconfigured or fails, do not break chat.
        logger.warning("default_chat_agent: search backend error", exc_info=True)
        search_snippet_block = ""

    if search_snippet_block:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Retrieved memory snippets and artifacts that may be relevant "
                    "to the user's current question:\n\n"
                    f"{search_snippet_block}\n\n"
                    "If they are not relevant, you may ignore them."
                ),
            }
        )

    # ------------------------------------------------------------------
    # 4) Build user message (with lightweight metadata hints for LLM)
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

    # Record user turn into memory (this becomes part of Layer 1 + 3 later)
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

    # Append current user turn to prompt
    messages.append({"role": "user", "content": user_content})

    print("---- Final LLM Prompt Messages ----")
    for m in messages:
        role = m.get("role", "unknown")
        content = m.get("content", "")
        if role == "system":
            print(f"[{role}] {content}")
        else:
            print(
                f"[{role}] {content[:200].replace(chr(10), ' ')}{'...' if len(content) > 200 else ''}"
            )

    # ------------------------------------------------------------------
    # 5) Single LLM call (all layers already baked into messages)
    # ------------------------------------------------------------------
    # await chan.send_phase(
    #     phase="reasoning",
    #     status="active",
    #     label="LLM call",
    #     detail="Calling LLM...",
    # )

    # resp, usage = await llm.chat(messages=messages)

    # # ------------------------------------------------------------------
    # # # 6) Send + auto-log assistant reply via channel (ChannelSession)
    # # # ------------------------------------------------------------------
    # try:
    #     memory_data = {"usage": usage} if usage else None
    #     await chan.send_text(
    #         resp,
    #         memory_tags=["session.chat"],
    #         memory_data=memory_data,
    #     )
    # except Exception:
    #     logger.warning("Failed to send/log assistant reply via channel", exc_info=True)

    # # Finalize "reasoning" phase
    # await chan.send_phase(
    #     phase="reasoning",
    #     status="done",
    #     label="LLM call",
    #     detail="LLM response finished.",
    # )

    # ------------------------------------------------------------------
    # 5) Single LLM call (streaming into ChannelSession)
    # ------------------------------------------------------------------
    try:
        # Mark the "reasoning" phase as active before calling the LLM
        try:
            await chan.send_phase(
                phase="reasoning",
                status="active",
                label="LLM call",
                detail="Calling LLM (streaming response)...",
            )

            await asyncio.sleep(0.6)  # slight delay to ensure phase event ordering
            await chan.send_phase(
                phase="llm",
                status="active",
                label="Planning generating response",
                detail="Planning is generating the response...",
            )
        except Exception:
            logger.debug("Failed to send LLM phase(active) state", exc_info=True)

        async with chan.stream() as s:
            # Hook for streaming deltas into the same message
            async def on_delta(piece: str) -> None:
                await s.delta(piece)

            # Streaming LLM call
            resp, usage = await llm.chat_stream(
                messages=messages,
                on_delta=on_delta,
            )

            # Finalize streaming + memory
            memory_data = {"usage": usage} if usage else None
            await s.end(
                full_text=resp,
                memory_tags=["session.chat"],
                memory_data=memory_data,
            )

        # Mark the "reasoning" phase as done
        try:
            await chan.send_phase(
                phase="reasoning",
                status="done",
                label="LLM call",
                detail="LLM response finished.",
            )
        except Exception:
            logger.debug("Failed to send LLM phase(done) state", exc_info=True)

    except Exception:
        logger.warning(
            "Failed to stream/log assistant reply via channel",
            exc_info=True,
        )

    # ------------------------------------------------------------------
    # 7) Periodic long-term distillation (maintains Layer 2)
    # ------------------------------------------------------------------
    try:
        await _maybe_distill_session(mem)
    except Exception:
        logger.warning("Chat agent memory distill error", exc_info=True)

    return {"reply": resp}
