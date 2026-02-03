# aethergraph/examples/agents/default_chat_agent.py

from __future__ import annotations

import asyncio
from contextlib import suppress
import time
from typing import Any

from aethergraph import NodeContext, graph_fn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _thread_tag(session_id: str | None) -> str | None:
    """
    Your custom, stable tag contract for "this conversation thread".
    Use something not special-cased by core memory.
    """
    if not session_id:
        return None
    return f"thread:session:{session_id}"


def _semantic_chat_tags(session_id: str | None) -> list[str]:
    """
    Tags that you control at agent level.
    Keep them semantic + thread-scoped, but not provenance (run/node/channel),
    since provenance can come from channel if you want.
    """
    tags = ["user.chat"]
    ttag = _thread_tag(session_id)
    if ttag:
        tags.append(ttag)
    return tags


async def _get_last_seen_thread(mem) -> str | None:
    # HotLog-only; good enough for "what was the last thread we chatted in?"
    evts = await mem.recent(kinds=["chat.thread_seen"], limit=5)
    for e in reversed(evts):  # newest last (per your recent() contract)
        t = (getattr(e, "data", None) or {}).get("thread")
        if isinstance(t, str) and t:
            return t
    return None


async def _mark_thread_seen(mem, ttag: str | None) -> None:
    if not ttag:
        return
    # Record as an ordinary event so it lives in user timeline
    await mem.record(
        kind="chat.thread_seen",
        text=ttag,
        data={"thread": ttag},
        tags=["thread_state"],
        severity=1,
        stage="system",
    )


async def _emit_handoff_capsule(mem, *, prev_ttag: str, prev_session_id: str | None = None) -> None:
    """
    Create a lightweight, immediate capsule for the previous thread.
    This is NOT long-term distill; it’s a quick “handoff”.
    """
    print(f"🍎 Emitting handoff capsule for thread: {prev_ttag}")
    # Pull the tail of the previous thread
    tail = await mem.recent_chat(limit=40, tags=[prev_ttag])

    if not tail:
        return

    # Ultra-cheap “summary” without an extra LLM call:
    # keep last few user+assistant messages in compact bullets
    lines: list[str] = []
    for m in tail[-12:]:
        role = m.get("role", "user")
        text = (m.get("text") or "").strip().replace("\n", " ")
        if not text:
            continue
        if len(text) > 180:
            text = text[:180] + "…"
        lines.append(f"{role}: {text}")

    capsule = "Previous session context (most recent tail):\n" + "\n".join(lines)

    # Store capsule in user-level memory (NOT thread-scoped), but tagged with prev thread
    await mem.record(
        kind="chat.handoff",
        text=capsule,
        data={"thread": prev_ttag, "session_id": prev_session_id, "text": capsule},
        tags=["handoff", prev_ttag],
        severity=2,
        stage="system",
    )


async def _maybe_handoff_on_thread_change(mem, *, session_id: str | None) -> str | None:
    """
    Detect thread change and emit capsule for previous thread.
    Returns prev_ttag if a change was detected.
    """
    print("🍎 Checking for thread change...")
    cur_ttag = _thread_tag(session_id)
    if not cur_ttag:
        return None

    last = await _get_last_seen_thread(mem)
    if last and last != cur_ttag:
        # Emit capsule for the previous thread right now
        with suppress(Exception):
            await _emit_handoff_capsule(mem, prev_ttag=last, prev_session_id=None)
        return last

    return None


async def _load_recent_handoff(mem, *, limit: int = 1) -> list[str]:
    """
    Pull most recent handoff capsules from user memory.
    """
    print("🍎 Loading recent handoff capsules...")
    data = await mem.recent_data(
        kinds=["chat.handoff"],
        tags=["handoff"],
        limit=max(limit * 5, 20),
    )
    # recent_data returns data or text; normalize to strings
    out: list[str] = []
    for x in data:
        if isinstance(x, dict):
            t = x.get("text") or x.get("summary") or ""
            if t:
                out.append(str(t))
        elif isinstance(x, str) and x.strip():
            out.append(x.strip())
    return out[-limit:]


async def _maybe_distill_session(mem, session_id: str | None) -> None:
    """
    Distill a per-session/thread summary, even though overall memory scope is user-level.

    We select only events from this thread via tag filtering and then write summary docs
    under a filesystem-safe tag path.
    """
    ttag = _thread_tag(session_id)
    print(f"🍎 Distilling session for thread tag: {ttag}")
    if not ttag:
        return

    # Pull more than needed and filter locally (keeps memory core unchanged)
    recent = await mem.recent(kinds=["chat.turn"], limit=250)
    recent = [e for e in recent if ttag in set(e.tags or [])]

    if len(recent) < 80:
        return

    # Store per-thread summaries under a filesystem-safe summary_tag
    safe_summary_tag = f"thread/session/{session_id}"

    await mem.distill_long_term(
        summary_tag=safe_summary_tag,
        summary_kind="long_term_summary",
        include_kinds=["chat.turn"],
        include_tags=["chat", ttag],  # only this thread
        max_events=200,
        use_llm=False,
    )


def _should_search_artifacts(
    message: str,
    files: list[Any] | None,
    context_refs: list[dict[str, Any]] | None,
) -> bool:
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
    lines: list[str] = []

    for r in event_results:
        meta = getattr(r, "metadata", None) or {}
        kind = meta.get("kind", "event")
        tags = meta.get("tags") or []
        text = meta.get("preview") or ""
        if not text:
            continue
        tag_str = f" tags={','.join(tags[:3])}" if tags else ""
        lines.append(f"- [event:{kind}]{tag_str} {text[:220]}")
        if len(lines) >= max_total:
            break

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
# Default chat agent (user memory + thread/session-scoped prompt)
# ---------------------------------------------------------------------------


@graph_fn(
    name="default_chat_agent_user_mem",
    inputs=["message", "files", "context_refs", "session_id", "user_meta"],
    outputs=["reply"],
    as_agent={
        "id": "chat_agent_user_mem",
        "title": "Chat",
        "short_description": "General-purpose chat agent (user memory + thread-scoped prompt).",
        "description": "Uses user-level memory across sessions; prompt history is scoped by a custom thread tag.",
        "icon": "message-circle",
        "color": "sky",
        "session_kind": "chat",
        "mode": "chat_v1",
        "memory_level": "user",  # ✅ global user memory scope
        "memory_scope": "global",
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
    logger = context.logger()
    llm = context.llm()
    chan = context.ui_session_channel()
    mem = context.memory()
    indices = context.indices()

    ttag = _thread_tag(session_id)

    # Detect thread change and emit a capsule for the previous thread (if any)
    try:
        await _maybe_handoff_on_thread_change(mem, session_id=session_id)
    except Exception:
        logger.debug("handoff capsule failed", exc_info=True)

    mem_tags = _semantic_chat_tags(session_id)

    # ------------------------------------------------------------------
    # 1) Layer 1 + 2:
    #    - user-level long-term summaries (cross-session)
    #    - thread/session-scoped recent chat (tag filter)
    # ------------------------------------------------------------------
    segments = await mem.build_prompt_segments(
        recent_chat_limit=20,
        include_long_term=True,
        summary_tag="user/global",  # ✅ user-level summaries
        max_summaries=3,
        include_recent_tools=False,
        # ✅ new generic tag filter; only include messages from this thread
        recent_chat_tags=[ttag] if ttag else None,
    )

    long_term_summary: str = segments.get("long_term") or ""
    recent_chat: list[dict[str, Any]] = segments.get("recent_chat") or []

    # ------------------------------------------------------------------
    # 2) Prompt assembly
    # ------------------------------------------------------------------
    system_prompt = (
        "You are AetherGraph's built-in helper.\n\n"
        "You can see:\n"
        "- A long-term summary of the user (across sessions).\n"
        "- A short window of recent messages from this thread.\n"
        "- Optionally, retrieved snippets from events and artifacts.\n\n"
        "Use them to answer questions, but do not invent details.\n"
        "If unsure, say so.\n"
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    if long_term_summary:
        messages.append({"role": "system", "content": "User memory summary:\n" + long_term_summary})

    try:
        print("🍎 Adding recent chat to prompt...")
        handoffs = await _load_recent_handoff(mem, limit=1)
        print(f"🍎 Loaded {len(handoffs)} handoff capsules.")
        if handoffs:
            print(handoffs[0])
            messages.append(
                {
                    "role": "system",
                    "content": "Recent context from your previous session:\n" + handoffs[0],
                }
            )
    except Exception:
        logger.debug("handoff load failed", exc_info=True)

    for item in recent_chat:
        role = item.get("role") or "user"
        text = item.get("text") or ""
        mapped_role = role if role in {"user", "assistant", "system"} else "assistant"
        if text:
            messages.append({"role": mapped_role, "content": text})

    # ------------------------------------------------------------------
    # 3) Layer 3: semantic search (user scope)
    # ------------------------------------------------------------------
    search_snippet_block = ""
    try:
        scope_id = getattr(mem, "memory_scope_id", None) or None
        filters: dict[str, Any] = {}
        if scope_id:
            filters["scope_id"] = scope_id

        now_ts = time.time()
        created_at_min = now_ts - 90 * 24 * 3600
        created_at_max = now_ts

        event_results = await indices.search_events(
            query=message,
            top_k=5,
            filters=filters or None,
            created_at_min=created_at_min,
            created_at_max=created_at_max,
        )

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
        logger.warning("default_chat_agent_user_mem: search backend error", exc_info=True)

    if search_snippet_block:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Retrieved snippets that may be relevant:\n\n"
                    f"{search_snippet_block}\n\n"
                    "Ignore if irrelevant."
                ),
            }
        )

    # ------------------------------------------------------------------
    # 4) Record user turn (semantic + thread tag), then call LLM
    # ------------------------------------------------------------------
    meta_lines: list[str] = []
    if files:
        meta_lines.append(f"(User attached {len(files)} file(s).)")
    if context_refs:
        meta_lines.append(f"(User attached {len(context_refs)} context reference(s).)")

    user_content = message + ("\n\n" + "\n".join(meta_lines) if meta_lines else "")

    user_data: dict[str, Any] = {}
    if files:
        user_data["files"] = [
            {
                "id": getattr(f, "id", None),
                "name": getattr(f, "name", None),
                "mimetype": getattr(f, "mimetype", None),
                "size": getattr(f, "size", None),
                "url": getattr(f, "url", None),
                "uri": getattr(f, "uri", None),
                "extra": getattr(f, "extra", None),
            }
            for f in files
        ]
    if context_refs:
        user_data["context_refs"] = context_refs

    # Record user turn under user memory scope, but tagged by thread
    try:
        await mem.record_chat_user(
            message,
            data=user_data,
            tags=mem_tags,
        )
    except Exception:
        logger.warning("Failed to record user chat message to memory", exc_info=True)

    messages.append({"role": "user", "content": user_content})

    # ------------------------------------------------------------------
    # 5) Stream response (assistant turn uses same semantic tags)
    # ------------------------------------------------------------------
    resp = ""
    try:
        try:
            await chan.send_phase(
                phase="reasoning",
                status="active",
                label="LLM call",
                detail="Calling LLM (streaming response)...",
            )
            await asyncio.sleep(0.2)
            await chan.send_phase(
                phase="llm",
                status="active",
                label="Generating",
                detail="LLM is generating the response...",
            )
        except Exception:
            logger.debug("Failed to send phase(active)", exc_info=True)

        async with chan.stream() as s:

            async def on_delta(piece: str) -> None:
                await s.delta(piece)

            resp, usage = await llm.chat_stream(
                messages=messages,
                on_delta=on_delta,
            )

            memory_data = {"usage": usage} if usage else None

            # IMPORTANT: use the same tag bundle so prompt filtering works
            await s.end(
                full_text=resp,
                memory_tags=mem_tags,
                memory_data=memory_data,
            )

        try:
            await chan.send_phase(
                phase="reasoning",
                status="done",
                label="LLM call",
                detail="LLM response finished.",
            )
        except Exception:
            logger.debug("Failed to send phase(done)", exc_info=True)

    except Exception:
        logger.warning("Failed to stream/log assistant reply via channel", exc_info=True)

    # ------------------------------------------------------------------
    # 6) Per-thread distillation (optional)
    # ------------------------------------------------------------------
    try:
        await _maybe_distill_session(mem, session_id=session_id)
    except Exception:
        logger.warning("Chat agent memory distill error", exc_info=True)

    try:
        await _mark_thread_seen(mem, ttag)
    except Exception:
        logger.debug("thread_seen record failed", exc_info=True)

    return {"reply": resp}
