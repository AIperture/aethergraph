from __future__ import annotations

from aethergraph import NodeContext


async def build_session_memory_prompt_segments(
    context: NodeContext,
    *,
    summary_tag: str = "session",
    recent_limit: int = 12,
) -> tuple[str, str]:
    """
    Build reusable 'memory segments' for LLM prompts:

      - session_summary: long-term summary text (may be empty)
      - recent_events: short list of recent events (chat, status, etc.)

    Any agent that wants cross-agent memory can call this, then
    feed these into its LLM prompt or structured prompt store.

    The underlying MemoryFacade is responsible for:
    - Choosing a scope_id (usually tied to the session).
    - Storing summaries in DocStore (mem/{scope_id}/summaries/{tag}/...).
    - Exposing hotlog events via mem.recent().
    """
    mem = None
    try:
        mem = context.memory()
    except TypeError:
        # Depending on how NodeContext is wired, .memory might be property or method
        mem = getattr(context, "memory", None)

    if mem is None:
        return "", ""

    # ---- 1) Long-term session summary (if summarization has been run) ----
    session_summary = ""
    try:
        summary = await mem.soft_hydrate_last_summary(
            summary_tag=summary_tag,
            summary_kind="long_term_summary",
        )
        if summary:
            session_summary = summary.get("text") or ""
    except Exception as e:
        logger = getattr(context, "logger", None)
        if logger:
            logger.warning(
                "build_session_memory_prompt_segments: soft_hydrate_last_summary failed",
                extra={"error": str(e)},
            )

    # ---- 2) Recent events across runs/agents in this session ----
    recent_events = ""
    try:
        events = await mem.recent(kinds=None, limit=recent_limit)
        lines: list[str] = []
        for evt in events:
            kind = getattr(evt, "kind", None) or "event"
            text = getattr(evt, "text", None)
            data = getattr(evt, "data", None)

            # Try to recover some text from data if text is empty
            if not text and isinstance(data, dict):
                text = data.get("text") or data.get("message") or data.get("summary")

            if not text:
                continue

            # Keep it short-ish; you can truncate here if needed
            lines.append(f"[{kind}] {text}")
        recent_events = "\n".join(lines)
    except Exception as e:
        logger = getattr(context, "logger", None)
        if logger:
            logger.warning(
                "build_session_memory_prompt_segments: mem.recent failed",
                extra={"error": str(e)},
            )

    return session_summary, recent_events
