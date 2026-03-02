from dataclasses import dataclass, field
from typing import Any

# -------------------------------------------------------------------------
# Distillation policy for Layer 2 (long-term session summaries)
# -------------------------------------------------------------------------


@dataclass
class DistillPolicy:
    """
    Simple, hard-coded policy for chat distillation.

    - min_turns: only distill when there are at least this many chat events
      in the session.
    - recent_limit: how many recent chat events to inspect when deciding.
    - max_events: how many events a single distill_long_term call will
      consider when building a summary.
    - use_llm: whether to use an LLM-based summarizer or a cheap non-LLM path.
    - include_tags / include_kinds: which events count as "chat" for
      distillation purposes.
    """

    enabled: bool = True
    min_turns: int = 80
    recent_limit: int = 120
    max_events: int = 200
    use_llm: bool = False

    # We standardize on tags="chat" for all chat events
    include_tags: list[str] = field(default_factory=lambda: ["chat"])
    # Optional: if your record_chat_* sets kind="chat.turn" or similar
    include_kinds: list[str] | None = None


DEFAULT_DISTILL_POLICY = DistillPolicy()


async def _maybe_distill_session(
    *,
    mem: Any,
    logger: Any,
    policy: DistillPolicy = DEFAULT_DISTILL_POLICY,
) -> None:
    """
    Layer-2 distillation hook for the built-in agent.

    Called at the end of chat turns to keep per-session long-term summaries
    up to date.

    - Uses MemoryFacade.recent_chat(level="session") as the signal for
      "do we have enough history".
    - Uses MemoryFacade.distill_long_term(...) to actually write a
      long_term_summary with summary_tag="session".
    """
    if not policy.enabled:
        return

    # 1) Check how much chat history we have in this session scope
    try:
        recent_for_distill = await mem.recent_chat(
            limit=policy.recent_limit,
            level="session",
        )
    except Exception as e:
        logger.warning(
            "distill_session: recent_chat failed; skipping distillation",
            extra={"error": str(e)},
        )
        return

    if len(recent_for_distill) < policy.min_turns:
        # Not enough history yet; do nothing.
        return

    # 2) If the MemoryFacade doesn't support distill_long_term yet, bail gracefully
    if not hasattr(mem, "distill_long_term"):
        logger.debug(
            "distill_session: memory facade has no distill_long_term; " "skipping distillation"
        )
        return

    # 3) Run distillation
    try:
        await mem.distill_long_term(
            summary_tag="session",
            summary_kind="long_term_summary",
            include_kinds=policy.include_kinds,
            include_tags=policy.include_tags,
            max_events=policy.max_events,
            use_llm=policy.use_llm,
        )
        logger.info(
            "distill_session: ran long_term_summary distillation",
            extra={
                "summary_tag": "session",
                "summary_kind": "long_term_summary",
                "recent_seen": len(recent_for_distill),
                "max_events": policy.max_events,
                "use_llm": policy.use_llm,
            },
        )
    except Exception as e:
        logger.warning(
            "distill_session: distill_long_term failed",
            extra={"error": str(e)},
        )
