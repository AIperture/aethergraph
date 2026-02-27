from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Literal

from aethergraph.contracts.services.memory import Event
from aethergraph.core.runtime.node_context import NodeContext
from aethergraph.plugins.agents.types import BUILTIN_AGENT_SKILL_ID

# ---------------------------------------------------------------------------
# Retrieval plan dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MemoryPlan:
    enabled: bool = True
    levels: list[str] = field(default_factory=lambda: ["user"])
    limit: int = 10
    use_embedding: bool = True
    time_window: str | None = None  # e.g. "1h", "24h", "7d"
    kinds: list[str] | None = None  # e.g. ["chat_user", "chat_assistant"]
    tags: list[str] | None = None  # e.g. ["important", "todo"]


@dataclass
class KBPlan:
    enabled: bool = False
    corpus_id: str | None = "ag.docs"  # logical corpus for AG docs
    kb_namespace: str | None = "ag.docs"  # kb_namespace param for search()
    top_k: int = 8
    mode: Literal["semantic", "lexical", "hybrid"] | str = "hybrid"
    reason: str | None = None  # for logging / debugging


@dataclass
class RetrievalPlan:
    # Chat context
    include_recent_chat: bool = True
    recent_chat_limit: int = 20

    include_session_summary: bool = True
    session_summary_limit: int = 3

    # Memory + KB
    memory: MemoryPlan = field(default_factory=MemoryPlan)
    kb: KBPlan = field(default_factory=KBPlan)


# ---------------------------------------------------------------------------
# Plan computation from skill config
# ---------------------------------------------------------------------------


def _compute_retrieval_plan(
    *,
    intent: Any | None,
    message: str,
    context: NodeContext,
) -> RetrievalPlan:
    """
    Build a RetrievalPlan using the active skill config and the current message.

    Reads from the `ag.builtin_agent` skill:

        config:
          retrieval:
            default:
              recent_chat: ...
              session_summary: ...
              memory: ...
              kb: ...

    and optionally trigger rules under `config.retrieval.default.kb.triggers`.
    """
    skills = context.skills()
    skill = skills.get(BUILTIN_AGENT_SKILL_ID)
    cfg = (skill.config or {}) if skill is not None else {}
    base = cfg.get("retrieval", {}).get("default", {})

    plan = RetrievalPlan()

    # --- 1) recent chat / session summary ---
    rc = base.get("recent_chat", {})
    plan.include_recent_chat = rc.get("enabled", True)
    plan.recent_chat_limit = rc.get("limit", 20)

    ss = base.get("session_summary", {})
    plan.include_session_summary = ss.get("enabled", True)
    plan.session_summary_limit = ss.get("limit", 3)

    # --- 2) memory ---
    mem_cfg = base.get("memory", {})
    plan.memory.enabled = mem_cfg.get("enabled", True)
    plan.memory.levels = mem_cfg.get("levels", ["user"])
    plan.memory.limit = mem_cfg.get("limit", 10)
    plan.memory.use_embedding = mem_cfg.get("use_embedding", True)
    plan.memory.time_window = mem_cfg.get("time_window")
    plan.memory.kinds = mem_cfg.get("kinds")
    plan.memory.tags = mem_cfg.get("tags")

    # --- 3) KB (config + trigger-based gating) ---
    kb_cfg = base.get("kb", {})
    plan.kb.enabled = kb_cfg.get("enabled", False)
    plan.kb.corpus_id = kb_cfg.get("corpus_id", "ag.docs")
    plan.kb.kb_namespace = kb_cfg.get("kb_namespace", plan.kb.corpus_id)
    plan.kb.top_k = kb_cfg.get("top_k", 8)
    plan.kb.mode = kb_cfg.get("mode", "hybrid")

    if plan.kb.enabled:
        triggers = kb_cfg.get("triggers", [])
        text = (message or "").lower()
        if triggers:
            should_use_kb = False
            for trig in triggers:
                kind = trig.get("kind")
                if kind == "contains_any":
                    terms = [t.lower() for t in trig.get("terms", [])]
                    if any(t in text for t in terms):
                        should_use_kb = True
                        plan.kb.reason = f"contains_any: {terms}"
                        break
                elif kind == "contains_both":
                    terms1 = [t.lower() for t in trig.get("terms1", [])]
                    terms2 = [t.lower() for t in trig.get("terms2", [])]
                    if any(t in text for t in terms1) and any(t in text for t in terms2):
                        should_use_kb = True
                        plan.kb.reason = f"contains_both: {terms1} & {terms2}"
                        break
                elif kind == "regex":
                    pattern = trig.get("pattern", "")
                    if pattern and re.search(pattern, text):
                        should_use_kb = True
                        plan.kb.reason = f"regex: {pattern}"
                        break

            if not should_use_kb:
                plan.kb.enabled = False
        # else: no triggers configured → always enabled

    # OPTIONAL: later you can overlay profile-specific overrides here based on
    # intent.subtype or similar.

    return plan


# ---------------------------------------------------------------------------
# Context gathering
# ---------------------------------------------------------------------------


async def gather_chat_context(
    *,
    message: str,
    session_id: str | None,
    context: NodeContext,
    plan: RetrievalPlan,
) -> dict[str, Any]:
    """
    Execute the given RetrievalPlan and gather contextual info for the agent.

    Returns:
        {
            "session_summary": str,
            "recent_chat": list[dict],
            "user_memory_snippets": str,
            "kb_snippets": str,
        }
    """
    mem = context.memory()
    logger = context.logger()

    # 1) Recent chat
    if plan.include_recent_chat:
        recent_chat = await mem.recent_chat(
            limit=plan.recent_chat_limit,
            level="session",
        )
    else:
        recent_chat = []

    # 2) Session summaries
    if plan.include_session_summary:
        try:
            session_summaries = await mem.load_recent_summaries(
                summary_tag="session",
                summary_kind="long_term_summary",
                level="session",
                limit=plan.session_summary_limit,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "gather_chat_context: load_recent_summaries failed",
                extra={"error": str(e)},
            )
            session_summaries = []
    else:
        session_summaries = []

    summary_parts: list[str] = []
    for s in session_summaries:
        st = next(
            (s.get(key) for key in ("summary", "text", "body", "value") if s.get(key)),
            "",
        )
        if st:
            summary_parts.append(st)
    session_summary = "\n\n".join(summary_parts)

    # 3) Memory search
    user_memory_snippets = ""
    if plan.memory.enabled:
        try:
            level = plan.memory.levels[0] if plan.memory.levels else "user"
            search_events = await mem.search(
                query=message,
                kinds=plan.memory.kinds,
                tags=plan.memory.tags,
                limit=plan.memory.limit,
                use_embedding=plan.memory.use_embedding,
                level=level,
                time_window=plan.memory.time_window,
            )
            user_memory_snippets = _format_event_search_snippets(
                search_events,
                max_total=plan.memory.limit,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "gather_chat_context: mem.search failed",
                extra={"error": str(e)},
            )
            user_memory_snippets = ""

    # 4) KB retrieval
    kb_snippets = ""
    if plan.kb.enabled:
        try:
            kb = context.kb()
            hits = await kb.search(
                corpus_id=plan.kb.corpus_id,
                query=message,
                top_k=plan.kb.top_k,
                kb_namespace=plan.kb.kb_namespace,
                mode=plan.kb.mode,
            )
            kb_snippets = _format_kb_search_snippets(hits)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "gather_chat_context: kb.search failed",
                extra={"error": str(e)},
            )
            kb_snippets = ""

    return {
        "session_summary": session_summary,
        "recent_chat": recent_chat,
        "user_memory_snippets": user_memory_snippets,
        "kb_snippets": kb_snippets,
    }


# ---------------------------------------------------------------------------
# Chat handler
# ---------------------------------------------------------------------------


async def basic_chat_handler(
    *,
    intent: Any | None,
    message: str,
    files: list[Any] | None,
    context_refs: list[dict[str, Any]] | None,
    session_id: str | None,
    user_meta: dict[str, Any] | None,
    context: NodeContext,
) -> str:
    """
    Main chat path for the built-in Aether Agent.

    Behavior is governed by the `ag.builtin_agent` skill, via sections:
      - chat.system
      - chat.retrieval
      - chat.style
    plus retrieval config in `config.retrieval.default`.
    """
    logger = context.logger()
    llm = context.llm()
    chan = context.ui_session_channel()

    msg_text = (message or "").strip()

    # 1) Compute retrieval plan based on skill config + message
    plan = _compute_retrieval_plan(
        intent=intent,
        message=msg_text,
        context=context,
    )

    # 2) Gather retrieval context
    ctx = await gather_chat_context(
        message=msg_text,
        session_id=session_id,
        context=context,
        plan=plan,
    )

    session_summary = ctx.get("session_summary", "")
    recent_chat = ctx.get("recent_chat", [])
    user_memory_snippets = ctx.get("user_memory_snippets", "")
    kb_snippets = ctx.get("kb_snippets", "")

    # 3) Build system prompt from skills
    skills = context.skills()
    try:
        print("🍎 Compiling system prompt for builtin agent from skills...")
        system_prompt = skills.compile_prompt(
            BUILTIN_AGENT_SKILL_ID,
            "chat.system",
            "chat.retrieval",
            "chat.style",
            separator="\n\n",
            fallback_keys=["chat.system"],
        )
        print("🍎 Compiled system prompt for builtin agent:\n", system_prompt)
    except Exception as e:  # noqa: BLE001
        logger.error(
            "basic_chat_handler: failed to compile system prompt for %s",
            BUILTIN_AGENT_SKILL_ID,
            extra={"error": str(e)},
        )
        system_prompt = (
            "You are the built-in Aether Agent.\n\n"
            "You help users:\n"
            "- Understand and use AetherGraph.\n"
            "- Recall past context from this session and previous sessions.\n"
            "- Use documentation snippets when provided.\n"
            "If you are unsure, ask the user for clarification instead of guessing."
        )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
    ]

    # 4) Attach retrieved context as system messages
    if session_summary:
        messages.append(
            {
                "role": "system",
                "content": "Session summary:\n" + session_summary,
            }
        )

    if user_memory_snippets:
        messages.append(
            {
                "role": "system",
                "content": (
                    "User-level memory snippets (may come from other sessions):\n\n"
                    f"{user_memory_snippets}\n\n"
                    "If irrelevant to the current question, you may ignore them."
                ),
            }
        )

    if kb_snippets:
        messages.append(
            {
                "role": "system",
                "content": (
                    "AetherGraph documentation snippets from the KB:\n\n"
                    f"{kb_snippets}\n\n"
                    "These describe the ground-truth behavior of AetherGraph. "
                    "Prefer them over your own assumptions."
                ),
            }
        )

    # 5) Replay recent chat for continuity
    for item in recent_chat:
        role = (item.get("role") or "user").lower()
        text = item.get("text") or ""
        mapped_role = role if role in {"user", "assistant", "system"} else "assistant"
        if text:
            messages.append({"role": mapped_role, "content": text})

    # 6) Append current user turn
    messages.append(
        {
            "role": "user",
            "content": msg_text,
        }
    )

    # 7) Streaming LLM call
    async with chan.stream() as s:

        async def on_delta(piece: str) -> None:
            await s.delta(piece)

        resp_text, _usage = await llm.chat_stream(
            messages=messages,
            on_delta=on_delta,
        )

        await s.end(
            full_text=resp_text,
            memory_log=False,  # we log memory via mem.record_chat_* in builtin_agent
        )

    return resp_text


# ---------------------------------------------------------------------------
# Formatting helpers (assuming you have versions of these; otherwise stub)
# ---------------------------------------------------------------------------


def _format_event_search_snippets(events: list[Event], max_total: int) -> str:
    """
    Turn MemoryFacade.search events into a compact text block for the LLM.

    You already seemed to have something like this; keep or adjust as needed.
    """
    lines: list[str] = []
    for e in events[:max_total]:
        text = (e.text or "").strip().replace("\n", " ")
        if not text:
            continue
        if len(text) > 220:
            text = text[:220] + "…"
        tags = ", ".join(e.tags or [])
        line = f"- [{e.kind or 'event'}] {text}"
        if tags:
            line += f" (tags: {tags})"
        lines.append(line)
    return "\n".join(lines)


def _format_kb_search_snippets(hits: list[Any]) -> str:
    """
    Turn KBSearchHit objects into a text block for the LLM.

    Assumes each hit has fields: text, score, doc_id, chunk_id.
    """
    lines: list[str] = []
    for h in hits:
        text = (h.text or "").replace("\n", " ")
        if len(text) > 220:
            text = text[:220] + "…"
        line = f"- [score={h.score:.3f} doc_id={h.doc_id} chunk_id={h.chunk_id}] " f"{text}"
        lines.append(line)
    return "\n".join(lines)
