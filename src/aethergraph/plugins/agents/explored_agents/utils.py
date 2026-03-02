from __future__ import annotations

from typing import Any

from aethergraph.core.runtime.node_context import NodeContext
from aethergraph.plugins.agents.types import ClassifiedIntent, SessionAgentState


async def get_session_agent_state(
    context: NodeContext, session_id: str | None
) -> SessionAgentState:
    """
    TODO:
      - Lookup session state from memory / store.
      - If none, return SessionAgentState(active_agent_id="aether_agent").
    """
    return SessionAgentState(active_agent_id="aether_agent")


async def set_session_agent_state(
    context: NodeContext,
    session_id: str | None,
    state: SessionAgentState,
) -> None:
    """
    TODO:
      - Persist state (e.g. in mem / docstore, keyed by session).
    """
    return


def parse_at_mention(message: str) -> tuple[str, str] | None:
    """
    Parse a leading @agent directive.

    Examples:
      "@deeplens design a lens" -> ("deeplens", "design a lens")
      "@finance: run a backtest" -> ("finance", "run a backtest")
      "hey @deeplens ..." -> None  (not leading -> ignore here)

    Returns:
      (agent_alias, remaining_message) or None if no leading @.
    """
    text = message.strip()
    if not text.startswith("@"):
        return None

    # Remove @ and split on first whitespace or colon
    without_at = text[1:]
    if not without_at:
        return None

    # Find first separator: space or colon
    sep_idx = None
    for i, ch in enumerate(without_at):
        if ch.isspace() or ch == ":":
            sep_idx = i
            break

    if sep_idx is None:
        # Whole word is the alias, no extra message
        alias = without_at.strip()
        rest = ""
    else:
        alias = without_at[:sep_idx].strip()
        rest = without_at[sep_idx + 1 :].lstrip(" :")

    if not alias:
        return None

    return alias, rest


async def invoke_agent_placeholder(
    agent_id: str,
    *,
    message: str,
    files: list[Any] | None,
    session_id: str | None,
    user_meta: dict[str, Any] | None,
    context: NodeContext,
) -> str:
    """
    TODO: Replace this with your actual agent invocation logic.

    Likely something like:
      - agent_svc = context.services().agent_service
      - result = await agent_svc.invoke(
            agent_id=agent_id,
            message=message,
            files=files,
            session_id=session_id,
            user_meta=user_meta,
        )
      - return result.reply_text
    """
    return f"[TODO] Would invoke agent '{agent_id}' with message: {message!r}"


async def resolve_agent_alias(
    alias: str,
    context: NodeContext,
) -> str | None:
    """
    TODO: Resolve user-facing alias (like 'deeplens') to a real agent_id.

    Designs:
      - Use a dedicated AgentRegistry where agents define:
          * id='deeplens.agent'
          * aliases=['deeplens', 'lens', 'dl']
      - Or a simple mapping in config.

    For now, return None (no resolution).
    """
    return None


async def _handle_route_stub(
    intent: ClassifiedIntent,
    message: str,
    context: NodeContext,
    files: list[Any] | None = None,
    session_id: str | None = None,
    user_meta: dict[str, Any] | None = None,
) -> str:
    """
    Routing to other agents.

    - If intent.target_agent_id is set: use that directly (persistent active agent).
    - Else if intent.target_agent_alias is set: resolve via registry and call once.
    - Else: return fallback text.
    """
    route_msg = intent.route_message or message

    # 1) Direct agent id (persistent mode)
    if intent.target_agent_id:
        agent_id = intent.target_agent_id
    elif intent.target_agent_alias:
        # 2) Alias mode (one-shot via @alias)
        agent_id = await resolve_agent_alias(intent.target_agent_alias, context=context)
        if not agent_id:
            return (
                f"Unknown agent alias `{intent.target_agent_alias}`.\n\n"
                "Use `/agents` to see available agents or `/help agents` for more info."
            )
    else:
        return (
            "Routing intent detected, but no target agent specified. "
            "This is likely a bug in the classifier."
        )

    # Placeholder invocation
    return await invoke_agent_placeholder(
        agent_id=agent_id,
        message=route_msg,
        files=files,
        session_id=session_id,
        user_meta=user_meta,
        context=context,
    )
