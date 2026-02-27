from __future__ import annotations

from typing import Any

from aethergraph import NodeContext, graph_fn
from aethergraph.plugins.agents.agent_functions.chat import basic_chat_handler
from aethergraph.plugins.agents.agent_functions.distillation import (
    _maybe_distill_session,
)
from aethergraph.plugins.agents.types import ClassifiedIntent, SessionAgentState
from aethergraph.plugins.agents.utils import parse_at_mention

# ---------------------------------------------------------------------------
# Intent / mode classification
# ---------------------------------------------------------------------------


async def _classify_intent(
    message: str,
    identity: Any | None = None,
    session_state: SessionAgentState | None = None,
) -> ClassifiedIntent:
    """
    Simple rule-based classifier for the built-in Aether Agent.

    Modes:
      - command: messages starting with '/' -> shell-style command
      - route: leading '@alias' OR a persisted active_agent_id != 'aether_agent'
      - chat: everything else (handled by the chat path)

    TODO (future):
      - extend to sub-modes (e.g. kb/api_help, builder, connector) once needed.
    """
    text = (message or "").strip()

    # 1) Slash commands (always handled by shell)
    if text.startswith("/"):
        without_slash = text[1:]
        if not without_slash:
            return ClassifiedIntent(
                mode="command",
                command="help",
                command_args="",
                debug_notes="empty slash -> /help",
            )

        parts = without_slash.split(maxsplit=1)
        cmd = parts[0]
        args = parts[1] if len(parts) > 1 else ""
        return ClassifiedIntent(
            mode="command",
            command=cmd,
            command_args=args,
            debug_notes="slash-prefixed /command",
        )

    # 2) Leading @alias one-shot route
    parsed = parse_at_mention(text)
    if parsed is not None:
        alias, rest = parsed
        return ClassifiedIntent(
            mode="route",
            target_agent_alias=alias,
            route_message=rest or text,  # if empty, send original
            debug_notes=f"leading @mention -> route to alias '{alias}'",
        )

    # 3) Session-level active agent (persistent mode)
    if (
        session_state is not None
        and session_state.active_agent_id
        and session_state.active_agent_id != "aether_agent"
    ):
        return ClassifiedIntent(
            mode="route",
            target_agent_id=session_state.active_agent_id,
            route_message=text,
            debug_notes=f"persistent active agent '{session_state.active_agent_id}'",
        )

    # 4) Everything else: normal chat
    return ClassifiedIntent(
        mode="chat",
        debug_notes="default chat mode",
    )


# ---------------------------------------------------------------------------
# Built-in agent entrypoint
# ---------------------------------------------------------------------------


@graph_fn(
    name="default_chat_agent",
    inputs=["message", "files", "context_refs", "session_id", "user_meta"],
    outputs=["reply"],
    as_agent={
        "id": "aether_agent",
        "title": "Aether Agent",
        "short_description": "Built-in Aether assistant with memory, KB, and routing.",
        "description": (
            "Default Aether agent for general chat, AG how-tos, graph building, "
            "system inspection, routing to specialist agents, and app connectors."
        ),
        "icon_key": "cpu",
        "color": "blue",
        "mode": "chat_v1",
        "memory_level": "user",  # conceptual: user-level memory
    },
)
async def builtin_agent(
    message: str,
    files: list[Any] | None = None,
    session_id: str | None = None,
    user_meta: dict[str, Any] | None = None,
    context_refs: list[dict[str, Any]] | None = None,
    *,
    context: NodeContext,
):
    """
    Aether default agent.

    Responsibilities:
      1. Basic chat with Aether identity & user-level memory.
      2. Answer "what/how in AG" using KB + docs.
      3. Construct / modify AG graphs & agents from user instructions (via other agents).
      4. Execute slash-style kernel commands (e.g., /runs, /graphs).
      5. Route to other registered agents when they are a better fit.
      6. Call external connectors (e.g. Gmail) when requested.
    """
    logger = context.logger()
    mem = context.memory()
    chan = context.ui_session_channel()

    raw_message = (message or "").strip()
    logger.debug("builtin_agent received message: %r (session_id=%s)", raw_message, session_id)

    print("🍎", mem.scope_id)
    print("🍎", mem.scope_info())

    # return {"reply": "Sorry, I couldn't process your message."}  # placeholder until we implement the logic below
    if not raw_message:
        reply = "Hi! What would you like to do with Aether today?"
        await mem.record_chat_assistant(text=reply, tags=["ag.agent_reply"])
        return {"reply": reply}

    # TODO (future): retrieve session_state from a dedicated state service
    session_state: SessionAgentState | None = None
    identity: Any | None = None

    # Classify intent (command / route / chat)
    intent = await _classify_intent(
        message=raw_message,
        identity=identity,
        session_state=session_state,
    )
    logger.debug("builtin_agent intent: %s", intent)

    # -----------------------------------------------------------------------
    # Dispatch by mode
    # -----------------------------------------------------------------------
    if intent.mode == "command":
        reply = await command_handler(
            intent=intent,
            message=raw_message,
            context=context,
        )
        # Commands: send to UI but typically not persisted as conversational memory
        await chan.send_text(reply, memory_log=False)

    elif intent.mode == "route":
        reply = await route_handler(
            intent=intent,
            message=raw_message,
            context=context,
            files=files,
            session_id=session_id,
            user_meta=user_meta,
        )
        await chan.send_text(reply, memory_log=False)

    else:
        # Chat path: delegated to the skills-driven chat handler
        reply = await basic_chat_handler(
            intent=intent,
            message=raw_message,
            files=files,
            context_refs=context_refs,
            session_id=session_id,
            user_meta=user_meta,
            context=context,
        )

    # Record user turn in hotlog (short-term memory) -- Do it after retrieval to avoid contaminating memory with the new message before it's processed.
    await mem.record_chat_user(
        text=raw_message,
        tags=["ag.user_message"],
    )

    # Record assistant reply for non-ephemeral paths
    await mem.record_chat_assistant(
        text=reply,
        tags=["ag.agent_reply"],
    )

    if intent.mode == "chat":
        await _maybe_distill_session(mem=mem, logger=logger)

    return {"reply": reply}


# ---------------------------------------------------------------------------
# Command / route handlers – shapes kept as-is (stubs)
# ---------------------------------------------------------------------------


async def command_handler(
    intent: ClassifiedIntent,
    message: str,
    context: NodeContext,
) -> str:
    """
    Slash command dispatcher.

    Example commands:
      - /graphs        -> list graphs/agents.
      - /runs          -> show recent runs.
      - /triggers      -> list triggers.
      - /logs          -> show recent log lines (or link to logs view).
      - /help          -> explain available commands.
    """
    from aethergraph.plugins.agents.agent_functions.commands import _find_command_spec

    logger = context.logger()
    cmd_name = (intent.command or "").strip().lower()
    args = intent.command_args or ""

    if not cmd_name:
        return 'Empty command. Type "/help" to see available commands.'

    spec = _find_command_spec(cmd_name)
    if not spec:
        return f"Unknown command: /{cmd_name}. Type /help to see available commands."

    try:
        return await spec.handler(intent=intent, args=args, context=context)
    except Exception as e:  # noqa: BLE001
        logger.exception("Error while handling command /%s: %s", cmd_name, e)
        return f"Error while executing /{cmd_name}: {e!r}"


async def route_handler(
    intent: ClassifiedIntent,
    message: str,
    context: NodeContext,
    files: list[Any] | None = None,
    session_id: str | None = None,
    user_meta: dict[str, Any] | None = None,
) -> str:
    """
    TODO: Implement routing to other agents.

    For now this just returns a placeholder message.
    """
    target = intent.target_agent_id or intent.target_agent_alias or "UNKNOWN"
    return f"TODO: route handler not implemented yet (would route to {target})."
