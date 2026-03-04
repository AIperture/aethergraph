# graph_builder.py
from __future__ import annotations

from typing import Any

from aethergraph import graph_fn
from aethergraph.core.runtime.node_context import NodeContext
from aethergraph.plugins.agents.graph_builder.branches_v2 import (
    _handle_chat_v2,
    _handle_generate_v2,
    _handle_plan_v2,
    _handle_register_app_v2,
)
from aethergraph.plugins.agents.graph_builder.router_v2 import route_v2
from aethergraph.plugins.agents.graph_builder.types import (
    GraphBuilderBranch,
)
from aethergraph.plugins.agents.graph_builder.utils import (
    _process_builder_files,
    _summarize_builder_files_for_llm,
)


@graph_fn(
    name="graph_builder",
    inputs=["message", "files", "context_refs", "session_id", "user_meta"],
    outputs=["reply"],
    as_agent={
        "id": "graph_builder",
        "title": "Graph Builder",
        "short_description": "Turns scripts and intent into @tool + @graphify workflows with checkpointing.",
        "description": (
            "Builds reliable AetherGraph workflows by extracting tools from scripts and composing them via graphify. "
            "Adds artifact checkpointing for expensive steps and can emit as_app registration snippets."
        ),
        "icon_key": "workflow",
        "color": "orange",
        "mode": "chat_v1",
        "memory_level": "session",
        "slash_commands": [
            {"name": "/plan", "description": "Draft or revise a build plan (no code)."},
            {"name": "/gen", "description": "Generate code from the current plan."},
            {"name": "/register", "description": "Register the generated graph as an app."},
            {"name": "/chat", "description": "Just chat about the builder without changing state."},
        ],
    },
)
async def graph_builder(
    message: str,
    files: list[Any] | None = None,
    context_refs: list[dict[str, Any]] | None = None,
    session_id: str | None = None,
    user_meta: dict[str, Any] | None = None,
    *,
    context: NodeContext,
) -> dict:
    logger = context.logger()
    mem = context.memory()
    chan = context.ui_session_channel()

    raw_message = (message or "").strip()
    if not raw_message and not (files or []):
        reply = (
            "Hi - I am the Graph Builder.\n\n"
            "- Describe the workflow you want, or\n"
            "- Upload Python scripts and tell me what to wrap.\n\n"
            "I will produce a plan plus @tool/@graphify code, and add checkpoints for expensive steps."
        )
        await mem.record_chat_assistant(
            text=reply,
            tags=["ag.graph_builder.reply"],
        )
        return {"reply": reply}

    await chan.send_phase(
        phase="routing",
        status="active",
        label="Routing request",
        detail="Processing files and conversation context.",
    )

    await mem.record_chat_user(
        text=raw_message,
        tags=["ag.graph_builder.user"],
    )

    try:
        code_files, text_files, other_files, notes = await _process_builder_files(
            files=files,
            context_refs=context_refs,
            context=context,
        )
    except Exception as e:
        await chan.send_phase(
            phase="routing",
            status="failed",
            label="Routing failed",
            detail="Failed while processing input files.",
        )
        await chan.send_text(f"Sorry, I had trouble processing the files you provided: {e}")
        logger.error("Error processing builder files: %s", e, exc_info=True)
        return {"reply": f"Sorry, I had trouble processing the files you provided: {e}"}

    files_summary = _summarize_builder_files_for_llm(
        code_files=code_files,
        text_files=text_files,
        other_files=other_files,
        notes=notes,
    )

    await chan.send_phase(
        phase="routing",
        status="active",
        label="Selecting branch",
        detail="Analyzing intent and current builder state.",
    )

    decision = await route_v2(
        message=raw_message,
        files_summary=files_summary,
        context=context,
    )

    await chan.send_phase(
        phase="routing",
        status="done",
        label="Route selected",
        detail=f"Selected branch: {decision['branch'].value}",
    )

    logger.info("graph_builder route: %s (%s)", decision["branch"].value, decision["reason"])

    if decision["branch"] == GraphBuilderBranch.PLAN:
        reply, _plan = await _handle_plan_v2(
            message=raw_message,
            files_summary=files_summary,
            context=context,
        )

    elif decision["branch"] == GraphBuilderBranch.GENERATE:
        reply, _plan, _generated_code, _generated_filename = await _handle_generate_v2(
            message=raw_message,
            files_summary=files_summary,
            files=[*code_files, *text_files, *other_files],
            context=context,
        )

    elif decision["branch"] == GraphBuilderBranch.REGISTER_APP:
        reply = await _handle_register_app_v2(
            message=raw_message,
            files=[*code_files, *text_files, *other_files],
            context=context,
        )

    else:
        reply = await _handle_chat_v2(message=raw_message, context=context)

    await chan.send_phase(
        phase="finishing",
        status="active",
        label="Finalizing response",
        detail="Saving assistant output to memory.",
    )
    await mem.record_chat_assistant(
        text=reply,
        tags=["ag.graph_builder.reply"],
    )
    await chan.send_phase(
        phase="finishing",
        status="done",
        label="Response ready",
        detail="Handing response back to the session.",
    )
    return {"reply": reply}
