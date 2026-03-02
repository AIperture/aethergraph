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
        "memory_level": "user",
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
        phase="thinking", status="active", label="Graph Builder is working on it..."
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
        await chan.send_phase(phase="thinking", status="failed", label="Error processing files")
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
        phase="thinking", status="active", label="Graph Builder is analyzing your input..."
    )

    decision = await route_v2(
        message=raw_message,
        files_summary=files_summary,
        context=context,
    )

    await chan.send_phase(
        phase="thinking",
        status="done",
        label=f"Graph Builder decided to {decision['branch'].value}...",
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
            context=context,
        )

    else:
        reply = await _handle_chat_v2(message=raw_message, context=context)

    await mem.record_chat_assistant(
        text=reply,
        tags=["ag.graph_builder.reply"],
    )
    return {"reply": reply}
