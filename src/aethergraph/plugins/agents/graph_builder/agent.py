# graph_builder.py
from __future__ import annotations

from typing import Any

from aethergraph import graph_fn
from aethergraph.core.runtime.node_context import NodeContext
from aethergraph.plugins.agents.graph_builder.branches import (
    _handle_chat,
    _handle_generate,
    _handle_plan,
    _handle_register_app,
)
from aethergraph.plugins.agents.graph_builder.router import (
    _heuristic_route,
    _llm_route_if_needed,
)
from aethergraph.plugins.agents.graph_builder.types import (
    GraphBuilderBranch,
    _hash_contract,
    _load_state,
    _save_state,
)

# adjust import path to wherever you placed these:
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
            "Hi — I’m the Graph Builder.\n\n"
            "- Describe the workflow you want, or\n"
            "- Upload Python scripts and tell me what to wrap.\n\n"
            "I’ll produce a plan + @tool/@graphify code, and add checkpoints for expensive steps."
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

    # Load state (user level usually makes sense for a builder)
    state = await _load_state(context=context, level="user")

    # Summarize files/context_refs using your helpers
    try:
        code_files, text_files, other_files, notes = await _process_builder_files(
            files=files,
            context_refs=context_refs,
            context=context,
        )
    except Exception as e:
        await chan.send_phase(phase="thinking", status="error", label="Error processing files")
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

    # Route
    decision = _heuristic_route(message=raw_message, files=files, state=state)
    decision = await _llm_route_if_needed(
        decision=decision,
        message=raw_message,
        state=state,
        files_summary=files_summary,
        context=context,
    )

    await chan.send_phase(
        phase="thinking",
        status="done",
        label=f"Graph Builder decided to {decision['branch'].value}...",
    )

    logger.debug("graph_builder route: %s (%s)", decision["branch"].value, decision["reason"])

    # Execute branch
    if decision["branch"] == GraphBuilderBranch.PLAN:
        reply, plan = await _handle_plan(
            message=raw_message,
            files_summary=files_summary,
            state=state,
            context=context,
        )
        if plan:
            state.plan_ver += 1
            state.last_plan_json = plan
            state.last_graph_name = (plan.get("graph") or {}).get("name")
            state.last_contract_hash = _hash_contract(plan)
            await _save_state(context=context, state=state)

    elif decision["branch"] == GraphBuilderBranch.GENERATE:
        reply, plan = await _handle_generate(
            message=raw_message,
            files_summary=files_summary,
            files=[*code_files, *text_files, *other_files],
            state=state,
            context=context,
        )
        if plan:
            state.plan_ver = max(state.plan_ver, 1)  # ensure nonzero if generating
            state.graph_ver += 1
            state.last_plan_json = plan
            state.last_graph_name = (plan.get("graph") or {}).get("name")
            state.last_contract_hash = _hash_contract(plan)
            await _save_state(context=context, state=state)

    elif decision["branch"] == GraphBuilderBranch.REGISTER_APP:
        reply = await _handle_register_app(
            message=raw_message,
            state=state,
            context=context,
        )

    else:
        reply = await _handle_chat(message=raw_message, context=context)

    # Record assistant turn
    await mem.record_chat_assistant(
        text=reply,
        tags=["ag.graph_builder.reply"],
    )
    return {"reply": reply}
