from __future__ import annotations

import json
import re
from typing import Any

from aethergraph.contracts.services.channel import Button
from aethergraph.core.runtime.node_context import NodeContext
from aethergraph.plugins.agents.graph_builder.registration_utils import (
    _ensure_graph_registered_from_code,
    _propose_app_config_via_llm,
    _resolve_registration_target,
)
from aethergraph.services.llm.generic_client import GenericLLMClient

from .types import (
    BUILDER_PLAN_WRAPPER_SCHEMA,
    GRAPH_BUILDER_SKILL_ID,
    GraphBuilderBranch,
    _detect_approval_intent,
    _hash_contract,
    _load_state,
    _save_state,
)

NODE_API_SKILL_ID = "ag-graph-builder-context-node-api"
CHANNEL_API_SKILL_ID = "ag-graph-builder-channel-api"
ARTIFACT_API_SKILL_ID = "ag-graph-builder-artifact-api"
GRAPHIFY_STYLE_SKILL_ID = "ag-graph-builder-graphify-style"
CHECKPOINT_SKILL_ID = "ag-graph-builder-checkpoint-pattern"


async def _recent_chat_for_llm(*, context: NodeContext, limit: int = 20) -> list[dict[str, str]]:
    mem = context.memory()
    rows = await mem.recent_chat(limit=limit)
    msgs: list[dict[str, str]] = []
    for r in rows:
        role = (r.get("role") or "user").lower()
        text = (r.get("text") or "").strip()
        if not text:
            continue
        if role not in {"user", "assistant", "system"}:
            role = "assistant"
        msgs.append({"role": role, "content": text})
    return msgs


def _compile_branch_prompt(*, context: NodeContext, branch: GraphBuilderBranch) -> str:
    skills = context.skills()
    if branch == GraphBuilderBranch.PLAN:
        base = skills.compile_prompt(
            GRAPH_BUILDER_SKILL_ID,
            "graph_builder.system",
            "graph_builder.plan",
            "graph_builder.style",
            separator="\n\n",
            fallback_keys=["graph_builder.system"],
        )
        node_api = skills.compile_prompt(NODE_API_SKILL_ID, separator="\n\n")
        chan_api = skills.compile_prompt(CHANNEL_API_SKILL_ID, separator="\n\n")
        return base + "\n\n----- API REFERENCE -----\n\n" + node_api + "\n\n" + chan_api

    if branch == GraphBuilderBranch.GENERATE:
        base = skills.compile_prompt(
            GRAPH_BUILDER_SKILL_ID,
            "graph_builder.system",
            "graph_builder.plan",
            "graph_builder.codegen",
            "graph_builder.style",
            separator="\n\n",
            fallback_keys=["graph_builder.system"],
        )
        node_api = skills.compile_prompt(NODE_API_SKILL_ID, separator="\n\n")
        chan_api = skills.compile_prompt(CHANNEL_API_SKILL_ID, separator="\n\n")
        art_api = skills.compile_prompt(ARTIFACT_API_SKILL_ID, separator="\n\n")
        graphify_style = skills.compile_prompt(GRAPHIFY_STYLE_SKILL_ID, separator="\n\n")
        ckpt = skills.compile_prompt(CHECKPOINT_SKILL_ID, separator="\n\n")
        return (
            base
            + "\n\n----- API REFERENCE -----\n\n"
            + node_api
            + "\n\n"
            + chan_api
            + "\n\n"
            + art_api
            + "\n\n----- PATTERNS -----\n\n"
            + graphify_style
            + "\n\n"
            + ckpt
        )

    if branch == GraphBuilderBranch.CHAT:
        return skills.compile_prompt(
            GRAPH_BUILDER_SKILL_ID,
            "graph_builder.system",
            "graph_builder.chat",
            "graph_builder.style",
            separator="\n\n",
            fallback_keys=["graph_builder.system"],
        )

    return skills.compile_prompt(
        GRAPH_BUILDER_SKILL_ID,
        "graph_builder.system",
        "graph_builder.register",
        "graph_builder.style",
        separator="\n\n",
        fallback_keys=["graph_builder.system"],
    )


def _plan_card(plan: dict[str, Any]) -> dict[str, Any]:
    graph = plan.get("graph") or {}
    tools = plan.get("tools") or []
    checkpoints = plan.get("checkpointing") or []
    needs = plan.get("needs") or {}
    return {
        "kind": "card",
        "title": f"Plan: {graph.get('name') or 'proposed_graph'}",
        "sections": [
            {"label": "Type", "value": str(graph.get("type") or "graphify")},
            {
                "label": "I/O",
                "value": f"in={graph.get('inputs') or []} | out={graph.get('outputs') or []}",
            },
            {"label": "Tools", "value": f"{len(tools)} planned"},
            {"label": "Checkpointing", "value": f"{len(checkpoints)} step(s)"},
            {"label": "Needs", "value": json.dumps(needs, ensure_ascii=False)},
            {
                "label": "Tool List",
                "value": [str(t.get("name") or "unnamed_tool") for t in tools] or ["(none)"],
            },
            {
                "label": "Checkpoint List",
                "value": [str(c.get("tool") or "unknown_tool") for c in checkpoints] or ["(none)"],
            },
        ],
    }


def _extract_python_code(markdown_text: str) -> str | None:
    m = re.search(r"```python\s*(.*?)\s*```", markdown_text or "", flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


async def _resolve_codegen_llm(*, context: NodeContext) -> GenericLLMClient:
    try:
        return context.llm("coding")
    except Exception:
        context.logger().warning(
            "graph_builder_v2: context.llm('coding') unavailable; fallback to default"
        )
        return context.llm()


async def _handle_plan_v2(
    *,
    message: str,
    files_summary: str,
    context: NodeContext,
) -> tuple[str, dict[str, Any] | None]:
    llm = context.llm()
    chan = context.ui_session_channel()
    system_prompt = _compile_branch_prompt(context=context, branch=GraphBuilderBranch.PLAN)
    history = await _recent_chat_for_llm(context=context, limit=20)

    user_prompt = (
        "Produce plan only. No implementation code.\n"
        'Return strict JSON with shape: {"explanation":"...","plan":{...}}.\n\n'
        f"User request:\n{message}\n\n"
        f"Files summary:\n{files_summary}\n"
    )

    await chan.send_phase(phase="thinking", status="active", label="Building plan...")
    try:
        resp, _usage = await llm.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                *history,
                {"role": "user", "content": user_prompt},
            ],
            output_format="json",
            json_schema=BUILDER_PLAN_WRAPPER_SCHEMA,
            schema_name="BuilderPlanWrapperV2",
            strict_schema=True,
            validate_json=True,
            max_output_tokens=2048,
        )
    except Exception:
        await chan.send_phase(phase="thinking", status="failed", label="Planning failed.")
        raise

    obj = json.loads(resp) if isinstance(resp, str) else resp
    explanation = str(obj.get("explanation") or "").strip()
    plan = obj.get("plan") or {}

    await chan.send_rich(
        text=explanation or "I drafted a build plan. Review and choose the next action.",
        rich=_plan_card(plan),
        memory_log=False,
    )
    await chan.send_buttons(
        text="Approve this plan to generate code, or replan with more details.",
        buttons=[
            Button(label="Proceed", value="proceed"),
            Button(label="Replan", value="replan"),
        ],
        memory_log=False,
    )
    await chan.send_phase(phase="thinking", status="done", label="Plan is ready.")

    if plan:
        state = await _load_state(context=context, level="user")
        state.plan_ver += 1
        state.pending_plan_json = plan
        state.pending_action = "awaiting_plan_approval"
        state.last_graph_name = (plan.get("graph") or {}).get("name")
        state.last_contract_hash = _hash_contract(plan)
        await _save_state(context=context, state=state)

    return (
        "Plan prepared. Click Proceed to generate code, or Replan and share additional requirements.",
        plan,
    )


async def _handle_generate_v2(
    *,
    message: str,
    files_summary: str,
    files: list[Any],
    context: NodeContext,
) -> tuple[str, dict[str, Any] | None, str | None, str | None]:
    chan = context.ui_session_channel()
    state = await _load_state(context=context, level="user")
    plan = state.pending_plan_json or state.last_plan_json
    if not plan:
        return (
            "I do not have an approved plan yet. Please run /plan first or provide requirements for planning.",
            None,
            None,
            None,
        )

    llm = await _resolve_codegen_llm(context=context)
    system_prompt = _compile_branch_prompt(context=context, branch=GraphBuilderBranch.GENERATE)
    history = await _recent_chat_for_llm(context=context, limit=16)

    user_prompt = (
        "Generate implementation code from the approved plan.\n"
        "Do not stream output instructions.\n"
        "Return Markdown with: brief explanation, plan JSON block, python code block, file manifest.\n\n"
        f"User message:\n{message}\n\n"
        f"Approved plan:\n{json.dumps(plan, indent=2, ensure_ascii=False)}\n\n"
        f"Files summary:\n{files_summary}\n"
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        *history,
        {"role": "user", "content": user_prompt},
    ]
    for f in files:
        try:
            text = await context.artifacts().load_text(uri=f.uri)
        except Exception:
            context.logger().warning(
                "graph_builder_v2: skipping non-text artifact in codegen prompt: %s", f
            )
            continue
        messages.append({"role": "system", "content": f"\n--- Relevant file: {f.name} ---\n{text}"})

    await chan.send_phase(phase="thinking", status="active", label="Generating code artifact...")
    try:
        resp_text, _usage = await llm.chat(
            messages=messages,
            max_output_tokens=4096,
            request_timeout_s=240,
        )
    except Exception:
        await chan.send_phase(phase="thinking", status="failed", label="Code generation failed.")
        raise
    print("🍎 Graph Builder: Code generation LLM response:\n", resp_text)

    new_plan: dict[str, Any] | None = None
    m_json = re.search(r"```json\s*(\{.*?\})\s*```", resp_text, flags=re.DOTALL)
    if m_json:
        try:
            wrapper = json.loads(m_json.group(1))
            if isinstance(wrapper, dict) and isinstance(wrapper.get("plan"), dict):
                new_plan = wrapper["plan"]
        except Exception:
            new_plan = None

    code = _extract_python_code(resp_text)
    if not code:
        await chan.send_phase(
            phase="thinking", status="failed", label="No Python code block found."
        )
        return (
            "I could not extract Python code from the generation output. Please replan or retry.",
            None,
            None,
            None,
        )

    graph_name = (
        (new_plan or plan).get("graph", {}).get("name")
        if isinstance((new_plan or plan).get("graph"), dict)
        else None
    ) or "generated_graph"
    filename = f"{graph_name}_v{max(state.graph_ver + 1, 1)}.py"
    await chan.send_file(
        file_bytes=code.encode("utf-8"),
        filename=filename,
        title=f"Generated code: {filename}",
        memory_log=False,
    )
    await chan.send_buttons(
        text="Code generated. Do you want to register it as an app?",
        buttons=[
            Button(label="Register App", value="register"),
            Button(label="Skip", value="skip"),
        ],
        memory_log=False,
    )
    await chan.send_phase(phase="thinking", status="done", label="Code generation complete.")

    latest_state = await _load_state(context=context, level="user")
    merged_plan = new_plan or plan
    if merged_plan:
        latest_state.plan_ver = max(latest_state.plan_ver, 1)
        latest_state.graph_ver += 1
        latest_state.last_plan_json = merged_plan
        latest_state.pending_plan_json = None
        latest_state.last_graph_name = (merged_plan.get("graph") or {}).get("name")
        latest_state.last_contract_hash = _hash_contract(merged_plan)
        latest_state.pending_action = "awaiting_register_decision"
        latest_state.last_generated_code = code
        latest_state.last_generated_filename = filename
        latest_state.last_generated_files = [{"name": filename, "kind": "python"}]
        await _save_state(context=context, state=latest_state)

    return (
        "I generated the code and sent it as a file.",
        merged_plan,
        code,
        filename,
    )


async def _handle_register_app_v2(
    *,
    message: str,
    files: list[Any] | None,
    context: NodeContext,
) -> str:
    """
    Register the selected graph as an app so the UI can discover it.

    - Chooses a graph (files → last generated).
    - Ensures the graph is actually registered (by executing its code if needed).
    - Uses LLM to propose an AppConfig-like dict.
    - Registers it in the UnifiedRegistry under nspace='app'.
    - Stores last_registered_app info in builder state.
    """
    logger = context.logger()
    chan = context.ui_session_channel()

    state = await _load_state(context=context, level="user")

    # Clear pending action if we were waiting for registration decision
    if state.pending_action == "awaiting_register_decision":
        state.pending_action = None

    graph_name, code_text = await _resolve_registration_target(
        context=context,
        files=files,
    )

    if not graph_name:
        await _save_state(context=context, state=state)
        msg = (
            "I couldn't find a graph to register.\n\n"
            "- Provide a file containing a @graphify/@graph_fn, or\n"
            "- Re-run generation so I have a recent graph to register."
        )
        return msg

    # 🔑 Make sure the graph actually exists in the registry
    await _ensure_graph_registered_from_code(
        context=context,
        graph_name=graph_name,
        code_text=code_text,
    )

    plan = state.last_plan_json or state.pending_plan_json or {}
    app_config = await _propose_app_config_via_llm(
        context=context,
        graph_name=graph_name,
        plan=plan,
        user_message=message or "",
    )

    app_id = app_config.get("id") or graph_name
    app_version = "0.1.0"  # you can evolve this to be smarter later

    # ---- register in UnifiedRegistry ----
    registry = context.registry()

    meta: dict[str, Any] = {
        "flow_id": app_config.get("flow_id") or graph_name,
        "graph_name": graph_name,
        "builder": "graph_builder_v2",
    }
    if state.last_generated_filename:
        meta["filename"] = state.last_generated_filename
    if state.last_generated_files:
        meta["generated_files"] = state.last_generated_files

    registry.register(
        nspace="app",
        name=app_id,
        version=app_version,
        obj=app_config,
        meta=meta,
    )

    # Optional: tag a 'stable' alias on first registration
    try:
        registry.alias(nspace="app", name=app_id, tag="stable", to_version=app_version)
    except Exception:
        logger.debug(
            "graph_builder_v2: could not alias app %s@%s as 'stable'",
            app_id,
            app_version,
        )

    # Update builder state for future convenience
    state.last_registered_app_id = app_id
    state.last_registered_app_version = app_version
    await _save_state(context=context, state=state)

    await chan.send_text(
        f"Registered app **{app_config.get('name', app_id)}** "
        f"(id=`{app_id}`, flow_id=`{app_config.get('flow_id', graph_name)}`).\n\n"
        "You can now run it from the App Gallery."
    )

    return (
        f"I've registered the app **{app_config.get('name', app_id)}** "
        f"with id `{app_id}`. You should now see it in the App Gallery, "
        f"backed by flow_id `{app_config.get('flow_id', graph_name)}`."
    )


async def _handle_chat_v2(*, message: str, context: NodeContext) -> str:
    msg = (message or "").strip().lower()
    intent = _detect_approval_intent(msg)
    state = await _load_state(context=context, level="user")
    has_plan = bool(state.pending_plan_json or state.last_plan_json)
    if intent == "approve" and not has_plan:
        return (
            "I do not have a plan yet. Use /plan or share requirements and I will draft one first, "
            "then we can proceed to generation."
        )
    if state.pending_action == "awaiting_plan_approval":
        return "If you want changes, tell me what to adjust and I will replan. If ready, click Proceed."
    if state.pending_action == "awaiting_register_decision" and intent == "decline":
        state.pending_action = None
        await _save_state(context=context, state=state)
        return "Okay, skipping app registration for now."

    llm = context.llm()
    system_prompt = _compile_branch_prompt(context=context, branch=GraphBuilderBranch.CHAT)
    history = await _recent_chat_for_llm(context=context, limit=18)
    text, _ = await llm.chat(
        messages=[
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": message},
        ],
        max_output_tokens=2048,
    )
    return text
