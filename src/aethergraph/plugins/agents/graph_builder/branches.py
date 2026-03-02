# branches.py
from __future__ import annotations

import json
import re
from typing import Any

from aethergraph.core.runtime.node_context import NodeContext
from aethergraph.services.llm.generic_client import GenericLLMClient

from .types import (
    BUILDER_PLAN_WRAPPER_SCHEMA,
    GRAPH_BUILDER_SKILL_ID,
    GraphBuilderBranch,
    GraphBuilderState,  # if you need it elsewhere
)

NODE_API_SKILL_ID = "ag-graph-builder-context-node-api"
CHANNEL_API_SKILL_ID = "ag-graph-builder-channel-api"
ARTIFACT_API_SKILL_ID = "ag-graph-builder-artifact-api"
GRAPHIFY_STYLE_SKILL_ID = "ag-graph-builder-graphify-style"
CHECKPOINT_SKILL_ID = "ag-graph-builder-checkpoint-pattern"


async def _recent_chat_for_llm(
    *,
    context: NodeContext,
    limit: int = 20,
) -> list[dict[str, str]]:
    mem = context.memory()
    rows = await mem.recent_chat(limit=limit)  # normalized dicts per docs
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

    # --- base branch prompt from SKILL.md ---
    if branch == GraphBuilderBranch.PLAN:
        base = skills.compile_prompt(
            GRAPH_BUILDER_SKILL_ID,
            "graph_builder.system",
            "graph_builder.plan",
            "graph_builder.style",
            separator="\n\n",
            fallback_keys=["graph_builder.system"],
        )
    elif branch == GraphBuilderBranch.GENERATE:
        base = skills.compile_prompt(
            GRAPH_BUILDER_SKILL_ID,
            "graph_builder.system",
            "graph_builder.plan",
            "graph_builder.codegen",
            "graph_builder.style",
            separator="\n\n",
            fallback_keys=["graph_builder.system"],
        )
    elif branch == GraphBuilderBranch.REGISTER_APP:
        base = skills.compile_prompt(
            GRAPH_BUILDER_SKILL_ID,
            "graph_builder.system",
            "graph_builder.register",
            "graph_builder.style",
            separator="\n\n",
            fallback_keys=["graph_builder.system"],
        )
    else:  # CHAT
        base = skills.compile_prompt(
            GRAPH_BUILDER_SKILL_ID,
            "graph_builder.system",
            "graph_builder.chat",
            "graph_builder.style",
            separator="\n\n",
            fallback_keys=["graph_builder.system"],
        )

    # --- attach references depending on branch ---
    # Plan needs schema + a bit of NodeContext awareness, but not the full pattern pack.
    if branch == GraphBuilderBranch.PLAN:
        node_api = skills.compile_prompt(NODE_API_SKILL_ID, separator="\n\n")
        return base + "\n\n----- NODECONTEXT API (CURATED) -----\n\n" + node_api

    # Codegen needs the full batteries: NodeContext + channel + artifacts + graphfiy style + checkpoint pattern.
    if branch == GraphBuilderBranch.GENERATE:
        node_api = skills.compile_prompt(NODE_API_SKILL_ID, separator="\n\n")
        chan_api = skills.compile_prompt(CHANNEL_API_SKILL_ID, separator="\n\n")
        art_api = skills.compile_prompt(ARTIFACT_API_SKILL_ID, separator="\n\n")
        graphify_style = skills.compile_prompt(GRAPHIFY_STYLE_SKILL_ID, separator="\n\n")
        ckpt = skills.compile_prompt(CHECKPOINT_SKILL_ID, separator="\n\n")

        return (
            base
            + "\n\n----- API REFERENCE (CURATED) -----\n\n"
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

    # Register + Chat can stay light-weight (or you can give them NodeContext reference too if you want).
    return base


# -------- Plan Branch --------------------------------------------------------


async def _handle_plan(
    *,
    message: str,
    files_summary: str,
    state: GraphBuilderState,
    context: NodeContext,
) -> tuple[str, dict[str, Any] | None]:
    """
    PLAN branch:

    - Use structured JSON output with BUILDER_PLAN_WRAPPER_SCHEMA.
    - Return a human-readable reply and the parsed inner plan dict.
    """
    llm = context.llm()
    system_prompt = _compile_branch_prompt(context=context, branch=GraphBuilderBranch.PLAN)
    history = await _recent_chat_for_llm(context=context, limit=20)

    user_prompt = (
        "You are producing a plan JSON that follows the builder plan schema.\n\n"
        f"User request:\n{message}\n\n"
        f"Files summary:\n{files_summary}\n\n"
        "Respond as JSON ONLY (no Markdown fences) with shape:\n"
        '{ "explanation": "...", "plan": { ... } }\n'
        "- `explanation`: a short natural-language explanation (5–12 lines).\n"
        "- `plan`: the builder plan object as described in your skill.\n"
    )

    resp, _usage = await llm.chat(
        messages=[
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": user_prompt},
        ],
        output_format="json",
        json_schema=BUILDER_PLAN_WRAPPER_SCHEMA,
        schema_name="BuilderPlanWrapper",
        strict_schema=True,
        validate_json=True,
        max_output_tokens=2048,
    )

    try:
        obj = json.loads(resp) if isinstance(resp, str) else resp
    except Exception:
        # If something goes wrong, fall back to treating resp as text.
        context.logger().warning("graph_builder: plan JSON parsing failed, returning raw text")
        return str(resp), None

    explanation = str(obj.get("explanation") or "").strip()
    plan = obj.get("plan") or {}

    # Construct the reply we show in chat: explanation + pretty-printed plan JSON
    pretty_plan = json.dumps(plan, indent=2, ensure_ascii=False)
    reply = (explanation + "\n\n") if explanation else "" + "```json\n" + pretty_plan + "\n```"

    return reply, plan  # plan is the inner plan dict used for state & hashing


# -------- Generate Branch ----------------------------------------------------


async def _handle_generate(
    *,
    message: str,
    files_summary: str,
    files: list[Any],
    state: GraphBuilderState,
    context: NodeContext,
) -> tuple[str, dict[str, Any] | None]:
    print("🍎 Graph Builder: starting _handle_generate with message:", message)
    llm: GenericLLMClient = context.llm()
    chan = context.ui_session_channel()

    system_prompt = _compile_branch_prompt(context=context, branch=GraphBuilderBranch.GENERATE)
    history = await _recent_chat_for_llm(context=context, limit=16)

    prior = ""
    if state.last_plan_json:
        prior = "Prior plan JSON (may be reused if still valid):\n" + json.dumps(
            state.last_plan_json,
            indent=2,
            ensure_ascii=False,
        )

    user_prompt = (
        "You are generating an AetherGraph workflow using @tool + @graphify.\n\n"
        f"User request:\n{message}\n\n"
        f"Files summary:\n{files_summary}\n\n"
        f"{prior}\n\n"
        "Requirements:\n"
        "- If scripts exist: wrap functions as @tool; do NOT rewrite scripts.\n"
        "- If expensive/iterative: add artifact checkpoint save/load around the expensive tool(s).\n"
        "- Use context.channel().send_text for progress.\n\n"
        "Return (as Markdown):\n"
        "1) brief explanation\n"
        '2) ```json\n{ "explanation": "...", "plan": { ... } }\n```\n'
        "3) ```python ...``` code (complete)\n"
        "4) file manifest list (paths + purpose)\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        *history,
        {"role": "user", "content": user_prompt},
    ]

    # append files summary to the last message
    for f in files:
        print(f"Graph Builder: loading content of file {f} for LLM input...")
        text = await context.artifacts().load_text(uri=f.uri)
        print(
            f"Graph Builder: appending relevant file {f.name} (first 200 chars): {text[:200]} to LLM messages..."
        )
        messages.append(
            {"role": "system", "content": f"\n\n--- Relevant file: {f.name} ---\n{text}"}
        )

    print(f"Graph Builder: sending {len(messages)} messages to LLM for code generation...")

    await chan.send_phase(
        phase="thinking", status="active", label="Graph Builder is generating code..."
    )

    # Stream to UI
    async with chan.stream() as s:

        async def on_delta(piece: str) -> None:
            await s.delta(piece)

        resp_text, _usage = await llm.chat_stream(
            messages=messages,
            on_delta=on_delta,
        )
        await s.end(full_text=resp_text, memory_log=False)

    # await chan.send_phase(phase="thinking", status="active", label="Graph Builder is generating code...")
    # resp_text, _usage = await llm.chat(
    #     messages=messages,
    #     max_output_tokens=4096,
    # )

    # await chan.send_phase(phase="thinking", status="done", label="Graph Builder finished generating.")
    # await chan.send_text(resp_text)

    # Parse plan JSON block from the full response
    plan: dict[str, Any] | None = None
    m = re.search(r"```json\s*(\{.*?\})\s*```", resp_text, flags=re.DOTALL)
    if m:
        try:
            wrapper = json.loads(m.group(1))
            # allow both raw plan and wrapper {explanation, plan}
            if "plan" in wrapper and isinstance(wrapper["plan"], dict):
                plan = wrapper["plan"]
            else:
                plan = wrapper
        except Exception:
            plan = None

    return resp_text, plan


# -------- Register-as-App Branch --------------------------------------------


async def _handle_register_app(
    *,
    message: str,
    state: GraphBuilderState,
    context: NodeContext,
) -> str:
    llm = context.llm()
    system_prompt = _compile_branch_prompt(context=context, branch=GraphBuilderBranch.REGISTER_APP)
    history = await _recent_chat_for_llm(context=context, limit=10)

    graph_name = state.last_graph_name
    if not graph_name and state.last_plan_json:
        graph_name = (state.last_plan_json.get("graph") or {}).get("name")

    user_prompt = (
        "Generate an AetherGraph app registration wrapper.\n\n"
        f"User request:\n{message}\n\n"
        f"Known graph_name:\n{graph_name or 'unknown'}\n\n"
        "Output:\n"
        "- Explain what will be registered\n"
        "- Provide a complete ```python``` snippet showing @graph_fn or @graphify with as_app={...}\n"
        "- Keep it minimal; do not invent non-existent APIs."
    )

    text, _ = await llm.chat(
        messages=[
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": user_prompt},
        ],
        max_output_tokens=2048,
    )
    return text


# -------- Chat Branch --------------------------------------------------------


async def _handle_chat(
    *,
    message: str,
    context: NodeContext,
) -> str:
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
