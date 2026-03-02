# router.py
from __future__ import annotations

import json
import re
from typing import Any

from aethergraph.core.runtime.node_context import NodeContext

from .types import (
    ROUTER_JSON_SCHEMA,
    GraphBuilderBranch,
    GraphBuilderState,
    RouterDecision,
)

# If you’ve placed the file helpers in a module, adjust this import accordingly:
# from .files import _infer_builder_mode


def _heuristic_route(
    *,
    message: str,
    files: list[Any] | None,
    state: GraphBuilderState,
) -> RouterDecision:
    msg = (message or "").strip().lower()

    # explicit commands
    if msg.startswith("/plan") or "plan:" in msg or "make a plan" in msg:
        return {"branch": GraphBuilderBranch.PLAN, "reason": "explicit plan request"}
    if msg.startswith("/gen") or "generate" in msg or "graphify" in msg or "code" in msg:
        return {"branch": GraphBuilderBranch.GENERATE, "reason": "explicit generation request"}
    if "register" in msg or "as_app" in msg or ("app" in msg and "register" in msg):
        return {
            "branch": GraphBuilderBranch.REGISTER_APP,
            "reason": "explicit register-as-app request",
        }

    # if user uploaded scripts, default to GENERATE (wrapping)
    if files:
        return {
            "branch": GraphBuilderBranch.GENERATE,
            "reason": "scripts provided; likely wrapping into tools/graphify",
        }

    # if we already have a plan (state), and user says "ok" / "looks good" / "confirm"
    if state.last_plan_json and re.search(r"\b(ok|yes|confirm|looks good|ship it)\b", msg):
        return {
            "branch": GraphBuilderBranch.GENERATE,
            "reason": "confirming prior plan; proceed to generate",
        }

    # default: chat
    return {"branch": GraphBuilderBranch.CHAT, "reason": "default to chat/help"}


async def _llm_route_if_needed(
    *,
    decision: RouterDecision,
    message: str,
    state: GraphBuilderState,
    files_summary: str,
    context: NodeContext,
) -> RouterDecision:
    """
    If the heuristic landed on CHAT but the message smells "buildy", call the LLM router.

    Uses context.llm().chat(..., output_format="json", json_schema=ROUTER_JSON_SCHEMA)
    so we get a strict JSON object back.
    """
    if decision["branch"] != GraphBuilderBranch.CHAT:
        return decision

    msg = (message or "").strip().lower()
    maybe_build = any(
        k in msg for k in ["workflow", "pipeline", "graph", "tool", "wrap", "script", "checkpoint"]
    )
    if not maybe_build:
        return decision

    llm = context.llm()
    skills = context.skills()

    # System prompt comes from the skill (graph_builder.system + graph_builder.router)
    system_prompt = skills.compile_prompt(
        "aethergraph-graph-builder",
        "graph_builder.system",
        "graph_builder.router",
        "graph_builder.style",
        separator="\n\n",
        fallback_keys=["graph_builder.system"],
    )

    state_summary = (
        f"plan_ver={state.plan_ver}, graph_ver={state.graph_ver}, "
        f"last_graph_name={state.last_graph_name or 'None'}, "
        f"has_plan={'yes' if state.last_plan_json else 'no'}"
    )

    user_content = (
        "You are routing a message for the AG Graph Builder.\n\n"
        f"Message:\n{message}\n\n"
        f"State summary:\n{state_summary}\n\n"
        f"Files summary:\n{files_summary}\n\n"
        "Return strict JSON ONLY, no prose.\n"
        'Shape: {"branch": "...", "reason": "..."}\n'
        'branch ∈ {"chat","plan","generate","register_app"}.'
    )

    try:
        resp, _usage = await llm.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            output_format="json",
            json_schema=ROUTER_JSON_SCHEMA,
            schema_name="GraphBuilderRoute",
            strict_schema=True,
            validate_json=True,
            max_output_tokens=256,
        )
    except Exception:
        context.logger().exception("graph_builder: LLM router failed, falling back to heuristic")
        return decision

    try:
        obj = json.loads(resp) if isinstance(resp, str) else resp
    except Exception:
        return decision

    branch_str = str(obj.get("branch", "")).strip().lower()
    if branch_str not in {b.value for b in GraphBuilderBranch}:
        return decision

    reason = str(obj.get("reason") or "llm_routed")
    return {
        "branch": GraphBuilderBranch(branch_str),
        "reason": reason,
    }
