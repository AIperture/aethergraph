from __future__ import annotations

import json

from aethergraph.core.runtime.node_context import NodeContext

from .types import (
    ROUTER_JSON_SCHEMA,
    GraphBuilderBranch,
    GraphBuilderState,
    RouterDecision,
    _detect_approval_intent,
    _load_state,
)


def _slash_route(message: str) -> RouterDecision | None:
    msg = (message or "").strip().lower()
    if not msg.startswith("/"):
        return None
    if msg.startswith("/plan"):
        return {"branch": GraphBuilderBranch.PLAN, "reason": "slash_override:/plan"}
    if msg.startswith("/gen"):
        return {"branch": GraphBuilderBranch.GENERATE, "reason": "slash_override:/gen"}
    if msg.startswith("/register"):
        return {"branch": GraphBuilderBranch.REGISTER_APP, "reason": "slash_override:/register"}
    if msg.startswith("/chat"):
        return {"branch": GraphBuilderBranch.CHAT, "reason": "slash_override:/chat"}
    return None


def _fallback_route(*, message: str, state: GraphBuilderState) -> RouterDecision:
    intent = _detect_approval_intent(message)
    if state.pending_action == "awaiting_plan_approval":
        if intent == "approve":
            return {
                "branch": GraphBuilderBranch.GENERATE,
                "reason": "fallback_pending_plan_approval_approved",
            }
        if intent == "revise":
            return {
                "branch": GraphBuilderBranch.PLAN,
                "reason": "fallback_pending_plan_approval_revise",
            }
        return {
            "branch": GraphBuilderBranch.PLAN,
            "reason": "fallback_pending_plan_approval_default",
        }

    if state.pending_action == "awaiting_register_decision":
        if intent == "decline":
            return {"branch": GraphBuilderBranch.CHAT, "reason": "fallback_register_declined"}
        return {"branch": GraphBuilderBranch.REGISTER_APP, "reason": "fallback_register_pending"}

    return {"branch": GraphBuilderBranch.CHAT, "reason": "fallback_default_chat"}


def _state_override(*, message: str, state: GraphBuilderState) -> RouterDecision | None:
    intent = _detect_approval_intent(message)
    if state.pending_action == "awaiting_plan_approval":
        if intent == "approve":
            return {"branch": GraphBuilderBranch.GENERATE, "reason": "state_override_plan_approved"}
        if intent == "revise":
            return {"branch": GraphBuilderBranch.PLAN, "reason": "state_override_plan_revise"}
    if state.pending_action == "awaiting_register_decision":
        if intent == "decline":
            return {"branch": GraphBuilderBranch.CHAT, "reason": "state_override_register_skip"}
        if intent == "approve":
            return {
                "branch": GraphBuilderBranch.REGISTER_APP,
                "reason": "state_override_register_approve",
            }
    return None


async def _llm_route(
    *,
    message: str,
    state: GraphBuilderState,
    files_summary: str,
    context: NodeContext,
) -> RouterDecision:
    llm = context.llm()
    skills = context.skills()
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
        f"has_plan={'yes' if state.last_plan_json else 'no'}, "
        f"has_pending_plan={'yes' if state.pending_plan_json else 'no'}, "
        f"pending_action={state.pending_action or 'none'}"
    )

    user_prompt = (
        "Route the user message for AG Graph Builder.\n"
        "Choose one branch: chat | plan | generate | register_app.\n"
        "Use user intent + state. If awaiting plan approval and user agrees, route generate. "
        "If awaiting plan approval and user revises scope, route plan.\n\n"
        f"Message:\n{message}\n\n"
        f"State summary:\n{state_summary}\n\n"
        f"Files summary:\n{files_summary}\n\n"
        "Return strict JSON only with shape:\n"
        '{"branch":"chat|plan|generate|register_app","reason":"..."}'
    )

    resp, _usage = await llm.chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        output_format="json",
        json_schema=ROUTER_JSON_SCHEMA,
        schema_name="GraphBuilderRouteV2",
        strict_schema=True,
        validate_json=True,
        max_output_tokens=256,
    )
    print("🍎 Graph Builder: LLM router response:\n", resp)
    obj = json.loads(resp) if isinstance(resp, str) else resp
    branch_str = str(obj.get("branch", "")).strip().lower()
    if branch_str not in {b.value for b in GraphBuilderBranch}:
        raise ValueError(f"invalid branch from llm route: {branch_str}")
    return {
        "branch": GraphBuilderBranch(branch_str),
        "reason": str(obj.get("reason") or "llm_route"),
    }


async def route_v2(
    *,
    message: str,
    files_summary: str,
    context: NodeContext,
) -> RouterDecision:
    state = await _load_state(context=context, level="user")

    slash = _slash_route(message)
    if slash:
        return slash

    state_override = _state_override(message=message, state=state)
    if state_override:
        return state_override

    try:
        decision = await _llm_route(
            message=message,
            state=state,
            files_summary=files_summary,
            context=context,
        )
    except Exception:
        context.logger().exception("graph_builder_v2: llm router failed; fallback engaged")
        return _fallback_route(message=message, state=state)

    if decision["branch"] == GraphBuilderBranch.GENERATE:
        has_plan = bool(state.pending_plan_json or state.last_plan_json)
        if not has_plan:
            return {
                "branch": GraphBuilderBranch.CHAT,
                "reason": "guard_generate_without_plan_redirect_to_chat",
            }
    return decision
