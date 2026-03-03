# types.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Literal, TypedDict, cast

from aethergraph.contracts.services.memory import Event
from aethergraph.core.runtime.node_context import NodeContext

GRAPH_BUILDER_SKILL_ID = "aethergraph-graph-builder"
BUILTIN_AGENT_SKILL_ID = "ag.builtin_agent"  # keep if you want to layer base chat behavior


class GraphBuilderBranch(str, Enum):
    CHAT = "chat"
    PLAN = "plan"
    GENERATE = "generate"
    REGISTER_APP = "register_app"


class RouterDecision(TypedDict):
    branch: GraphBuilderBranch
    reason: str


ApprovalIntent = Literal["approve", "revise", "decline", "unknown"]


def _detect_approval_intent(message: str) -> ApprovalIntent:
    msg = (message or "").strip().lower()
    if not msg:
        return "unknown"

    approve_terms = {"proceed", "generate", "yes", "ok", "approve", "ship it"}
    revise_terms = {"replan", "revise", "change", "update", "modify"}
    decline_terms = {"skip", "no", "decline", "cancel", "reject"}

    if any(term in msg for term in approve_terms):
        return "approve"
    if any(term in msg for term in revise_terms):
        return "revise"
    if any(term in msg for term in decline_terms):
        return "decline"
    return "unknown"


@dataclass
class GraphBuilderState:
    # simple versioning
    plan_ver: int = 0
    graph_ver: int = 0

    # last artifacts produced by the agent (for debugging / introspection)
    last_plan_json: dict[str, Any] | None = None
    last_generated_files: list[dict[str, Any]] | None = None
    last_graph_name: str | None = None

    # small contract hash to detect plan changes
    last_contract_hash: str | None = None

    # v2 pending interaction state
    pending_action: str | None = None  # awaiting_plan_approval | awaiting_register_decision
    pending_plan_json: dict[str, Any] | None = None

    # latest generated code metadata
    last_generated_code: str | None = None
    last_generated_filename: str | None = None

    # last registered app metadata
    last_registered_app_id: str | None = None
    last_registered_app_version: str | None = None


STATE_KEY = "ag.graph_builder.state"  # stable key for memory state snapshots


async def _load_state(
    *,
    context: NodeContext,
    level: str | None = None,
) -> GraphBuilderState:
    """
    Load the latest GraphBuilderState from memory.

    `level` is usually "user" for user-level persistence. When level=="user",
    we also enable user_persistence so it survives process restarts.
    """
    mem = context.memory()
    tag = f"state:{STATE_KEY}"
    recent_events: list[Event] = []
    try:
        recent_events = await mem.recent(
            kinds=["state.snapshot"],
            limit=80,
            level=level,
            return_event=True,
        )
    except Exception:
        context.logger().exception("graph_builder: recent() failed while loading state")

    for event in reversed(recent_events):
        tags = event.tags or []
        if tag not in tags:
            continue
        payload = event.data or {}
        raw = payload.get("value")
        if raw is None:
            continue
        try:
            return GraphBuilderState(**cast(dict[str, Any], raw))
        except Exception:
            context.logger().warning("graph_builder: invalid state payload in recent(); skipping")

    try:
        raw = await mem.latest_state(
            STATE_KEY,
            level=level,
            user_persistence=(level == "user"),
        )
    except Exception:
        context.logger().exception("graph_builder: latest_state failed, returning default")
        return GraphBuilderState()

    if not raw:
        return GraphBuilderState()

    try:
        return GraphBuilderState(**cast(dict[str, Any], raw))
    except Exception:
        # if schema evolves, fail soft
        context.logger().warning(
            "graph_builder: failed to load GraphBuilderState, returning default",
        )
        return GraphBuilderState()


async def _save_state(
    *,
    context: NodeContext,
    state: GraphBuilderState,
) -> None:
    """
    Persist GraphBuilderState as a state snapshot.

    Scope / level is inferred from the current NodeContext scope; we don't
    override it explicitly here.
    """
    await context.memory().record_state(
        key=STATE_KEY,
        value=asdict(state),
        tags=["ag", "graph_builder", "state"],
        meta={"schema": "GraphBuilderState", "ver": 1},
        severity=1,
    )


def _hash_contract(plan: dict[str, Any]) -> str:
    """
    A stable hash of the semantic contract: tools + graph IO + checkpointing policy.
    """
    contract = {
        "graph": plan.get("graph"),
        "tools": plan.get("tools"),
        "needs": plan.get("needs"),
        "checkpointing": plan.get("checkpointing"),
    }
    s = json.dumps(contract, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ---- JSON schemas for structured LLM output ---------------------------------

# Inner builder plan schema (rough shape; actual schema lives in references/)
BUILDER_PLAN_INNER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "graph": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "type": {"type": "string"},
                "inputs": {"type": "array", "items": {"type": "string"}},
                "outputs": {"type": "array", "items": {"type": "string"}},
                "description": {"type": "string"},
            },
            "required": ["name", "type", "inputs", "outputs"],
            "additionalProperties": True,
        },
        "tools": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "kind": {"type": "string"},
                    "source": {"type": "string"},
                    "inputs": {"type": "array", "items": {"type": "string"}},
                    "outputs": {"type": "array", "items": {"type": "string"}},
                    "expensive": {"type": "boolean"},
                    "notes": {"type": "string"},
                },
                "required": ["name"],
                "additionalProperties": True,
            },
        },
        "needs": {
            "type": "object",
            "properties": {
                "channel": {"type": "boolean"},
                "artifacts": {"type": "boolean"},
                "memory": {"type": "boolean"},
            },
            "additionalProperties": True,
        },
        "checkpointing": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string"},
                    "ckpt_key": {"type": "string"},
                    "resume_strategy": {"type": "string"},
                },
                "required": ["tool", "ckpt_key"],
                "additionalProperties": True,
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["graph", "tools", "needs"],
    "additionalProperties": True,
}

# Wrapper schema used for plan branch: explanation + plan
BUILDER_PLAN_WRAPPER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
        "plan": BUILDER_PLAN_INNER_SCHEMA,
    },
    "required": ["plan"],
    "additionalProperties": True,
}

# Router schema (branch classifier)
ROUTER_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "branch": {
            "type": "string",
            "enum": [b.value for b in GraphBuilderBranch],
        },
        "reason": {"type": "string"},
    },
    "required": ["branch"],
    "additionalProperties": True,
}
