# aethergraph/runtime/agent_app_meta.py
from __future__ import annotations

from collections.abc import Callable
import inspect
from typing import Any, Literal, TypedDict

from aethergraph.core.runtime.run_types import RunImportance, RunVisibility

# ---------------------------------------------------------------------
# Config schemas used by decorators
# ---------------------------------------------------------------------


class AgentConfig(TypedDict, total=False):
    # Identity & basic UI
    id: str
    title: str
    description: str

    icon: str
    color: str
    badge: str
    category: str
    status: str  # "available" | "coming-soon" | "hidden" | ...

    # Behavior & wiring
    mode: str  # "chat_v1", etc.
    session_kind: str  # "chat", "batch", ...

    flow_id: str
    tags: list[str]
    tool_graphs: list[str]

    # Runtime behavior
    run_visibility: RunVisibility
    run_importance: RunImportance

    # Memory policy
    memory_level: Literal["user", "session", "run"]
    memory_scope: str

    # Optional metadata
    github_url: str


class AppConfig(TypedDict, total=False):
    # Identity & UI
    id: str
    name: str
    badge: str
    short_description: str
    description: str
    category: str  # "Core" | "R&D Lab" | "Infra" | "Productivity" | ...
    status: str  # "available" | "coming-soon" | "hidden" | ...
    icon_key: str
    tags: list[str]

    # UX hints
    features: list[str]

    # Runtime behavior
    run_visibility: RunVisibility
    run_importance: RunImportance
    flow_id: str

    # Optional metadata
    github_url: str


AGENT_CORE_KEYS = {
    "id",
    "title",
    "description",
    "icon",
    "color",
    "badge",
    "category",
    "status",
    "mode",
    "session_kind",
    "flow_id",
    "tags",
    "tool_graphs",
    "run_visibility",
    "run_importance",
    "memory_level",
    "memory_scope",
    "github_url",
}

APP_CORE_KEYS = {
    "id",
    "name",
    "badge",
    "short_description",
    "description",
    "category",
    "status",
    "icon_key",
    "tags",
    "features",
    "run_visibility",
    "run_importance",
    "flow_id",
    "github_url",
}

# ---------------------------------------------------------------------
# Shared constants / validators
# ---------------------------------------------------------------------

CHAT_V1_REQUIRED_INPUTS = [
    "message",
    "files",
    "context_refs",
    "session_id",
    "user_meta",
]


def validate_agent_signature(
    *,
    graph_name: str,
    fn: Callable[..., Any],
    inputs: list[str] | None,
    agent_cfg: AgentConfig | None,
) -> list[str] | None:
    """
    Small validator to enforce contracts for specific agent modes.

    For now:
      - If mode == "chat_v1", we enforce that:
        1) `inputs` exactly matches CHAT_V1_REQUIRED_INPUTS
           (if inputs is None, we auto-fill with this list).
        2) The function signature has parameters with these names.
    """
    if not agent_cfg:
        return inputs

    mode = agent_cfg.get("mode")
    if mode != "chat_v1":
        return inputs

    expected = CHAT_V1_REQUIRED_INPUTS

    # 1) Validate / fill the .inputs list
    if inputs is None:
        # auto-fill if user forgot; this keeps ergonomics nice
        inputs = expected.copy()
    else:
        if list(inputs) != expected:
            raise ValueError(
                f"Agent '{graph_name}' is mode='chat_v1' but inputs={inputs!r}; "
                f"expected exactly {expected!r}."
            )

    # 2) Validate function signature has params with these names
    sig = inspect.signature(fn)
    missing = [name for name in expected if name not in sig.parameters]
    if missing:
        raise ValueError(
            f"Agent '{graph_name}' (mode='chat_v1') is missing parameters {missing!r} "
            f"in function signature. Required parameters are {expected!r}."
        )

    return inputs


# ---------------------------------------------------------------------
# Normalization helpers used by decorators & other runtime code
# ---------------------------------------------------------------------


def build_agent_meta(
    *,
    graph_name: str,
    version: str,
    graph_meta: dict[str, Any],
    agent_cfg: AgentConfig | None,
) -> dict[str, Any] | None:
    """
    Normalize AgentConfig + graph metadata into a registry-ready meta dict.

    Returns None if agent_cfg is None.
    """
    if agent_cfg is None:
        return None

    cfg = dict(agent_cfg)
    base_tags = graph_meta.get("tags") or []

    agent_id = cfg.get("id", graph_name)
    agent_title = cfg.get("title", f"Agent for {graph_name}")
    agent_flow_id = cfg.get("flow_id", graph_meta.get("flow_id", graph_name))
    agent_tags = cfg.get("tags", base_tags)

    extra = {k: v for k, v in cfg.items() if k not in AGENT_CORE_KEYS}

    memory_level = cfg.get("memory_level", "none")
    memory_scope = cfg.get("memory_scope")

    description = cfg.get("description")

    # unified icon key + accent color (optional)
    icon_key = cfg.get("icon_key") or cfg.get("icon")  # allow both
    accent_color = cfg.get("color")

    meta: dict[str, Any] = {
        "kind": "agent",
        "id": agent_id,
        "title": agent_title,
        "description": description,
        "icon": cfg.get("icon"),
        "icon_key": icon_key,  # to match apps, preferred
        "color": accent_color,
        "badge": cfg.get("badge"),
        "category": cfg.get("category"),
        "status": cfg.get("status", "available"),
        "mode": cfg.get("mode"),
        "session_kind": cfg.get("session_kind", "chat"),
        "flow_id": agent_flow_id,
        "tags": agent_tags,
        "tool_graphs": cfg.get("tool_graphs", []),
        "run_visibility": cfg.get("run_visibility", "inline"),
        "run_importance": cfg.get("run_importance", "normal"),
        "memory": {
            "level": memory_level,
            "scope": memory_scope,
        },
        "github_url": cfg.get("github_url"),
        "backing": {
            "type": "graphfn",
            "name": graph_name,
            "version": version,
        },
        "extra": extra,
    }

    # unified gallery view
    meta["gallery"] = {
        "kind": "agent",
        "id": agent_id,
        "title": agent_title,
        "subtitle": cfg.get("session_kind") or cfg.get("mode"),
        "badge": cfg.get("badge"),
        "category": cfg.get("category"),
        "status": meta["status"],
        "short_description": description,
        "description": description,
        "icon_key": icon_key,
        "accent_color": accent_color,
        "tags": agent_tags,
        "github_url": cfg.get("github_url"),
        "flow_id": agent_flow_id,
        "backing": meta["backing"],
        "extra": extra,
    }

    return meta


def build_app_meta(
    *,
    graph_name: str,
    version: str,
    graph_meta: dict[str, Any],
    app_cfg: AppConfig | None,
) -> dict[str, Any] | None:
    """
    Normalize AppConfig + graph metadata into a registry-ready meta dict.

    Returns None if app_cfg is None.
    """
    if app_cfg is None:
        return None

    cfg = dict(app_cfg)
    base_tags = graph_meta.get("tags") or []

    app_id = cfg.get("id", graph_name)
    app_flow_id = cfg.get("flow_id", graph_meta.get("flow_id", graph_name))
    app_name = cfg.get("name", f"App for {graph_name}")
    app_tags = cfg.get("tags", base_tags)

    extra = {k: v for k, v in cfg.items() if k not in APP_CORE_KEYS}

    short_description = cfg.get("short_description") or cfg.get("description")
    description = cfg.get("description")

    icon_key = cfg.get("icon_key")
    accent_color = cfg.get("color")

    meta: dict[str, Any] = {
        "kind": "app",
        "id": app_id,
        "name": app_name,
        "graph_id": graph_name,
        "flow_id": app_flow_id,
        "badge": cfg.get("badge"),
        "category": cfg.get("category"),
        "short_description": short_description,
        "description": description,
        "status": cfg.get("status", "available"),
        "icon_key": icon_key,
        "color": accent_color,
        "tags": app_tags,
        "features": cfg.get("features", []),
        "run_visibility": cfg.get("run_visibility", "normal"),
        "run_importance": cfg.get("run_importance", "normal"),
        "github_url": cfg.get("github_url"),
        "backing": {
            "type": "graphfn",
            "name": graph_name,
            "version": version,
        },
        "extra": extra,
    }

    # unified gallery view
    meta["gallery"] = {
        "kind": "app",
        "id": app_id,
        "title": app_name,
        "subtitle": cfg.get("category"),
        "badge": cfg.get("badge"),
        "category": cfg.get("category"),
        "status": meta["status"],
        "short_description": short_description,
        "description": description,
        "icon_key": icon_key,
        "accent_color": accent_color,
        "tags": app_tags,
        "github_url": cfg.get("github_url"),
        "flow_id": app_flow_id,
        "backing": meta["backing"],
        "extra": extra,
    }

    return meta
