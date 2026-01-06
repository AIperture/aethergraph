# aethergraph/runtime/agent_app_meta.py
from __future__ import annotations

from collections.abc import Callable
import inspect
from typing import Any, Literal, TypedDict

from aethergraph.core.runtime.run_types import RunImportance, RunVisibility

# ---------------------------------------------------------------------
# Config schemas used by decorators
# ---------------------------------------------------------------------
SUPPORTED_AGENT_MODES = {"chat_v1"}


class AgentConfig(TypedDict, total=False):
    """
    Configuration metadata for an agent. Register an agent with `as_agent`
    parameter in `@graphify` or `@graph_fn`.

    All fields are optional except `id` in practice; anything omitted gets
    reasonable defaults in build_agent_meta.

    Attributes:
        id (str): Unique identifier for the agent. Defaults to graph name.
        title (str): Display name of the agent. Optional, shown in the UI.
        description (str): Brief description of the agent. Optional, shown in the UI.
        short_description (str): Shorter summary (used in cards). Optional.
        icon_key (str): Icon key used in the UI (e.g. "message-circle").
        color (str): Accent color token (e.g. "emerald").
        badge (str): Badge label, e.g. "Chat Agent".
        category (str): Category, e.g. "Core", "R&D Lab", "Infra", "Productivity".
        status (str): "available" | "coming-soon" | "hidden" | "error" | ...
        mode (str): Operational mode. Defaults to "chat_v1" for chat agents.
        session_kind (str): Session type, e.g. "chat". Defaults to "chat".
        flow_id (str): Flow identifier for wiring. Defaults to graph name.
        tags (list[str]): Tags used for search / grouping.
        tool_graphs (list[str]): Related tool graph identifiers.
        features (list[str]): Optional feature bullets for UI.
        run_visibility (RunVisibility): "normal" | "inline" | ...
        run_importance (RunImportance): "normal" | "high" | ...
        memory_level (Literal["user","session","run"]): Memory scope level.
        memory_scope (str): Logical scope, e.g. "session.global", "user.all".
        github_url (str): Optional GitHub link.
    """

    # Identity & basic UI
    id: str
    title: str
    description: str
    short_description: str
    icon_key: str
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
    features: list[str]

    # Runtime behavior
    run_visibility: RunVisibility
    run_importance: RunImportance

    # Memory policy
    memory_level: Literal["user", "session", "run"]
    memory_scope: str

    # Optional metadata
    github_url: str


class AppConfig(TypedDict, total=False):
    """
    Configuration metadata for an application. Register an app with `as_app`
    parameter in `@graphify` or `@graph_fn`.

    Attributes:
        id (str): Unique identifier for the app. Defaults to graph name.
        name (str): Human-readable name of the app. Defaults to "App for <graph>".
        badge (str): Short badge or label for the app. Optional, shown in the UI.
        short_description (str): Brief summary of the app's purpose. Optional.
        description (str): Detailed description of the app. Optional.
        category (str): Category, e.g. "Core", "R&D Lab", "Infra", "Productivity".
        status (str): "available" | "coming-soon" | "hidden" | "error" | ...
        icon_key (str): Icon key for the app.
        color (str): Accent color token.
        mode (str): App mode, e.g. "no_input_v1". Defaults to "no_input_v1".
        tags (list[str]): Tags for search / grouping.
        features (list[str]): Notable features for the app.
        run_visibility (RunVisibility): "normal" | "inline" | ...
        run_importance (RunImportance): "normal" | "high" | ...
        flow_id (str): Flow identifier. Defaults to graph name.
        github_url (str): Optional GitHub link.
    """

    # Identity & UI
    id: str
    name: str
    badge: str
    short_description: str
    description: str
    category: str
    status: str
    icon_key: str
    color: str
    mode: str
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
    "short_description",
    "icon_key",
    "color",
    "badge",
    "category",
    "status",
    "mode",
    "session_kind",
    "flow_id",
    "tags",
    "tool_graphs",
    "features",
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
    "color",
    "mode",
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


def normalize_agent_mode(agent_cfg: AgentConfig) -> str:
    # Default behavior: if user doesn't specify, it's chat_v1
    mode = (agent_cfg.mode or "chat_v1").strip()

    if mode not in SUPPORTED_AGENT_MODES:
        # this will be caught and turned into status="error" for now
        raise ValueError(
            f"Unsupported agent mode '{mode}'. "
            "Currently only 'chat_v1' is supported. "
            "Omit 'mode' or set mode='chat_v1'."
        )
    return mode


SUPPORTED_APP_MODES = {"no_input_v1"}


def normalize_app_mode(app_cfg: AppConfig) -> str:
    mode = (app_cfg.mode or "no_input_v1").strip()
    if mode not in SUPPORTED_APP_MODES:
        raise ValueError(
            f"Unsupported app mode '{mode}'. "
            "Currently only 'no_input_v1' is supported for gallery apps. "
            "Omit 'mode' or set mode='no_input_v1'."
        )
    return mode


def validate_agent_signature(
    graph_name: str, fn: Callable, agent_cfg: AgentConfig
) -> tuple[str, list[str]]:
    mode = normalize_agent_mode(agent_cfg)

    sig = inspect.signature(fn)
    param_names = list(sig.parameters.keys())

    if mode == "chat_v1":
        missing = [p for p in CHAT_V1_REQUIRED_INPUTS if p not in param_names]
        if missing:
            raise ValueError(
                f"chat_v1 agent '{graph_name}' is missing parameters: {missing}. "
                f"Expected parameters: {CHAT_V1_REQUIRED_INPUTS}"
            )
        # TODO future: could be more flexible here:
        #  - use CHAT_V1_REQUIRED_INPUTS as canonical inputs, or
        #  - accept superset but keep these first.
        inputs = CHAT_V1_REQUIRED_INPUTS
    else:
        # Currently unreachable because normalize_agent_mode rejects unknowns,
        # but future-proof if add more modes.
        inputs = param_names

    return mode, inputs


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

    cfg: dict[str, Any] = dict(agent_cfg)
    base_tags = graph_meta.get("tags") or []

    agent_id = cfg.get("id", graph_name)
    agent_title = cfg.get("title", f"Agent for {graph_name}")
    agent_flow_id = cfg.get("flow_id", graph_meta.get("flow_id", graph_name))
    agent_tags = cfg.get("tags", base_tags)

    # Anything not in core keys becomes "extra" for future use
    extra = {k: v for k, v in cfg.items() if k not in AGENT_CORE_KEYS}

    # Memory policy
    memory_level = cfg.get("memory_level", "session")
    memory_scope = cfg.get("memory_scope")

    # Text fields
    description = cfg.get("description")
    short_description = cfg.get("short_description") or description

    # Visuals
    icon_key = cfg.get("icon_key")
    accent_color = cfg.get("color")

    # Behavior
    agent_mode = cfg.get("mode") or "chat_v1"
    session_kind = cfg.get("session_kind", "chat")

    meta: dict[str, Any] = {
        "kind": "agent",
        "id": agent_id,
        "title": agent_title,
        "description": description,
        "short_description": short_description,
        "icon_key": icon_key,
        "color": accent_color,
        "badge": cfg.get("badge"),
        "category": cfg.get("category"),
        "status": cfg.get("status", "available"),
        "mode": agent_mode,
        "session_kind": session_kind,
        "flow_id": agent_flow_id,
        "tags": agent_tags,
        "tool_graphs": cfg.get("tool_graphs", []),
        "features": cfg.get("features", []),
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
        "subtitle": session_kind or agent_mode,
        "badge": cfg.get("badge"),
        "category": cfg.get("category"),
        "status": meta["status"],
        "short_description": short_description,
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

    cfg: dict[str, Any] = dict(app_cfg)
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
    app_mode = cfg.get("mode") or "no_input_v1"

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
        "mode": app_mode,
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
