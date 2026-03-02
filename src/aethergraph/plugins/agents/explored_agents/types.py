from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BUILTIN_AGENT_SKILL_ID = "ag.builtin_agent"


AgentMode = Literal[
    "chat",  # default general-purpose chat
    "kb",  # AG "what/how" questions, KB-heavy
    "builder",  # build / modify agents & graphs
    "command",  # slash commands like /runs, /graphs
    "route",  # hand off to other agent
    "connector",  # Gmail, Calendar, etc.
]


@dataclass
class ClassifiedIntent:
    mode: AgentMode
    # Optional target agent for routing
    target_agent_id: str | None = None
    # Parsed command (for /commands)
    command: str | None = None
    command_args: str | None = None
    # Connector hint, e.g. "gmail", "calendar"
    connector_kind: str | None = None
    # Whether this looks like an AG how-to / docs question
    is_ag_howto: bool = False
    # Free-form reasons / notes (for debugging / logging)
    debug_notes: str | None = None
    # message to send when routing
    route_message: str | None = None


@dataclass
class SessionAgentState:
    active_agent_id: str  # e.g. "aether_agent", "deeplens_agent", "finance_agent"
