from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

IntentMode = Literal["chat_only", "quick_action", "plan_and_execute"]


@dataclass
class RoutedIntent:
    """
    Result of routing a user turn:
      - How should we handle this? (mode)
      - If planning: which flows are in scope?
      - If quick_action: which quick action?
    """

    mode: IntentMode

    # For planning
    flow_ids: list[str] | None = None

    # For quick action, e.g. `list_recent_runs`
    quick_action_id: str | None = None

    # Freeform extention field.
    # safety flags, strategy hints, etc. without changing the dataclass structure.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionState:
    # placeholder for future session-level state
    last_flow_ids: list[str] | None = None


class IntentRouter(Protocol):
    async def route(
        self,
        *,
        user_message: str,
        session_state: SessionState,
    ) -> RoutedIntent: ...


class PlanningContextBuilderProtocol:
    async def build(
        self,
        *,
        user_message: str,
        routed: RoutedIntent,
        session_state: SessionState,
    ) -> Any: ...
