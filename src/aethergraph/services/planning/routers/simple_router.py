from __future__ import annotations

from dataclasses import dataclass

from aethergraph.contracts.services.planning import IntentRouter, RoutedIntent, SessionState


@dataclass
class SimpleIntentRouter(IntentRouter):
    """
    Docstring for SimpleIntentRouter
    """

    async def route(
        self,
        *,
        user_message: str,
        session_state: SessionState,
    ) -> RoutedIntent:
        # Simple routing logic: always plan and execute with no specific flows
        return RoutedIntent(
            mode="plan_and_execute",
            flow_ids=None,
            quick_action_id=None,
            metadata={},
        )
