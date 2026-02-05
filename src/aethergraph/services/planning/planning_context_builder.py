from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aethergraph.contracts.services.planning import (
    PlanningContextBuilderProtocol,
    RoutedIntent,
    SessionState,
)
from aethergraph.services.planning.planner import PlanningContext


@dataclass
class SimplePlanningContextBuilder(PlanningContextBuilderProtocol):
    """
    Baseline builder: treat the raw user message as goal and
    pass through user_inputs as-is (you can enrich with RAG later).
    """

    def _extract_user_inputs(self, user_message: str) -> dict[str, Any]:
        # For now: rely on external caller to provide user_inputs;
        # later add LLM-based extraction or explicit forms.
        return {}

    async def build(
        self,
        *,
        user_message: str,
        routed: RoutedIntent,
        session_state: SessionState,
    ) -> PlanningContext:
        goal = user_message.strip()
        user_inputs = self._extract_user_inputs(user_message)

        return PlanningContext(
            goal=goal,
            user_inputs=user_inputs,
            external_slots={},  # you can thread structured external inputs later
            memory_snippets=[],  # future: fetch from ctx.memory()
            artifact_snippets=[],  # future: fetch from indices/artifacts
            flow_ids=routed.flow_ids,
        )
