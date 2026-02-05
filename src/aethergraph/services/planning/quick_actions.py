from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class QuickActionSpec:
    id: str
    description: str
    # e.g. `list_recent_runs`, `open_notebook`, `answer_with_rag`
    requires_confirmation: bool = False
    action_ref: str | None = None  # optional link to ActionSpec.ref


class QuickActionHandler(Protocol):
    async def __call__(self, *, context: Any) -> Any: ...


@dataclass
class QuickActionRegistry:
    actions: dict[str, QuickActionSpec]
    handlers: dict[str, QuickActionHandler]

    def get_spec(self, action_id: str) -> QuickActionSpec | None:
        return self.actions.get(action_id)

    def get_handler(self, action_id: str) -> QuickActionHandler | None:
        return self.handlers.get(action_id)
