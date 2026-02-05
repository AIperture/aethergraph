# aethergraph/services/planning/dependency_index.py
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from aethergraph.core.graph.action_spec import ActionSpec, IOSlot
from aethergraph.services.planning.action_catalog import ActionCatalog


@dataclass
class DependencyIndex:
    actions: list[ActionSpec]

    @classmethod
    def from_catalog(
        cls,
        catalog: ActionCatalog,
        *,
        flow_ids: list[str] | None = None,
        kinds: Iterable[Literal["graph", "graphfn"]] | None = ("graph", "graphfn"),
        include_global: bool = True,
    ) -> DependencyIndex:
        return cls(
            actions=list(
                catalog.list_actions(
                    flow_ids=flow_ids,
                    kinds=kinds,
                    include_global=include_global,
                )
            )
        )

    def find_producers(
        self,
        needed: IOSlot,
        *,
        flow_ids: list[str] | None = None,
    ) -> list[tuple[ActionSpec, IOSlot]]:
        """
        Find (action, output_slot) pairs whose outputs are compatible with the given input slot.
        If flow_ids is provided, only consider actions within those flows or global ones (the
        actions list is usually pre-filtered by from_catalog).
        """
        matches: list[tuple[ActionSpec, IOSlot]] = []
        for act in self.actions:
            if flow_ids is not None:  # noqa: SIM102
                # respect pre-filtering convention: allow actions with flow_id in flow_ids or None
                if act.flow_id not in flow_ids and act.flow_id is not None:
                    continue
            for out_slot in act.outputs:
                if self._compatible(needed, out_slot):
                    matches.append((act, out_slot))
        return matches

    @staticmethod
    def _compatible(inp: IOSlot, out: IOSlot) -> bool:
        if inp.type is None or out.type is None:
            return True
        if inp.type == out.type:
            return True
        if inp.type in {"object", "any"} or out.type in {"object", "any"}:  # noqa: SIM103
            return True
        return False
