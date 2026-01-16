from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from aethergraph.core.graph.action_spec import ActionSpec, IOSlot
from aethergraph.services.planning.action_catalog import ActionCatalog


@dataclass
class DependencyIndex:
    """
    DependencyIndex manages a collection of ActionSpec objects and provides methods to find compatible producers for a given IOSlot.
    It supports construction from an ActionCatalog and filtering by flow and action kinds.
    """

    actions: list[ActionSpec]

    @classmethod
    def from_catalog(
        cls,
        catalog: ActionCatalog,
        *,
        flow_id: str | None = None,
        kinds: Iterable[Literal["graph", "graphfn"]] | None = ("graph", "graphfn"),
    ) -> DependencyIndex:
        return cls(actions=list(catalog.list_actions(flow_id=flow_id, kinds=kinds)))

    def find_producers(
        self,
        needed: IOSlot,
        *,
        flow_id: str | None = None,
    ) -> list[tuple[ActionSpec, IOSlot]]:
        """
        Finds and returns a list of (action, output_slot) pairs whose output slots are compatible with the given `needed` input slot.
        If `flow_id` is specified, only considers actions with the matching flow ID.

        Args:
            needed (IOSlot): The input slot that needs to be satisfied.
            flow_id (str | None, optional): If provided, restricts search to actions with this flow ID.

        Returns:
            list[tuple[ActionSpec, IOSlot]]: List of (action, output_slot) pairs that can produce the needed input.
        """
        matches: list[tuple[ActionSpec, IOSlot]] = []
        for act in self.actions:
            if flow_id is not None and act.flow_id != flow_id:
                continue
            for out_slot in act.outputs:
                if self._compatible(needed, out_slot):
                    matches.append((act, out_slot))
        return matches

    @staticmethod
    def _compatible(inp: IOSlot, out: IOSlot) -> bool:
        """
        Docstring for _compatible

        :param inp: Description
        :type inp: IOSlot
        :param out: Description
        :type out: IOSlot
        :return: Description
        :rtype: bool
        """
        # name match (could be relaxed later)
        if inp.type is None or out.type is None:
            return True
        if inp.type == out.type:  # noqa: SIM103
            return True
        if inp.type == "object" or out.type == "object":  # noqa: SIM103
            return True
        return False
