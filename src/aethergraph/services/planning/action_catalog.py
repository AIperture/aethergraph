from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from aethergraph.core.graph.action_spec import ActionSpec, IOSlot
from aethergraph.core.graph.graph_fn import GraphFunction
from aethergraph.services.planning.graph_io_adapter import graph_io_to_slots
from aethergraph.services.registry.unified_registry import UnifiedRegistry


@dataclass
class ActionCatalog:
    registry: UnifiedRegistry

    def _build_graphfn_spec(self, name: str, version: str | None = None) -> ActionSpec:
        gf: GraphFunction = self.registry.get_graphfn(name, version=version)
        meta = self.registry.get_metadata("graphfn", name, version=version) or {}
        io = gf.io_signature()

        flow_id = meta.get("flow_id", None)
        tags = meta.get("tags", [])
        description = meta.get("description", None)

        # io_signature from GraphFunction are already IOSlot
        inputs: list[IOSlot] = io.get("inputs", [])
        outputs: list[IOSlot] = io.get("outputs", [])

        # resolve final version used
        latest_version = self.registry.list_graphfns().get(f"graphfn:{name}", version or "0.0.0")

        return ActionSpec(
            name=name,
            ref=f"graphfn:{name}:{latest_version}",
            kind="graphfn",
            version=latest_version,
            inputs=inputs,
            outputs=outputs,
            description=description,
            tags=tags,
            flow_id=flow_id,
        )

    def _build_graph_spec(self, name: str, version: str | None = None) -> ActionSpec:
        g = self.registry.get_graph(name, version=version)
        meta = self.registry.get_meta("graph", name, version=version) or {}

        flow_id = meta.get("flow_id")
        tags = meta.get("tags") or []
        description = meta.get("description") or name

        io_slots = graph_io_to_slots(g)
        inputs = io_slots["inputs"]
        outputs = io_slots["outputs"]

        latest_version = self.registry.list_graphs().get(f"graph:{name}", version or "0.0.0")

        return ActionSpec(
            name=name,
            ref=f"graph:{name}@{latest_version}",
            kind="graph",
            version=latest_version,
            description=description,
            tags=list(tags),
            flow_id=flow_id,
            inputs=inputs,
            outputs=outputs,
        )

    def list_actions(
        self,
        *,
        flow_id: str | None = None,
        kinds: Iterable[Literal["graph", "graphfn"]] | None = ("graph", "graphfn"),
    ) -> list[ActionSpec]:
        """
        Docstring for list_actions

        :param self: Description
        :param flow_id: Description
        :type flow_id: str | None
        :param kinds: Description
        :type kinds: Iterable[Literal["graph", "graphfn"]] | None
        :return: Description
        :rtype: list[ActionSpec]
        """

        specs: list[ActionSpec] = []

        if "graphfn" in kinds:
            for key, ver in self.registry.list_graphfns().items():
                _, name = key.split(":", 1)
                spec = self._build_graphfn_spec(name, version=ver)
                if flow_id is not None and spec.flow_id != flow_id:
                    continue
                specs.append(spec)

        if "graph" in kinds:
            for key, ver in self.registry.list_graphs().items():
                _, name = key.split(":", 1)
                spec = self._build_graph_spec(name, version=ver)
                if flow_id is not None and spec.flow_id != flow_id:
                    continue
                specs.append(spec)

        # stable order
        specs.sort(key=lambda s: (s.flow_id or "", s.name, s.version))
        return specs

    def to_llm_prompt(
        self,
        *,
        flow_id: str | None = None,
        kinds: Iterable[Literal["graph", "graphfn"]] | None = ("graph", "graphfn"),
    ) -> str:
        actions = self.list_actions(flow_id=flow_id, kinds=kinds)
        lines: list[str] = []
        for a in actions:
            lines.append(f"- {a.name} ({a.kind})")
            lines.append(f"  ref: {a.ref}")
            lines.append(f"  description: {a.description}")
            if a.tags:
                lines.append(f"  tags: {', '.join(a.tags)}")
            if a.inputs:
                lines.append("  inputs:")
                for inp in a.inputs:
                    t = inp.type or "any"
                    req = "required" if inp.required else f"optional (default={inp.default!r})"
                    lines.append(f"    - {inp.name}: {t}, {req}")
            if a.outputs:
                lines.append("  outputs:")
                for out in a.outputs:
                    t = out.type or "any"
                    lines.append(f"    - {out.name}: {t}")
            lines.append("")
        return "\n".join(lines)
