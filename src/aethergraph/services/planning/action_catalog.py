from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

from aethergraph.core.graph.action_spec import ActionSpec, IOSlot
from aethergraph.core.graph.graph_fn import GraphFunction
from aethergraph.services.planning.graph_io_adapter import graph_io_to_slots
from aethergraph.services.registry.unified_registry import UnifiedRegistry


@dataclass
class ActionCatalog:
    registry: UnifiedRegistry

    def _build_graphfn_spec(self, name: str, version: str | None = None) -> ActionSpec:
        gf: GraphFunction = self.registry.get_graphfn(name, version=version)
        meta = self.registry.get_meta("graphfn", name, version=version) or {}
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

        # NEW: pass meta to adapter so it can use io_types
        io_slots = graph_io_to_slots(g, meta=meta)
        inputs = io_slots["inputs"]
        outputs = io_slots["outputs"]

        latest_version = self.registry.list_graphs().get(
            f"graph:{name}",
            version or "0.0.0",
        )

        return ActionSpec(
            name=name,
            ref=f"graph:{name}:{latest_version}",
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

    def iter_actions(
        self,
        *,
        flow_id: str | None = None,
        kinds: Iterable[Literal["graph", "graphfn"]] | None = ("graph", "graphfn"),
    ) -> Iterator[ActionSpec]:
        for spec in self.list_actions(flow_id=flow_id, kinds=kinds):  # noqa: UP028
            yield spec

    def get_action(self, ref: str) -> ActionSpec | None:
        kind, rest = ref.split(":", 1)
        name, _, version = rest.rpartition(":")
        version = version or None
        if kind == "graphfn":
            return self._build_graphfn_spec(name, version=version)
        elif kind == "graph":
            return self._build_graph_spec(name, version=version)
        else:
            raise ValueError(f"Unknown action kind in ref: {ref}")

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

    def format_actions_table(
        self,
        *,
        group_by_flow: bool = True,
        include_types: bool = True,
    ) -> str:
        """
        Return a human-readable table of all registered actions.

        - Grouped by flow_id when requested.
        - Shows inputs/outputs with types when available.
        """
        actions = self.list_actions()
        if not actions:
            return "No actions registered."

        # Group by flow_id
        by_flow: dict[str | None, list[Any]] = {}
        for a in actions:
            by_flow.setdefault(a.flow_id, []).append(a)

        lines: list[str] = []
        for flow_id, flow_actions in sorted(by_flow.items(), key=lambda kv: (kv[0] or "")):
            header = f"Flow: {flow_id or '<none>'}"
            lines.append(header)
            lines.append("-" * len(header))

            # sort actions by kind + name for stable view
            for act in sorted(flow_actions, key=lambda a: (a.kind or "", a.name)):
                lines.append(f"[{act.kind}] {act.name} @ {act.version}")
                desc = getattr(act, "description", None)
                if desc:
                    lines.append(f"  desc: {desc}")

                # ---- Inputs ----
                in_slots = getattr(act, "inputs", None) or []
                if in_slots:
                    lines.append("  inputs:")
                    for slot in in_slots:
                        t = getattr(slot, "type", None) if include_types else None
                        req_flag = "required" if getattr(slot, "required", True) else "optional"
                        if t and t != "any":
                            lines.append(f"    - {slot.name}: {t} ({req_flag})")
                        else:
                            lines.append(f"    - {slot.name} ({req_flag})")
                else:
                    lines.append("  inputs: <none>")

                # ---- Outputs ----
                out_slots = getattr(act, "outputs", None) or []
                if out_slots:
                    lines.append("  outputs:")
                    for slot in out_slots:
                        t = getattr(slot, "type", None) if include_types else None
                        if t and t != "any":
                            lines.append(f"    - {slot.name}: {t}")
                        else:
                            lines.append(f"    - {slot.name}")
                else:
                    lines.append("  outputs: <none>")

                tags = getattr(act, "tags", None) or []
                if tags:
                    lines.append(f"  tags: {', '.join(tags)}")

                lines.append("")  # blank line between actions

            lines.append("")  # blank line between flows

        return "\n".join(lines)

    def pretty_print(
        self,
        *,
        flow_id: str | None = None,
        kinds: Iterable[Literal["graph", "graphfn"]] | None = ("graph", "graphfn"),
    ) -> str:
        """
        Human-readable table for planner prompts.
        """
        actions = self.list_actions(flow_id=flow_id, kinds=kinds)
        lines = []
        for a in actions:
            inputs = ", ".join(f"{s.name}:{s.type or 'any'}" for s in a.inputs)
            outputs = ", ".join(f"{s.name}:{s.type or 'any'}" for s in a.outputs)
            tag_str = ", ".join(a.tags or [])
            lines.append(
                f"- {a.name} [{a.kind}] (ref={a.ref})\n"
                f"  inputs: {inputs or 'none'}\n"
                f"  outputs: {outputs or 'none'}\n"
                f"  tags: {tag_str or '-'}\n"
                f"  desc: {a.description or '-'}"
            )
        return "\n".join(lines)
