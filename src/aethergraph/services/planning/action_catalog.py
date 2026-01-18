# aethergraph/services/planning/action_catalog.py
from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Literal

from aethergraph.core.graph.action_spec import ActionSpec, IOSlot
from aethergraph.core.graph.graph_fn import GraphFunction
from aethergraph.services.planning.graph_io_adapter import graph_io_to_slots
from aethergraph.services.registry.registry_key import Key
from aethergraph.services.registry.unified_registry import UnifiedRegistry


@dataclass
class ActionCatalog:
    registry: UnifiedRegistry

    # --- builders ---------------------------------------------------------

    def _build_graphfn_spec(self, name: str, version: str | None = None) -> ActionSpec:
        gf: GraphFunction = self.registry.get_graphfn(name, version=version)
        meta = self.registry.get_meta("graphfn", name, version=version) or {}
        io = gf.io_signature()

        flow_id = meta.get("flow_id", None)
        tags = meta.get("tags", []) or []
        description = meta.get("description", None)

        # io_signature from GraphFunction are already IOSlot
        inputs: list[IOSlot] = io.get("inputs", [])
        outputs: list[IOSlot] = io.get("outputs", [])

        # resolve final version used
        latest_version = self.registry.list_graphfns().get(
            f"graphfn:{name}",
            version or "0.0.0",
        )

        return ActionSpec(
            name=name,
            ref=Key(nspace="graphfn", name=name, version=latest_version).canonical(),
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

        # pass meta to adapter so it can use io_types
        io_slots = graph_io_to_slots(g, meta=meta)
        inputs = io_slots["inputs"]
        outputs = io_slots["outputs"]

        latest_version = self.registry.list_graphs().get(
            f"graph:{name}",
            version or "0.0.0",
        )

        return ActionSpec(
            name=name,
            ref=Key(nspace="graph", name=name, version=latest_version).canonical(),
            kind="graph",
            version=latest_version,
            description=description,
            tags=list(tags),
            flow_id=flow_id,
            inputs=inputs,
            outputs=outputs,
        )

    # --- listing / filtering ---------------------------------------------

    def _flow_filter(
        self,
        spec: ActionSpec,
        *,
        flow_ids: list[str] | None,
        include_global: bool,
    ) -> bool:
        """
        Decide whether to include this spec given flow_ids and include_global.
        - spec.flow_id is a single string or None.
        - flow_ids is the set of flows we care about, or None for 'no filtering'.
        """
        if flow_ids is None:
            # no restriction → include everything
            return True

        if spec.flow_id in flow_ids:
            return True

        # allow "global" actions when requested
        if include_global and spec.flow_id is None:  # noqa: SIM103
            return True

        return False

    def list_actions(
        self,
        *,
        flow_ids: list[str] | None = None,
        kinds: Iterable[Literal["graph", "graphfn"]] | None = ("graph", "graphfn"),
        include_global: bool = True,
    ) -> list[ActionSpec]:
        """
        Return all ActionSpecs, optionally filtered by:
          - kinds (graph vs graphfn)
          - flow_ids (one or more flow ids)
          - include_global: if True, also include actions with flow_id=None
        """
        specs: list[ActionSpec] = []

        if kinds is None:
            kinds = ("graph", "graphfn")

        if "graphfn" in kinds:
            for key, ver in self.registry.list_graphfns().items():
                _, name = key.split(":", 1)
                spec = self._build_graphfn_spec(name, version=ver)
                if not self._flow_filter(spec, flow_ids=flow_ids, include_global=include_global):
                    continue
                specs.append(spec)

        if "graph" in kinds:
            for key, ver in self.registry.list_graphs().items():
                _, name = key.split(":", 1)
                spec = self._build_graph_spec(name, version=ver)
                if not self._flow_filter(spec, flow_ids=flow_ids, include_global=include_global):
                    continue
                specs.append(spec)

        # stable order
        specs.sort(key=lambda s: (s.flow_id or "", s.name, s.version))
        return specs

    def iter_actions(
        self,
        *,
        flow_ids: list[str] | None = None,
        kinds: Iterable[Literal["graph", "graphfn"]] | None = ("graph", "graphfn"),
        include_global: bool = True,
    ) -> Iterator[ActionSpec]:
        for spec in self.list_actions(  # noqa: UP028
            flow_ids=flow_ids,
            kinds=kinds,
            include_global=include_global,
        ):
            yield spec

    # --- lookups ---------------------------------------------------------

    def get_action(self, ref: str) -> ActionSpec | None:
        kind, rest = ref.split(":", 1)
        name, sep, version = rest.partition("@")
        version = version or None

        if kind == "graphfn":
            return self._build_graphfn_spec(name, version=version)
        if kind == "graph":
            return self._build_graph_spec(name, version=version)
        raise ValueError(f"Unknown action kind in ref: {ref}")

    def get_action_by_name(
        self,
        name: str,
        *,
        kind: Literal["graph", "graphfn"] | None = None,
        flow_ids: list[str] | None = None,
        include_global: bool = True,
    ) -> ActionSpec | None:
        """
        Convenience lookup: find an ActionSpec by its logical name.
        """
        if kind is None:
            kinds: Iterable[Literal["graph", "graphfn"]] = ("graph", "graphfn")
        else:
            kinds = (kind,)

        for spec in self.list_actions(
            flow_ids=flow_ids,
            kinds=kinds,
            include_global=include_global,
        ):
            if spec.name == name:
                return spec
        return None

    # --- LLM-facing renderers -------------------------------------------

    def to_llm_prompt(
        self,
        *,
        flow_ids: list[str] | None = None,
        kinds: Iterable[Literal["graph", "graphfn"]] | None = ("graph", "graphfn"),
        include_global: bool = True,
    ) -> str:
        actions = self.list_actions(
            flow_ids=flow_ids,
            kinds=kinds,
            include_global=include_global,
        )
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

    def pretty_print(
        self,
        *,
        flow_ids: list[str] | None = None,
        kinds: Iterable[Literal["graph", "graphfn"]] | None = ("graph", "graphfn"),
        include_global: bool = True,
    ) -> str:
        """
        Human-readable table for planner prompts.
        """
        actions = self.list_actions(
            flow_ids=flow_ids,
            kinds=kinds,
            include_global=include_global,
        )
        lines: list[str] = []
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
