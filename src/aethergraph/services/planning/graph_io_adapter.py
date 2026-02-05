# aethergraph/services/planning/graph_io_adapter.py
from __future__ import annotations

from typing import Any

from aethergraph.core.graph.action_spec import IOSlot, _map_py_type_to_json_type
from aethergraph.core.graph.task_graph import TaskGraph


def graph_io_to_slots(
    graph: TaskGraph,
    meta: dict[str, Any] | None = None,
) -> dict[str, list[IOSlot]]:
    """
    Adapter: TaskGraph.io_signature() + IOSpec + optional registry meta -> IOSlot lists.

    Priority for types:
      1) registry meta["io_types"]["inputs"/"outputs"]  (set by @graphify)
      2) ParamSpec.annotation from IOSpec
      3) None (treated as "any" by planner)
    """
    sig = graph.io_signature(include_values=False)
    io_spec = getattr(graph.spec, "io", None)  # IOSpec if present

    inputs_info = sig.get("inputs", {}) or {}
    outputs_info = sig.get("outputs", {}) or {}

    # ---- io_types from registry meta (preferred) ----
    io_types = (meta or {}).get("io_types") or {}
    input_type_map: dict[str, str] = io_types.get("inputs", {}) or {}
    output_type_map: dict[str, str] = io_types.get("outputs", {}) or {}

    # --- INPUTS ---

    req_raw = inputs_info.get("required") or []
    opt_raw = inputs_info.get("optional") or {}

    required_names = list(req_raw.keys()) if isinstance(req_raw, dict) else list(req_raw)
    optional_names = list(opt_raw.keys()) if isinstance(opt_raw, dict) else list(opt_raw)

    input_slots: list[IOSlot] = []

    def _param_for(name: str):
        if io_spec is None:
            return None
        if hasattr(io_spec, "required") and name in io_spec.required:
            return io_spec.required[name]
        if hasattr(io_spec, "optional") and name in io_spec.optional:
            return io_spec.optional[name]
        return None

    # required inputs
    for name in required_names:
        ps = _param_for(name)

        # 1) type from meta.io_types if present
        t_from_meta = input_type_map.get(name)

        # 2) else type from ParamSpec.annotation
        j_type = None
        default = None
        description = None
        required_flag = True

        if ps is not None:
            anno = getattr(ps, "annotation", None)
            default = getattr(ps, "default", None)
            required_flag = getattr(ps, "required", True)
            description = getattr(ps, "description", None)
            if t_from_meta is None and anno is not None:
                j_type = _map_py_type_to_json_type(anno)

        # final type choice
        final_type = t_from_meta or j_type

        input_slots.append(
            IOSlot(
                name=name,
                type=final_type,
                required=required_flag,
                default=None if required_flag else default,
                description=description,
            )
        )

    # optional inputs
    for name in optional_names:
        ps = _param_for(name)

        t_from_meta = input_type_map.get(name)

        j_type = None
        default = None
        description = None

        if ps is not None:
            anno = getattr(ps, "annotation", None)
            default = getattr(ps, "default", None)
            description = getattr(ps, "description", None)
            if t_from_meta is None and anno is not None:
                j_type = _map_py_type_to_json_type(anno)

        final_type = t_from_meta or j_type

        input_slots.append(
            IOSlot(
                name=name,
                type=final_type,
                required=False,
                default=default,
                description=description,
            )
        )

    # --- OUTPUTS ---

    output_keys = outputs_info.get("keys") or []
    output_slots: list[IOSlot] = []

    def _output_param_for(name: str):
        if io_spec is None or not hasattr(io_spec, "outputs"):
            return None
        return io_spec.outputs.get(name)

    for name in output_keys:
        ps = _output_param_for(name)

        t_from_meta = output_type_map.get(name)

        j_type = None
        description = None

        if ps is not None:
            anno = getattr(ps, "annotation", None)
            description = getattr(ps, "description", None)
            if t_from_meta is None and anno is not None:
                j_type = _map_py_type_to_json_type(anno)

        final_type = t_from_meta or j_type

        output_slots.append(
            IOSlot(
                name=name,
                type=final_type,
                required=True,  # outputs are logically “present”
                description=description,
            )
        )

    return {"inputs": input_slots, "outputs": output_slots}
