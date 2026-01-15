# aethergraph/services/planning/graph_io_adapter.py

from __future__ import annotations

from aethergraph.core.graph.action_spec import IOSlot, _map_py_type_to_json_type
from aethergraph.core.graph.task_graph import TaskGraph


def graph_io_to_slots(graph: TaskGraph) -> dict[str, list[IOSlot]]:
    """
    Adapter: TaskGraph.io_signature() + IOSpec -> {inputs: [...], outputs: [...]}
    for planner / ActionSpec usage.

    - This does NOT change TaskGraph.
    - It is deliberately defensive against multiple shapes.
    """
    sig = graph.io_signature(include_values=False)
    io_spec = getattr(graph.spec, "io", None)  # IOSpec if present

    inputs_info = sig.get("inputs", {}) or {}
    outputs_info = sig.get("outputs", {}) or {}

    # --- INPUTS ---

    # required can be list[str] *or* dict name->ParamSpec depending on future changes
    req_raw = inputs_info.get("required") or []
    opt_raw = inputs_info.get("optional") or {}

    required_names = list(req_raw.keys()) if isinstance(req_raw, dict) else list(req_raw)
    optional_names = list(opt_raw.keys()) if isinstance(opt_raw, dict) else list(opt_raw)

    input_slots: list[IOSlot] = []

    def _param_for(name: str):
        if io_spec is None:
            return None
        # io_spec.required & io_spec.optional are dict[str, ParamSpec]
        if hasattr(io_spec, "required") and name in io_spec.required:
            return io_spec.required[name]
        if hasattr(io_spec, "optional") and name in io_spec.optional:
            return io_spec.optional[name]
        return None

    for name in required_names:
        ps = _param_for(name)
        if ps is not None:
            anno = getattr(ps, "annotation", None)
            default = getattr(ps, "default", None)
            # If ParamSpec has a "required" flag use that, else default to True
            required_flag = getattr(ps, "required", True)
            j_type = _map_py_type_to_json_type(anno) if anno is not None else None
            input_slots.append(
                IOSlot(
                    name=name,
                    type=j_type,
                    required=required_flag,
                    default=None if required_flag else default,
                    description=getattr(ps, "description", None),
                )
            )
        else:
            # No ParamSpec info: just name + required=True
            input_slots.append(IOSlot(name=name, type=None, required=True))

    for name in optional_names:
        ps = _param_for(name)
        if ps is not None:
            anno = getattr(ps, "annotation", None)
            default = getattr(ps, "default", None)
            j_type = _map_py_type_to_json_type(anno) if anno is not None else None
            input_slots.append(
                IOSlot(
                    name=name,
                    type=j_type,
                    required=False,
                    default=default,
                    description=getattr(ps, "description", None),
                )
            )
        else:
            input_slots.append(IOSlot(name=name, type=None, required=False))

    # --- OUTPUTS ---

    # We expose outputs via IOSpec.expose/expose_bindings, but io_signature
    # already gave us "keys".
    output_keys = outputs_info.get("keys") or []
    output_slots: list[IOSlot] = []

    def _output_param_for(name: str):
        if io_spec is None or not hasattr(io_spec, "outputs"):
            return None
        return io_spec.outputs.get(name)

    for name in output_keys:
        ps = _output_param_for(name)
        if ps is not None:
            anno = getattr(ps, "annotation", None)
            j_type = _map_py_type_to_json_type(anno) if anno is not None else None
            output_slots.append(
                IOSlot(
                    name=name,
                    type=j_type,
                    required=True,  # outputs are logically always “present”
                    description=getattr(ps, "description", None),
                )
            )
        else:
            output_slots.append(IOSlot(name=name, type=None, required=True))

    return {"inputs": input_slots, "outputs": output_slots}
