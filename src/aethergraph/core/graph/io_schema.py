from __future__ import annotations

from typing import Any

from aethergraph.core.graph.action_spec import IOSlot, _map_py_type_to_json_type
from aethergraph.core.graph.task_graph import TaskGraph


def graph_io_to_slots(
    graph: TaskGraph,
    meta: dict[str, Any] | None = None,
) -> dict[str, list[IOSlot]]:
    """Project a TaskGraph I/O signature into typed API schema slots."""

    signature = graph.io_signature(include_values=False)
    io_spec = getattr(graph.spec, "io", None)

    inputs_info = signature.get("inputs", {}) or {}
    outputs_info = signature.get("outputs", {}) or {}
    io_types = (meta or {}).get("io_types") or {}
    input_type_map: dict[str, str] = io_types.get("inputs", {}) or {}
    output_type_map: dict[str, str] = io_types.get("outputs", {}) or {}

    required_raw = inputs_info.get("required") or []
    optional_raw = inputs_info.get("optional") or {}
    required_names = (
        list(required_raw.keys()) if isinstance(required_raw, dict) else list(required_raw)
    )
    optional_names = (
        list(optional_raw.keys()) if isinstance(optional_raw, dict) else list(optional_raw)
    )

    def _input_param(name: str) -> Any | None:
        if io_spec is None:
            return None
        if hasattr(io_spec, "required") and name in io_spec.required:
            return io_spec.required[name]
        if hasattr(io_spec, "optional") and name in io_spec.optional:
            return io_spec.optional[name]
        return None

    input_slots: list[IOSlot] = []
    for name in required_names:
        param = _input_param(name)
        type_from_meta = input_type_map.get(name)
        json_type = None
        default = None
        description = None
        required = True
        if param is not None:
            annotation = getattr(param, "annotation", None)
            default = getattr(param, "default", None)
            required = getattr(param, "required", True)
            description = getattr(param, "description", None)
            if type_from_meta is None and annotation is not None:
                json_type = _map_py_type_to_json_type(annotation)
        input_slots.append(
            IOSlot(
                name=name,
                type=type_from_meta or json_type,
                required=required,
                default=None if required else default,
                description=description,
            )
        )

    for name in optional_names:
        param = _input_param(name)
        type_from_meta = input_type_map.get(name)
        json_type = None
        default = None
        description = None
        if param is not None:
            annotation = getattr(param, "annotation", None)
            default = getattr(param, "default", None)
            description = getattr(param, "description", None)
            if type_from_meta is None and annotation is not None:
                json_type = _map_py_type_to_json_type(annotation)
        input_slots.append(
            IOSlot(
                name=name,
                type=type_from_meta or json_type,
                required=False,
                default=default,
                description=description,
            )
        )

    def _output_param(name: str) -> Any | None:
        if io_spec is None or not hasattr(io_spec, "outputs"):
            return None
        return io_spec.outputs.get(name)

    output_slots: list[IOSlot] = []
    for name in outputs_info.get("keys") or []:
        param = _output_param(name)
        type_from_meta = output_type_map.get(name)
        json_type = None
        description = None
        if param is not None:
            annotation = getattr(param, "annotation", None)
            description = getattr(param, "description", None)
            if type_from_meta is None and annotation is not None:
                json_type = _map_py_type_to_json_type(annotation)
        output_slots.append(
            IOSlot(
                name=name,
                type=type_from_meta or json_type,
                required=True,
                description=description,
            )
        )

    return {"inputs": input_slots, "outputs": output_slots}
