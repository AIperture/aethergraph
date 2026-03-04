from __future__ import annotations

from typing import Any

from aethergraph.core.graph.action_spec import IOSlot
from aethergraph.services.planning.graph_io_adapter import graph_io_to_slots
from aethergraph.services.registry.unified_registry import UnifiedRegistry

from .schemas.input_schema import InputFieldSpec


def _normalize_json_type(value: Any) -> str | None:
    if value is None:
        return None
    v = str(value).strip().lower()
    if v in {"string", "number", "boolean", "object", "array", "any"}:
        return v
    if v in {"int", "float"}:
        return "number"
    if v in {"bool"}:
        return "boolean"
    return "any"


def _slot_get(slot: IOSlot | dict[str, Any], key: str) -> Any:
    if isinstance(slot, dict):
        return slot.get(key)
    return getattr(slot, key, None)


def _slot_to_input_field(slot: IOSlot | dict[str, Any]) -> InputFieldSpec:
    return InputFieldSpec(
        name=str(_slot_get(slot, "name") or ""),
        type=_normalize_json_type(_slot_get(slot, "type")),
        required=bool(_slot_get(slot, "required")),
        default=_slot_get(slot, "default"),
        description=_slot_get(slot, "description"),
    )


def _graphfn_input_slots(gf: Any) -> list[IOSlot | dict[str, Any]]:
    if gf is None or not hasattr(gf, "io_signature"):
        return []
    sig = gf.io_signature() or {}
    slots = sig.get("inputs") or []
    return list(slots)


def resolve_graph_input_schema(
    reg: UnifiedRegistry,
    *,
    graph_id: str,
) -> list[InputFieldSpec]:
    """
    Resolve typed input fields from either graph or graphfn registry entries.
    """
    try:
        graph_obj = reg.get_graph(name=graph_id, version=None)
        meta = reg.get_meta(nspace="graph", name=graph_id, version=None) or {}
        slots = graph_io_to_slots(graph_obj, meta).get("inputs", [])
        fields = [_slot_to_input_field(s) for s in slots]
        return [f for f in fields if f.name]
    except KeyError:
        pass
    except Exception:
        # Keep apps/graphs listing resilient if a single graph has malformed metadata.
        pass

    try:
        gf = reg.get_graphfn(name=graph_id, version=None)
        slots = _graphfn_input_slots(gf)
        fields = [_slot_to_input_field(s) for s in slots]
        return [f for f in fields if f.name]
    except KeyError:
        return []
    except Exception:
        return []


_OVERRIDE_KEYS = {"label", "placeholder", "widget", "description", "default"}


def _normalize_override_item(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name") or "").strip()
    if not name:
        return None
    out: dict[str, Any] = {"name": name}
    for key in _OVERRIDE_KEYS:
        if key in raw:
            out[key] = raw[key]
    return out


def merge_input_schema_overrides(
    base: list[InputFieldSpec],
    *,
    app_meta: dict[str, Any] | None,
) -> list[InputFieldSpec]:
    """
    Merge app-level input_schema overrides into graph-derived schema by field name.
    """
    meta = app_meta or {}
    extra = meta.get("extra") if isinstance(meta.get("extra"), dict) else {}
    override_raw = meta.get("input_schema")
    if override_raw is None:
        override_raw = extra.get("input_schema")
    if not isinstance(override_raw, list):
        return base

    overrides_by_name: dict[str, dict[str, Any]] = {}
    for item in override_raw:
        normalized = _normalize_override_item(item)
        if normalized is None:
            continue
        overrides_by_name[normalized["name"]] = normalized

    merged: list[InputFieldSpec] = []
    for field in base:
        patch = overrides_by_name.get(field.name)
        if not patch:
            merged.append(field)
            continue
        data = field.model_dump()
        for key in _OVERRIDE_KEYS:
            if key in patch:
                data[key] = patch[key]
        merged.append(InputFieldSpec(**data))
    return merged
