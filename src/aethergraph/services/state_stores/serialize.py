from __future__ import annotations
from dataclasses import asdict
from typing import Any, Mapping, Sequence, Tuple, Optional, Dict
from .externalize import externalize_to_artifact


_JSON_SCALARS = (str, int, float, bool, type(None))

def is_json_pure(obj: Any) -> bool:
    if isinstance(obj, _JSON_SCALARS):
        return True
    if isinstance(obj, Mapping):
        # disallow references in "pure" mode
        if "__aether_ref__" in obj or obj.get("__externalized__") is True:
            return False
        for k, v in obj.items():
            if not isinstance(k, str):
                return False
            if not is_json_pure(v):
                return False
        return True
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return all(is_json_pure(v) for v in obj)
    return False


def jsonish_or_ref(
    obj: Any,
    *,
    mk_ref: Optional[callable[[Any], Dict[str, Any]]] = None,
) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """
    Returns (jsonish, ref) where:
      - if obj is JSON-pure → (obj, None)
      - else → ({"__aether_ref__": <...>}, ref_meta)
    mk_ref(obj) must return a dict with at least {"__aether_ref__": "<uri-or-id>"}.
    """
    if is_json_pure(obj):
        return obj, None

    if mk_ref is None:
        # Default: opaque marker (will block resume; artifacts still saved by caller if desired)
        return {"__aether_ref__": "opaque:nonjson"}, {"__aether_ref__": "opaque:nonjson"}

    ref = mk_ref(obj) or {"__aether_ref__": "opaque:nonjson"}
    return {"__aether_ref__": ref.get("__aether_ref__", "opaque:nonjson"), **{k: v for k, v in ref.items() if k != "__aether_ref__"}}, ref


def map_jsonish_or_ref(
    payload: Any,
    *,
    mk_ref: Optional[callable[[Any], Dict[str, Any]]] = None,
) -> Tuple[Any, bool]:
    """
    Walk nested structures. Returns (jsonish_payload, had_refs).
    Any non-JSON leaf becomes a {"__aether_ref__": ...} marker via mk_ref.
    """
    if is_json_pure(payload):
        return payload, False

    # dict
    if isinstance(payload, Mapping):
        out = {}
        had_ref = False
        for k, v in payload.items():
            if not isinstance(k, str):
                # stringify non-string keys to keep snapshot JSON-safe
                k = str(k)
            vv, r = map_jsonish_or_ref(v, mk_ref=mk_ref)
            out[k] = vv
            had_ref = had_ref or r
        return out, had_ref

    # list/tuple
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        out = []
        had_ref = False
        for v in payload:
            vv, r = map_jsonish_or_ref(v, mk_ref=mk_ref)
            out.append(vv)
            had_ref = had_ref or r
        return out, had_ref

    # leaf non-JSON → ref
    jsonish, _ref_meta = jsonish_or_ref(payload, mk_ref=mk_ref)
    return jsonish, True



async def _externalize_leaf_to_artifact(obj: Any, *, run_id: str, graph_id: str, node_id: str, tool_name: str | None, tool_version: str | None, artifacts):
    ref = await externalize_to_artifact(
        obj,
        run_id=run_id, graph_id=graph_id, node_id=node_id,
        tool_name=tool_name, tool_version=tool_version,
        artifacts=artifacts
    )
    return ref

async def _jsonish_outputs_with_refs(
    *,
    outputs: Dict[str, Any] | None,
    run_id: str,
    graph_id: str,
    node_id: str,
    tool_name: str | None,
    tool_version: str | None,
    artifacts,                     # AsyncArtifactStore or None
    allow_externalize: bool,       # toggle
) -> Dict[str, Any] | None:
    if outputs is None:
        return None

    def mk_ref(obj):
        # If we can't (or shouldn't) externalize, mark opaque
        if not allow_externalize or artifacts is None:
            return {"__aether_ref__": "opaque:nonjson"}
        # We'll externalize synchronously at call site; placeholder here—replaced below
        return {"__aether_ref__": "pending:externalize"}

    # First pass: mark structure with refs
    jsonish, had_refs = map_jsonish_or_ref(outputs, mk_ref=mk_ref)

    if not had_refs or not allow_externalize or artifacts is None:
        # Nothing to externalize (or disabled) → return as-is
        return jsonish

    # Second pass: walk and replace "pending:externalize" leaves with real URIs
    async def _resolve_refs(x):
        if isinstance(x, dict):
            if x.get("__aether_ref__") == "pending:externalize" and len(x) == 1:
                # We need original object to externalize; this simple walker can’t see it anymore.
                # Approach: re-walk original outputs in parallel to find leaves that caused refs.
                # For simplicity & performance, do a single-pass direct externalization below instead.
                return x
            return {k: await _resolve_refs(v) for k, v in x.items()}
        if isinstance(x, list):
            return [await _resolve_refs(v) for v in x]
        return x

    # Simpler approach: do a second real pass that externalizes by diffing original leaves.
    # Implement a focused externalizer that walks original outputs again.
    async def _externalize_in_place(orig):
        # returns jsonish w/ real refs
        from collections.abc import Mapping, Sequence
        if isinstance(orig, (str, int, float, bool, type(None))):
            return orig
        if isinstance(orig, Mapping):
            out = {}
            for k, v in orig.items():
                out[str(k)] = await _externalize_in_place(v)
            return out
        if isinstance(orig, Sequence) and not isinstance(orig, (str, bytes, bytearray)):
            return [await _externalize_in_place(v) for v in orig]

        # leaf non-JSON → actual artifact ref
        ref = await _externalize_leaf_to_artifact(
            orig,
            run_id=run_id, graph_id=graph_id, node_id=node_id,
            tool_name=tool_name, tool_version=tool_version,
            artifacts=artifacts
        )
        return {"__aether_ref__": ref["__aether_ref__"], **{k: v for k, v in ref.items() if k != "__aether_ref__"}}

    return await _externalize_in_place(outputs)


async def state_to_json_safe(
    state_obj,
    *,
    run_id: str,
    graph_id: str,
    artifacts=None,
    allow_externalize: bool = False, # Do not externalize by default until fixing artifacts writer
    include_wait_spec: bool = True,
) -> Dict[str, Any]:
    """
    Convert TaskGraphState to a JSON-safe dict.
    - JSON outputs inlined
    - non-JSON leaves become {"__aether_ref__": "..."} and (optionally) are externalized to artifacts
    """
    nodes_block = {}
    for nid, ns in state_obj.nodes.items():
        status = getattr(ns, "status", None)
        status_name = getattr(status, "name", status)  # Enum.name or string
        tool_name = getattr(ns, "tool_name", None) or getattr(getattr(ns, "spec", None), "tool_name", None)
        tool_version = getattr(ns, "tool_version", None) or getattr(getattr(ns, "spec", None), "tool_version", None)

        outputs_json = await _jsonish_outputs_with_refs(
            outputs=getattr(ns, "outputs", None),
            run_id=run_id,
            graph_id=graph_id,
            node_id=nid,
            tool_name=tool_name,
            tool_version=tool_version,
            artifacts=artifacts,
            allow_externalize=allow_externalize,
        )

        entry = {
            "status": status_name,
            "outputs": outputs_json,
            "error": getattr(ns, "error", None),
            "attempts": getattr(ns, "attempts", 0),
            "next_wakeup_at": getattr(ns, "next_wakeup_at", None),
            "wait_token": getattr(ns, "wait_token", None),
        }
        if include_wait_spec:
            ws = getattr(ns, "wait_spec", None)
            if ws:
                # ensure JSON-safe wait_spec (it should be strings/lists/dicts already)
                entry["wait_spec"] = ws
        nodes_block[nid] = entry

    return {
        "run_id": getattr(state_obj, "run_id", run_id),
        "rev": getattr(state_obj, "rev", None),
        "patches": [asdict(p) for p in getattr(state_obj, "patches", [])],
        "_bound_inputs": getattr(state_obj, "_bound_inputs", None),
        "nodes": nodes_block,
    }