# aethergraph/core/persist/resume_policy.py
from __future__ import annotations
from typing import Any, Dict, Iterable, Tuple
from aethergraph.contracts.errors.errors import ResumeIncompatibleSnapshot

_JSON_PRIMITIVES = (str, int, float, bool, type(None))

def _is_json_like(x: Any) -> bool:
    if isinstance(x, _JSON_PRIMITIVES):
        return True
    if isinstance(x, list):
        return all(_is_json_like(v) for v in x)
    if isinstance(x, dict):
        # Treat any dict that has __aether_ref__ as NOT allowed in strict JSON-only policy
        if "__aether_ref__" in x:
            return False
        return all(isinstance(k, str) and _is_json_like(v) for k, v in x.items())
    return False


def _walk_non_json(obj, path="$"):
    from collections.abc import Mapping, Sequence
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return
    if isinstance(obj, Mapping):
        if "__aether_ref__" in obj:
            yield path
            return
        for k, v in obj.items():
            yield from _walk_non_json(v, f"{path}.{k}")
        return
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        for i, v in enumerate(obj):
            yield from _walk_non_json(v, f"{path}[{i}]")
        return
    yield path  # non-JSON leaf


def assert_snapshot_json_only(
    run_id: str,
    snap_json: dict,
    *,
    mode: str = "reuse_only",           # "strict" | "reuse_only"
    ignore_nodes: set[str] | None = None,  # node_ids to skip (e.g., graph output producers)
) -> None:
    """
    - mode="strict": scan ALL nodes; forbid any non-JSON/ref anywhere.
    - mode="reuse_only": ONLY check nodes whose outputs reuse (status DONE/SKIPPED).
    - ignore_nodes: always skip these node_ids (e.g., final/sink nodes that feed graph outputs).
    """
    ignore_nodes = ignore_nodes or set()
    reasons: list[str] = []

    state = snap_json.get("state") or snap_json
    nodes = state.get("nodes", {})

    def _should_check(nid: str, ns: dict) -> bool:
        if nid in ignore_nodes:
            return False
        if mode == "strict":
            return True
        st = (ns.get("status") or "").upper()
        return st in {"DONE", "SKIPPED"}

    for nid, ns in nodes.items():
        if not _should_check(nid, ns):
            continue
        outs = ns.get("outputs")
        if not outs:
            continue
        bad_paths = list(_walk_non_json(outs))
        if bad_paths:
            reasons.append(
                f"node '{nid}' outputs contain non-JSON or refs at: "
                + ", ".join(bad_paths[:8]) + (" ..." if len(bad_paths) > 8 else "")
            )

    if reasons:
        raise ResumeIncompatibleSnapshot(run_id, reasons)
    

def output_node_ids_from_graph(graph) -> set[str]:
    """
    Collect node_ids that directly feed the declared graph outputs.
    Expected binding shape:
      {'sum': {'_type': 'ref', 'from': 'combine_3', 'key': 'sum'}}
    """
    try:
        bindings = graph.io_signature().get("outputs", {}).get("bindings", {}) or {}
    except Exception:
        bindings = {}

    out_nodes: set[str] = set()

    def _collect(ref):
        # Handle canonical case
        if isinstance(ref, dict) and ref.get("_type") == "ref":
            src = ref.get("from")
            if isinstance(src, str) and src:
                out_nodes.add(src)
            return

        # Be forgiving if future formats appear (list of refs, nested, etc.)
        if isinstance(ref, dict):
            for v in ref.values():
                _collect(v)
        elif isinstance(ref, (list, tuple)):
            for v in ref:
                _collect(v)

    for v in bindings.values():
        _collect(v)

    return out_nodes

