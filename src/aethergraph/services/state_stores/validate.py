from __future__ import annotations
from typing import Any, Mapping, Sequence

class ResumptionNotSupported(Exception):
    """Raised when a snapshot contains non-JSON-pure outputs and resume is disabled."""
    pass

_JSON_SCALARS = (str, int, float, bool, type(None))

def _is_json_pure(obj: Any) -> bool:
    """
    Strict check: only JSON scalars, lists/tuples, and dicts with string keys.
    No custom objects, no bytes, no externalization markers.
    """
    if isinstance(obj, _JSON_SCALARS):
        return True

    if isinstance(obj, Mapping):
        # Disallow future externalization markers proactively
        if "__aether_ref__" in obj or obj.get("__externalized__") is True:
            return False
        # JSON requires string keys
        for k, v in obj.items():
            if not isinstance(k, str):
                return False
            if not _is_json_pure(v):
                return False
        return True

    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return all(_is_json_pure(v) for v in obj)

    return False


def assert_snapshot_json_pure(snapshot_state: dict, *, run_id: str, graph_id: str,
                              allow_non_json: bool = False) -> None:
    """
    Validate the *serialized* (dict) snapshot state produced by snapshot_from_graph(...).
    If any node outputs are not strictly JSON-pure, raise ResumptionNotSupported
    (unless allow_non_json=True).
    """
    if allow_non_json:
        return

    if not isinstance(snapshot_state, dict):
        raise ResumptionNotSupported(
            f"Resume blocked: snapshot state is not a dict (run_id={run_id}, graph_id={graph_id})."
        )

    nodes = snapshot_state.get("nodes", {})
    bad_nodes = []

    for nid, ns in nodes.items():
        if not isinstance(ns, dict):
            bad_nodes.append(nid)
            continue
        outs = ns.get("outputs", None)
        if outs is None:
            continue  # output not produced yet is fine
        if not _is_json_pure(outs):
            bad_nodes.append(nid)

    if bad_nodes:
        listed = ", ".join(bad_nodes)
        raise ResumptionNotSupported(
            "Resume blocked: snapshot contains non-JSON outputs; "
            f"nodes=({listed}) run_id={run_id} graph_id={graph_id}. "
            "Ensure tools return JSON-friendly outputs (dict/list/str/float/int/bool/null), "
            "or disable strict resume checks once tested."
        )
