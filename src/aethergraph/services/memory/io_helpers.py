from __future__ import annotations
from typing import Any 
from aethergraph.contracts.services.memory import Value


"""Create a Value of vtype 'ref' pointing to the given kind and uri.
Args:
    name: name of the Value slot
    kind: kind of the referenced artifact, e.g. "spec", "design", "output", "tool_result"
    uri: URI of the referenced artifact, e.g. "file://...", "mem://...", "db://..."
    meta: optional additional metadata for the Ref
Returns:
    Value dict with vtype 'ref'

Example:
    v = ref(
        name="my_ref",
        kind="spec",
        uri="file://path/to/spec",
        title="My Spec",
        mime="application/json"
    )
    print(v)
    # Output: {
    #   "name": "my_ref",
    #   "vtype": "ref",
    #   "value": {
    #       "kind": "spec",
    #       "uri": "file://path/to/spec",
    #       "title": "My Spec",
    #       "mime": "application/json"
    #   }
    # }
"""

def ref(name: str, kind: str, uri: str, **meta) -> Value:

    v: Value = {
        "name": name, 
        "vtype": "ref",
        "value": {
            "kind": kind,
            "uri": uri,
            **meta
        },
    }
    return v

def num(name: str, x: float) -> Value:
    """Create a Value of vtype 'number'."""
    return {"name": name, "vtype": "number", "value": float(x)}

def text(name: str, s: str) -> Value:
    """Create a Value of vtype 'string'."""
    return {"name": name, "vtype": "string", "value": str(s)}

def flag(name: str, b: bool) -> Value:
    """Create a Value of vtype 'boolean'."""
    return {"name": name, "vtype": "boolean", "value": bool(b)}


def obj(name: str, d: dict) -> Value:
    """Create a Value of vtype 'object'."""
    return {"name": name, "vtype": "object", "value": dict(d)}

def arr(name: str, lst: list) -> Value:
    """Create a Value of vtype 'array'."""
    return {"name": name, "vtype": "array", "value": list(lst)}

def null(name: str) -> Value:
    """Create a Value of vtype 'null'."""
    return {"name": name, "vtype": "null", "value": None}
