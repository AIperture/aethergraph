from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple, TypedDict, Union 

GRAPH_INPUTS_NODE_ID = "__graph_inputs__" # special node_id for graph inputs
RESERVED_INJECTABLES = {"resume", "context", "self"}


REF_TYPE = "ref"
class RefDict(TypedDict):
    _type: str
    from_: str  # 'from' is reserved in Python keyword args, keep key as 'from' in payload though
    key: str


Ref = Dict[str, str] # {"_type": "ref", "from": "<node_id>", "key": "<output_key>"}

# ---------- Constructors ----------

def ref(node_id: str, key: str) -> Ref:
    return {"_type": "ref", "from": node_id, "key": key}

def arg(name: str) -> Ref:
    return ref(GRAPH_INPUTS_NODE_ID, name)


# ---------- Type checks / Normalizations ----------
def is_ref(x: Any) -> bool:
    """True if x is a dict that looks like a Ref."""
    return isinstance(x, Mapping) and x.get("_type") == REF_TYPE and "from" in x and "key" in x

def is_arg_ref(x: Any) -> bool:
    """True if x is a ref pointing to __graph_inputs__."""
    return is_ref(x) and x.get("from") == GRAPH_INPUTS_NODE_ID

def to_tuple(x: Union[Ref, Tuple[str, str]]) -> Optional[Tuple[str, str]]:
    """Return (node_id, key) if x is a Ref/tuple; else None."""
    if isinstance(x, tuple) and len(x) == 2 and all(isinstance(s, str) for s in x):
        return x  # already canonical enough
    if is_ref(x):
        return x["from"], x["key"]
    return None

def from_tuple(node_key: Tuple[str, str]) -> Ref:
    """Build a Ref from (node_id, key)."""
    node_id, key = node_key
    return ref(node_id, key)

def normalize_binding(x: Any) -> Any:
    """
    Normalize a binding value (Ref | (node, key) | literal) into Ref|literal.
    - If tuple -> Ref
    - If Ref -> ensure minimal canonical shape
    - Else literal passthrough
    """
    t = to_tuple(x)
    if t is not None:
        return from_tuple(t)
    if is_ref(x):
        # keep only the keys we care about (defensive)
        return {"_type": REF_TYPE, "from": x["from"], "key": x["key"]}
    return x  # literal


# ---------- Resolution ----------
def resolve_ref(reference: Ref, node_outputs: Mapping[str, Mapping[str, Any]]) -> Any:
    """
    Resolve a Ref against the current node_outputs: {node_id: {output_key: value}}.
    Returns None if missing.
    """
    src = reference["from"]
    key = reference["key"]

    if src is None or key is None:
        raise KeyError(f"Bad Ref: {ref}")
    if src not in node_outputs:
        raise KeyError(f"Upstream node '{src}' has no outputs yet")
    if key not in node_outputs[src]:
        raise KeyError(f"Output '{key}' not found on node '{src}'")
    
    outs = node_outputs.get(src)
    return outs.get(key) if isinstance(outs, Mapping) else None

    
def resolve_any(val, *, graph_inputs: Dict[str, Any], outputs_by_node: Dict[str, Dict[str, Any]]):
    """Recursively resolve any value that may contain Refs or Args. This function is used 
    to resolve inputs for a node before execution.
    Args:
        val: The value to resolve. Can be a literal, dict, list, or Ref/Arg.
        graph_inputs: The dict of graph inputs for Arg resolution.
        outputs_by_node: The dict of node_id to outputs for Ref resolution.
    Returns:
        The fully resolved value.
    """
    # Arg shape: {"_type":"arg","key":"<input_key>"}
    if isinstance(val, dict):
        t = val.get("_type")
        if t == "arg":
            k = val.get("key")
            if k not in graph_inputs:
                raise KeyError(f"Graph input '{k}' was not provided")
            return graph_inputs[k]
        if t == "ref":
            return resolve_ref(val, outputs_by_node)
        # regular dict → recurse
        return {k: resolve_any(v, graph_inputs=graph_inputs, outputs_by_node=outputs_by_node)
                for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        cast = list if isinstance(val, list) else tuple
        return cast(resolve_any(v, graph_inputs=graph_inputs, outputs_by_node=outputs_by_node) for v in val)
    return val  # literal


def resolve_binding(binding: Any, node_outputs: Mapping[str, Mapping[str, Any]]) -> Any:
    """
    Resolve a binding that can be Ref or literal. Literals pass through unchanged.
    """
    if is_ref(binding):
        return resolve_ref(binding, node_outputs)
    return binding

# ---------- Pretty helpers ----------

def ref_str(x: Union[Ref, Tuple[str, str], Any]) -> str:
    """Human-friendly string for logs."""
    t = to_tuple(x)
    if t is None:
        return repr(x)
    node_id, key = t
    return f"{node_id}.{key}"

# --------- Marker checks ----------
def is_arg_marker(x: Any) -> bool:
    return isinstance(x, Mapping) and x.get("_type") == "arg" and "key" in x

def is_context_marker(x: Any) -> bool:
    return isinstance(x, Mapping) and x.get("_type") == "context" and "key" in x
