from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

JsonType = Literal["string", "number", "boolean", "object", "array"]


@dataclass
class IOSlot:
    """
    One input or output slot for a graph action.

    - `type` is a coarse JSON-like type for LLM/tool matching.
    """

    name: str
    type: JsonType | None = None
    description: str | None = None
    required: bool = True
    default: Any | None = None


@dataclass
class ActionSpec:
    """
    Docstring for ActionSpec
    """

    name: str  # hunman name: usually the graph/graphfn name
    ref: str  # canonical ref: e.g. "graph:mygraph:1.0.0"
    kind: Literal["graph", "graphfn"]  # kind of action
    version: str  # version string

    inputs: list[IOSlot]  # input slots
    outputs: list[IOSlot]  # output slots

    description: str | None = None  # human description for LLM/tool matching
    tags: list[str] = None  # optional tags
    flow_id: str | None = None  # optional flow ID. If None, treat it globally.


def _map_py_type_to_json_type(tp: Any) -> JsonType | None:
    """
    Docstring for _map_py_type_to_json_type

    :param tp: Description
    :type tp: Any
    :return: Description
    :rtype: JsonType | None
    """
    origin = getattr(tp, "__origin__", None)

    if tp in (str, bytes):
        return "string"
    if tp in (int, float):
        return "number"
    if tp is bool:
        return "boolean"

    # collections
    if origin in (list, tuple, set) or tp in (list, tuple, set):
        return "array"

    # dict / mapping -> treat as object
    if origin in (dict,) or tp in (dict,):
        return "object"

    # Optional[T] / Union[T, None] etc: peel outer layer and re-map
    if origin is __import__("typing").Union:
        args = [a for a in tp.__args__ if a is not type(None)]  # noqa: E721
        if len(args) == 1:
            return _map_py_type_to_json_type(args[0])

    # Fallback = "object" or None; for now we return object
    return "object"
