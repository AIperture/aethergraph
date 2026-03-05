from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import uuid
import warnings

import pytest

from aethergraph.contracts.errors.errors import GraphValidationError


def _exec_module_source(tmp_path: Path, source: str):
    module_name = f"_ag_fail_fast_{uuid.uuid4().hex}"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(module_name, None)


def test_graphify_fails_fast_on_async_def(tmp_path: Path):
    source = """
from aethergraph import graphify

@graphify(name="bad_async_graph", inputs=["x"], outputs=["x"])
async def bad_async_graph(x: int):
    return {"x": x}
"""
    with pytest.raises(GraphValidationError) as exc:
        _exec_module_source(tmp_path, source)
    msg = str(exc.value)
    assert "graphify_async_def" in msg


def test_graphify_fails_fast_on_invalid_condition_expr(tmp_path: Path):
    source = """
from aethergraph import graphify, tool

@tool(name="echo_for_failfast", outputs=["x"])
def echo_for_failfast(x: int):
    return {"x": x}

@graphify(name="bad_condition_graph", inputs=["x"], outputs=["x"])
def bad_condition_graph(x: int):
    out = echo_for_failfast(x=x, _condition=lambda _: True)
    return {"x": out.x}
"""
    with pytest.raises(GraphValidationError) as exc:
        _exec_module_source(tmp_path, source)
    msg = str(exc.value)
    assert "graphify_unsupported_condition_expr" in msg


def test_graphify_fails_fast_on_non_deterministic_control_flow(tmp_path: Path):
    source = """
from aethergraph import graphify, tool

@tool(name="echo_for_cf", outputs=["x"])
def echo_for_cf(x: int):
    return {"x": x}

@graphify(name="bad_control_flow_graph", inputs=["x"], outputs=["x"])
def bad_control_flow_graph(x: int):
    for _ in [1]:
        out = echo_for_cf(x=x)
    return {"x": out.x}
"""
    with pytest.raises(GraphValidationError) as exc:
        _exec_module_source(tmp_path, source)
    msg = str(exc.value)
    assert "graphify_control_flow_non_deterministic" in msg


def test_graphify_valid_source_loads(tmp_path: Path):
    source = """
from aethergraph import graphify, tool

@tool(name="double_for_valid", outputs=["value"])
def double_for_valid(x: int):
    return {"value": x * 2}

@graphify(name="valid_failfast_graph", inputs=["x"], outputs=["value"])
def valid_failfast_graph(x: int):
    out = double_for_valid(x=x)
    return {"value": out.value}
"""
    module = _exec_module_source(tmp_path, source)
    assert hasattr(module, "valid_failfast_graph")


def test_graphify_accepts_class_based_tool_calls(tmp_path: Path):
    source = """
from typing import Any
from aethergraph import graphify, tool
from aethergraph.core.tools.waitable import DualStageTool, WaitSpec

@tool(outputs=["decision"])
class ApproveWait(DualStageTool):
    outputs = ["decision"]

    async def setup(self, **kwargs):
        return WaitSpec(kind="approval", prompt={"text": "ok"}, resume_schema={"type": "object"})

    async def on_resume(self, resume: dict[str, Any], **kwargs):
        return {"decision": "Yes"}

@tool(outputs=["result"])
def emit(x: int):
    return {"result": x}

@graphify(name="wait_graph", inputs=["x"], outputs=["decision"])
def wait_graph(x: int):
    a = emit(x=x)
    w = ApproveWait(prompt="Proceed?")
    return {"decision": w.decision}
"""
    module = _exec_module_source(tmp_path, source)
    assert hasattr(module, "wait_graph")


def test_graphify_emits_warning_for_unused_local_assignment(tmp_path: Path):
    source = """
from aethergraph import graphify, tool

@tool(outputs=["value"])
def emit(x: int):
    return {"value": x}

@graphify(name="warn_unused_local", inputs=["x"], outputs=["value"])
def warn_unused_local(x: int):
    out = emit(x=x)
    dead_local = {"k": out.value}
    return {"value": out.value}
"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _exec_module_source(tmp_path, source)
    msgs = [str(w.message) for w in caught]
    assert any("graphify_unused_local_assignment" in m for m in msgs)


def test_graphify_emits_warning_for_risky_subscript_on_tool_output(tmp_path: Path):
    source = """
from aethergraph import graphify, tool

@tool(outputs=["payload"])
def build_payload(x: int):
    return {"payload": {"model": x}}

@graphify(name="warn_risky_subscript", inputs=["x"], outputs=["value"])
def warn_risky_subscript(x: int):
    node = build_payload(x=x)
    value = node.payload["model"]
    return {"value": value}
"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _exec_module_source(tmp_path, source)
    msgs = [str(w.message) for w in caught]
    assert any("graphify_risky_subscript_on_tool_output" in m for m in msgs)


def test_graph_fn_fails_fast_on_missing_required_kwargs(tmp_path: Path):
    source = """
from aethergraph import graph_fn

@graph_fn(name="bad_graph_fn", inputs=["x"])
def bad_graph_fn(x: int):
    return {"x": x}
"""
    with pytest.raises(GraphValidationError) as exc:
        _exec_module_source(tmp_path, source)
    msg = str(exc.value)
    assert "missing_decorator_kw" in msg
