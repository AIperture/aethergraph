# tests/test_graph_endpoints.py

from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

import aethergraph.api.v1.graphs as graphs_api
from aethergraph.services.registry.unified_registry import UnifiedRegistry

# --- Fake specs/graph objects for testing ---


@dataclass
class FakeIOSpec:
    required: dict[str, Any] = field(default_factory=dict)
    optional: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeTaskNodeSpec:
    node_id: str
    type: str
    dependencies: list[str] = field(default_factory=list)
    expected_input_keys: list[str] = field(default_factory=list)
    expected_output_keys: list[str] = field(default_factory=lambda: ["result"])
    output_keys: list[str] = field(default_factory=lambda: ["result"])
    tool_name: str | None = None
    tool_version: str | None = None


@dataclass
class FakeTaskGraphSpec:
    graph_id: str
    version: str = "0.1.0"
    nodes: dict[str, FakeTaskNodeSpec] = field(default_factory=dict)
    io: FakeIOSpec = field(default_factory=FakeIOSpec)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeTaskGraph:
    spec: FakeTaskGraphSpec


class FakeGraphFunction:
    def __init__(
        self, name: str, inputs: list[str] | None = None, outputs: list[str] | None = None
    ):
        self.name = name
        self._inputs = inputs or []
        self._outputs = outputs or []

    @property
    def fn(self):
        def _dummy_fn(**kwargs):
            return {}

        return _dummy_fn

    @property
    def inputs(self) -> list[str]:
        return self._inputs

    @property
    def outputs(self) -> list[str]:
        return self._outputs


@pytest.fixture()
def client(monkeypatch) -> TestClient:
    """
    Build a small FastAPI app with the /graphs router, and a fresh registry
    patched into the graphs module.
    """
    reg = UnifiedRegistry()

    # Patch current_registry used inside graphs_api
    monkeypatch.setattr(graphs_api, "current_registry", lambda: reg)

    # Register a fake static graph (ns="graph")
    spec = FakeTaskGraphSpec(
        graph_id="test_graph",
        nodes={
            "n1": FakeTaskNodeSpec(
                node_id="n1",
                type="tool",
                dependencies=[],
                expected_input_keys=["x"],
                expected_output_keys=["result"],
                output_keys=["result"],
                tool_name="work",
                tool_version="0.1.0",
            ),
            "n2": FakeTaskNodeSpec(
                node_id="n2",
                type="tool",
                dependencies=["n1"],
                expected_input_keys=["y"],
                expected_output_keys=["result"],
                output_keys=["result"],
                tool_name="reduce",
                tool_version="0.1.0",
            ),
        },
        io=FakeIOSpec(
            required={"items": None},
            optional={},
            outputs={"sum": None},
        ),
        meta={"description": "A fake test graph", "tags": ["test", "graph"]},
    )
    task_graph = FakeTaskGraph(spec=spec)
    reg.register(nspace="graph", name="test_graph", version="0.1.0", obj=task_graph)

    # Register a fake GraphFunction (ns="graphfn")
    gf = FakeGraphFunction(name="batch_agent", inputs=["items"], outputs=["ys"])
    reg.register(nspace="graphfn", name="batch_agent", version="0.1.0", obj=gf)

    # Build app
    app = FastAPI()
    app.include_router(graphs_api.router, prefix="/api/v1")
    return TestClient(app)


def test_list_graphs_contains_both_graph_and_graphfn(client: TestClient):
    resp = client.get("/api/v1/graphs")
    assert resp.status_code == 200

    data = resp.json()
    ids = {g["graph_id"] for g in data}
    assert "test_graph" in ids
    assert "batch_agent" in ids

    # Basic shape checks
    tg = next(g for g in data if g["graph_id"] == "test_graph")
    assert tg["inputs"] == ["items"]
    assert tg["outputs"] == ["sum"]
    assert "graph" in tg["tags"]

    gf = next(g for g in data if g["graph_id"] == "batch_agent")
    assert gf["inputs"] == ["items"]
    assert gf["outputs"] == ["ys"]
    assert "graphfn" in gf["tags"]


def test_get_task_graph_detail_includes_nodes_and_edges(client: TestClient):
    resp = client.get("/api/v1/graphs/test_graph")
    assert resp.status_code == 200

    data = resp.json()
    assert data["graph_id"] == "test_graph"
    assert data["description"] == "A fake test graph"
    assert data["inputs"] == ["items"]
    assert data["outputs"] == ["sum"]

    # Nodes
    nodes = {n["id"]: n for n in data["nodes"]}
    assert set(nodes.keys()) == {"n1", "n2"}

    n1 = nodes["n1"]
    assert n1["tool_name"] == "work"
    assert n1["expected_inputs"] == ["x"]

    # Edges: n1 -> n2 from dependencies
    edges = data["edges"]
    assert {"source": "n1", "target": "n2"} in edges


def test_get_graphfn_detail_has_empty_nodes_and_edges(client: TestClient):
    resp = client.get("/api/v1/graphs/batch_agent")
    assert resp.status_code == 200

    data = resp.json()
    assert data["graph_id"] == "batch_agent"
    assert data["inputs"] == ["items"]
    assert data["outputs"] == ["ys"]
    assert data["nodes"] == []
    assert data["edges"] == []
