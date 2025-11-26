# /graphs


from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from aethergraph.core.graph.graph_fn import GraphFunction
from aethergraph.core.graph.task_graph import TaskGraph
from aethergraph.core.runtime.runtime_registry import current_registry
from aethergraph.services.registry.unified_registry import UnifiedRegistry

from .deps import RequestIdentity, get_identity
from .schemas import GraphDetail, GraphListItem

router = APIRouter(tags=["graphs"])


GRAPH_NS = "graph"
GRAPHFN_NS = "graphfn"


def _is_task_graph(obj: Any) -> bool:
    if isinstance(obj, TaskGraph):
        return True
    # Fallback check -- used in tests
    return hasattr(obj, "spec")


def _is_graph_function(obj: Any) -> bool:
    if isinstance(obj, GraphFunction):
        return True
    # Fallback check -- used in tests
    return hasattr(obj, "fn") and hasattr(obj, "name")


@router.get("/graphs", response_model=list[GraphListItem])
async def list_graphs(
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> list[GraphListItem]:
    """
    List available graphs (both static TaskGraphs and imperative GraphFunctions).

    - graph (ns=graph): a TaskGraph with a TaskGraphSpec.
    - graphfn (ns=graphfn): an imperative GraphFunction without a static DAG spec.
    """
    reg: UnifiedRegistry = current_registry()

    items: list[GraphListItem] = []

    # 1) Static TaskGraphs (ns="graph")
    latest_graphs = reg.list(nspace=GRAPH_NS)  # {"graph:batch_agent": "0.1.0", ...}
    for key, version in latest_graphs.items():
        ns, name = key.split(":", 1)
        if ns != GRAPH_NS:
            continue
        graph_obj = reg.get_graph(name=name, version=version)

        # Assume TaskGraph has .spec: TaskGraphSpec
        spec = getattr(graph_obj, "spec", None)
        if spec is None:
            # Fallback: shallow info
            items.append(
                GraphListItem(
                    graph_id=name,
                    name=name,
                    description=None,
                    inputs=[],
                    outputs=[],
                    tags=["graph"],
                )
            )
            continue

        # Inputs/outputs from spec.io
        inputs = list(spec.io.required.keys()) + list(spec.io.optional.keys())
        outputs = list(spec.io.outputs.keys())

        items.append(
            GraphListItem(
                graph_id=name,
                name=name,
                description=spec.meta.get("description") if hasattr(spec, "meta") else None,
                inputs=inputs,
                outputs=outputs,
                tags=list(spec.meta.get("tags", [])) if hasattr(spec, "meta") else ["graph"],
            )
        )

    # 2) Imperative GraphFunctions (ns="graphfn")
    latest_graphfns = reg.list(nspace=GRAPHFN_NS)  # {"graphfn:batch_agent": "0.1.0", ...}
    for key, version in latest_graphfns.items():
        ns, name = key.split(":", 1)
        if ns != GRAPHFN_NS:
            continue
        gf = reg.get_graphfn(name=name, version=version)
        if not _is_graph_function(gf):
            continue

        # GraphFunction should expose .inputs and .outputs (possibly inferred)
        inputs = list(getattr(gf, "inputs", []) or [])
        outputs = list(getattr(gf, "outputs", []) or [])

        items.append(
            GraphListItem(
                graph_id=name,
                name=name,
                description=getattr(gf, "description", None),
                inputs=inputs,
                outputs=outputs,
                tags=["graphfn"],
            )
        )

    return items


@router.get("/graphs/{graph_id}", response_model=GraphDetail)
async def get_graph_detail(
    graph_id: str,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> GraphDetail:
    """
    Get detailed information about a specific graph.

    Resolution order:
      1) Try static TaskGraph (ns="graph").
      2) If not found, try GraphFunction (ns="graphfn").
    """
    reg: UnifiedRegistry = current_registry()

    # 1) Try TaskGraph
    try:
        graph_obj = reg.get_graph(name=graph_id, version=None)
        spec = getattr(graph_obj, "spec", None)

        if spec is None:
            return GraphDetail(
                graph_id=graph_id,
                name=graph_id,
                description=None,
                inputs=[],
                outputs=[],
                tags=["graph"],
                nodes=[],
                edges=[],
            )

        # ---- Nodes from TaskNodeSpec ----
        nodes_list: list[dict[str, Any]] = []
        for node_id, node_spec in spec.nodes.items():
            # node_spec is TaskNodeSpec
            node_info: dict[str, Any] = {
                "id": node_id,
                "type": str(node_spec.type),
                "tool_name": node_spec.tool_name,
                "tool_version": node_spec.tool_version,
                "expected_inputs": list(node_spec.expected_input_keys),
                "expected_outputs": list(node_spec.expected_output_keys),
                "output_keys": list(node_spec.output_keys),
            }
            # can add more later (metadata, condition, etc.)
            nodes_list.append(node_info)

        # ---- Edges from dependencies ----
        # For each node B with dependencies [A, C], create edges A->B, C->B.
        edge_set: set[tuple[str, str]] = set()
        for node_id, node_spec in spec.nodes.items():
            for dep_id in node_spec.dependencies:
                edge_set.add((dep_id, node_id))

        edges_list: list[dict[str, Any]] = [
            {"from": src, "to": dst} for (src, dst) in sorted(edge_set)
        ]

        # Inputs/outputs + meta
        inputs = list(spec.io.required.keys()) + list(spec.io.optional.keys())
        outputs = list(spec.io.outputs.keys())
        desc = spec.meta.get("description") if hasattr(spec, "meta") else None
        tags = list(spec.meta.get("tags", [])) if hasattr(spec, "meta") else ["graph"]

        return GraphDetail(
            graph_id=graph_id,
            name=graph_id,
            description=desc,
            inputs=inputs,
            outputs=outputs,
            tags=tags,
            nodes=nodes_list,
            edges=edges_list,
        )

    except KeyError:
        # Not a TaskGraph, fall through to graphfn
        pass

    # 2) Try GraphFunction
    try:
        gf = reg.get_graphfn(name=graph_id, version=None)
    except KeyError as e:
        raise HTTPException(status_code=404, detail="Graph not found") from e

    inputs = list(getattr(gf, "inputs", []) or [])
    outputs = list(getattr(gf, "outputs", []) or [])
    desc = getattr(gf, "description", None)

    return GraphDetail(
        graph_id=graph_id,
        name=graph_id,
        description=desc,
        inputs=inputs,
        outputs=outputs,
        tags=["graphfn"],
        nodes=[],
        edges=[],
    )
