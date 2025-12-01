# /graphs


from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query

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
    flow_id: Annotated[str | None, Query()] = None,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> list[GraphListItem]:
    """
    List available graphs (TaskGraphs and GraphFunctions).

    Optional:
      - flow_id: filter to graphs whose registry metadata has this flow_id.
    """
    reg: UnifiedRegistry = current_registry()

    items: list[GraphListItem] = []

    # ---- 1) Static TaskGraphs (ns="graph") ----
    latest_graphs = reg.list(nspace=GRAPH_NS)
    for key, version in latest_graphs.items():
        ns, name = key.split(":", 1)
        if ns != GRAPH_NS:
            continue

        graph_obj = reg.get_graph(name=name, version=version)
        spec = getattr(graph_obj, "spec", None)

        meta = reg.get_meta(nspace=GRAPH_NS, name=name, version=version) or {}
        meta_flow_id: str | None = meta.get("flow_id")
        meta_entrypoint: bool = bool(meta.get("entrypoint", False))
        meta_tags = list(meta.get("tags", []))

        # flow filter
        if flow_id is not None and meta_flow_id != flow_id:
            continue

        if spec is None:
            items.append(
                GraphListItem(
                    graph_id=name,
                    name=name,
                    description=None,
                    inputs=[],
                    outputs=[],
                    tags=meta_tags or ["graph"],
                    kind="graph",
                    flow_id=meta_flow_id,
                    entrypoint=meta_entrypoint,
                )
            )
            continue

        inputs = list(spec.io.required.keys()) + list(spec.io.optional.keys())
        outputs = list(spec.io.outputs.keys())

        desc = spec.meta.get("description") if hasattr(spec, "meta") else None
        spec_tags = list(spec.meta.get("tags", [])) if hasattr(spec, "meta") else []

        tags = meta_tags or spec_tags or ["graph"]

        items.append(
            GraphListItem(
                graph_id=name,
                name=name,
                description=desc,
                inputs=inputs,
                outputs=outputs,
                tags=tags,
                kind="graph",
                flow_id=meta_flow_id,
                entrypoint=meta_entrypoint,
            )
        )

    # ---- 2) Imperative GraphFunctions (ns="graphfn") ----
    latest_graphfns = reg.list(nspace=GRAPHFN_NS)
    for key, version in latest_graphfns.items():
        ns, name = key.split(":", 1)
        if ns != GRAPHFN_NS:
            continue

        gf = reg.get_graphfn(name=name, version=version)
        if not _is_graph_function(gf):
            continue

        meta = reg.get_meta(nspace=GRAPHFN_NS, name=name, version=version) or {}
        meta_flow_id: str | None = meta.get("flow_id")
        meta_entrypoint: bool = bool(meta.get("entrypoint", False))
        meta_tags = list(meta.get("tags", []))

        if flow_id is not None and meta_flow_id != flow_id:
            continue

        inputs = list(getattr(gf, "inputs", []) or [])
        outputs = list(getattr(gf, "outputs", []) or [])
        desc = getattr(gf, "description", None)

        items.append(
            GraphListItem(
                graph_id=name,
                name=name,
                description=desc,
                inputs=inputs,
                outputs=outputs,
                tags=meta_tags or ["graphfn"],
                kind="graphfn",
                flow_id=meta_flow_id,
                entrypoint=meta_entrypoint,
            )
        )

    return items


@router.get("/graphs/{graph_id}", response_model=GraphDetail)
async def get_graph_detail(
    graph_id: str,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> GraphDetail:
    """
    Get detailed information about a specific graph (structure only).
    """
    reg: UnifiedRegistry = current_registry()

    # 1) Try TaskGraph
    try:
        graph_obj = reg.get_graph(name=graph_id, version=None)
        spec = getattr(graph_obj, "spec", None)
        meta = reg.get_meta(nspace=GRAPH_NS, name=graph_id, version=None) or {}

        flow_id = meta.get("flow_id")
        entrypoint = bool(meta.get("entrypoint", False))
        meta_tags = list(meta.get("tags", []))

        if spec is None:
            return GraphDetail(
                graph_id=graph_id,
                name=graph_id,
                description=None,
                inputs=[],
                outputs=[],
                tags=meta_tags or ["graph"],
                kind="graph",
                flow_id=flow_id,
                entrypoint=entrypoint,
                nodes=[],
                edges=[],
            )

        # ---- Nodes from TaskNodeSpec ----
        nodes_list: list[dict[str, Any]] = []
        for node_id, node_spec in spec.nodes.items():
            node_info: dict[str, Any] = {
                "id": node_id,
                "type": str(getattr(node_spec, "type", "")),
                "tool_name": getattr(node_spec, "tool_name", None),
                "tool_version": getattr(node_spec, "tool_version", None),
                "expected_inputs": list(getattr(node_spec, "expected_input_keys", []) or []),
                "expected_outputs": list(getattr(node_spec, "expected_output_keys", []) or []),
                "output_keys": list(getattr(node_spec, "output_keys", []) or []),
            }
            nodes_list.append(node_info)

        # ---- Edges from dependencies ----
        edge_set: set[tuple[str, str]] = set()
        for node_id, node_spec in spec.nodes.items():
            for dep_id in getattr(node_spec, "dependencies", []):
                edge_set.add((str(dep_id), str(node_id)))

        edges_list: list[dict[str, Any]] = [
            {"source": src, "target": dst} for (src, dst) in sorted(edge_set)
        ]

        inputs = list(spec.io.required.keys()) + list(spec.io.optional.keys())
        outputs = list(spec.io.outputs.keys())
        desc = spec.meta.get("description") if hasattr(spec, "meta") else None
        spec_tags = list(spec.meta.get("tags", [])) if hasattr(spec, "meta") else []

        tags = meta_tags or spec_tags or ["graph"]

        return GraphDetail(
            graph_id=graph_id,
            name=graph_id,
            description=desc,
            inputs=inputs,
            outputs=outputs,
            tags=tags,
            kind="graph",
            flow_id=flow_id,
            entrypoint=entrypoint,
            nodes=nodes_list,
            edges=edges_list,
        )

    except KeyError:
        pass

    # 2) Try GraphFunction
    try:
        gf = reg.get_graphfn(name=graph_id, version=None)
    except KeyError as e:
        raise HTTPException(status_code=404, detail="Graph not found") from e

    meta = reg.get_meta(nspace=GRAPHFN_NS, name=graph_id, version=None) or {}
    flow_id = meta.get("flow_id")
    entrypoint = bool(meta.get("entrypoint", False))
    meta_tags = list(meta.get("tags", []))

    inputs = list(getattr(gf, "inputs", []) or [])
    outputs = list(getattr(gf, "outputs", []) or [])
    desc = getattr(gf, "description", None)

    return GraphDetail(
        graph_id=graph_id,
        name=graph_id,
        description=desc,
        inputs=inputs,
        outputs=outputs,
        tags=meta_tags or ["graphfn"],
        kind="graphfn",
        flow_id=flow_id,
        entrypoint=entrypoint,
        nodes=[],  # GraphFunction has no static node DAG
        edges=[],
    )
