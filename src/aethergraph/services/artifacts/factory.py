import os
from .fs_store import FSArtifactStore
from .jsonl_index import JsonlArtifactIndex
from .sqlite_index import SqliteArtifactIndex  # if present

def make_artifact_store() -> FSArtifactStore:
    base = os.getenv("ARTIFACTS_DIR", "./artifacts")
    return FSArtifactStore(base)

def make_artifact_index():
    kind = (os.getenv("ARTIFACT_INDEX", "jsonl")).lower()
    if kind == "sqlite":
        path = os.getenv("ARTIFACT_INDEX_SQLITE", "./artifacts/index.sqlite")
        return SqliteArtifactIndex(path)
    path = os.getenv("ARTIFACT_INDEX_JSONL", "./artifacts/index.jsonl")
    return JsonlArtifactIndex(path)

def make_facade_for_node(*, env, node, store=None, index=None):
    store = store or make_artifact_store()
    index = index or make_artifact_index()
    from aethergraph.services.artifacts.facade import ArtifactFacade
    return ArtifactFacade(
        run_id=env.run_id, graph_id=env.graph_id, node_id=node.node_id,
        tool_name=node.tool_name, tool_version=node.tool_version,
        store=store, index=index
    )