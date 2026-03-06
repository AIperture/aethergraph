from __future__ import annotations

import pytest

from aethergraph.contracts.services.artifacts import Artifact
from aethergraph.core.runtime.runtime_registry import set_current_registry
from aethergraph.services.registry.registration_service import RegistrationService
from aethergraph.services.registry.unified_registry import UnifiedRegistry
from aethergraph.storage.docstore.fs_doc import FSDocStore
from aethergraph.storage.registry.registration_docstore import RegistrationManifestStore

GOOD_SOURCE = """
from aethergraph import graphify, tool

@tool(name="double_value", outputs=["value"])
def double_value(x: int):
    return {"value": x * 2}

@graphify(name="demo_graph", inputs=["x"], outputs=["value"])
def demo_graph(x):
    out = double_value(x=x)
    return {"value": out.value}
"""


class _StubArtifactStore:
    def __init__(self, text_by_uri: dict[str, str] | None = None):
        self.text_by_uri = dict(text_by_uri or {})

    async def load_text(self, uri: str, *, encoding: str = "utf-8", errors: str = "strict") -> str:
        _ = (encoding, errors)
        return self.text_by_uri[uri]


class _StubArtifactIndex:
    def __init__(self, uri_by_id: dict[str, str] | None = None):
        self.uri_by_id = dict(uri_by_id or {})

    async def get(self, artifact_id: str) -> Artifact | None:
        uri = self.uri_by_id.get(artifact_id)
        if uri is None:
            return None
        return Artifact(artifact_id=artifact_id, uri=uri)


def _make_service(
    tmp_path,
    *,
    registry: UnifiedRegistry,
    artifact_uri_by_id: dict[str, str] | None = None,
    artifact_text_by_uri: dict[str, str] | None = None,
) -> RegistrationService:
    docs = FSDocStore(root=str(tmp_path / "docs"))
    manifests = RegistrationManifestStore(doc_store=docs)
    return RegistrationService(
        registry=registry,
        manifest_store=manifests,
        artifact_store=_StubArtifactStore(artifact_text_by_uri),
        artifact_index=_StubArtifactIndex(artifact_uri_by_id),
    )


@pytest.mark.asyncio
async def test_register_by_file_success(tmp_path):
    reg = UnifiedRegistry()
    set_current_registry(reg)
    service = _make_service(tmp_path, registry=reg)

    src_path = tmp_path / "demo_graph.py"
    src_path.write_text(GOOD_SOURCE, encoding="utf-8")

    result = await service.register_by_file(str(src_path), persist=True, strict=True)
    assert result.success is True
    assert result.graph_name == "demo_graph"
    assert reg.get_graph("demo_graph") is not None

    rows = await service.manifest_store.list_entries()  # type: ignore[union-attr]
    assert len(rows) == 1
    assert rows[0]["source_kind"] == "file"


@pytest.mark.asyncio
async def test_register_by_artifact_success(tmp_path):
    reg = UnifiedRegistry()
    set_current_registry(reg)

    service = _make_service(
        tmp_path,
        registry=reg,
        artifact_uri_by_id={"art-1": "artifact://art-1.py"},
        artifact_text_by_uri={"artifact://art-1.py": GOOD_SOURCE},
    )
    result = await service.register_by_artifact(artifact_id="art-1", persist=True, strict=True)
    assert result.success is True
    assert result.graph_name == "demo_graph"
    assert reg.get_graph("demo_graph") is not None


@pytest.mark.asyncio
async def test_register_by_folder_mixed(tmp_path):
    reg = UnifiedRegistry()
    set_current_registry(reg)
    service = _make_service(tmp_path, registry=reg)

    folder = tmp_path / "graphs"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "good.py").write_text(GOOD_SOURCE, encoding="utf-8")
    (folder / "bad.py").write_text("def f():\n    return 1\n", encoding="utf-8")

    results = await service.register_by_folder(str(folder), strict=False, persist=True)
    assert len(results) == 2
    assert any(r.success for r in results)
    assert any(not r.success for r in results)


def test_validate_graphify_source_errors():
    reg = UnifiedRegistry()
    set_current_registry(reg)
    service = RegistrationService(registry=reg)

    bad_source = """
from aethergraph import graphify

@graphify(name="x", inputs=["a"])
async def x(a):
    return {"a": a}
"""
    result = service.validate_graphify_source(bad_source, filename="bad.py", strict=True)
    assert result.ok is False
    issue_codes = {i.code for i in result.issues}
    assert "missing_decorator_kw" in issue_codes
    assert "graphify_async_def" in issue_codes


def test_validate_graphify_source_strict_dag_rules():
    reg = UnifiedRegistry()
    set_current_registry(reg)
    service = RegistrationService(registry=reg)

    source = """
from aethergraph import graphify, tool

@tool(name="inner", outputs=["x"])
def inner(x: int):
    return {"x": x}

@tool(name="outer", outputs=["x"])
def outer(x: int):
    y = inner(x=x)
    return {"x": y["x"]}

def plain(x):
    return {"x": x}

@graphify(name="g", inputs=["x"], outputs=["x"])
def g(x):
    h = plain(x)
    z = inner(x=x, _condition=lambda n: True)
    if h["x"] > 1:
        return {"x": z.x}
    return {"x": 0}
"""
    result = service.validate_graphify_source(source, filename="strict_bad.py", strict=True)
    assert result.ok is False
    issue_codes = {i.code for i in result.issues}
    assert "tool_nested_tool_call_disallowed" in issue_codes
    assert "graphify_plain_call_used_as_handle" in issue_codes
    assert "graphify_unsupported_condition_expr" in issue_codes
    assert "graphify_control_flow_non_deterministic" in issue_codes


def test_validate_graphify_source_result_shape_regression():
    reg = UnifiedRegistry()
    set_current_registry(reg)
    service = RegistrationService(registry=reg)

    result = service.validate_graphify_source(GOOD_SOURCE, filename="good.py", strict=True)
    assert result.ok is True
    assert isinstance(result.issues, list)
    assert isinstance(result.graph_names, list)
    assert isinstance(result.graphfn_names, list)
    assert "demo_graph" in result.graph_names


@pytest.mark.asyncio
async def test_replay_registered_sources(tmp_path):
    reg1 = UnifiedRegistry()
    set_current_registry(reg1)

    docs = FSDocStore(root=str(tmp_path / "docs"))
    manifests = RegistrationManifestStore(doc_store=docs)
    artifact_uri = "artifact://artifact-demo.py"
    artifact_id = "artifact-demo"

    writer = RegistrationService(
        registry=reg1,
        manifest_store=manifests,
        artifact_store=_StubArtifactStore({artifact_uri: GOOD_SOURCE}),
        artifact_index=_StubArtifactIndex({artifact_id: artifact_uri}),
    )
    create_result = await writer.register_by_artifact(
        artifact_id=artifact_id, persist=True, strict=True
    )
    assert create_result.success is True

    reg2 = UnifiedRegistry()
    set_current_registry(reg2)
    replayer = RegistrationService(
        registry=reg2,
        manifest_store=manifests,
        artifact_store=_StubArtifactStore({artifact_uri: GOOD_SOURCE}),
        artifact_index=_StubArtifactIndex({artifact_id: artifact_uri}),
    )
    report = await replayer.replay_registered_sources(strict=False)
    assert report.loaded == 1
    assert report.failed == 0
    assert reg2.get_graph("demo_graph") is not None
