from __future__ import annotations

import ast
from hashlib import sha256
import json
from pathlib import Path

import pytest

from aethergraph.services.host.compiled_build import (
    CompiledBuildError,
    inspect_compiled_build,
)


def _write_build(parent: Path) -> Path:
    build_id = "0123456789abcdef01234567"
    root = parent / build_id
    generated = root / "src" / "demo_compiled" / "entry.py"
    generated.parent.mkdir(parents=True)
    generated.write_text("VALUE = 1\n", encoding="utf-8")
    resolved = {
        "schema_version": "aethergraph.resolved-system/v10",
        "semantic_event_protocol_version": "aethergraph.semantic-event/v2",
        "logical_output_requirements": ["origin"],
        "source_digest": "a" * 64,
        "catalog_digest": "b" * 64,
        "system_id": "demo",
        "entry_agent_ref": "agent.demo",
        "surface": {"graph_fn_name": "demo.graph", "ignored": True},
        "agents": [{"resource_ref": "agent.demo", "ignored": True}],
        "ignored": True,
    }
    resolved_path = root / "resolved-system.json"
    resolved_path.write_text(json.dumps(resolved), encoding="utf-8")
    indexed = []
    for path in sorted((generated, resolved_path)):
        content = path.read_bytes()
        indexed.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": len(content),
                "sha256": sha256(content).hexdigest(),
                "origin": "generated",
            }
        )
    manifest = {
        "schema_version": "aethergraph.compiled-system-manifest/v13",
        "build_id": build_id,
        "package_name": "demo_compiled",
        "entrypoint_module": "demo_compiled.entry",
        "entrypoint_symbol": "demo_entry",
        "source_digest": "a" * 64,
        "engine_version": "0.1.0a1",
        "compiler_version": "30",
        "semantic_event_protocol_version": "aethergraph.semantic-event/v2",
        "logical_output_requirements": ["origin"],
        "catalog_digest": "b" * 64,
        "resolved_definition_digest": sha256(
            json.dumps(
                resolved,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "resolved_definition_path": "resolved-system.json",
        "files": sorted(indexed, key=lambda item: item["path"]),
        "manifest_self_hash_excluded": True,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_inspect_compiled_build_without_engine_package(tmp_path) -> None:
    root = _write_build(tmp_path)

    result = inspect_compiled_build(root)

    assert result.manifest.build_id == root.name
    assert result.resolved_definition.entry_agent_ref == "agent.demo"


def test_inspect_compiled_build_rejects_tampered_file(tmp_path) -> None:
    root = _write_build(tmp_path)
    (root / "src" / "demo_compiled" / "entry.py").write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(CompiledBuildError, match="integrity"):
        inspect_compiled_build(root)


def test_inspect_compiled_build_rejects_resolved_definition_digest_mismatch(
    tmp_path,
) -> None:
    root = _write_build(tmp_path)
    resolved_path = root / "resolved-system.json"
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    resolved["ignored"] = "changed-with-a-reindexed-file"
    resolved_path.write_text(json.dumps(resolved), encoding="utf-8")
    content = resolved_path.read_bytes()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    indexed = next(item for item in manifest["files"] if item["path"] == "resolved-system.json")
    indexed["size"] = len(content)
    indexed["sha256"] = sha256(content).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CompiledBuildError, match="definition digest"):
        inspect_compiled_build(root)


def test_aethergraph_production_has_no_engine_imports() -> None:
    source_root = Path(__file__).parents[1] / "src" / "aethergraph"
    violations: list[str] = []
    for path in source_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            if any(
                name == "aethergraph_engine" or name.startswith("aethergraph_engine.")
                for name in modules
            ):
                violations.append(str(path.relative_to(source_root)))

    assert violations == []
