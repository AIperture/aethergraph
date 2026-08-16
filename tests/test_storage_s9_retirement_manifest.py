from __future__ import annotations

import ast
import json
from pathlib import Path

from aethergraph.config.config import AppSettings

_ROOT = Path(__file__).parents[1]
_SOURCE_ROOT = _ROOT / "src" / "aethergraph"
_MANIFEST_PATH = _ROOT / "docs" / "storage_provider_s9_retirement_manifest.json"


def _manifest() -> dict[str, object]:
    return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))


def _source_identifiers() -> set[str]:
    identifiers: set[str] = set()
    for path in _SOURCE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                identifiers.add(node.id)
            elif isinstance(node, ast.Attribute):
                identifiers.add(node.attr)
            elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                identifiers.add(node.name)
    return identifiers


def test_s9_manifest_records_and_removes_every_legacy_service_call_site() -> None:
    manifest = _manifest()
    identifiers = _source_identifiers()

    assert manifest["schema_version"] == 2
    assert manifest["status"] == "completed_s9_clean_cut"
    assert manifest["retired_call_sites"]
    for item in manifest["retired_call_sites"]:
        assert str(item["symbol"]) not in identifiers, item


def test_s9_manifest_proves_legacy_settings_factories_and_paths_are_absent() -> None:
    manifest = _manifest()

    assert set(manifest["legacy_app_settings_fields"]) == {"cont", "search", "storage"}
    assert set(manifest["legacy_app_settings_fields"]).isdisjoint(AppSettings.model_fields)
    for relative in (
        *manifest["legacy_settings_modules"],
        *manifest["legacy_factory_modules"],
        *manifest["whole_module_retirements"],
    ):
        assert not (_ROOT / str(relative)).exists(), relative
    for item in manifest["legacy_physical_paths"]:
        path = _ROOT / str(item["path"])
        if path.exists():
            assert str(item["literal"]) not in path.read_text(encoding="utf-8"), item
    for item in manifest["partial_symbol_retirements"]:
        path = _ROOT / str(item["path"])
        if path.exists():
            source = path.read_text(encoding="utf-8")
            for symbol in item["symbols"]:
                assert str(symbol) not in source, (path, symbol)


def test_s9_manifest_preserves_project_inputs_and_retires_only_runtime_history() -> None:
    manifest = _manifest()

    assert manifest["policy"] == "clean_cut_no_fallback_no_history_migration"
    assert set(manifest["preserved_project_inputs"]) == {
        "graph source",
        "agent source",
        "project-data",
        "provider configuration without secrets",
    }
    assert "legacy workspace databases" in manifest["retired_runtime_history"]
    assert "project-data" not in manifest["retired_runtime_history"]


def test_registration_cli_is_not_a_legacy_storage_owner() -> None:
    source = (_ROOT / "src/aethergraph/cli/commands/register.py").read_text(encoding="utf-8")

    assert "FSDocStore" not in source
    assert "RegistrationManifestStore" not in source
    assert "_register_via_local" not in source
    assert 'choices=["auto", "api", "local"]' not in source
