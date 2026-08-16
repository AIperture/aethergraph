from __future__ import annotations

import ast
from collections import Counter
import json
from pathlib import Path

from aethergraph.config.config import AppSettings

_ROOT = Path(__file__).parents[1]
_SOURCE_ROOT = _ROOT / "src" / "aethergraph"
_MANIFEST_PATH = _ROOT / "docs" / "storage_provider_s9_retirement_manifest.json"


def _manifest() -> dict[str, object]:
    return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))


def _legacy_symbols(manifest: dict[str, object]) -> set[str]:
    return {str(item["symbol"]) for item in manifest["legacy_call_sites"]}


def _source_call_sites(symbols: set[str]) -> Counter[tuple[str, str]]:
    calls: Counter[tuple[str, str]] = Counter()
    for path in _SOURCE_ROOT.rglob("*.py"):
        relative = path.relative_to(_ROOT).as_posix()
        if relative.startswith("src/aethergraph/storage/"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            else:
                continue
            if name in symbols:
                calls[(relative, name)] += 1
    return calls


def test_s9_manifest_freezes_every_remaining_legacy_service_call_site() -> None:
    manifest = _manifest()
    expected = Counter(
        {
            (str(item["path"]), str(item["symbol"])): int(item["count"])
            for item in manifest["legacy_call_sites"]
        }
    )

    assert manifest["schema_version"] == 1
    assert manifest["status"] == "frozen_pre_s9"
    assert _source_call_sites(_legacy_symbols(manifest)) == expected


def test_s9_manifest_freezes_legacy_settings_factories_and_physical_paths() -> None:
    manifest = _manifest()

    assert set(manifest["legacy_app_settings_fields"]) == {"cont", "search", "storage"}
    assert set(manifest["legacy_app_settings_fields"]) <= set(AppSettings.model_fields)
    for relative in (
        *manifest["legacy_settings_modules"],
        *manifest["legacy_factory_modules"],
        *manifest["whole_module_retirements"],
    ):
        assert (_ROOT / str(relative)).is_file(), relative
    for item in manifest["legacy_physical_paths"]:
        source = (_ROOT / str(item["path"])).read_text(encoding="utf-8")
        assert str(item["literal"]) in source, item


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
