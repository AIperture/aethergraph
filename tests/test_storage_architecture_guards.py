from __future__ import annotations

import ast
from dataclasses import fields, is_dataclass
import importlib
from pathlib import Path
from typing import get_type_hints

from aethergraph.config.config import AppSettings
from aethergraph.storage.contracts import StorageOpenRequest, StorageStartupDiagnostic

CONTRACT_ROOT = Path(__file__).parents[1] / "src" / "aethergraph" / "storage" / "contracts"
STORAGE_ROOT = CONTRACT_ROOT.parent
REPOSITORY_ROOT = CONTRACT_ROOT.parents[3]
CANONICAL_FILES = (
    *sorted(CONTRACT_ROOT.glob("*.py")),
    CONTRACT_ROOT.parent / "composition.py",
    CONTRACT_ROOT.parent / "provider_registry.py",
)
FORBIDDEN_IMPORT_PREFIXES = (
    "aethergraph.api",
    "aethergraph.core",
    "aethergraph.observability",
    "aethergraph.services",
    "aethergraph_engine",
    "ag_studio",
)
DEPRECATED_IDENTITY_FIELDS = {"app_id", "application_id", "client_id"}


def _imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.lineno, node.module))
    return found


def test_canonical_storage_layer_does_not_import_runtime_or_legacy_implementations() -> None:
    violations = []
    for path in CANONICAL_FILES:
        for line, imported in _imports(path):
            if imported.startswith(FORBIDDEN_IMPORT_PREFIXES):
                violations.append(f"{path.name}:{line}: {imported}")

    assert violations == []


def test_canonical_records_exclude_deprecated_application_identity_fields() -> None:
    violations = []
    for path in sorted(CONTRACT_ROOT.glob("*.py")):
        module = importlib.import_module(f"aethergraph.storage.contracts.{path.stem}")
        for name, value in vars(module).items():
            if not isinstance(value, type) or value.__module__ != module.__name__:
                continue
            if is_dataclass(value):
                aliases = DEPRECATED_IDENTITY_FIELDS.intersection(
                    field.name for field in fields(value)
                )
                if aliases:
                    violations.append(f"{name}: {sorted(aliases)}")

    assert violations == []


def test_workspace_root_is_the_only_canonical_physical_path_field_name() -> None:
    path_fields = []
    for path in sorted(CONTRACT_ROOT.glob("*.py")):
        module = importlib.import_module(f"aethergraph.storage.contracts.{path.stem}")
        for name, value in vars(module).items():
            if not isinstance(value, type) or value.__module__ != module.__name__:
                continue
            if is_dataclass(value):
                hints = get_type_hints(value)
                path_fields.extend(
                    (name, field.name) for field in fields(value) if hints.get(field.name) is Path
                )

    assert path_fields == [
        (StorageStartupDiagnostic.__name__, "workspace_root"),
        (StorageOpenRequest.__name__, "workspace_root"),
    ]


def test_provider_registry_contains_no_implicit_default_or_dynamic_discovery() -> None:
    path = CONTRACT_ROOT.parent / "provider_registry.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert calls.isdisjoint({"getattr", "hasattr", "__import__", "import_module"})


def test_superseded_copied_sqlite_vector_implementations_are_absent() -> None:
    vector_root = STORAGE_ROOT / "vector_index"

    assert not (vector_root / "sqlite_index copy.py").exists()
    assert not (vector_root / "sqlite_index_vanila.py").exists()
    assert not any(vector_root.glob("*.py"))


def test_independent_rag_vector_configuration_and_factory_are_absent() -> None:
    vector_root = STORAGE_ROOT / "vector_index"
    factory = STORAGE_ROOT / "factory.py"

    assert "storage" not in AppSettings.model_fields
    assert "search" not in AppSettings.model_fields
    assert not factory.exists()
    assert not (vector_root / "chroma_index.py").exists()
    assert not (STORAGE_ROOT.parent / "config" / "search.py").exists()


def test_silent_null_search_backend_is_absent() -> None:
    search_root = STORAGE_ROOT / "search_backend"

    assert not (search_root / "null_backend.py").exists()
    assert not any(search_root.glob("*.py"))
    assert not (STORAGE_ROOT / "search_factory.py").exists()


def test_memory_contract_has_no_duplicate_vector_or_embedding_protocol() -> None:
    memory_contract = STORAGE_ROOT.parent / "contracts" / "services" / "memory.py"
    class_names = {
        node.name
        for node in ast.walk(ast.parse(memory_contract.read_text(encoding="utf-8")))
        if isinstance(node, ast.ClassDef)
    }

    assert "VectorIndex" not in class_names
    assert "EmbeddingsClient" not in class_names


def test_final_storage_tree_has_no_migration_or_legacy_runtime_scaffolding() -> None:
    retired_paths = (
        "docs/storage_provider_s9_retirement_manifest.json",
        "docs/storage_provider_migration_s0.md",
        "docs/observability_legacy_cleanup.md",
        "scripts/storage_provider_s0_baseline.py",
        "src/aethergraph/cli/commands/observability.py",
        "src/aethergraph/observability/legacy_cleanup.py",
        "src/aethergraph/services/state_stores/json_store.py",
        "tests/cli/test_observability_cli.py",
        "tests/test_graph_state_store.py",
        "tests/test_legacy_observability_cleanup.py",
        "tests/test_storage_s9_retirement_manifest.py",
    )

    assert [path for path in retired_paths if (REPOSITORY_ROOT / path).exists()] == []


def test_registration_cli_remains_api_only() -> None:
    source = (REPOSITORY_ROOT / "src/aethergraph/cli/commands/register.py").read_text(
        encoding="utf-8"
    )

    assert "FSDocStore" not in source
    assert "RegistrationManifestStore" not in source
    assert "_register_via_local" not in source
    assert 'choices=["auto", "api", "local"]' not in source
