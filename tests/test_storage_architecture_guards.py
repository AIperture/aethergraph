from __future__ import annotations

import ast
from dataclasses import fields, is_dataclass
import importlib
from pathlib import Path
from typing import get_type_hints

from aethergraph.storage.contracts import StorageOpenRequest

CONTRACT_ROOT = Path(__file__).parents[1] / "src" / "aethergraph" / "storage" / "contracts"
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


def test_workspace_root_is_the_only_canonical_physical_path_field() -> None:
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

    assert path_fields == [(StorageOpenRequest.__name__, "workspace_root")]


def test_provider_registry_contains_no_implicit_default_or_dynamic_discovery() -> None:
    path = CONTRACT_ROOT.parent / "provider_registry.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert calls.isdisjoint({"getattr", "hasattr", "__import__", "import_module"})
