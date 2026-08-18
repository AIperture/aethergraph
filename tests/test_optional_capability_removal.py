from pathlib import Path
from types import SimpleNamespace

import pytest

from aethergraph.config.config import AppSettings
from aethergraph.core.runtime.node_context import NodeContext
from aethergraph.core.runtime.node_services import NodeServices
from aethergraph.core.runtime.runtime_services import (
    get_ext_context_service,
    register_context_service,
    use_services,
)
import aethergraph.runtime as runtime_api
from aethergraph.services.container.default_container import DefaultContainer

REMOVED_ACCESSORS = ("execute", "kb", "mcp", "web_search")
REMOVED_FIELDS = ("execution", "kb", "mcp", "web_search")
REMOVED_RUNTIME_HELPERS = (
    "get_mcp_service",
    "list_mcp_clients",
    "register_mcp_client",
    "set_mcp_service",
)


def test_optional_capability_implementation_paths_are_absent() -> None:
    package_root = Path(__file__).parents[1] / "src" / "aethergraph"

    for name in ("execution", "harness", "knowledge", "mcp", "websearch"):
        assert not any((package_root / "services" / name).rglob("*.py"))

    assert not any((package_root / "plugins" / "mcp").rglob("*.py"))
    assert not (package_root / "plugins" / "utils" / "data_io.py").exists()

    contracts = package_root / "contracts" / "services"
    for name in ("execution.py", "knowledge.py", "mcp.py"):
        assert not (contracts / name).exists()


def test_optional_capability_runtime_surfaces_are_absent() -> None:
    for name in REMOVED_ACCESSORS:
        assert not hasattr(NodeContext, name)

    for name in REMOVED_FIELDS:
        assert name not in NodeServices.__dataclass_fields__
        assert name not in DefaultContainer.__dataclass_fields__

    for name in REMOVED_RUNTIME_HELPERS:
        assert not hasattr(runtime_api, name)

    settings = AppSettings()
    assert not hasattr(settings, "knowledge")
    assert not hasattr(settings, "rag")


@pytest.mark.parametrize(
    "name",
    [
        "execute",
        "execution",
        "harness",
        "kb",
        "knowledge",
        "mcp",
        "planner",
        "planning",
        "skills",
        "web_search",
        "websearch",
    ],
)
def test_removed_capabilities_cannot_reappear_through_dynamic_services(name: str) -> None:
    services = SimpleNamespace(ext_services={name: object()})

    with use_services(services):
        with pytest.raises(KeyError, match="Removed first-class capability"):
            get_ext_context_service(name)
        with pytest.raises(ValueError, match="reserved for a removed"):
            register_context_service(name, object())  # type: ignore[arg-type]
