from pathlib import Path

from aethergraph.core.runtime.node_context import NodeContext
from aethergraph.services.container.default_container import DefaultContainer
from aethergraph.storage.kv.inmem_kv import InMemoryKV


def test_dead_leaf_service_packages_are_absent() -> None:
    services_root = Path(__file__).parents[1] / "src" / "aethergraph" / "services"
    for name in ("eventbus", "features", "kv", "redactor"):
        assert not (services_root / name).exists()

    assert not (services_root / "__init__.pu").exists()
    assert not (services_root.parent / "contracts" / "services" / "eventbus.py").exists()


def test_kv_uses_the_canonical_storage_provider() -> None:
    assert hasattr(NodeContext, "kv")
    assert InMemoryKV.__module__ == "aethergraph.storage.kv.inmem_kv"


def test_dead_optional_container_fields_are_absent() -> None:
    assert "event_bus" not in DefaultContainer.__dataclass_fields__
    assert "redactor" not in DefaultContainer.__dataclass_fields__
