import importlib.util
from pathlib import Path

from aethergraph.core.runtime.node_context import NodeContext
from aethergraph.services.container.default_container import DefaultContainer


def test_dead_leaf_service_packages_are_absent() -> None:
    services_root = Path(__file__).parents[1] / "src" / "aethergraph" / "services"
    for name in ("eventbus", "features", "kv", "redactor"):
        assert not (services_root / name).exists()

    assert not (services_root / "__init__.pu").exists()
    assert not (services_root.parent / "contracts" / "services" / "eventbus.py").exists()


def test_kv_uses_the_canonical_storage_provider() -> None:
    assert hasattr(NodeContext, "kv")
    try:
        spec = importlib.util.find_spec("aethergraph.storage.kv.inmem_kv")
    except ModuleNotFoundError:
        spec = None
    assert spec is None


def test_dead_optional_container_fields_are_absent() -> None:
    assert "event_bus" not in DefaultContainer.__dataclass_fields__
    assert "redactor" not in DefaultContainer.__dataclass_fields__
