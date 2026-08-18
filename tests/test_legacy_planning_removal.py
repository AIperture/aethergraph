from pathlib import Path

from aethergraph.core.runtime.node_context import NodeContext
from aethergraph.core.runtime.node_services import NodeServices
from aethergraph.services.container.default_container import DefaultContainer


def test_ag_planning_service_tree_and_contract_are_absent() -> None:
    package_root = Path(__file__).parents[1] / "src" / "aethergraph"
    assert not (package_root / "services" / "planning").exists()
    assert not (package_root / "contracts" / "services" / "planning.py").exists()


def test_legacy_planner_runtime_surface_is_absent() -> None:
    assert not hasattr(NodeContext, "planner")
    assert "_planner_facade" not in NodeContext.__dataclass_fields__
    assert "planner_service" not in NodeServices.__dataclass_fields__
    assert "planner_service" not in DefaultContainer.__dataclass_fields__
