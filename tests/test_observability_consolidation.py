from __future__ import annotations

import importlib.util
from pathlib import Path

from aethergraph.observability import (
    EventLogMeteringService,
    OperationObserver,
    StdLoggerService,
)
from aethergraph.services.container.default_container import DefaultContainer


def test_legacy_observation_service_paths_are_absent() -> None:
    package_root = Path(__file__).parents[1] / "src" / "aethergraph"

    for name in ("logger", "metering", "tracing"):
        assert not any((package_root / "services" / name).rglob("*.py"))
        assert importlib.util.find_spec(f"aethergraph.services.{name}") is None


def test_observation_implementations_have_one_public_owner() -> None:
    assert EventLogMeteringService.__module__ == "aethergraph.observability.metering"
    assert OperationObserver.__module__ == "aethergraph.observability.operations"
    assert StdLoggerService.__module__ == "aethergraph.observability.logger.std"
    assert "tracer" not in DefaultContainer.__dataclass_fields__
