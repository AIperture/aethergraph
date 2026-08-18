from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

from aethergraph.observability import (
    CanonicalMeteringService,
    LoggingConfig,
    OperationObserver,
    StdLoggerService,
)
from aethergraph.observability.logging import ObservationLogHandler
from aethergraph.services.container.default_container import DefaultContainer


class _ObservationSink:
    async def append_observation(self, _observation) -> None:
        return None


def test_legacy_observation_service_paths_are_absent() -> None:
    package_root = Path(__file__).parents[1] / "src" / "aethergraph"

    for name in ("logger", "metering", "tracing"):
        assert not any((package_root / "services" / name).rglob("*.py"))
        assert importlib.util.find_spec(f"aethergraph.services.{name}") is None


def test_observation_implementations_have_one_public_owner() -> None:
    assert CanonicalMeteringService.__module__ == "aethergraph.observability.metering"
    assert OperationObserver.__module__ == "aethergraph.observability.operations"
    assert StdLoggerService.__module__ == "aethergraph.observability.logger.std"
    assert "tracer" not in DefaultContainer.__dataclass_fields__


def test_logger_close_detaches_only_its_owned_observation_handler() -> None:
    root_name = "aethergraph.test.storage-cutover.logger-lifecycle"
    cfg = LoggingConfig(root_ns=root_name, console_level="CRITICAL")
    first = StdLoggerService.build(cfg, observation_store=_ObservationSink())
    first_observation = next(
        handler for handler in first.base().handlers if isinstance(handler, ObservationLogHandler)
    )

    replacement = StdLoggerService.build(cfg, observation_store=_ObservationSink())
    replacement_handlers = tuple(replacement.base().handlers)
    replacement_observation = next(
        handler for handler in replacement_handlers if isinstance(handler, ObservationLogHandler)
    )

    assert first_observation not in replacement.base().handlers
    first.close()
    assert replacement_observation in replacement.base().handlers
    assert tuple(replacement.base().handlers) == replacement_handlers

    replacement.close()
    assert replacement.base().handlers == []
    logging.Logger.manager.loggerDict.pop(root_name, None)
