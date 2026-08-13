from __future__ import annotations

import logging
from typing import Any

import pytest

from aethergraph.services.llm.types import ImageGenerationUsage
from aethergraph.services.llm.usage_metering import _record_image_generation_metering
from aethergraph.services.metering.eventlog_metering import EventLogMeteringService


class _Store:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def append(self, event: dict[str, Any]) -> None:
        self.events.append(dict(event))

    async def query(self, *, kinds=None, **kwargs) -> list[dict[str, Any]]:
        del kwargs
        selected = set(kinds or ())
        return [event for event in self.events if not selected or event.get("kind") in selected]


@pytest.mark.asyncio
async def test_image_meter_event_and_stats_remain_operation_specific() -> None:
    store = _Store()
    service = EventLogMeteringService(store)  # type: ignore[arg-type]

    await _record_image_generation_metering(
        service,
        provider="openai",
        model="gpt-image-1",
        usage=ImageGenerationUsage.from_provider_usage({"input_tokens": 4, "output_tokens": 6}),
        image_count=2,
        size="1024x1024",
        quality="high",
        latency_ms=25,
        dimensions={"run_id": "run-1"},
        logger=logging.getLogger(__name__),
    )

    assert [event["kind"] for event in store.events] == ["meter.image_generation"]
    assert store.events[0]["tags"] == ["meter.image_generation"]
    assert store.events[0]["total_tokens"] == 10
    assert await service.get_image_generation_stats(run_ids={"run-1"}) == {
        "gpt-image-1": {
            "calls": 1,
            "images": 2,
            "input_tokens": 4,
            "output_tokens": 6,
            "total_tokens": 10,
        }
    }
    overview = await service.get_overview(run_ids={"run-1"})
    assert overview["llm_calls"] == 0
    assert overview["image_generation_calls"] == 1
    assert overview["images_generated"] == 2
