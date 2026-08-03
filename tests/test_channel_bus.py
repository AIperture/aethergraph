from __future__ import annotations

import pytest

from aethergraph.contracts.services.channel import Button, OutEvent
from aethergraph.services.channel.channel_bus import ChannelBus


class _CapturingAdapter:
    capabilities = {"text"}

    def __init__(self) -> None:
        self.events: list[OutEvent] = []

    async def send(self, event: OutEvent) -> dict[str, str]:
        self.events.append(event)
        return {"delivery_id": "delivery-1"}


@pytest.mark.asyncio
async def test_publish_delivers_unsupported_event_unchanged() -> None:
    adapter = _CapturingAdapter()
    bus = ChannelBus(adapters={"test": adapter})
    event = OutEvent(
        type="session.need_approval",
        channel="test:conversation/one",
        text="Approve?",
        buttons=[Button(label="Approve", value="approve")],
    )

    result = await bus.publish(event)

    assert result == {"delivery_id": "delivery-1"}
    assert adapter.events == [event]
    assert adapter.events[0].type == "session.need_approval"
    assert adapter.events[0].buttons == [Button(label="Approve", value="approve")]
