from __future__ import annotations

import pytest

from aethergraph.contracts.services.channel import Button, ChannelRoutingError, OutEvent
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


@pytest.mark.asyncio
async def test_publish_reports_structured_missing_adapter() -> None:
    bus = ChannelBus(adapters={"ui": _CapturingAdapter()})
    event = OutEvent(type="agent.message", channel="slack:team/T:chan/C", text="Done")

    with pytest.raises(ChannelRoutingError) as exc_info:
        await bus.publish(event)

    assert exc_info.value.code == "channel.adapter_not_found"
    assert exc_info.value.channel_key == "slack:team/T:chan/C"
    assert exc_info.value.known_prefixes == ("ui",)
