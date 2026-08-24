from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from aethergraph.contracts.integration import IntegrationKind
from aethergraph.contracts.services.channel import Button, OutEvent
from aethergraph.plugins.channel.adapters.slack import SlackChannelAdapter
from aethergraph.plugins.channel.adapters.telegram import TelegramChannelAdapter
from aethergraph.plugins.channel.utils import slack_utils, telegram_utils
from aethergraph.services.integration import ResourceIngressPolicy


class _Coordinator:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.resource_ingress = SimpleNamespace(policy=ResourceIngressPolicy())
        self.receipt = SimpleNamespace(accepted=True)

    async def accept(self, *, verified, envelope):
        self.calls.append({"verified": verified, "envelope": envelope})
        return self.receipt


class _DeliveryAdapter:
    def __init__(self) -> None:
        self.events = []

    async def send(self, event) -> None:
        self.events.append(event)


class _Logger:
    def for_run(self):
        return self

    def debug(self, *args, **kwargs) -> None:
        pass


class _Container:
    def __init__(self) -> None:
        self.integration_ingress = _Coordinator()
        self.channels = SimpleNamespace(adapters={})
        self.logger = _Logger()


def _slack_settings():
    return SimpleNamespace(
        slack=SimpleNamespace(
            integration_id="slack-main",
            bot_token=None,
        )
    )


def _slack_message_payload(*, event_id: str = "Ev-1") -> dict[str, Any]:
    return {
        "event_id": event_id,
        "team_id": "T1",
        "event": {
            "type": "message",
            "channel": "C1",
            "user": "U1",
            "ts": "100.1",
            "text": "hello from Slack",
        },
    }


def _telegram_message_payload(*, update_id: int = 123) -> dict[str, Any]:
    return {
        "update_id": update_id,
        "message": {
            "message_id": 7,
            "date": 10,
            "from": {"id": 42, "is_bot": False},
            "chat": {"id": 99},
            "text": "hello from Telegram",
        },
    }


@pytest.mark.asyncio
async def test_slack_message_translates_to_canonical_ingress() -> None:
    container = _Container()

    await slack_utils.handle_slack_events_common(
        container,
        _slack_settings(),
        _slack_message_payload(),
    )

    call = container.integration_ingress.calls[0]
    envelope = call["envelope"]
    assert call["verified"].integration_kind is IntegrationKind.SLACK
    assert envelope.integration_id == "slack-main"
    assert envelope.external_identity.tenant_id == "T1"
    assert envelope.external_identity.conversation_id == "team/T1:chan/C1"
    assert envelope.external_identity.thread_id == "100.1"
    assert envelope.external_identity.user_id == "U1"
    assert envelope.text == "hello from Slack"
    assert envelope.origin_address.channel_key == "slack:team/T1:chan/C1:thread/100.1"


@pytest.mark.asyncio
async def test_telegram_message_translates_to_canonical_ingress() -> None:
    container = _Container()

    await telegram_utils._process_update(
        container,
        _telegram_message_payload(),
        token="",
        integration_id="telegram-main",
    )

    call = container.integration_ingress.calls[0]
    envelope = call["envelope"]
    assert call["verified"].integration_kind is IntegrationKind.TELEGRAM
    assert envelope.integration_id == "telegram-main"
    assert envelope.external_identity.tenant_id == "telegram"
    assert envelope.external_identity.conversation_id == "chat/99"
    assert envelope.external_identity.user_id == "42"
    assert envelope.text == "hello from Telegram"
    assert envelope.origin_address.channel_key == "tg:chat/99"


@pytest.mark.asyncio
async def test_provider_retries_emit_stable_idempotency_identities() -> None:
    slack_container = _Container()
    payload = _slack_message_payload(event_id="Ev-retry")
    await slack_utils.handle_slack_events_common(slack_container, _slack_settings(), payload)
    await slack_utils.handle_slack_events_common(slack_container, _slack_settings(), payload)

    telegram_container = _Container()
    update = _telegram_message_payload(update_id=777)
    await telegram_utils._process_update(
        telegram_container,
        update,
        token="",
        integration_id="telegram-main",
    )
    await telegram_utils._process_update(
        telegram_container,
        update,
        token="",
        integration_id="telegram-main",
    )

    slack_keys = [
        call["envelope"].idempotency_key for call in slack_container.integration_ingress.calls
    ]
    telegram_keys = [
        call["envelope"].idempotency_key for call in telegram_container.integration_ingress.calls
    ]
    assert slack_keys == ["Ev-retry", "Ev-retry"]
    assert telegram_keys == ["update-777", "update-777"]


@pytest.mark.asyncio
async def test_slack_callback_submits_exact_public_interaction() -> None:
    container = _Container()
    payload = {
        "team": {"id": "T1"},
        "user": {"id": "U1"},
        "channel": {"id": "C1"},
        "message": {"ts": "100.3"},
        "actions": [
            {
                "action_id": "ag_button_0",
                "action_ts": "100.4",
                "value": json.dumps(
                    {
                        "choice": "ship",
                        "choice_label": "Ship It",
                        "interaction_id": "interaction-public-1",
                    }
                ),
            }
        ],
    }

    await slack_utils.handle_slack_interactive_common(
        container,
        payload,
        integration_id="slack-main",
    )

    envelope = container.integration_ingress.calls[0]["envelope"]
    assert envelope.choice.interaction_id == "interaction-public-1"
    assert envelope.choice.option_ids == ("ship",)
    assert "token" not in json.dumps(envelope.model_dump(mode="json"))


@pytest.mark.asyncio
async def test_telegram_callback_submits_exact_public_interaction() -> None:
    acknowledgments: list[str] = []

    class _TelegramAdapter:
        async def _api(self, method: str, **kwargs) -> None:
            acknowledgments.append(method)

    container = _Container()
    container.channels.adapters["tg"] = _TelegramAdapter()
    payload = {
        "update_id": 125,
        "callback_query": {
            "id": "callback-1",
            "from": {"id": 42},
            "data": "i=interaction-public-2|o=2",
            "message": {"message_id": 9, "chat": {"id": 99}},
        },
    }

    await telegram_utils._process_update(
        container,
        payload,
        token="",
        integration_id="telegram-main",
    )

    envelope = container.integration_ingress.calls[0]["envelope"]
    assert acknowledgments == ["answerCallbackQuery"]
    assert envelope.choice.interaction_id == "interaction-public-2"
    assert envelope.choice.option_ids == ("2",)


@pytest.mark.asyncio
async def test_slack_file_bytes_are_verified_but_not_staged_by_edge(monkeypatch) -> None:
    async def _download(url: str, token: str, **kwargs) -> bytes:
        assert url == "https://files.slack.test/F1"
        assert token == "xoxb-test"
        assert kwargs["max_bytes"] == 25 * 1024 * 1024
        return b"contents"

    monkeypatch.setattr(slack_utils, "_download_slack_file", _download)
    container = _Container()
    settings = SimpleNamespace(
        slack=SimpleNamespace(
            integration_id="slack-main",
            bot_token=SimpleNamespace(get_secret_value=lambda: "xoxb-test"),
        )
    )
    payload = _slack_message_payload()
    payload["event"]["files"] = [
        {
            "id": "F1",
            "name": "brief.txt",
            "mimetype": "text/plain",
            "url_private": "https://files.slack.test/F1",
        }
    ]

    await slack_utils.handle_slack_events_common(container, settings, payload)

    call = container.integration_ingress.calls[0]
    assert call["envelope"].attachments[0].source_id == "F1"
    assert call["envelope"].attachments[0].size_bytes == 8
    assert call["verified"].attachments[0].data == b"contents"


@pytest.mark.asyncio
async def test_telegram_file_bytes_are_verified_but_not_staged_by_edge(monkeypatch) -> None:
    async def _path(file_id: str, token: str) -> str:
        assert file_id == "TG1"
        assert token == "telegram-token"
        return "documents/TG1.txt"

    async def _download(file_path: str, token: str, **kwargs) -> bytes:
        assert file_path == "documents/TG1.txt"
        assert token == "telegram-token"
        assert kwargs["max_bytes"] == 25 * 1024 * 1024
        return b"telegram contents"

    monkeypatch.setattr(telegram_utils, "_tg_get_file_path", _path)
    monkeypatch.setattr(telegram_utils, "_tg_download_file", _download)
    container = _Container()
    payload = _telegram_message_payload()
    payload["message"].pop("text")
    payload["message"]["caption"] = "please read"
    payload["message"]["document"] = {
        "file_id": "TG1",
        "file_name": "brief.txt",
        "mime_type": "text/plain",
        "file_size": 17,
    }

    await telegram_utils._process_update(
        container,
        payload,
        token="telegram-token",
        integration_id="telegram-main",
    )

    call = container.integration_ingress.calls[0]
    assert call["envelope"].attachments[0].source_id == "TG1"
    assert call["envelope"].attachments[0].content_type == "text/plain"
    assert call["verified"].attachments[0].data == b"telegram contents"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "prefix"),
    (("slack", "slack"), ("telegram", "tg")),
)
async def test_provider_rejections_are_reported_to_the_origin(provider: str, prefix: str) -> None:
    container = _Container()
    delivery = _DeliveryAdapter()
    container.channels.adapters[prefix] = delivery
    container.integration_ingress.receipt = SimpleNamespace(
        accepted=False,
        rejection_code="integration.attachment_type_rejected",
        rejection_message="That attachment type is not enabled.",
    )

    if provider == "slack":
        await slack_utils.handle_slack_events_common(
            container,
            _slack_settings(),
            _slack_message_payload(),
        )
    else:
        await telegram_utils._process_update(
            container,
            _telegram_message_payload(),
            token="",
            integration_id="telegram-main",
        )

    assert len(delivery.events) == 1
    assert "That attachment type is not enabled." in delivery.events[0].text
    assert "integration.attachment_type_rejected" in delivery.events[0].text


@pytest.mark.asyncio
async def test_slack_oversized_file_is_rejected_before_download(monkeypatch) -> None:
    async def _unexpected_download(*args, **kwargs) -> bytes:
        raise AssertionError("oversized provider files must not be downloaded")

    monkeypatch.setattr(slack_utils, "_download_slack_file", _unexpected_download)
    container = _Container()
    delivery = _DeliveryAdapter()
    container.channels.adapters["slack"] = delivery
    settings = SimpleNamespace(
        slack=SimpleNamespace(
            integration_id="slack-main",
            bot_token=SimpleNamespace(get_secret_value=lambda: "xoxb-test"),
        )
    )
    payload = _slack_message_payload()
    payload["event"]["files"] = [
        {
            "id": "F-large",
            "name": "large.bin",
            "mimetype": "application/octet-stream",
            "size": 25 * 1024 * 1024 + 1,
            "url_private": "https://files.slack.test/F-large",
        }
    ]

    await slack_utils.handle_slack_events_common(container, settings, payload)

    assert container.integration_ingress.calls == []
    assert "integration.attachment_too_large" in delivery.events[0].text


@pytest.mark.asyncio
async def test_slack_file_shared_uses_the_registered_delivery_client(monkeypatch) -> None:
    async def _download(*args, **kwargs) -> bytes:
        return b"shared contents"

    class _SlackClient:
        async def files_info(self, *, file: str):
            assert file == "F-shared"
            return {
                "file": {
                    "id": file,
                    "user": "U1",
                    "name": "shared.txt",
                    "mimetype": "text/plain",
                    "size": 15,
                    "url_private": "https://files.slack.test/F-shared",
                }
            }

    monkeypatch.setattr(slack_utils, "_download_slack_file", _download)
    container = _Container()
    delivery = _DeliveryAdapter()
    delivery.client = _SlackClient()
    container.channels.adapters["slack"] = SimpleNamespace(downstream=delivery)
    settings = SimpleNamespace(
        slack=SimpleNamespace(
            integration_id="slack-main",
            bot_token=SimpleNamespace(get_secret_value=lambda: "xoxb-test"),
        )
    )
    payload = {
        "event_id": "Ev-shared",
        "team_id": "T1",
        "event": {
            "type": "file_shared",
            "event_ts": "100.5",
            "channel_id": "C1",
            "file": {"id": "F-shared"},
        },
    }

    await slack_utils.handle_slack_events_common(container, settings, payload)

    assert container.integration_ingress.calls[0]["envelope"].attachments[0].source_id == (
        "F-shared"
    )


@pytest.mark.asyncio
async def test_malformed_provider_callbacks_reject_without_guessing() -> None:
    container = _Container()
    slack_payload = {
        "team": {"id": "T1"},
        "user": {"id": "U1"},
        "channel": {"id": "C1"},
        "message": {"ts": "100.3"},
        "actions": [{"action_id": "a", "action_ts": "1", "value": "approve"}],
    }
    telegram_payload = {
        "update_id": 1,
        "callback_query": {
            "id": "callback-1",
            "from": {"id": 42},
            "data": "i=1|k=legacy-prefix",
            "message": {"message_id": 9, "chat": {"id": 99}},
        },
    }

    with pytest.raises(ValueError, match="not JSON"):
        await slack_utils.handle_slack_interactive_common(
            container,
            slack_payload,
            integration_id="slack-main",
        )
    with pytest.raises(ValueError, match="Malformed Telegram interaction"):
        await telegram_utils._process_update(
            container,
            telegram_payload,
            token="",
            integration_id="telegram-main",
        )
    assert container.integration_ingress.calls == []


@pytest.mark.asyncio
async def test_slack_buttons_carry_choice_and_public_interaction_id() -> None:
    posted: list[dict[str, Any]] = []

    class _SlackClient:
        async def chat_postMessage(self, **kwargs):
            posted.append(kwargs)
            return {"ts": "100.5"}

    adapter = object.__new__(SlackChannelAdapter)
    adapter.client = _SlackClient()
    adapter._first_ts_by_chan = {}
    await adapter.send(
        OutEvent(
            type="link.buttons",
            channel="slack:team/T1:chan/C1:thread/100.1",
            text="Choose",
            buttons=[Button(label="Ship It", value="ship")],
            meta={"interaction_id": "interaction-public-1"},
        )
    )

    value = json.loads(posted[0]["blocks"][1]["elements"][0]["value"])
    assert value == {
        "choice": "ship",
        "choice_label": "Ship It",
        "interaction_id": "interaction-public-1",
    }


@pytest.mark.asyncio
async def test_telegram_buttons_keep_compact_public_interaction_id() -> None:
    sent: list[dict[str, Any]] = []

    async def _api(method: str, **kwargs):
        sent.append({"method": method, **kwargs})
        return {"result": {"message_id": 10}}

    adapter = object.__new__(TelegramChannelAdapter)
    adapter._msg_id_cache = {}
    adapter._api = _api
    await adapter.send(
        OutEvent(
            type="link.buttons",
            channel="tg:chat/99",
            text="Choose",
            buttons=[
                Button(label="One", value="one"),
                Button(label="Two", value="two"),
                Button(label="Three", value="three"),
                Button(label="Four", value="four"),
            ],
            meta={"interaction_id": "interaction-1234567890abcdef"},
        )
    )

    keyboard = sent[0]["reply_markup"]["inline_keyboard"]
    callback_data = [row[0]["callback_data"] for row in keyboard]
    assert callback_data == [
        "i=interaction-1234567890abcdef|o=1",
        "i=interaction-1234567890abcdef|o=2",
        "i=interaction-1234567890abcdef|o=3",
        "i=interaction-1234567890abcdef|o=4",
    ]
    assert all(len(value.encode("utf-8")) <= 64 for value in callback_data)
