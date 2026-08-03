"""Slack transport extraction for the canonical integration ingress boundary."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import hmac
import json
import time
from typing import Any

import aiohttp
from fastapi import HTTPException, Request

from aethergraph.api.v1.deps import RequestIdentity
from aethergraph.contracts.integration import (
    ExternalIdentity,
    IngressAttachment,
    IngressChoice,
    IngressEnvelope,
    IntegrationKind,
    OriginAddress,
)
from aethergraph.services.integration import (
    VerifiedAttachment,
    VerifiedIntegrationContext,
)


async def _download_slack_file(url: str, token: str) -> bytes:
    async with (
        aiohttp.ClientSession() as session,
        session.get(url, headers={"Authorization": f"Bearer {token}"}) as response,
    ):
        response.raise_for_status()
        return await response.read()


def _verify_sig(request: Request, body: bytes) -> None:
    signing_secret = (
        request.app.state.settings.slack.signing_secret.get_secret_value()
        if request.app.state.settings.slack.signing_secret
        else ""
    )
    if not signing_secret:
        raise HTTPException(401, "no slack signing secret configured")
    timestamp = request.headers.get("X-Slack-Request-Timestamp")
    signature = request.headers.get("X-Slack-Signature")
    if not timestamp or not signature or abs(time.time() - int(timestamp)) > 300:
        raise HTTPException(400, "stale or missing signature")
    signature_base = f"v0:{timestamp}:{body.decode()}"
    expected = (
        "v0="
        + hmac.new(
            signing_secret.encode(),
            signature_base.encode(),
            hashlib.sha256,
        ).hexdigest()
    )
    if not hmac.compare_digest(expected, signature):
        raise HTTPException(401, "bad signature")


def _channel_key(team_id: str, channel_id: str, thread_ts: str | None) -> str:
    key = f"slack:team/{team_id}:chan/{channel_id}"
    return f"{key}:thread/{thread_ts}" if thread_ts else key


def _required(value: Any, field: str) -> str:
    if value is None or value == "":
        raise ValueError(f"Malformed Slack ingress: missing {field}.")
    return str(value)


def _identity(
    *,
    team_id: str,
    channel_id: str,
    thread_ts: str | None,
    user_id: str,
) -> ExternalIdentity:
    return ExternalIdentity(
        tenant_id=team_id,
        conversation_id=f"team/{team_id}:chan/{channel_id}",
        thread_id=thread_ts,
        user_id=user_id,
    )


def _verified_context(
    *,
    integration_id: str,
    team_id: str,
    user_id: str,
    attachments: tuple[VerifiedAttachment, ...] = (),
) -> VerifiedIntegrationContext:
    return VerifiedIntegrationContext(
        integration_id=integration_id,
        integration_kind=IntegrationKind.SLACK,
        external_tenant_id=team_id,
        attachments=attachments,
        request_identity=RequestIdentity(user_id=user_id, org_id=team_id, mode="local"),
    )


async def _collect_files(
    files: list[dict[str, Any]],
    *,
    bot_token: str,
) -> tuple[tuple[IngressAttachment, ...], tuple[VerifiedAttachment, ...]]:
    declared: list[IngressAttachment] = []
    verified: list[VerifiedAttachment] = []
    for item in files:
        if item.get("mode") == "tombstone":
            continue
        file_id = _required(item.get("id"), "file.id")
        filename = str(item.get("name") or item.get("title") or "file")
        content_type = str(item.get("mimetype") or "application/octet-stream")
        url = _required(
            item.get("url_private_download") or item.get("url_private"),
            "file.url_private",
        )
        if not bot_token:
            raise ValueError("Slack file ingress requires an authenticated bot token.")
        attachment_id = f"slack-file-{file_id}"
        data = await _download_slack_file(url, bot_token)
        declared.append(
            IngressAttachment(
                attachment_id=attachment_id,
                source_kind="provider_file",
                source_id=file_id,
                filename=filename,
                content_type=content_type,
                size_bytes=len(data),
            )
        )
        verified.append(VerifiedAttachment(attachment_id=attachment_id, data=data))
    return tuple(declared), tuple(verified)


async def _accept_message(
    container,
    *,
    integration_id: str,
    bot_token: str,
    payload: dict[str, Any],
    event: dict[str, Any],
    files: list[dict[str, Any]],
) -> None:
    team_id = _required(payload.get("team_id"), "team_id")
    channel_id = _required(event.get("channel") or event.get("channel_id"), "event.channel")
    user_id = _required(event.get("user"), "event.user")
    event_id = _required(payload.get("event_id"), "event_id")
    thread_ts = str(event.get("thread_ts") or event.get("ts") or event.get("event_ts") or "")
    if not thread_ts:
        raise ValueError("Malformed Slack ingress: missing event timestamp.")
    attachments, verified_attachments = await _collect_files(files, bot_token=bot_token)
    text = str(event.get("text") or "")
    envelope = IngressEnvelope(
        integration_id=integration_id,
        external_identity=_identity(
            team_id=team_id,
            channel_id=channel_id,
            thread_ts=thread_ts,
            user_id=user_id,
        ),
        external_event_id=event_id,
        idempotency_key=event_id,
        received_at=datetime.now(UTC),
        text=text if text else None,
        attachments=attachments,
        transport_metadata={
            "provider": "slack",
            "event_type": str(event.get("type") or "message"),
            "event_ts": str(event.get("event_ts") or event.get("ts") or ""),
        },
        origin_address=OriginAddress(
            channel_key=_channel_key(team_id, channel_id, thread_ts),
            capability_profile_id="slack-v1",
        ),
    )
    await container.integration_ingress.accept(
        verified=_verified_context(
            integration_id=integration_id,
            team_id=team_id,
            user_id=user_id,
            attachments=verified_attachments,
        ),
        envelope=envelope,
    )


async def handle_slack_events_common(container, settings, payload: dict) -> dict:
    """Translate one authenticated Slack event into canonical ingress.

    The handler extracts Slack identity and retrieves protected file bytes, while
    route selection, session binding, artifact scope, and dispatch remain in AG.

    Examples:
        Accept a Socket Mode message:
        ```python
        await handle_slack_events_common(container, settings, payload)
        ```

        Accept a verified Events API file event:
        ```python
        result = await handle_slack_events_common(container, settings, file_payload)
        ```

    Args:
        container: Host container with an installed `integration_ingress` coordinator.
        settings: Host settings containing the authenticated Slack bot token.
        payload: Parsed Slack Events API payload.

    Returns:
        dict: Empty Slack acknowledgment payload.

    Notes:
        Unsupported event kinds are acknowledged without creating ingress. Malformed
        supported events fail explicitly and never invoke a legacy dispatch path.
    """
    event = payload.get("event") or {}
    event_type = event.get("type")
    integration_id = _required(settings.slack.integration_id, "settings.slack.integration_id")
    if event_type == "message" and not event.get("bot_id"):
        bot_token = settings.slack.bot_token.get_secret_value() if settings.slack.bot_token else ""
        await _accept_message(
            container,
            integration_id=integration_id,
            bot_token=bot_token,
            payload=payload,
            event=event,
            files=list(event.get("files") or []),
        )
        return {}

    if event_type == "file_shared":
        file_id = _required((event.get("file") or {}).get("id"), "event.file.id")
        info = await container.slack.client.files_info(file=file_id)
        file_record = info.get("file") or {}
        file_record["id"] = file_id
        synthetic = dict(event)
        synthetic["user"] = file_record.get("user")
        synthetic["channel_id"] = event.get("channel_id") or (event.get("channel") or {}).get("id")
        bot_token = settings.slack.bot_token.get_secret_value() if settings.slack.bot_token else ""
        await _accept_message(
            container,
            integration_id=integration_id,
            bot_token=bot_token,
            payload=payload,
            event=synthetic,
            files=[file_record],
        )
    return {}


async def handle_slack_interactive_common(
    container,
    payload: dict,
    *,
    integration_id: str,
) -> dict:
    """Translate one Slack button action into an exact canonical interaction.

    Only the public interaction identity issued in the button value is accepted;
    continuation tokens, run IDs, and fuzzy lookup are not transport inputs.

    Examples:
        Accept a Socket Mode action:
        ```python
        await handle_slack_interactive_common(
            container,
            payload,
            integration_id="slack-main",
        )
        ```

        Accept a verified HTTP action:
        ```python
        response = await handle_slack_interactive_common(
            container,
            payload,
            integration_id="slack-main",
        )
        ```

    Args:
        container: Host container with an installed `integration_ingress` coordinator.
        payload: Parsed Slack interactive-component payload.
        integration_id: Exact configured Slack connection identity.

    Returns:
        dict: Empty Slack acknowledgment payload.

    Notes:
        The action value must be valid JSON with `interaction_id` and `choice`.
        Malformed values reject instead of being reinterpreted as free text.
    """
    integration_id = _required(integration_id, "integration_id")
    actions = payload.get("actions") or []
    if len(actions) != 1:
        raise ValueError("Malformed Slack interaction: exactly one action is required.")
    action = actions[0]
    try:
        value = json.loads(_required(action.get("value"), "action.value"))
    except json.JSONDecodeError as exc:
        raise ValueError("Malformed Slack interaction: action value is not JSON.") from exc
    if not isinstance(value, dict):
        raise ValueError("Malformed Slack interaction: action value must be an object.")

    team_id = _required((payload.get("team") or {}).get("id"), "team.id")
    channel_id = _required(
        (payload.get("channel") or {}).get("id")
        or (payload.get("container") or {}).get("channel_id"),
        "channel.id",
    )
    user_id = _required((payload.get("user") or {}).get("id"), "user.id")
    thread_ts = _required(
        (payload.get("message") or {}).get("thread_ts")
        or (payload.get("message") or {}).get("ts")
        or (payload.get("container") or {}).get("thread_ts")
        or (payload.get("container") or {}).get("message_ts"),
        "message thread timestamp",
    )
    interaction_id = _required(value.get("interaction_id"), "action.value.interaction_id")
    choice = _required(value.get("choice"), "action.value.choice")
    action_id = _required(action.get("action_id"), "action.action_id")
    action_ts = _required(action.get("action_ts"), "action.action_ts")
    event_id = f"action-{action_ts}-{action_id}-{user_id}"
    envelope = IngressEnvelope(
        integration_id=integration_id,
        external_identity=_identity(
            team_id=team_id,
            channel_id=channel_id,
            thread_ts=thread_ts,
            user_id=user_id,
        ),
        external_event_id=event_id,
        idempotency_key=event_id,
        received_at=datetime.now(UTC),
        choice=IngressChoice(interaction_id=interaction_id, option_ids=(choice,)),
        transport_metadata={
            "provider": "slack",
            "action_id": action_id,
            "action_ts": action_ts,
        },
        origin_address=OriginAddress(
            channel_key=_channel_key(team_id, channel_id, thread_ts),
            capability_profile_id="slack-v1",
        ),
    )
    await container.integration_ingress.accept(
        verified=_verified_context(
            integration_id=integration_id,
            team_id=team_id,
            user_id=user_id,
        ),
        envelope=envelope,
    )
    return {}
