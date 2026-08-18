"""Telegram transport extraction for the canonical integration ingress boundary."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import aiohttp

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

_aiohttp_session: aiohttp.ClientSession | None = None


def _http_session() -> aiohttp.ClientSession:
    global _aiohttp_session
    if _aiohttp_session is None or _aiohttp_session.closed:
        timeout = aiohttp.ClientTimeout(total=40, connect=5, sock_read=35)
        connector = aiohttp.TCPConnector(limit=50, ttl_dns_cache=300)
        _aiohttp_session = aiohttp.ClientSession(timeout=timeout, connector=connector)
    return _aiohttp_session


def _required(value: Any, field: str) -> str:
    if value is None or value == "":
        raise ValueError(f"Malformed Telegram ingress: missing {field}.")
    return str(value)


def _channel_key(chat_id: str, topic_id: str | None) -> str:
    base = f"tg:chat/{chat_id}"
    return f"{base}:topic/{topic_id}" if topic_id else base


def _external_identity(
    *,
    chat_id: str,
    topic_id: str | None,
    user_id: str,
) -> ExternalIdentity:
    return ExternalIdentity(
        tenant_id="telegram",
        conversation_id=f"chat/{chat_id}",
        thread_id=topic_id,
        user_id=user_id,
    )


def _verified_context(
    *,
    integration_id: str,
    user_id: str,
    attachments: tuple[VerifiedAttachment, ...] = (),
) -> VerifiedIntegrationContext:
    return VerifiedIntegrationContext(
        integration_id=integration_id,
        integration_kind=IntegrationKind.TELEGRAM,
        external_tenant_id="telegram",
        attachments=attachments,
        request_identity=RequestIdentity(user_id=user_id, org_id="telegram", mode="local"),
    )


async def _tg_get_file_path(file_id: str, token: str) -> str:
    if not token:
        raise ValueError("Telegram file ingress requires an authenticated bot token.")
    api = f"https://api.telegram.org/bot{token}/getFile"
    async with _http_session().post(api, json={"file_id": file_id}) as response:
        response.raise_for_status()
        data = await response.json()
    file_path = (data.get("result") or {}).get("file_path") if data.get("ok") else None
    return _required(file_path, "getFile.result.file_path")


async def _tg_download_file(file_path: str, token: str) -> bytes:
    url = f"https://api.telegram.org/file/bot{token}/{file_path}"
    async with _http_session().get(url) as response:
        response.raise_for_status()
        return await response.read()


def _normalize_mime_by_name(name: str | None, hint: str | None) -> str:
    extmap = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
        "tif": "image/tiff",
        "tiff": "image/tiff",
        "bmp": "image/bmp",
        "svg": "image/svg+xml",
        "pdf": "application/pdf",
        "csv": "text/csv",
        "json": "application/json",
        "yaml": "text/yaml",
        "yml": "text/yaml",
        "txt": "text/plain",
        "md": "text/markdown",
        "zip": "application/zip",
        "gz": "application/gzip",
        "tar": "application/x-tar",
        "7z": "application/x-7z-compressed",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    }
    if hint:
        return hint.lower()
    if name and "." in name:
        return extmap.get(name.lower().rsplit(".", 1)[-1], "application/octet-stream")
    return "application/octet-stream"


async def _download_message_files(
    message: dict[str, Any],
    *,
    token: str,
) -> tuple[tuple[IngressAttachment, ...], tuple[VerifiedAttachment, ...]]:
    candidates: list[tuple[str, str, str, int | None]] = []
    photos = message.get("photo") or []
    if photos:
        photo = photos[-1]
        file_id = _required(photo.get("file_id"), "photo.file_id")
        candidates.append((file_id, f"photo_{file_id}.jpg", "image/jpeg", photo.get("file_size")))
    document = message.get("document")
    if document:
        file_id = _required(document.get("file_id"), "document.file_id")
        filename = str(document.get("file_name") or f"document_{file_id}")
        candidates.append(
            (
                file_id,
                filename,
                _normalize_mime_by_name(filename, document.get("mime_type")),
                document.get("file_size"),
            )
        )

    declared: list[IngressAttachment] = []
    verified: list[VerifiedAttachment] = []
    for file_id, filename, content_type, _declared_size in candidates:
        file_path = await _tg_get_file_path(file_id, token)
        data = await _tg_download_file(file_path, token)
        attachment_id = f"telegram-file-{file_id}"
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


def _parse_callback(data: str) -> tuple[str, str]:
    parts = data.split("|")
    if len(parts) != 2 or not parts[0].startswith("i=") or not parts[1].startswith("o="):
        raise ValueError("Malformed Telegram interaction callback.")
    interaction_id = _required(parts[0][2:], "callback interaction_id")
    option_id = _required(parts[1][2:], "callback option_id")
    return interaction_id, option_id


async def _process_update(
    container,
    payload: dict,
    token: str,
    integration_id: str,
) -> None:
    """Translate one authenticated Telegram update into canonical ingress."""
    integration_id = _required(integration_id, "integration_id")
    update_id = _required(payload.get("update_id"), "update_id")
    callback = payload.get("callback_query")
    if callback:
        message = callback.get("message") or {}
        chat_id = _required((message.get("chat") or {}).get("id"), "message.chat.id")
        topic_id = (
            str(message["message_thread_id"])
            if message.get("message_thread_id") is not None
            else None
        )
        user_id = _required((callback.get("from") or {}).get("id"), "callback_query.from.id")
        interaction_id, option_id = _parse_callback(
            _required(callback.get("data"), "callback_query.data")
        )
        callback_id = _required(callback.get("id"), "callback_query.id")

        adapter = container.channels.adapters.get("tg")
        if adapter is not None:
            await adapter._api("answerCallbackQuery", callback_query_id=callback_id)

        envelope = IngressEnvelope(
            integration_id=integration_id,
            external_identity=_external_identity(
                chat_id=chat_id,
                topic_id=topic_id,
                user_id=user_id,
            ),
            external_event_id=f"update-{update_id}",
            idempotency_key=f"update-{update_id}",
            received_at=datetime.now(UTC),
            choice=IngressChoice(interaction_id=interaction_id, option_ids=(option_id,)),
            transport_metadata={
                "provider": "telegram",
                "update_id": update_id,
                "callback_id": callback_id,
                "message_id": str(message.get("message_id") or ""),
            },
            origin_address=OriginAddress(
                channel_key=_channel_key(chat_id, topic_id),
                capability_profile_id="telegram-v1",
            ),
        )
        await container.integration_ingress.accept(
            verified=_verified_context(
                integration_id=integration_id,
                user_id=user_id,
            ),
            envelope=envelope,
        )
        return

    message = payload.get("message")
    if not message or (message.get("from") or {}).get("is_bot"):
        return
    chat_id = _required((message.get("chat") or {}).get("id"), "message.chat.id")
    topic_id = (
        str(message["message_thread_id"]) if message.get("message_thread_id") is not None else None
    )
    user_id = _required((message.get("from") or {}).get("id"), "message.from.id")
    message_id = _required(message.get("message_id"), "message.message_id")
    attachments, verified_attachments = await _download_message_files(message, token=token)
    text = str(message.get("text") or message.get("caption") or "")
    envelope = IngressEnvelope(
        integration_id=integration_id,
        external_identity=_external_identity(
            chat_id=chat_id,
            topic_id=topic_id,
            user_id=user_id,
        ),
        external_event_id=f"update-{update_id}",
        idempotency_key=f"update-{update_id}",
        received_at=datetime.now(UTC),
        text=text if text else None,
        attachments=attachments,
        transport_metadata={
            "provider": "telegram",
            "update_id": update_id,
            "message_id": message_id,
        },
        origin_address=OriginAddress(
            channel_key=_channel_key(chat_id, topic_id),
            capability_profile_id="telegram-v1",
        ),
    )
    await container.integration_ingress.accept(
        verified=_verified_context(
            integration_id=integration_id,
            user_id=user_id,
            attachments=verified_attachments,
        ),
        envelope=envelope,
    )
