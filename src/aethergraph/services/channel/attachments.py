from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class InputAttachment:
    """
    Canonical graph-facing attachment.

    Current supported kind is "artifact". The shape is intentionally extensible.
    """

    kind: str
    source: str
    name: str | None = None
    mimetype: str | None = None
    size: int | None = None
    artifact_id: str | None = None
    uri: str | None = None
    url: str | None = None
    labels: dict[str, Any] | None = None
    meta: dict[str, Any] | None = None


def attachment_to_dict(a: InputAttachment) -> dict[str, Any]:
    return {
        "kind": a.kind,
        "source": a.source,
        "name": a.name,
        "mimetype": a.mimetype,
        "size": a.size,
        "artifact_id": a.artifact_id,
        "uri": a.uri,
        "url": a.url,
        "labels": dict(a.labels or {}),
        "meta": dict(a.meta or {}),
    }


def parse_input_attachments(raw: Any) -> list[InputAttachment]:
    """
    Parse and validate inbound attachment JSON payload.

    Supported:
      - kind == "artifact" with required artifact_id
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("attachments_json must be a JSON list")

    out: list[InputAttachment] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"attachments[{i}] must be an object")

        kind = str(item.get("kind") or "").strip()
        if not kind:
            raise ValueError(f"attachments[{i}].kind is required")
        if kind != "artifact":
            raise ValueError(f"attachments[{i}].kind '{kind}' is not supported")

        source = str(item.get("source") or "context_ref").strip() or "context_ref"
        artifact_id = item.get("artifact_id")
        if not artifact_id:
            raise ValueError(f"attachments[{i}].artifact_id is required for kind=artifact")

        labels = item.get("labels")
        meta = item.get("meta")
        out.append(
            InputAttachment(
                kind=kind,
                source=source,
                name=item.get("name"),
                mimetype=item.get("mimetype"),
                size=item.get("size"),
                artifact_id=str(artifact_id),
                uri=item.get("uri"),
                url=item.get("url"),
                labels=labels if isinstance(labels, dict) else {},
                meta=meta if isinstance(meta, dict) else {},
            )
        )
    return out
