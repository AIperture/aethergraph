"""Project a prompt manifest into the per-cycle context snapshot DTO.

Sections are content-addressed fragments (`hash` = fragment id, `key` =
semantic kind), which is exactly what the UI's fragment-hash delta needs to diff
consecutive cycles. Body sections are populated only when the manifest captured
a provider request (manifest/full capture modes).
"""

from __future__ import annotations

from typing import Any

from .models import ContextBodySection, ContextSection, ContextSnapshot
from .reader import EngineEvent


def build_context_snapshot(
    events: list[EngineEvent],
    manifest_id: str,
    manifest: dict[str, Any],
) -> ContextSnapshot:
    """Build one context snapshot DTO for a prompt manifest.

    The decision event that produced the manifest supplies step/agent identity;
    the manifest supplies capture mode, section fragments, and any body.
    """
    decision = _decision_for_manifest(events, manifest_id)
    data = decision.data if decision else {}
    parts = list(manifest.get("parts") or [])
    provider_request = manifest.get("provider_request")
    body_present = isinstance(provider_request, dict)
    sections = [
        ContextSection(
            key=str(part.get("semantic_kind") or f"part:{index}"),
            value_type=str(part.get("content_kind") or "json"),
            char_count=_int(part.get("byte_count")),
            hash=str(part.get("fragment_id") or ""),
            omitted=not body_present,
        )
        for index, part in enumerate(parts)
    ]
    body_sections: list[ContextBodySection] = []
    if body_present:
        for index, message in enumerate(provider_request.get("messages") or []):
            role = message.get("role") if isinstance(message, dict) else "unknown"
            body_sections.append(
                ContextBodySection(key=f"message:{index}:{role or 'unknown'}", value=message)
            )
    return ContextSnapshot(
        snapshot_id=manifest_id,
        run_id=decision.run_id if decision else "",
        agent_instance_id=decision.agent_instance_id if decision else "",
        step_index=_int(data.get("step_index")),
        capture_mode=str(manifest.get("capture_mode") or "metadata"),
        created_at=decision.iso if decision else "",
        total_chars=_int(manifest.get("total_chars")),
        sections=sections,
        body_sections=body_sections,
    )


def _decision_for_manifest(events: list[EngineEvent], manifest_id: str) -> EngineEvent | None:
    return next(
        (
            event
            for event in events
            if event.kind == "agent_engine.decision"
            and str(event.data.get("prompt_manifest_id") or "") == manifest_id
        ),
        None,
    )


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


__all__ = ["build_context_snapshot"]
