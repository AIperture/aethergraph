from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import uuid

from .models import CaptureMode, LLMObservationRecord
from .policy import ObservationPolicy
from .redaction import canonical_json, sanitize_content

_SEMANTIC_KINDS = {
    "frozen_header",
    "ledger_entry",
    "ledger_checkpoint",
    "volatile_frame",
    "repair_output",
    "repair_instruction",
    "direct_message",
}


def content_hash(*, content_kind: str, body: str) -> str:
    return sha256(f"{content_kind}\0v1\0{body}".encode()).hexdigest()


@dataclass(frozen=True)
class PreparedFragment:
    fragment_id: str
    content_kind: str
    canonical_hash: str
    byte_count: int
    body: str


@dataclass(frozen=True)
class PreparedManifestPart:
    ordinal: int
    semantic_kind: str
    role: str | None
    fragment_id: str
    source_event_id: str | None = None


@dataclass(frozen=True)
class PreparedPromptCapture:
    capture_mode: CaptureMode
    assembled_request_hash: str
    total_chars: int
    total_bytes: int
    roles: tuple[str, ...]
    manifest_id: str | None = None
    fragments: tuple[PreparedFragment, ...] = ()
    parts: tuple[PreparedManifestPart, ...] = ()
    response_fragment: PreparedFragment | None = None
    omission_reason: str | None = None


class PromptStore:
    """Prepare content-addressed LLM prompt captures for atomic persistence.

    Intro:
        Canonicalizes provider requests into off, metadata, manifest, or full capture.

    Examples:
        Prepare metadata-only capture:
        ```python
        capture = PromptStore(ObservationPolicy()).prepare(record)
        ```

        Prepare reconstructable manifest capture:
        ```python
        capture = PromptStore(
            ObservationPolicy(capture_mode="manifest")
        ).prepare(record)
        ```

    Args:
        policy: Capture mode and bounded payload limits.

    Returns:
        PromptStore: Deterministic prompt preparation service.

    Notes:
        Preparation performs no I/O; SQLite persistence stays atomic with the
        owning LLM observation.
    """

    def __init__(self, policy: ObservationPolicy) -> None:
        policy.validate()
        self.policy = policy

    def prepare(self, record: LLMObservationRecord) -> PreparedPromptCapture:
        request = {
            "messages": sanitize_content(record.messages),
            "provider_request_args": sanitize_content(record.provider_request_args),
        }
        request_body = canonical_json(request)
        request_bytes = request_body.encode("utf-8")
        roles = tuple(str(message.get("role") or "message") for message in record.messages)
        common = {
            "capture_mode": self.policy.capture_mode,
            "assembled_request_hash": sha256(request_bytes).hexdigest(),
            "total_chars": len(request_body),
            "total_bytes": len(request_bytes),
            "roles": roles,
        }
        if self.policy.capture_mode == "off":
            return PreparedPromptCapture(**common)

        manifest_id = str(uuid.uuid4())
        if self.policy.capture_mode == "metadata":
            return PreparedPromptCapture(manifest_id=manifest_id, **common)

        if len(request_bytes) > self.policy.max_fragment_bytes:
            return PreparedPromptCapture(
                manifest_id=manifest_id,
                omission_reason="request_exceeds_capture_limit",
                **common,
            )

        if self.policy.capture_mode == "full":
            fragment = self._fragment("provider_request", request_body)
            fragments = (fragment,)
            parts = (
                PreparedManifestPart(
                    ordinal=0,
                    semantic_kind="direct_message",
                    role=None,
                    fragment_id=fragment.fragment_id,
                ),
            )
        else:
            prepared_fragments: list[PreparedFragment] = []
            prepared_parts: list[PreparedManifestPart] = []
            for ordinal, message in enumerate(request["messages"]):
                body = canonical_json(message)
                fragment = self._fragment("prompt_message", body)
                semantic_kind = str(message.get("semantic_kind") or "direct_message")
                if semantic_kind not in _SEMANTIC_KINDS:
                    raise ValueError(f"Unsupported prompt semantic kind: {semantic_kind}")
                prepared_fragments.append(fragment)
                prepared_parts.append(
                    PreparedManifestPart(
                        ordinal=ordinal,
                        semantic_kind=semantic_kind,
                        role=str(message.get("role") or "message"),
                        fragment_id=fragment.fragment_id,
                        source_event_id=message.get("source_event_id"),
                    )
                )
            config_body = canonical_json(request["provider_request_args"])
            config_fragment = self._fragment("provider_request_config", config_body)
            prepared_fragments.append(config_fragment)
            prepared_parts.append(
                PreparedManifestPart(
                    ordinal=len(prepared_parts),
                    semantic_kind="direct_message",
                    role="provider_config",
                    fragment_id=config_fragment.fragment_id,
                )
            )
            fragments = tuple(prepared_fragments)
            parts = tuple(prepared_parts)

        response_fragment = None
        if record.raw_text is not None:
            response_body = canonical_json({"text": record.raw_text})
            if len(response_body.encode("utf-8")) <= self.policy.max_fragment_bytes:
                response_fragment = self._fragment("llm_response", response_body)
                fragments = (*fragments, response_fragment)

        return PreparedPromptCapture(
            manifest_id=manifest_id,
            fragments=fragments,
            parts=parts,
            response_fragment=response_fragment,
            **common,
        )

    @staticmethod
    def _fragment(content_kind: str, body: str) -> PreparedFragment:
        digest = content_hash(content_kind=content_kind, body=body)
        return PreparedFragment(
            fragment_id=digest,
            content_kind=content_kind,
            canonical_hash=digest,
            byte_count=len(body.encode("utf-8")),
            body=body,
        )
