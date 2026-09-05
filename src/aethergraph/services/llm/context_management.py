"""Provider-neutral contracts for session-scoped model context management."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Literal

MODEL_CONTEXT_CHECKPOINT_VERSION = "aethergraph.model-context-checkpoint/v1"


@dataclass(frozen=True)
class ModelContextManagement:
    """Request provider-owned compaction for a long-running model context."""

    strategy: Literal["server_compaction"] = "server_compaction"
    trigger_tokens: int = 150_000
    instructions: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.trigger_tokens, bool) or not isinstance(self.trigger_tokens, int):
            raise TypeError("context-management trigger_tokens must be an integer")
        if self.trigger_tokens <= 0:
            raise ValueError("context-management trigger_tokens must be positive")
        if self.instructions is not None:
            value = str(self.instructions).strip()
            if not value:
                raise ValueError("context-management instructions must not be empty")
            object.__setattr__(self, "instructions", value)


@dataclass(frozen=True)
class ModelContextCheckpoint:
    """Carry opaque provider compaction state across semantic turns."""

    provider: str
    model: str
    protocol: str
    source_message_count: int
    source_message_digest: str
    payload: dict[str, Any] = field(default_factory=dict, repr=False)
    revision: int = 1
    contract_version: str = MODEL_CONTEXT_CHECKPOINT_VERSION

    def __post_init__(self) -> None:
        for name in ("provider", "model", "protocol"):
            value = str(getattr(self, name) or "").strip()
            if not value:
                raise ValueError(f"model context checkpoint {name} must be non-empty")
            object.__setattr__(self, name, value)
        if self.contract_version != MODEL_CONTEXT_CHECKPOINT_VERSION:
            raise ValueError("unsupported model context checkpoint contract")
        if self.source_message_count < 0:
            raise ValueError("source_message_count must be non-negative")
        if len(self.source_message_digest) != 64:
            raise ValueError("source_message_digest must be a SHA-256 digest")
        if self.revision < 1:
            raise ValueError("model context checkpoint revision must be positive")
        if not isinstance(self.payload, dict) or not self.payload:
            raise ValueError("model context checkpoint payload must be a non-empty object")
        object.__setattr__(self, "payload", copy.deepcopy(self.payload))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the checkpoint for Engine-private persistence."""

        return {
            "contract_version": self.contract_version,
            "provider": self.provider,
            "model": self.model,
            "protocol": self.protocol,
            "source_message_count": self.source_message_count,
            "source_message_digest": self.source_message_digest,
            "payload": copy.deepcopy(self.payload),
            "revision": self.revision,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ModelContextCheckpoint:
        """Restore a validated Engine-private checkpoint mapping."""

        return cls(**dict(value or {}))


def model_context_message_digest(messages: list[dict[str, Any]]) -> str:
    """Return a stable digest for a complete provider-projected message prefix."""

    encoded = json.dumps(
        messages,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_model_context_prefix(
    messages: list[dict[str, Any]],
    checkpoint: ModelContextCheckpoint,
) -> list[dict[str, Any]]:
    """Return messages appended after an unchanged compacted source prefix."""

    count = checkpoint.source_message_count
    if count > len(messages):
        raise ValueError("model context checkpoint source exceeds the current history")
    if model_context_message_digest(messages[:count]) != checkpoint.source_message_digest:
        raise ValueError("model context checkpoint source history diverged")
    return [copy.deepcopy(item) for item in messages[count:]]


__all__ = [
    "MODEL_CONTEXT_CHECKPOINT_VERSION",
    "ModelContextCheckpoint",
    "ModelContextManagement",
    "model_context_message_digest",
    "validate_model_context_prefix",
]
