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
    pending_result_ids: tuple[str, ...] = ()
    replay_results: tuple[dict[str, Any], ...] = field(default=(), repr=False)

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
        ids = tuple(self.pending_result_ids)
        if any(not isinstance(value, str) or not value.strip() for value in ids):
            raise ValueError("model context pending result ids must be non-empty strings")
        if len(ids) != len(set(ids)):
            raise ValueError("model context pending result ids must be unique")
        if not all(isinstance(value, dict) for value in self.replay_results):
            raise TypeError("model context replay results must be objects")
        object.__setattr__(self, "pending_result_ids", ids)
        object.__setattr__(self, "replay_results", tuple(copy.deepcopy(self.replay_results)))

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
            "pending_result_ids": list(self.pending_result_ids),
            "replay_results": copy.deepcopy(list(self.replay_results)),
        }

    def validated_replay_results(self, pending_ids: tuple[str, ...]) -> dict[str, dict[str, Any]]:
        """Validate durable results against the native calls retained in a checkpoint.

        Adapters provide the exact pending identities from their native payload.
        Engine supplies results restored from canonical execution Events.

        Examples:
            Validate a checkpoint without pending calls:
                ```python
                assert checkpoint.validated_replay_results(()) == {}
                ```
            Validate one completed call:
                ```python
                results = checkpoint.validated_replay_results(("call_1",))
                assert results["call_1"]["kind"] == "tool_output"
                ```

        Args:
            self: Provider-bound context checkpoint with restored result records.
            pending_ids: Native identities requiring results during explicit replay.

        Returns:
            dict[str, dict[str, Any]]: Exact, unique completed results by call identity.

        Notes:
            Missing or conflicting results fail before provider I/O. They never
            authorize re-execution of an already completed Tool.
        """
        from .tool_calling import LLMToolCallResponseError
        from .tool_discovery import ToolDiscoveryError, ToolDiscoveryResult

        results: dict[str, dict[str, Any]] = {}
        for value in self.replay_results:
            try:
                if value.get("kind") == "tool_output":
                    if not isinstance(value.get("call_id"), str) or not isinstance(
                        value.get("output"), str
                    ):
                        raise ValueError("Tool replay requires an exact serialized output")
                elif value.get("kind") == "discovery_result":
                    discovery = {key: item for key, item in value.items() if key != "kind"}
                    if discovery.get("error") is not None:
                        discovery["error"] = ToolDiscoveryError(**discovery["error"])
                    ToolDiscoveryResult(**discovery)
                else:
                    raise ValueError("Unknown replay result kind")
            except (TypeError, ValueError) as exc:
                raise LLMToolCallResponseError(
                    code="model_context_replay_result_invalid",
                    message="Provider context replay contains an invalid durable result.",
                ) from exc
            identity = str(value.get("call_id") or value.get("provider_reference_id") or "")
            if identity not in pending_ids or (identity in results and results[identity] != value):
                raise LLMToolCallResponseError(
                    code="model_context_replay_result_conflict",
                    message="Provider context replay has an unexpected or conflicting result.",
                )
            results[identity] = copy.deepcopy(value)
        missing = [identity for identity in pending_ids if identity not in results]
        if missing:
            raise LLMToolCallResponseError(
                code="model_context_replay_result_missing",
                message="Provider context replay is missing durable results for: "
                + ", ".join(missing),
            )
        return results

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
