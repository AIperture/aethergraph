"""Strict immutable contracts for the packaged model capability catalog."""

from __future__ import annotations

from datetime import date
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

from ..registry import ModelOperation

CatalogEvidenceStatus = Literal["verified", "conservative", "unknown"]
CatalogCapabilityState = Literal["supported", "unsupported", "unknown"]
CatalogCapability = Literal[
    "chat_tools",
    "native_tool_search",
    "structured_output",
    "prompt_cache",
    "embeddings",
    "image_generation",
]


class CatalogContract(BaseModel):
    """Base class for closed immutable model catalog records."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class CatalogNativeToolSearchMode(CatalogContract):
    """Evidence-backed native Tool-search mode for one model binding."""

    mode: Literal["native_hosted", "native_client"]
    replay_requirement: Literal["none", "previous_response", "full_history"]
    result_limit_behavior: Literal["request_bound", "provider_fixed", "post_validated"]
    max_results: int = Field(ge=1, le=50)
    protocol_version: str = Field(min_length=1, max_length=256)
    selection_owner: Literal["provider", "application"]
    tool_representation: Literal["full_definitions", "search_schema_manifest"]
    inventory_timing: Literal["request", "search", "preloaded"]
    path_transport: Literal["native_group", "metadata", "manifest", "none"]


class CatalogChatToolCapabilities(CatalogContract):
    """Evidence-backed Chat tool-loop capabilities for one model binding.

    Intro:
        Tool calling, returning Tool results, and multiple calls in one model
        turn are independent model facts. Keeping them together in one domain
        prevents endpoint adapter support from manufacturing model support.

    Examples:
        Declare a complete Tool loop:
            ```python
            facts = CatalogChatToolCapabilities(
                native_tool_calling="supported",
                tool_result_continuation="supported",
                parallel_tool_calls="supported",
            )
            ```

        Preserve an unverified parallel-call fact:
            ```python
            facts = CatalogChatToolCapabilities(
                native_tool_calling="supported",
                tool_result_continuation="supported",
                parallel_tool_calls="unknown",
            )
            ```

    Args:
        native_tool_calling: Whether the model can emit structured Tool calls.
        tool_result_continuation: Whether a Tool result can be returned for the
            model to continue the same logical turn.
        parallel_tool_calls: Whether one model turn may emit multiple Tool calls.

    Returns:
        CatalogChatToolCapabilities: Immutable validated Chat Tool facts.

    Notes:
        `unknown` is intentional and does not satisfy fail-closed requirements.
        Engine-projected Tool discovery is outside this model capability domain.
    """

    native_tool_calling: CatalogCapabilityState
    tool_result_continuation: CatalogCapabilityState
    parallel_tool_calls: CatalogCapabilityState


class CatalogStructuredOutput(CatalogContract):
    """Evidence-backed structured-output modes for one model binding."""

    native_strict_schema: bool
    native_schema: bool
    json_object: bool
    prompt_json: bool = True
    capability_source: str = Field(min_length=1, max_length=256)


class CatalogPromptCache(CatalogContract):
    """Evidence-backed prompt-cache semantics for one model binding."""

    mode: Literal["explicit", "implicit", "unavailable"]
    capability_source: str = Field(min_length=1, max_length=256)
    max_total_boundaries: int | None = Field(default=None, ge=1, le=64)
    max_new_writes_per_request: int | None = Field(default=None, ge=1, le=64)


class CatalogEmbeddingCapabilities(CatalogContract):
    """Evidence-backed embedding capabilities for one model binding."""

    text_embeddings: CatalogCapabilityState
    dimensions: CatalogCapabilityState


class CatalogImageGenerationCapabilities(CatalogContract):
    """Evidence-backed image-generation capabilities for one model binding."""

    text_to_image: CatalogCapabilityState
    image_editing: CatalogCapabilityState
    multiple_outputs: CatalogCapabilityState


class ModelCatalogEntry(CatalogContract):
    """One exact or narrowly matched model-operation capability record."""

    catalog_key: str = Field(pattern=r"^[a-z0-9][a-z0-9._/-]+$")
    provider_id: str = Field(min_length=1, max_length=128)
    operation: ModelOperation
    endpoint_ids: tuple[str, ...]
    model_id: str | None = Field(default=None, min_length=1, max_length=512)
    model_pattern: str | None = Field(default=None, min_length=1, max_length=1024)
    chat_tools: CatalogChatToolCapabilities | None = None
    native_tool_search: tuple[CatalogNativeToolSearchMode, ...] = ()
    structured_output: CatalogStructuredOutput | None = None
    prompt_cache: CatalogPromptCache | None = None
    embeddings: CatalogEmbeddingCapabilities | None = None
    image_generation: CatalogImageGenerationCapabilities | None = None
    sources: tuple[HttpUrl, ...]
    verified_at: date
    catalog_revision: int = Field(ge=1)
    evidence_status: CatalogEvidenceStatus
    priority: int = Field(default=0, ge=0, le=10_000)
    stale_after: date | None = None

    @field_validator("endpoint_ids")
    @classmethod
    def _validate_endpoint_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Require a non-empty unique ordered endpoint selection.

        Intro:
            Every catalog entry binds facts to explicit endpoint adapters and
            cannot float across provider protocols.

        Examples:
            Accept one endpoint:
                ```python
                validated = ModelCatalogEntry.model_validate(payload)
                ```

            Reject duplicates:
                ```python
                try:
                    ModelCatalogEntry.model_validate(duplicate_payload)
                except ValueError:
                    pass
                ```

        Args:
            value: Parsed endpoint-adapter identities.

        Returns:
            tuple[str, ...]: Unchanged valid endpoint identities.

        Notes:
            Endpoint existence is validated by the catalog loader against the
            production provider registry.
        """

        if not value or len(value) != len(set(value)):
            raise ValueError("catalog endpoint_ids must be non-empty and unique")
        return value

    @model_validator(mode="after")
    def _validate_match_and_evidence(self) -> ModelCatalogEntry:
        """Require one model selector and evidence for positive capabilities.

        Intro:
            Entries use either an exact model ID or a full-match regular
            expression. Native Tool-search support requires verified URL
            evidence and unique native modes.

        Examples:
            Validate an exact entry:
                ```python
                entry = ModelCatalogEntry.model_validate(exact_payload)
                ```

            Reject an unverified positive claim:
                ```python
                try:
                    ModelCatalogEntry.model_validate(unverified_payload)
                except ValueError:
                    pass
                ```

        Args:
            self: Fully parsed catalog entry.

        Returns:
            ModelCatalogEntry: Unchanged evidence-consistent entry.

        Notes:
            Regular expressions are compiled during validation and are always
            applied with `fullmatch` by the loader.
        """

        if (self.model_id is None) == (self.model_pattern is None):
            raise ValueError("catalog entry requires exactly one model_id or model_pattern")
        if self.model_pattern is not None:
            try:
                re.compile(self.model_pattern)
            except re.error as exc:
                raise ValueError("catalog model_pattern is invalid") from exc
        modes = tuple(item.mode for item in self.native_tool_search)
        if len(modes) != len(set(modes)):
            raise ValueError("catalog native Tool-search modes must be unique")
        capability_count = sum(
            (
                bool(self.native_tool_search),
                self.chat_tools is not None,
                self.structured_output is not None,
                self.prompt_cache is not None,
                self.embeddings is not None,
                self.image_generation is not None,
            )
        )
        if capability_count != 1:
            raise ValueError("catalog entry must declare exactly one capability domain")
        declared_operation = (
            "embeddings"
            if self.embeddings is not None
            else "image_generation"
            if self.image_generation is not None
            else "chat"
        )
        if self.operation != declared_operation:
            raise ValueError("catalog capability domain does not match operation")
        positive_capability = bool(self.native_tool_search)
        if self.chat_tools is not None:
            positive_capability = positive_capability or "supported" in {
                self.chat_tools.native_tool_calling,
                self.chat_tools.tool_result_continuation,
                self.chat_tools.parallel_tool_calls,
            }
        if self.structured_output is not None:
            positive_capability = positive_capability or any(
                (
                    self.structured_output.native_strict_schema,
                    self.structured_output.native_schema,
                    self.structured_output.json_object,
                )
            )
        if self.prompt_cache is not None:
            positive_capability = positive_capability or self.prompt_cache.mode != "unavailable"
        if self.embeddings is not None:
            positive_capability = positive_capability or "supported" in {
                self.embeddings.text_embeddings,
                self.embeddings.dimensions,
            }
        if self.image_generation is not None:
            positive_capability = positive_capability or "supported" in {
                self.image_generation.text_to_image,
                self.image_generation.image_editing,
                self.image_generation.multiple_outputs,
            }
        if positive_capability and (self.evidence_status != "verified" or not self.sources):
            raise ValueError("positive capability facts require verified URL evidence")
        if self.stale_after is not None and self.stale_after < self.verified_at:
            raise ValueError("catalog stale_after must not precede verified_at")
        return self

    def matches(self, model_id: str) -> bool:
        """Return whether this entry selects one exact provider model ID.

        Intro:
            Exact entries compare directly. Pattern entries use full regular
            expression matching so partial names cannot inherit capability.

        Examples:
            Match an exact model:
                ```python
                assert exact_entry.matches(exact_entry.model_id)
                ```

            Reject an unrelated model:
                ```python
                assert not exact_entry.matches("unrelated")
                ```

        Args:
            model_id: Exact configured provider model identity.

        Returns:
            bool: True only when the complete model identity matches.

        Notes:
            Provider and endpoint matching are performed separately by the
            catalog resolver.
        """

        candidate = str(model_id or "").strip()
        if self.model_id is not None:
            return candidate == self.model_id
        return re.fullmatch(self.model_pattern or r"(?!)", candidate) is not None

    def declares(self, capability: CatalogCapability) -> bool:
        """Return whether this entry owns the requested capability domain.

        Intro:
            Catalog entries deliberately contain one capability domain so
            independently revised facts may overlap on the same model safely.

        Examples:
            Detect a structured-output entry:
                ```python
                assert structured_entry.declares("structured_output")
                ```

            Reject an unrelated domain:
                ```python
                assert not structured_entry.declares("prompt_cache")
                ```

        Args:
            capability: Catalog capability domain to inspect.

        Returns:
            bool: True only when this entry declares that domain.

        Notes:
            Validation guarantees exactly one domain per entry.
        """

        if capability == "native_tool_search":
            return bool(self.native_tool_search)
        return getattr(self, capability) is not None


class ModelCatalog(CatalogContract):
    """Versioned production collection of model capability records."""

    schema_version: Literal["aethergraph.model-catalog/v1"]
    catalog_revision: int = Field(ge=1)
    entries: tuple[ModelCatalogEntry, ...]

    @model_validator(mode="after")
    def _validate_unique_keys(self) -> ModelCatalog:
        """Require unique keys and monotonically bounded entry revisions.

        Intro:
            Catalog keys are stable provenance identities. Entry revisions
            cannot be newer than the containing catalog revision.

        Examples:
            Validate a production catalog:
                ```python
                catalog = ModelCatalog.model_validate(payload)
                ```

            Reject duplicate keys:
                ```python
                try:
                    ModelCatalog.model_validate(duplicate_payload)
                except ValueError:
                    pass
                ```

        Args:
            self: Fully parsed model catalog.

        Returns:
            ModelCatalog: Unchanged catalog with unique provenance keys.

        Notes:
            Ambiguous model match precedence is checked by resolution tests.
        """

        keys = tuple(entry.catalog_key for entry in self.entries)
        if len(keys) != len(set(keys)):
            raise ValueError("model catalog keys must be unique")
        if any(entry.catalog_revision > self.catalog_revision for entry in self.entries):
            raise ValueError("model catalog entry revision exceeds catalog revision")
        return self


__all__ = [
    "CatalogContract",
    "CatalogCapability",
    "CatalogCapabilityState",
    "CatalogChatToolCapabilities",
    "CatalogEmbeddingCapabilities",
    "CatalogEvidenceStatus",
    "CatalogImageGenerationCapabilities",
    "CatalogNativeToolSearchMode",
    "CatalogPromptCache",
    "CatalogStructuredOutput",
    "ModelCatalog",
    "ModelCatalogEntry",
]
