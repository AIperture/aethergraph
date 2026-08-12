"""Fail-closed model and adapter capability resolution with provenance."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from .catalog import catalog_digest, resolve_model_catalog_entry
from .profiles import CapabilityState, ChatProfile
from .registry import get_endpoint_adapter, get_provider_descriptor

ChatCapabilityName = Literal[
    "image_input",
    "streaming",
    "native_tool_calling",
    "tool_result_continuation",
    "parallel_tool_calls",
    "structured_output",
    "prompt_cache",
    "native_tool_search_hosted",
    "native_tool_search_client",
]


class CapabilityContract(BaseModel):
    """Base class for closed immutable resolved-capability records."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class CapabilityEvidence(CapabilityContract):
    """One ordered reason contributing to an effective capability state."""

    source: Literal["catalog", "override", "adapter", "unknown"]
    reference: str
    state: CapabilityState


class EffectiveCapability(CapabilityContract):
    """One effective capability state with complete ordered provenance."""

    state: CapabilityState
    provenance: tuple[CapabilityEvidence, ...]


class CapabilityDiagnostic(CapabilityContract):
    """One deterministic capability-resolution or requirement diagnostic."""

    code: str
    capability: ChatCapabilityName
    message: str


class ResolvedChatCapabilities(CapabilityContract):
    """Effective Chat capabilities for one exact model and adapter binding."""

    image_input: EffectiveCapability
    streaming: EffectiveCapability
    native_tool_calling: EffectiveCapability
    tool_result_continuation: EffectiveCapability
    parallel_tool_calls: EffectiveCapability
    structured_output: EffectiveCapability
    prompt_cache: EffectiveCapability
    native_tool_search_hosted: EffectiveCapability
    native_tool_search_client: EffectiveCapability


class ResolvedModelBinding(CapabilityContract):
    """Pinned model binding, effective Chat facts, and preflight diagnostics."""

    operation: Literal["chat"] = "chat"
    provider_id: str
    endpoint_id: str
    model_id: str
    catalog_key: str | None
    catalog_digest: str
    capabilities: ResolvedChatCapabilities
    diagnostics: tuple[CapabilityDiagnostic, ...] = ()

    @property
    def valid(self) -> bool:
        """Return whether all requested capabilities passed preflight.

        Intro:
            A resolved binding is valid only when no fail-closed requirement
            diagnostic was produced.

        Examples:
            Check a valid binding:
                ```python
                assert binding.valid is True
                ```

            Check a rejected binding:
                ```python
                if not binding.valid:
                    assert binding.diagnostics
                ```

        Args:
            None.

        Returns:
            bool: `True` when the binding has no diagnostics.

        Notes:
            Resolution itself is side-effect free and never calls a provider.
        """

        return not self.diagnostics


_CAPABILITY_NAMES: tuple[ChatCapabilityName, ...] = (
    "image_input",
    "streaming",
    "native_tool_calling",
    "tool_result_continuation",
    "parallel_tool_calls",
    "structured_output",
    "prompt_cache",
    "native_tool_search_hosted",
    "native_tool_search_client",
)

_ADAPTER_FLAGS: dict[ChatCapabilityName, str] = {
    "image_input": "image_input",
    "streaming": "streaming",
    "native_tool_calling": "native_tools",
    "tool_result_continuation": "native_tools",
    "parallel_tool_calls": "native_tools",
    "structured_output": "structured_output",
    "native_tool_search_hosted": "native_tool_search",
    "native_tool_search_client": "native_tool_search",
}


def resolve_chat_profile(
    profile: ChatProfile,
    *,
    required: tuple[ChatCapabilityName, ...] = (),
) -> ResolvedModelBinding:
    """Resolve effective Chat capabilities for one canonical profile.

    Intro:
        Resolution combines catalog facts, explicit model overrides, and the
        selected adapter implementation. Adapter limitations always clamp a
        positive catalog or override assertion. Unknown never satisfies a
        required capability.

    Examples:
        Resolve ordinary Chat without advanced requirements:
            ```python
            binding = resolve_chat_profile(profile)
            assert binding.provider_id == profile.connection.provider_id
            ```

        Require native client Tool search:
            ```python
            binding = resolve_chat_profile(
                profile,
                required=("native_tool_search_client",),
            )
            ```

    Args:
        profile: Canonical immutable Chat profile.
        required: Capabilities that must resolve to `supported`.

    Returns:
        ResolvedModelBinding: Pinned binding, effective facts, provenance, and
        deterministic fail-closed diagnostics.

    Notes:
        Engine-projected Tool search is not a model capability and is therefore
        absent from both `required` and the resolved capability record.
    """

    if not isinstance(profile, ChatProfile):
        raise TypeError("Chat capability resolution requires ChatProfile")
    if len(required) != len(set(required)):
        raise ValueError("required Chat capabilities must be unique")
    adapter = get_endpoint_adapter(profile.connection.endpoint_id)
    provider = get_provider_descriptor(profile.connection.provider_id)
    if profile.connection.endpoint_id not in provider.endpoint_ids:
        raise ValueError("Chat profile endpoint is not registered for its provider")
    if "chat" not in adapter.implemented_operations:
        raise ValueError("Chat profile endpoint does not implement Chat")
    entry = resolve_model_catalog_entry(
        profile.connection.provider_id,
        profile.model.model_id,
        "chat",
        profile.connection.endpoint_id,
    )
    catalog_states: dict[ChatCapabilityName, CapabilityState] = {
        name: "unknown" for name in _CAPABILITY_NAMES
    }
    if entry is not None:
        native_modes = {item.mode for item in entry.native_tool_search}
        catalog_states["native_tool_search_hosted"] = (
            "supported" if "native_hosted" in native_modes else "unsupported"
        )
        catalog_states["native_tool_search_client"] = (
            "supported" if "native_client" in native_modes else "unsupported"
        )
    override_values = profile.capability_overrides.model_dump()
    effective: dict[str, EffectiveCapability] = {}
    for name in _CAPABILITY_NAMES:
        state = catalog_states[name]
        evidence = [
            CapabilityEvidence(
                source="catalog" if entry is not None else "unknown",
                reference=entry.catalog_key if entry is not None else "no_catalog_match",
                state=state,
            )
        ]
        override = override_values[name]
        if override != "unknown":
            state = override
            evidence.append(
                CapabilityEvidence(
                    source="override",
                    reference=f"profile.capability_overrides.{name}",
                    state=override,
                )
            )
        adapter_flag = _ADAPTER_FLAGS.get(name)
        if adapter_flag is not None and adapter_flag not in adapter.implementation_capabilities:
            state = "unsupported"
            evidence.append(
                CapabilityEvidence(
                    source="adapter",
                    reference=f"{adapter.adapter_id}:{adapter_flag}:unimplemented",
                    state="unsupported",
                )
            )
        effective[name] = EffectiveCapability(state=state, provenance=tuple(evidence))
    capabilities = ResolvedChatCapabilities(**effective)
    diagnostics = tuple(
        CapabilityDiagnostic(
            code=(
                "required_capability_unknown"
                if getattr(capabilities, name).state == "unknown"
                else "required_capability_unsupported"
            ),
            capability=name,
            message=(
                f"Required Chat capability {name!r} resolved to "
                f"{getattr(capabilities, name).state!r}."
            ),
        )
        for name in required
        if getattr(capabilities, name).state != "supported"
    )
    return ResolvedModelBinding(
        provider_id=profile.connection.provider_id,
        endpoint_id=profile.connection.endpoint_id,
        model_id=profile.model.model_id,
        catalog_key=entry.catalog_key if entry is not None else None,
        catalog_digest=catalog_digest(),
        capabilities=capabilities,
        diagnostics=diagnostics,
    )


__all__ = [
    "CapabilityDiagnostic",
    "CapabilityEvidence",
    "ChatCapabilityName",
    "EffectiveCapability",
    "ResolvedChatCapabilities",
    "ResolvedModelBinding",
    "resolve_chat_profile",
]
