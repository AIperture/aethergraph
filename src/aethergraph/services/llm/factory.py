import logging
import os

from pydantic import SecretStr

from aethergraph.config.llm import LLMProfile, LLMSettings

from ..secrets.base import Secrets
from .credentials import resolve_provider_credential
from .generic_client import GenericLLMClient
from .observability import CaptureMode, LLMObservationSink
from .provider_transport import ProviderRateGate
from .registry import provider_default_base_url


def _apply_env_overrides_to_profile(
    name: str,
    p: LLMProfile,
    *,
    is_default: bool,
    secrets: Secrets,
) -> LLMProfile:
    """
    Mutate + return profile with env-based overrides.
    - For the default profile, allow generic LLM_* env vars.
    - For all profiles, fill missing base_url / api_key from provider-specific env.
    """
    # 1) Generic overrides for DEFAULT profile (if user wants a quick global switch)
    if is_default:
        provider_env = os.getenv("LLM_PROVIDER")
        model_env = os.getenv("LLM_MODEL")
        base_env = os.getenv("LLM_BASE_URL")
        timeout_env = os.getenv("LLM_TIMEOUT")
        reasoning_effort_env = os.getenv("LLM_REASONING_EFFORT")
        thinking_mode_env = os.getenv("LLM_THINKING_MODE")
        compat_env = os.getenv("LLM_COMPATIBILITY_POLICY")
        structured_output_env = os.getenv("LLM_STRUCTURED_OUTPUT_POLICY")

        if provider_env:
            p.provider = provider_env.lower()  # type: ignore[assignment]
        if model_env:
            p.model = model_env
        if base_env:
            p.base_url = base_env
        if timeout_env:
            try:
                p.timeout = float(timeout_env)
            except ValueError:
                logger = logging.getLogger("aethergraph.services.llm")
                logger.warning(f"Invalid LLM_TIMEOUT value: {timeout_env}")
        if reasoning_effort_env:
            p.reasoning_effort = reasoning_effort_env.lower()  # type: ignore[assignment]
        if thinking_mode_env:
            p.thinking_mode = thinking_mode_env.lower()  # type: ignore[assignment]
        if compat_env:
            p.compatibility_policy = compat_env.lower()  # type: ignore[assignment]
        if structured_output_env:
            p.structured_output_policy = structured_output_env.lower()  # type: ignore[assignment]

    # 2) Provider-specific base_url fallback
    if not p.base_url:
        p.base_url = provider_default_base_url(p.provider)

    # 3) API key resolution:
    #    - prefer explicit api_key on profile
    #    - else api_key_ref + Secrets
    #    - else provider-specific env name
    credential = resolve_provider_credential(
        provider_id=p.provider,
        direct=p.api_key,
        secret_ref=p.api_key_ref,
        secrets=secrets,
    )
    api_key = credential.value
    if api_key and not p.api_key_ref and credential.source_ref:
        p.api_key_ref = credential.source_ref

    # Finally, store the resolved key back into api_key for the client factory
    if api_key:
        p.api_key = SecretStr(api_key)

    return p


def client_from_profile(
    p: LLMProfile,
    secrets: Secrets,
    *,
    profile_name: str | None = None,
    observation_sink: LLMObservationSink | None = None,
    observation_capture_mode: CaptureMode = "manifest",
    rate_gate: ProviderRateGate | None = None,
) -> GenericLLMClient:
    # At this point, _apply_env_overrides_to_profile has already filled
    # p.base_url, p.api_key, etc. as much as possible.
    api_key = resolve_provider_credential(
        provider_id=p.provider,
        direct=p.api_key,
        secret_ref=p.api_key_ref,
        secrets=secrets,
    ).value

    return GenericLLMClient(
        provider=p.provider,
        model=p.model,
        base_url=p.base_url,
        api_key=api_key,
        azure_deployment=p.azure_deployment,
        timeout=p.timeout,
        retry_settings=p.retry,
        rate_limit_group=p.rate_limit_group,
        rate_gate=rate_gate,
        reasoning_effort=p.reasoning_effort,
        thinking_mode=p.thinking_mode,
        compatibility_policy=p.compatibility_policy,
        structured_output_policy=p.structured_output_policy,
        context_window_tokens=p.context_window_tokens,
        thinking_budget=p.thinking_budget,
        reasoning_summary=p.reasoning_summary,
        observation_sink=observation_sink,
        observation_capture_mode=observation_capture_mode,
        profile_name=profile_name,
    )


def build_llm_clients(
    cfg: LLMSettings,
    secrets: Secrets,
    *,
    observation_sink: LLMObservationSink | None = None,
    observation_capture_mode: CaptureMode = "manifest",
    rate_gate: ProviderRateGate | None = None,
) -> dict[str, GenericLLMClient]:
    """Returns dict of {profile_name: client}, always includes 'default' if enabled."""
    if not cfg.enabled:
        return {}

    shared_rate_gate = rate_gate or ProviderRateGate()

    # Mutate cfg.llm.default in-place with env defaults
    default_profile = _apply_env_overrides_to_profile(
        name="default",
        p=cfg.default,
        is_default=True,
        secrets=secrets,
    )
    clients: dict[str, GenericLLMClient] = {
        "default": client_from_profile(
            default_profile,
            secrets,
            profile_name="default",
            observation_sink=observation_sink,
            observation_capture_mode=observation_capture_mode,
            rate_gate=shared_rate_gate,
        )
    }

    # Extra profiles
    for name, prof in (cfg.profiles or {}).items():
        prof = _apply_env_overrides_to_profile(
            name=name,
            p=prof,
            is_default=False,
            secrets=secrets,
        )
        clients[name] = client_from_profile(
            prof,
            secrets,
            profile_name=name,
            observation_sink=observation_sink,
            observation_capture_mode=observation_capture_mode,
            rate_gate=shared_rate_gate,
        )

    return clients
