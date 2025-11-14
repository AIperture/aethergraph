import logging
import os

from pydantic import SecretStr

from aethergraph.config.llm import LLMProfile, LLMSettings

from ..secrets.base import Secrets
from .generic_client import GenericLLMClient
from .providers import Provider


def _resolve_key(direct: SecretStr | None, ref: str | None, secrets: Secrets) -> str | None:
    if direct:
        return direct.get_secret_value()
    if ref:
        return secrets.get(ref)
    return None


def _provider_default_base_url(provider: Provider) -> str | None:
    # Fallback base URLs if not given in config or env
    if provider == "openai":
        return "https://api.openai.com/v1"
    if provider == "azure":
        # Must still rely on env/config for endpoint
        return os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/") or None
    if provider == "anthropic":
        return "https://api.anthropic.com"
    if provider == "google":
        return "https://generativelanguage.googleapis.com"
    if provider == "openrouter":
        return "https://openrouter.ai/api/v1"
    if provider == "lmstudio":
        return os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    if provider == "ollama":
        return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    return None


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
                logger.warning(f"Invalid LLM_TIMEOUT value: {timeout_env}", exc_info=True)

    # 2) Provider-specific base_url fallback
    if not p.base_url:
        p.base_url = _provider_default_base_url(p.provider)  # type: ignore[arg-type]

    # 3) API key resolution:
    #    - prefer explicit api_key on profile
    #    - else api_key_ref + Secrets
    #    - else provider-specific env name
    api_key = _resolve_key(p.api_key, p.api_key_ref, secrets)

    if not api_key:
        # Fallback to provider-specific env if nothing else was set
        if p.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        elif p.provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
        elif p.provider == "google":
            api_key = os.getenv("GOOGLE_API_KEY")
        elif p.provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
        elif p.provider == "azure":
            api_key = os.getenv("AZURE_OPENAI_KEY")

        # If we found one, and no api_key_ref was configured, we can
        # optionally set api_key_ref so it's visible in config
        if api_key and not p.api_key_ref:
            # Optional: record that this profile is using that env key
            p.api_key_ref = {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "google": "GOOGLE_API_KEY",
                "openrouter": "OPENROUTER_API_KEY",
                "azure": "AZURE_OPENAI_KEY",
            }.get(p.provider, None)  # type: ignore[index]

    # Finally, store the resolved key back into api_key for the client factory
    if api_key:
        p.api_key = SecretStr(api_key)

    return p


def client_from_profile(p: LLMProfile, secrets: Secrets) -> GenericLLMClient:
    # At this point, _apply_env_overrides_to_profile has already filled
    # p.base_url, p.api_key, etc. as much as possible.
    api_key = _resolve_key(p.api_key, p.api_key_ref, secrets)

    return GenericLLMClient(
        provider=p.provider,
        model=p.model,
        embed_model=p.embed_model,
        base_url=p.base_url,
        api_key=api_key,
        azure_deployment=p.azure_deployment,
        timeout=p.timeout,
    )


def build_llm_clients(cfg: LLMSettings, secrets: Secrets) -> dict[str, GenericLLMClient]:
    """Returns dict of {profile_name: client}, always includes 'default' if enabled."""
    if not cfg.enabled:
        return {}

    # Mutate cfg.llm.default in-place with env defaults
    default_profile = _apply_env_overrides_to_profile(
        name="default",
        p=cfg.default,
        is_default=True,
        secrets=secrets,
    )
    clients: dict[str, GenericLLMClient] = {
        "default": client_from_profile(default_profile, secrets)
    }

    # Extra profiles
    for name, prof in (cfg.profiles or {}).items():
        prof = _apply_env_overrides_to_profile(
            name=name,
            p=prof,
            is_default=False,
            secrets=secrets,
        )
        clients[name] = client_from_profile(prof, secrets)

    return clients
