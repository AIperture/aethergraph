"""Shared secret selection for every model operation factory."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import SecretStr

from aethergraph.server.security.credentials import EnvironmentSecretStore, SecretStore

from .registry import get_provider_descriptor


@dataclass(frozen=True)
class ResolvedCredential:
    """Selected secret value and its non-secret source reference."""

    value: str | None
    source_ref: str | None


def resolve_provider_credential(
    *,
    provider_id: str,
    direct: SecretStr | str | None,
    secret_ref: str | None,
    secrets: SecretStore | None,
    environ: Mapping[str, str] | None = None,
) -> ResolvedCredential:
    """Resolve one provider credential with deterministic precedence.

    Intro:
        Inline secret material wins, followed by the named secret store entry,
        then registry-declared environment variables. The function never logs
        or embeds secret material in errors.

    Examples:
        Resolve an inline key:
            ```python
            result = resolve_provider_credential(
                provider_id="openai",
                direct=SecretStr("inline"),
                secret_ref=None,
                secrets=store,
                environ={},
            )
            assert result.value == "inline"
            ```

        Resolve a provider environment key:
            ```python
            result = resolve_provider_credential(
                provider_id="openai",
                direct=None,
                secret_ref=None,
                secrets=store,
                environ={"OPENAI_API_KEY": "environment"},
            )
            assert result.source_ref == "OPENAI_API_KEY"
            ```

    Args:
        provider_id: Registered provider identity.
        direct: Optional inline secret value.
        secret_ref: Optional secret-store reference.
        secrets: Optional secret store used for named lookup.
        environ: Optional environment mapping used instead of `os.environ`.

    Returns:
        ResolvedCredential: Selected value and its non-secret source reference.

    Notes:
        A missing named secret permits the exact provider environment fallback
        for compatibility with existing `.env` behavior. Passing no store is
        valid only when there is no named secret reference to resolve.
    """

    descriptor = get_provider_descriptor(provider_id)
    if direct is not None:
        value = direct.get_secret_value() if isinstance(direct, SecretStr) else str(direct)
        return ResolvedCredential(value, None)
    if secret_ref and secrets is None:
        raise ValueError("named credential resolution requires a secret store")
    if secret_ref:
        assert secrets is not None
        value = secrets.get(secret_ref)
        if value:
            return ResolvedCredential(value, secret_ref)
    environment_store = EnvironmentSecretStore(environ)
    for name in descriptor.credential_envs:
        value = environment_store.get(name)
        if value:
            return ResolvedCredential(value, name)
    return ResolvedCredential(None, secret_ref)


__all__ = ["ResolvedCredential", "resolve_provider_credential"]
