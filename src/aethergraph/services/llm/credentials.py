"""Shared secret selection for every model operation factory."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from typing import Protocol

from pydantic import SecretStr

from .registry import get_provider_descriptor


class SecretStore(Protocol):
    """Minimal secret-store surface required by model factories."""

    def get(self, name: str) -> str | None:
        """Return one secret value by reference without exposing enumeration.

        Intro:
            Model factories require only exact named lookup and never inspect
            the full secret-store inventory.

        Examples:
            Resolve a present secret:
                ```python
                value = store.get("OPENAI_API_KEY")
                ```

            Resolve an absent secret:
                ```python
                assert store.get("MISSING") is None
                ```

        Args:
            name: Exact configured secret reference.

        Returns:
            str | None: Secret value when present, otherwise `None`.

        Notes:
            Implementations must not log the returned value.
        """

        ...


@dataclass(frozen=True)
class ResolvedCredential:
    """Selected secret value and its non-secret source reference."""

    value: str | None
    source_ref: str | None


def resolve_provider_credential(
    *,
    provider_id: str,
    direct: SecretStr | None,
    secret_ref: str | None,
    secrets: SecretStore,
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
        secrets: Secret store used for named lookup.
        environ: Optional environment mapping used instead of `os.environ`.

    Returns:
        ResolvedCredential: Selected value and its non-secret source reference.

    Notes:
        A missing named secret permits the provider environment fallback for
        compatibility with existing `.env` behavior.
    """

    descriptor = get_provider_descriptor(provider_id)
    if direct is not None:
        return ResolvedCredential(direct.get_secret_value(), None)
    if secret_ref:
        value = secrets.get(secret_ref)
        if value:
            return ResolvedCredential(value, secret_ref)
    values = os.environ if environ is None else environ
    for name in descriptor.credential_envs:
        value = values.get(name)
        if value:
            return ResolvedCredential(value, name)
    return ResolvedCredential(None, secret_ref)


__all__ = ["ResolvedCredential", "SecretStore", "resolve_provider_credential"]
