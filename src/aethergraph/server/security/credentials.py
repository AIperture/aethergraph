from __future__ import annotations

from collections.abc import Mapping
import os
from typing import Protocol

from pydantic import SecretStr


class SecretStore(Protocol):
    """Resolve exact credential references synchronously at server boundaries."""

    def get(self, name: str) -> str | None:
        """Return one credential without exposing the store inventory.

        Intro:
            Resolves one exact configured reference through the synchronous
            credential contract used by every model-operation factory.

        Examples:
            Resolve a configured credential:
            ```python
            value = store.get("OPENAI_API_KEY")
            ```

            Handle a missing credential:
            ```python
            assert store.get("MISSING_KEY") is None
            ```

        Args:
            name: Exact credential reference.

        Returns:
            str | None: Credential value when present, otherwise `None`.

        Notes:
            Implementations must not log returned values or support enumeration
            through this protocol.
        """
        ...


class EnvironmentSecretStore(SecretStore):
    """Resolve exact credential names from one environment mapping."""

    def __init__(self, environ: Mapping[str, str] | None = None) -> None:
        self._environ = os.environ if environ is None else environ

    def get(self, name: str) -> str | None:
        """Return one environment credential by exact name.

        Intro:
            Performs a synchronous exact-key lookup without copying or exposing
            unrelated process environment values.

        Examples:
            Resolve a present value:
            ```python
            store = EnvironmentSecretStore({"API_KEY": "secret"})
            assert store.get("API_KEY") == "secret"
            ```

            Resolve an absent value:
            ```python
            store = EnvironmentSecretStore({})
            assert store.get("API_KEY") is None
            ```

        Args:
            name: Exact environment variable name.

        Returns:
            str | None: Environment value when present, otherwise `None`.

        Notes:
            The store is read-only and intentionally has no list or write API.
        """
        return self._environ.get(name)


def resolve_auth_secret(*, deploy_mode: str, configured: SecretStr | str | None) -> str:
    """Resolve the server authentication signing secret for one deployment.

    Intro:
        Requires explicit signing material in shared demo/cloud deployments and
        limits the fixed development key to explicit local mode.

    Examples:
        Resolve an explicit production secret:
        ```python
        secret = resolve_auth_secret(deploy_mode="cloud", configured="configured")
        ```

        Resolve the local development secret:
        ```python
        secret = resolve_auth_secret(deploy_mode="local", configured=None)
        ```

    Args:
        deploy_mode: Configured `local`, `demo`, or `cloud` deployment mode.
        configured: Optional explicit signing secret.

    Returns:
        str: Exact signing secret installed in the authentication service.

    Notes:
        Missing signing material fails before operational stores are constructed
        for demo and cloud deployments.
    """
    if configured is not None:
        return (
            configured.get_secret_value() if isinstance(configured, SecretStr) else str(configured)
        )
    if deploy_mode == "local":
        return "aethergraph-dev-secret"
    raise ValueError("auth.secret is required when deploy_mode is 'demo' or 'cloud'")
