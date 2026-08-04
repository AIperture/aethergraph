"""Bounded endpoint credentials for immutable AG Host browser clients."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from hmac import compare_digest
import secrets

from aethergraph.contracts.integration import HostManifest, IntegrationKind

ENDPOINT_COOKIE_NAME = "ag_endpoint_session"
ENDPOINT_CREDENTIAL_TTL_SECONDS = 8 * 60 * 60


@dataclass(frozen=True)
class _CredentialRecord:
    token_digest: str
    expires_at: datetime


class EndpointCredentialRegistry:
    """Issue and validate one bounded credential per enabled endpoint route.

    Examples:
        Issue credentials for an immutable Host manifest:
            ```python
            registry = EndpointCredentialRegistry.from_manifest(manifest)
            token = registry.take_launch_credentials()["studio-ui"]
            ```

        Validate the credential at the endpoint boundary:
            ```python
            assert registry.validate("studio-ui", token)
            ```

    Args:
        records: Endpoint-scoped token records keyed by public endpoint identity.
        launch_tokens: Plain launch credentials transferred once to the supervisor.

    Returns:
        None: The registry retains only bounded endpoint authorities for this Host.

    Notes:
        Plain credentials are never exposed by health or diagnostics. The Host-ready
        handshake is private supervisor IPC and transfers each token exactly once.
    """

    def __init__(
        self,
        *,
        records: dict[str, _CredentialRecord],
        launch_tokens: dict[str, str],
    ) -> None:
        self._records = dict(records)
        self._launch_tokens = dict(launch_tokens)

    @classmethod
    def from_manifest(
        cls,
        manifest: HostManifest,
        *,
        ttl_seconds: int = ENDPOINT_CREDENTIAL_TTL_SECONDS,
    ) -> EndpointCredentialRegistry:
        """Issue endpoint-scoped credentials for one immutable Host launch.

        Examples:
            Create the default eight-hour browser authorities:
                ```python
                registry = EndpointCredentialRegistry.from_manifest(manifest)
                ```

        Args:
            manifest: Sealed Host manifest containing the only endpoint routes.
            ttl_seconds: Positive lifetime for every credential issued at launch.

        Returns:
            EndpointCredentialRegistry: Registry with one token per enabled endpoint.

        Notes:
            Provider routes receive no browser credential. Restarting the Host issues
            a new credential set, so stale browser authorities fail closed.
        """

        if ttl_seconds <= 0:
            raise ValueError("Endpoint credential lifetime must be positive.")
        expires_at = datetime.now(UTC) + timedelta(seconds=ttl_seconds)
        tokens: dict[str, str] = {}
        records: dict[str, _CredentialRecord] = {}
        for route in manifest.integration_routes:
            if not route.enabled or route.integration_kind not in {
                IntegrationKind.AG_UI,
                IntegrationKind.AGENT_ENDPOINT,
            }:
                continue
            endpoint_id = route.endpoint_id
            if endpoint_id is None:
                raise ValueError("Enabled endpoint route is missing endpoint_id.")
            token = secrets.token_urlsafe(48)
            tokens[endpoint_id] = token
            records[endpoint_id] = _CredentialRecord(
                token_digest=sha256(token.encode()).hexdigest(),
                expires_at=expires_at,
            )
        return cls(records=records, launch_tokens=tokens)

    def take_launch_credentials(self) -> dict[str, str]:
        """Take the one-time private supervisor handoff for browser launch URLs.

        Examples:
            Take the credentials once for the readiness handshake:
                ```python
                credentials = registry.take_launch_credentials()
                ```

        Args:
            None.

        Returns:
            dict[str, str]: Endpoint identities mapped to bounded bearer tokens.

        Notes:
            This clears every plaintext token from the registry. A second call returns
            an empty mapping. Callers must redact the returned values from process logs.
        """

        credentials = self._launch_tokens
        self._launch_tokens = {}
        return credentials

    def validate(self, endpoint_id: str, token: str | None) -> bool:
        """Validate one token against its exact endpoint and expiry.

        Examples:
            Reject a token copied to another endpoint:
                ```python
                assert not registry.validate("another-endpoint", token)
                ```

        Args:
            endpoint_id: Exact public endpoint identity from the request path.
            token: Cookie or bearer credential supplied by the browser.

        Returns:
            bool: True only for the matching unexpired launch credential.

        Notes:
            Validation has no local-mode bypass and never falls through to AG auth.
        """

        record = self._records.get(endpoint_id)
        if record is None or not token or datetime.now(UTC) >= record.expires_at:
            return False
        return compare_digest(record.token_digest, sha256(token.encode()).hexdigest())

    def ttl_seconds(self, endpoint_id: str) -> int:
        """Return the remaining whole-second lifetime for a valid endpoint.

        Examples:
            Bound the cookie to the credential lifetime:
                ```python
                response.set_cookie(max_age=registry.ttl_seconds(endpoint_id))
                ```

        Args:
            endpoint_id: Exact endpoint whose cookie is being established.

        Returns:
            int: Remaining positive seconds, or zero after expiry/for unknown routes.

        Notes:
            This method never extends or rotates a credential.
        """

        record = self._records.get(endpoint_id)
        if record is None:
            return 0
        return max(0, int((record.expires_at - datetime.now(UTC)).total_seconds()))


__all__ = [
    "ENDPOINT_COOKIE_NAME",
    "ENDPOINT_CREDENTIAL_TTL_SECONDS",
    "EndpointCredentialRegistry",
]
