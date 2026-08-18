from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
import pytest

from aethergraph.api.v1.deps import (
    RequestIdentity,
    enforce_run_rate_limits,
    get_identity,
)
from aethergraph.config.config import AppSettings, RateLimitSettings
from aethergraph.core.runtime.node_services import NodeServices
from aethergraph.server.admission import RunBurstLimiter
from aethergraph.server.security.credentials import (
    EnvironmentSecretStore,
    resolve_auth_secret,
)
from aethergraph.server.security.redaction import (
    REDACTED_CREDENTIAL,
    is_masked_secret,
    mask_secret,
    sanitize_content,
    sanitize_text,
)
from aethergraph.services.container.default_container import (
    DefaultContainer,
    build_default_container,
)
from aethergraph.services.llm.credentials import resolve_provider_credential


def test_run_burst_limiter_uses_atomic_sliding_window() -> None:
    now = [100.0]
    limiter = RunBurstLimiter(2, 10, monotonic=lambda: now[0])
    assert limiter.allow("org-a")
    assert limiter.allow("org-a")
    assert not limiter.allow("org-a")
    assert limiter.allow("org-b")
    now[0] = 110.0
    assert limiter.allow("org-a")


@pytest.mark.parametrize("deploy_mode", ["demo", "cloud"])
def test_real_container_burst_limiter_returns_http_429(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    deploy_mode: str,
) -> None:
    root = tmp_path / deploy_mode
    cfg = AppSettings(
        workspace=str(root),
        deploy_mode=deploy_mode,
        auth={"secret": "test-auth-secret"},
        rate_limit={
            "enabled": True,
            "max_runs_per_window": 100,
            "burst_max_runs": 1,
            "burst_window_seconds": 60,
        },
    )
    container = build_default_container(root=str(root), cfg=cfg, channel_adapters={})
    container.metering = None
    monkeypatch.setattr("aethergraph.api.v1.deps.current_services", lambda: container)

    app = FastAPI()

    @app.post("/runs", dependencies=[Depends(enforce_run_rate_limits)])
    async def admitted() -> dict[str, bool]:
        return {"ok": True}

    async def identity() -> RequestIdentity:
        return RequestIdentity(user_id="user-a", org_id="org-a", mode=deploy_mode)

    app.dependency_overrides[get_identity] = identity
    with TestClient(app) as client:
        assert client.post("/runs").status_code == 200
        assert client.post("/runs").status_code == 429


def test_container_security_fields_have_one_canonical_shape(tmp_path: Path) -> None:
    fields = DefaultContainer.__dataclass_fields__
    assert "run_burst_limiter" in fields
    assert "rate_limiter" not in fields
    assert "secrets" not in fields
    assert {"redactor", "rate_limit", "secrets"}.isdisjoint(NodeServices.__dataclass_fields__)

    container = build_default_container(root=str(tmp_path), channel_adapters={})
    assert isinstance(container.run_burst_limiter, RunBurstLimiter)


@pytest.mark.parametrize("deploy_mode", ["demo", "cloud"])
def test_shared_deployments_require_explicit_auth_secret(tmp_path: Path, deploy_mode: str) -> None:
    cfg = AppSettings(workspace=str(tmp_path), deploy_mode=deploy_mode)
    with pytest.raises(ValueError, match="auth.secret is required"):
        build_default_container(root=str(tmp_path), cfg=cfg, channel_adapters={})


def test_fixed_auth_secret_is_local_only() -> None:
    assert resolve_auth_secret(deploy_mode="local", configured=None) == "aethergraph-dev-secret"
    assert resolve_auth_secret(deploy_mode="cloud", configured="configured") == "configured"


def test_environment_secret_store_is_synchronous_and_shared_by_model_resolution() -> None:
    store = EnvironmentSecretStore({"OPENAI_API_KEY": "stored"})
    assert store.get("OPENAI_API_KEY") == "stored"
    resolved = resolve_provider_credential(
        provider_id="openai",
        direct=None,
        secret_ref="OPENAI_API_KEY",
        secrets=store,
        environ={},
    )
    assert resolved.value == "stored"
    assert resolved.source_ref == "OPENAI_API_KEY"


def test_security_redaction_masks_credentials_and_persistence_payloads() -> None:
    sanitized = sanitize_content(
        {
            "authorization": "Bearer top-secret",
            "nested": {"api_key": "sk-private"},
            "safe": "api_key=sk-inline data:text/plain;base64,c2VjcmV0",
        }
    )
    assert sanitized["authorization"] == REDACTED_CREDENTIAL
    assert sanitized["nested"]["api_key"] == REDACTED_CREDENTIAL
    assert "sk-inline" not in sanitized["safe"]
    assert "c2VjcmV0" not in sanitized["safe"]
    assert "top-secret" not in sanitize_text("Authorization: Bearer top-secret")
    assert mask_secret("abcdefghijk") == "abcd****hijk"
    assert is_masked_secret("abcd****hijk")


def test_removed_security_service_modules_are_absent() -> None:
    import importlib.util

    assert importlib.util.find_spec("aethergraph.services.rate_limit") is None
    assert importlib.util.find_spec("aethergraph.services.secrets") is None
    assert importlib.util.find_spec("aethergraph.observability.redaction") is None


def test_rate_limit_settings_reject_non_positive_boundaries() -> None:
    with pytest.raises(ValueError):
        RateLimitSettings(burst_max_runs=0)
    with pytest.raises(ValueError):
        RateLimitSettings(burst_window_seconds=0)
