from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from inspect import getdoc
from pathlib import Path

import pytest

from aethergraph.api.v1.deps import get_identity
from aethergraph.services.auth.authn import AuthenticationRejected, AuthnService, DemoGrant
from aethergraph.services.auth.canonical_authn import CanonicalAuthnService
from aethergraph.services.auth.canonical_store import CanonicalAuthStore
from aethergraph.storage.contracts import StorageOpenMode, StorageScope
from aethergraph.storage.providers.local_sqlite import (
    LocalDatabaseRole,
    LocalKeyValueStore,
    LocalSQLiteDatabase,
)

_NOW = datetime(2026, 8, 16, 20, tzinfo=UTC)
_OWNER = StorageScope(tenant_id="tenant-1", project_id="project-1")


class _Clock:
    def __init__(self) -> None:
        self.value = _NOW

    def now(self) -> datetime:
        return self.value


def _service(
    root: Path,
    *,
    session_ids: tuple[str, ...] = ("session-1",),
) -> tuple[CanonicalAuthnService, CanonicalAuthStore, LocalSQLiteDatabase, _Clock]:
    clock = _Clock()
    database = LocalSQLiteDatabase.open(
        workspace_root=root,
        role=LocalDatabaseRole.CONTROL,
        mode=StorageOpenMode.READ_WRITE,
    )
    repository = LocalKeyValueStore(database=database, clock=clock)
    store = CanonicalAuthStore(
        grant_repository=repository,
        invite_repository=repository,
        owner_scope=_OWNER,
        clock=clock.now,
    )
    available_session_ids = iter(session_ids)
    available_guest_ids = iter(f"guest-{index}" for index in range(1, len(session_ids) + 1))
    authn = CanonicalAuthnService(
        store=store,
        secret="test-secret",
        clock=clock.now,
        guest_id_factory=lambda: next(available_guest_ids),
        session_id_factory=lambda: next(available_session_ids),
        invite_code_factory=lambda: "DEMO-GENERATED",
    )
    return authn, store, database, clock


@pytest.mark.asyncio
async def test_canonical_authn_redeems_and_resolves_provider_grant(tmp_path: Path) -> None:
    authn, _store, database, _clock = _service(tmp_path)
    grant = DemoGrant(
        grant_id="grant-1",
        org_id="org-1",
        allowed_apps=["legacy-app"],
        allowed_agents=["agent-1"],
    )
    try:
        invite = await authn.create_invite_code(grant, max_uses=1)
        session = await authn.redeem_invite_code(invite.code, client_id="browser-1")
        resolved = await authn.resolve(
            deploy_mode="demo",
            session_id=session.session_id,
            client_id=None,
            x_user_id=None,
            x_org_id=None,
        )

        assert invite.code == "DEMO-GENERATED"
        assert resolved.mode == "demo_guest"
        assert resolved.auth_source == "demo_guest_session"
        assert resolved.client_id == "browser-1"
        assert resolved.grant is not None
        assert resolved.grant.allowed_apps == ["legacy-app"]
        assert resolved.grant.allowed_agents == ["agent-1"]
    finally:
        await database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_state", ["missing", "revoked", "expired"])
async def test_canonical_authn_invalid_session_grant_fails_without_mode_fallback(
    tmp_path: Path,
    invalid_state: str,
) -> None:
    authn, store, database, clock = _service(tmp_path)
    expiry = _NOW + timedelta(seconds=1) if invalid_state == "expired" else None
    grant = DemoGrant(grant_id="grant-1", org_id="org-1", expires_at=expiry)
    try:
        session = await authn.create_demo_session(grant=grant)
        if invalid_state == "missing":
            await store.delete_grant(grant.grant_id)
        elif invalid_state == "revoked":
            await authn.revoke_grant(grant.grant_id)
        else:
            clock.value = _NOW + timedelta(seconds=2)

        with pytest.raises(AuthenticationRejected, match="no longer valid"):
            await authn.resolve(
                deploy_mode="demo",
                session_id=session.session_id,
                client_id="browser-1",
                x_user_id="cloud-user",
                x_org_id="cloud-org",
                x_mode="demo",
            )

        assert authn.get_session(session.session_id) is None
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_canonical_authn_invite_claims_remain_atomic(tmp_path: Path) -> None:
    session_ids = tuple(f"session-{index}" for index in range(1, 13))
    authn, store, database, _clock = _service(tmp_path, session_ids=session_ids)
    grant = DemoGrant(grant_id="grant-1", org_id="org-1")
    try:
        invite = await authn.create_invite_code(grant, max_uses=3, code="DEMO-LIMITED")
        outcomes = await asyncio.gather(
            *(authn.redeem_invite_code(invite.code) for _ in range(12)),
            return_exceptions=True,
        )
        successes = [item for item in outcomes if not isinstance(item, BaseException)]
        failures = [item for item in outcomes if isinstance(item, BaseException)]
        persisted = await store.get_invite(invite.code)

        assert len(successes) == 3
        assert len({item.session_id for item in successes}) == 3
        assert len(failures) == 9
        assert all(isinstance(item, ValueError) for item in failures)
        assert persisted is not None
        assert persisted.uses == 3
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_canonical_authn_admin_updates_use_provider_authority(tmp_path: Path) -> None:
    authn, _store, database, _clock = _service(tmp_path)
    grant = DemoGrant(grant_id="grant-1", org_id="org-1")
    try:
        invite = await authn.create_invite_code(grant, code="DEMO-ADMIN")
        updated_grant = await authn.update_grant(
            grant.grant_id,
            {"allowed_agents": ["agent-1"], "client_label": "Research"},
        )
        updated_invite = await authn.update_invite_code(invite.code, {"max_uses": 2})
        deactivated = await authn.deactivate_invite_code(invite.code)

        assert updated_grant.allowed_agents == ["agent-1"]
        assert updated_grant.client_label == "Research"
        assert updated_invite.max_uses == 2
        assert deactivated.active is False
        assert [item.grant_id for item in await authn.list_grants()] == [grant.grant_id]
        assert [item.code for item in await authn.list_invite_codes()] == [invite.code]
        with pytest.raises(ValueError, match="unsupported"):
            await authn.update_grant(grant.grant_id, {"org_id": "other"})

        await authn.delete_invite_code(invite.code)
        await authn.delete_grant(grant.grant_id)
        assert await authn.list_invite_codes() == []
        assert await authn.list_grants() == []
    finally:
        await database.close()


def test_canonical_authn_has_provider_neutral_structure_and_strict_docstrings() -> None:
    source = Path(CanonicalAuthnService.__module__.replace(".", "/") + ".py")
    source_text = (Path(__file__).parents[1] / "src" / source).read_text(encoding="utf-8")

    assert "SQLiteKVSync" not in source_text
    assert "self._grants" not in source_text
    assert "self._invite_codes" not in source_text
    assert "load_persisted" not in source_text
    assert '"_index"' not in source_text
    assert "except Exception" not in source_text
    for method_name in (
        "__init__",
        "create_demo_session",
        "get_session",
        "delete_session",
        "get_grant",
        "create_invite_code",
        "redeem_invite_code",
        "list_invite_codes",
        "deactivate_invite_code",
        "delete_invite_code",
        "update_invite_code",
        "list_grants",
        "revoke_grant",
        "delete_grant",
        "update_grant",
        "resolve",
    ):
        doc = getdoc(getattr(CanonicalAuthnService, method_name)) or ""
        assert doc.splitlines()[0]
        assert doc.index("Intro:") < doc.index("Examples:")
        assert doc.index("Examples:") < doc.index("Args:")
        assert doc.index("Args:") < doc.index("Returns:")
        assert doc.index("Returns:") < doc.index("Notes:")
        assert doc.count("```python") >= 2

    for member in (AuthnService.resolve, get_identity):
        doc = getdoc(member) or ""
        assert doc.index("Intro:") < doc.index("Examples:")
        assert doc.index("Examples:") < doc.index("Args:")
        assert doc.index("Args:") < doc.index("Returns:")
        assert doc.index("Returns:") < doc.index("Notes:")
        assert doc.count("```python") >= 2
