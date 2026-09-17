"""Session cleanup must retain siblings, shared blobs and restart evidence."""

from dataclasses import replace

import pytest
from test_canonical_artifact_facade import _facade, _open_bundle, _owner_scope
from test_local_control_repositories import NOW, _run, _session

from aethergraph.storage.contracts import RunStatus, StorageConflictError, StorageScope
from aethergraph.storage.providers.local_sqlite import LocalBlobStore
from aethergraph.storage.providers.local_sqlite.session_purge import LocalSessionPurge


@pytest.mark.asyncio
async def test_session_purge_retains_sibling_and_shared_content(tmp_path):
    bundle = _open_bundle(tmp_path)
    owner = StorageScope(tenant_id="tenant-1", project_id="project-1")
    try:
        for sid in ("session-1", "session-2"):
            await bundle.sessions.create(_session(sid))
        run = replace(_run("run-1"), status=RunStatus.SUCCEEDED, finished_at=NOW)
        await bundle.runs.create(run)
        first = _facade(
            bundle,
            execution_scope=StorageScope(
                **_owner_scope().as_filter(),
                session_id="session-1",
                run_id="run-1",
                graph_id="graph-1",
            ),
        )
        second = _facade(
            bundle,
            execution_scope=StorageScope(**_owner_scope().as_filter(), session_id="session-2"),
        )
        await first.save_text(
            "exclusive bytes",
            kind="report",
            artifact_id="exclusive",
            occurrence_id="exclusive-occurrence",
        )
        await first.save_text(
            "shared bytes",
            kind="report",
            artifact_id="shared",
            occurrence_id="shared-first",
        )
        await second.save_text(
            "shared bytes",
            kind="report",
            artifact_id="shared-sibling",
            occurrence_id="shared-second",
        )
        await LocalSessionPurge(bundle).purge(owner, ("session-1",))
        assert await bundle.sessions.get(owner, "session-1") is None
        assert await bundle.runs.get(owner, "run-1") is None
        assert await bundle.sessions.get(owner, "session-2") is not None
        assert await second.load_bytes("shared-sibling") == b"shared bytes"
        assert await bundle.artifacts.get(_owner_scope(), "exclusive") is None
        assert await bundle.artifacts.get(_owner_scope(), "shared") is None
        await LocalSessionPurge(bundle).purge(owner, ("session-1",))
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_session_purge_refuses_active_runs(tmp_path):
    bundle = _open_bundle(tmp_path)
    try:
        await bundle.sessions.create(_session("session-1"))
        await bundle.runs.create(_run("run-1"))
        with pytest.raises(StorageConflictError, match="active runs"):
            await LocalSessionPurge(bundle).purge(
                StorageScope(tenant_id="tenant-1", project_id="project-1"),
                ("session-1",),
            )
        assert (
            await bundle.sessions.get(StorageScope(project_id="project-1"), "session-1") is not None
        )
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_session_purge_resumes_after_blob_metadata_commit(tmp_path, monkeypatch):
    owner = StorageScope(tenant_id="tenant-1", project_id="project-1")
    bundle = _open_bundle(tmp_path)
    await bundle.sessions.create(_session("session-1"))
    facade = _facade(
        bundle,
        execution_scope=StorageScope(**_owner_scope().as_filter(), session_id="session-1"),
    )
    receipt = await facade.save_text(
        "remove after restart",
        kind="report",
        artifact_id="owned",
        occurrence_id="owned-event",
    )
    path = bundle.blobs._content_path(receipt.record.content_hash)
    assert path.is_file()

    async def interrupted(*args, **kwargs):
        raise PermissionError("injected locked blob")

    with monkeypatch.context() as patch:
        patch.setattr(LocalBlobStore, "_drain_gc_tombstones", interrupted)
        with pytest.raises(PermissionError, match="locked blob"):
            await LocalSessionPurge(bundle).purge(owner, ("session-1",))
    assert path.is_file()
    await bundle.close()
    reopened = _open_bundle(tmp_path)
    try:
        await LocalSessionPurge(reopened).purge(owner, ("session-1",))
        assert not path.exists()
        assert await reopened.sessions.get(owner, "session-1") is None
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_confirmed_process_loss_allows_stale_running_history(tmp_path):
    from test_canonical_artifact_facade import _SECRET_REF

    from aethergraph.maintenance import purge_local_sessions
    from aethergraph.storage.contracts import StorageProviderSelection

    bundle = _open_bundle(tmp_path)
    await bundle.sessions.create(_session("session-1"))
    await bundle.runs.create(_run("run-1"))
    await bundle.close()
    await purge_local_sessions(
        tmp_path,
        owner_scope=StorageScope(tenant_id="tenant-1", project_id="project-1"),
        session_ids=("session-1",),
        stopped_session_ids=("session-1",),
        selection=StorageProviderSelection(
            provider="local.sqlite",
            config={"continuation_token_secret_ref": _SECRET_REF},
        ),
    )
    reopened = _open_bundle(tmp_path)
    try:
        assert await reopened.runs.get(StorageScope(project_id="project-1"), "run-1") is None
    finally:
        await reopened.close()
