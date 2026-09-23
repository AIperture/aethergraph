"""Direct graph callers retain their exact storage scope in managed child runs."""

import asyncio
from uuid import uuid4

import pytest

from aethergraph import NodeContext
from aethergraph.api.v1.deps import RequestIdentity
from aethergraph.config.config import AppSettings
from aethergraph.core.graph.graph_fn import GraphFunction
from aethergraph.core.runtime.runtime_registry import current_registry
from aethergraph.core.runtime.runtime_services import use_services
from aethergraph.runner import run_async
from aethergraph.services.container.default_container import build_default_container
from aethergraph.services.runner.facade import RunFacade


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "identity", [None, RequestIdentity(org_id="org", user_id="user", client_id="client")]
)
async def test_direct_parent_and_child_share_exact_session_state(tmp_path, identity):
    container = build_default_container(
        cfg=AppSettings(workspace=str(tmp_path), embed={"enabled": False}),
        root=str(tmp_path),
    )
    scopes = []

    async def child(*, context: NodeContext):
        scopes.append((context.scope.org_id, context.scope.user_id, context.scope.client_id))
        value = await context.state("parent-receipt", model=dict, level="session").load()
        return {"value": value}

    child_graph = GraphFunction(
        name=f"test.child_scope.{uuid4().hex}", fn=child, inputs=[], outputs=["value"]
    )

    async def parent(*, context: NodeContext):
        scopes.append((context.scope.org_id, context.scope.user_id, context.scope.client_id))
        handle = context.state("parent-receipt", model=dict, level="session")
        await handle.load()
        await handle.commit({"value": "retained"}, reason="parent receipt")
        run_id = await context.runner().spawn_run(child_graph.name, inputs={})
        record, output = await context.runner().wait_run(run_id, return_outputs=True)
        assert record.status.value == "succeeded", record.error
        assert (await context.runner().inspect_run(run_id)).run_id == run_id
        return {"value": output["value"]}

    parent_graph = GraphFunction(
        name=f"test.parent_scope.{uuid4().hex}", fn=parent, inputs=[], outputs=["value"]
    )
    with use_services(container):
        try:
            current_registry().register(
                nspace="graphfn",
                name=child_graph.name,
                version=child_graph.version,
                obj=child_graph,
            )
            result = await asyncio.wait_for(
                run_async(parent_graph, {}, identity=identity, session_id="scope-test"), 10
            )
            assert scopes[0] == scopes[1]
            assert result == {"value": {"value": "retained"}}
        finally:
            await container.run_manager.close()
            await container.close_storage()


@pytest.mark.asyncio
@pytest.mark.parametrize("launch", ["spawn", "waiting"])
async def test_isolated_child_is_owned_by_parent_session_after_run_completion(tmp_path, launch):
    container = build_default_container(
        cfg=AppSettings(workspace=str(tmp_path), embed={"enabled": False}), root=str(tmp_path)
    )
    observed = {}

    async def child(*, context: NodeContext):
        observed["session"] = context.session_id
        observed["origin_session"] = context.origin_binding.session_id
        value = await context.state("parent-only", model=dict, level="session").load()
        return {"value": value}

    child_graph = GraphFunction(
        name=f"test.isolated_child.{uuid4().hex}", fn=child, inputs=[], outputs=["value"]
    )

    async def parent(*, context: NodeContext):
        handle = context.state("parent-only", model=dict, level="session")
        await handle.load()
        await handle.commit({"private": "parent state"}, reason="parent fixture")
        if launch == "spawn":
            run_id = await context.runner().spawn_run(
                child_graph.name, inputs={}, session_id="isolated-child"
            )
        else:
            run_id, _, _, _ = await context.runner().run_and_wait(
                child_graph.name, inputs={}, session_id="isolated-child"
            )
        record, output = await context.runner().wait_run(run_id, return_outputs=True)
        assert record.status.value == "succeeded", record.error
        assert output == {"value": {}}
        assert (await context.runner().inspect_run(run_id)).run_id == run_id
        later = RunFacade(
            container.run_manager, session_id="parent-session", identity=context.identity
        )
        stored = await container.run_manager._store.get(run_id)
        assert stored.parent.run_id == context.run_id
        assert stored.parent.session_id == "parent-session"
        assert (await later.inspect_run(run_id)).run_id == run_id
        foreign = RunFacade(
            container.run_manager,
            session_id="parent-session",
            identity=RequestIdentity(org_id="other-org", user_id="other-user"),
        )
        with pytest.raises(LookupError, match="caller scope"):
            await foreign.inspect_run(run_id)
        assert context.origin_binding.session_id == "parent-session"
        outsider = RunFacade(
            container.run_manager, session_id="unrelated-session", identity=context.identity
        )
        for operation in (outsider.inspect_run, outsider.wait_run, outsider.cancel_run):
            with pytest.raises(LookupError, match="caller scope"):
                await operation(run_id)
        observed["child_run_id"] = run_id
        return {"done": True}

    parent_graph = GraphFunction(
        name=f"test.isolated_parent.{uuid4().hex}", fn=parent, inputs=[], outputs=["done"]
    )
    with use_services(container):
        try:
            current_registry().register(
                nspace="graphfn",
                name=child_graph.name,
                version=child_graph.version,
                obj=child_graph,
            )
            result = await asyncio.wait_for(
                run_async(
                    parent_graph,
                    {},
                    session_id="parent-session",
                    origin_binding={
                        "integration_id": "test",
                        "route_id": "test",
                        "session_id": "parent-session",
                        "channel_key": "console:stdout",
                        "external_conversation_id": "parent-session",
                        "capability_profile_id": "test/v1",
                    },
                ),
                10,
            )
            assert result == {"done": True}
            assert observed["session"] == observed["origin_session"] == "isolated-child"
        finally:
            await container.run_manager.close()
            await container.close_storage()
