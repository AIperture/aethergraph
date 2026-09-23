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
