"""Direct graph callers retain their exact storage scope in managed child runs."""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from aethergraph import NodeContext
from aethergraph.api.v1.deps import RequestIdentity
from aethergraph.config.config import AppSettings
from aethergraph.core.graph.graph_fn import GraphFunction
from aethergraph.core.runtime.run_types import RunParent, RunRecord, RunStatus
from aethergraph.core.runtime.runtime_registry import current_registry
from aethergraph.core.runtime.runtime_services import use_services
from aethergraph.runner import run_async
from aethergraph.services.container.default_container import build_default_container
from aethergraph.services.continuations.continuation import (
    ContinuationQuery,
    ContinuationStatus,
    Correlator,
)
from aethergraph.services.integration.interactions import (
    InteractionResolutionError,
    InteractionResolver,
)
from aethergraph.services.runner.facade import RunFacade
from aethergraph.services.triggers.trigger_facade import TriggerConfig


@pytest.mark.asyncio
async def test_scheduled_child_retains_root_authority_for_later_grandchildren(tmp_path):
    container = build_default_container(
        cfg=AppSettings(workspace=str(tmp_path), embed={"enabled": False}), root=str(tmp_path)
    )
    captured = {}

    async def grandchild(*, context: NodeContext):
        return {"value": "grandchild completed"}

    leaf = GraphFunction(
        name=f"test.scheduled-leaf.{uuid4().hex}", fn=grandchild, inputs=[], outputs=["value"]
    )

    async def scheduled(*, context: NodeContext):
        return {
            "child_id": await context.runner().spawn_run(
                leaf.name, inputs={}, session_id="grandchild"
            )
        }

    wake = GraphFunction(
        name=f"test.scheduled-wake.{uuid4().hex}", fn=scheduled, inputs=[], outputs=["child_id"]
    )

    async def child(*, context: NodeContext):
        captured["child_run_id"] = context.run_id
        trigger = await context.triggers().create(
            graph_id=wake.name,
            default_inputs={},
            config=TriggerConfig(kind="interval", interval_seconds=10),
            idempotency_key="child-wakeup",
        )
        return {"trigger_id": trigger.trigger_id}

    relay = GraphFunction(
        name=f"test.scheduled-owner.{uuid4().hex}", fn=child, inputs=[], outputs=["trigger_id"]
    )

    async def parent(*, context: NodeContext):
        captured["root_run_id"] = context.run_id
        child_id = await context.runner().spawn_run(relay.name, inputs={}, session_id="child")
        _, result = await context.runner().wait_run(child_id, return_outputs=True)
        return result

    root = GraphFunction(
        name=f"test.scheduled-root.{uuid4().hex}", fn=parent, inputs=[], outputs=["trigger_id"]
    )
    with use_services(container):
        try:
            for graph in (leaf, wake, relay):
                current_registry().register(
                    nspace="graphfn", name=graph.name, version=graph.version, obj=graph
                )
            result = await asyncio.wait_for(run_async(root, {}, session_id="root"), 10)
            trigger = await container.trigger_store.get(result["trigger_id"])
            fired = await container.trigger_engine._submit(
                trigger,
                inputs={},
                run_id="scheduled-child-wake",
                fire_id="fire-1",
            )
            ended = await asyncio.wait_for(container.run_manager.wait_run(fired.run_id), 10)
            assert ended.status is RunStatus.succeeded, ended.error
            root_facade = RunFacade(container.run_manager, session_id="root")
            # Both records must remain accessible after the original parent and
            # child have completed; a shared channel is neither needed nor used.
            record, output = await root_facade.wait_run(fired.run_id, return_outputs=True)
            assert record.parent == RunParent(captured["child_run_id"], "child")
            assert output["child_id"]
            leaf_record, leaf_output = await asyncio.wait_for(
                root_facade.wait_run(output["child_id"], return_outputs=True), 10
            )
            assert leaf_record.session_id == "grandchild"
            assert leaf_output == {"value": "grandchild completed"}
            with pytest.raises(LookupError):
                await RunFacade(container.run_manager, session_id="unrelated").inspect_run(
                    leaf_record.run_id
                )
        finally:
            await container.run_manager.close()
            await container.close_storage()


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


@pytest.mark.asyncio
@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize(("immediate", "inline"), [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("question_kind", ["text", "choice"])
async def test_parent_answers_live_isolated_child_after_parent_run_finishes(
    tmp_path, depth, immediate, inline, question_kind
):
    container = build_default_container(
        cfg=AppSettings(workspace=str(tmp_path), embed={"enabled": False}), root=str(tmp_path)
    )
    questions = asyncio.Queue()
    executions = []
    answer_payload = {"text": "Titanium"} if question_kind == "text" else {"choice": "Ti"}

    class Questions:
        capabilities = {"text", "input"}

        async def send(self, event):
            if event.type in {"session.need_input", "session.need_approval"}:
                if question_kind == "choice":
                    assert [(button.value, button.label) for button in event.buttons] == [
                        ("Titanium", "Titanium")
                    ]
                questions.put_nowait(event)
                if inline:
                    return {"payload": answer_payload}
                if immediate:
                    selected = await InteractionResolver(
                        container.cont_store,
                        run_manager=container.run_manager,
                    ).resolve_exact(
                        session_id="parent-session",
                        interaction_id=event.meta["interaction_id"],
                        expected_kinds={"user_input", "choice"},
                    )
                    await container.resume_router.resume_continuation(
                        selected.continuation,
                        answer_payload,
                    )
            return {"correlator": Correlator("test", event.channel, "", "question")}

    container.channels.register_adapter("test", Questions())

    async def child(*, context: NodeContext):
        executions.append(context.run_id)
        if question_kind == "text":
            answer = await context.channel("test:parent").ask_text("Which material?")
        else:
            reply = await context.channel("test:parent").ask_choices(
                "Which material?",
                [{"id": "Titanium", "label": "Titanium", "aliases": ["Ti"]}],
            )
            assert reply.matched
            answer = reply.choice
        return {"answer": answer}

    child_graph = GraphFunction(
        name=f"test.child_question.{uuid4().hex}", fn=child, inputs=[], outputs=["answer"]
    )

    async def relay(*, context: NodeContext):
        run_id = await context.runner().spawn_run(
            child_graph.name, inputs={}, session_id="isolated-grandchild"
        )
        _, output = await context.runner().wait_run(run_id, return_outputs=True)
        return output

    relay_graph = GraphFunction(
        name=f"test.relay_question.{uuid4().hex}", fn=relay, inputs=[], outputs=["answer"]
    )

    async def parent(*, context: NodeContext):
        child_run = await context.runner().spawn_run(
            child_graph.name if depth == 1 else relay_graph.name,
            inputs={},
            session_id="isolated-child",
        )
        return {"child_run": child_run}

    parent_graph = GraphFunction(
        name=f"test.parent_question.{uuid4().hex}", fn=parent, inputs=[], outputs=["child_run"]
    )
    with use_services(container):
        try:
            for graph in (child_graph, relay_graph):
                current_registry().register(
                    nspace="graphfn",
                    name=graph.name,
                    version=graph.version,
                    obj=graph,
                )
            result = await asyncio.wait_for(
                run_async(parent_graph, {}, session_id="parent-session"), 10
            )
            question = await asyncio.wait_for(questions.get(), 10)
            expected_session = "isolated-child" if depth == 1 else "isolated-grandchild"
            assert question.meta["session_id"] == expected_session

            resolver = InteractionResolver(container.cont_store, run_manager=container.run_manager)
            with pytest.raises(InteractionResolutionError, match="bound AG session|not open"):
                await resolver.resolve_exact(
                    session_id="unrelated-session",
                    interaction_id=question.meta["interaction_id"],
                    expected_kinds={"user_input", "choice"},
                )
            if not immediate and not inline:
                resolved = await resolver.resolve_exact(
                    session_id="parent-session",
                    interaction_id=question.meta["interaction_id"],
                    expected_kinds={"user_input", "choice"},
                )
                await container.resume_router.resume_continuation(
                    resolved.continuation,
                    answer_payload,
                )
            record, output = await asyncio.wait_for(
                RunFacade(container.run_manager, session_id="parent-session").wait_run(
                    result["child_run"],
                    return_outputs=True,
                ),
                10,
            )
            assert record.status.value == "succeeded", record.error
            assert output == {"answer": "Titanium"}
            assert executions == [question.meta["run_id"]]
            page = await container.cont_store.query(
                ContinuationQuery(
                    correlator=Correlator(
                        "interaction", "public", message=question.meta["interaction_id"]
                    ),
                    statuses=(ContinuationStatus.RESUMED,),
                    limit=1,
                )
            )
            assert len(page.items[0].correlators) == 1
            assert page.items[0].status.value == "resumed"
            assert all(page.items[0].payload[key] == value for key, value in answer_payload.items())
        finally:
            await container.run_manager.close()
            await container.close_storage()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [None, "missing", "cycle", "org", "user", "session"])
async def test_nested_run_authority_requires_intact_same_tenant_lineage(fault):
    leaf = RunRecord(
        run_id="leaf",
        graph_id="graph",
        kind="graphfn",
        status=RunStatus.running,
        started_at=datetime.now(UTC),
        session_id="leaf-session",
        org_id="org",
        user_id="user",
        parent=RunParent(run_id="relay", session_id="relay-session"),
    )
    relay = replace(
        leaf,
        run_id="relay",
        session_id="relay-session",
        status=RunStatus.succeeded,
        parent=RunParent(run_id="root", session_id="root-session"),
    )
    if fault == "cycle":
        relay = replace(relay, parent=RunParent(run_id="leaf", session_id="leaf-session"))
    elif fault in {"org", "user", "session"}:
        relay = replace(relay, **{f"{fault}_id": "unrelated"})
    records = {"leaf": leaf, "relay": relay}
    if fault == "missing":
        del records["relay"]
    manager = SimpleNamespace(get_record=AsyncMock(side_effect=records.get))
    facade = RunFacade(
        manager, session_id="root-session", identity=RequestIdentity(org_id="org", user_id="user")
    )
    if fault is None:
        assert await facade.inspect_run("leaf") == leaf
    else:
        with pytest.raises(LookupError, match="caller scope"):
            await facade.inspect_run("leaf")


@pytest.mark.asyncio
@pytest.mark.parametrize("termination", ["cancel", "delivery_failure", "cleanup_failure"])
async def test_stopped_child_closes_its_question_without_stopping_sibling(
    tmp_path, termination, monkeypatch, capfd
):
    container = build_default_container(
        cfg=AppSettings(workspace=str(tmp_path), embed={"enabled": False}), root=str(tmp_path)
    )
    questions = asyncio.Queue()
    sibling_release = asyncio.Event()
    executions = []

    class Adapter:
        capabilities = {"text", "input"}

        async def send(self, event):
            questions.put_nowait(event)
            if termination in {"delivery_failure", "cleanup_failure"}:
                raise RuntimeError("question delivery failed")
            return {}

    container.channels.register_adapter("test", Adapter())
    if termination == "cleanup_failure":
        monkeypatch.setattr(
            container.cont_store, "close", AsyncMock(side_effect=OSError("storage unavailable"))
        )

    async def child(question, *, context: NodeContext):
        executions.append(context.run_id)
        if question:
            value = await context.channel("test:question").ask_text("Material?")
        else:
            await sibling_release.wait()
            value = "sibling completed"
        return {"value": value}

    child_graph = GraphFunction(
        name=f"test.stopped_question.{uuid4().hex}",
        fn=child,
        inputs=["question"],
        outputs=["value"],
    )

    async def parent(*, context: NodeContext):
        ids = [
            await context.runner().spawn_run(
                child_graph.name,
                inputs={"question": question},
                session_id=f"child-{question}",
            )
            for question in (True, False)
        ]
        return {"ids": ids}

    with use_services(container):
        try:
            current_registry().register(
                nspace="graphfn",
                name=child_graph.name,
                version=child_graph.version,
                obj=child_graph,
            )
            result = await run_async(
                GraphFunction(
                    name=f"test.stop_parent.{uuid4().hex}", fn=parent, inputs=[], outputs=["ids"]
                ),
                {},
                session_id="parent",
            )
            event = await asyncio.wait_for(questions.get(), 10)
            facade = RunFacade(container.run_manager, session_id="parent")
            child_id, sibling_id = result["ids"]
            if termination == "cancel":
                await facade.cancel_run(child_id)
            stopped = await asyncio.wait_for(facade.wait_run(child_id), 10)
            assert stopped.status.value == ("canceled" if termination == "cancel" else "failed")
            retained = await container.cont_store.get(event.meta["run_id"], event.meta["node_id"])
            if termination == "cleanup_failure":
                assert retained.status is ContinuationStatus.WAITING
                assert "Failed to close interrupted interaction" in capfd.readouterr().err
                assert "question delivery failed" in stopped.error
            else:
                assert retained.status is ContinuationStatus.CANCELED
            assert not container.wait_registry.has(retained.continuation_id)
            if termination != "cleanup_failure":
                with pytest.raises(InteractionResolutionError, match="not open"):
                    await InteractionResolver(
                        container.cont_store,
                        run_manager=container.run_manager,
                    ).resolve_exact(
                        session_id="parent",
                        interaction_id=event.meta["interaction_id"],
                        expected_kinds={"user_input"},
                    )
            assert (await facade.inspect_run(sibling_id)).status is RunStatus.running
            sibling_release.set()
            record, output = await asyncio.wait_for(
                facade.wait_run(sibling_id, return_outputs=True),
                10,
            )
            assert record.status is RunStatus.succeeded
            assert output == {"value": "sibling completed"}
            assert sorted(executions) == sorted(result["ids"])
        finally:
            sibling_release.set()
            await container.run_manager.close()
            await container.close_storage()
