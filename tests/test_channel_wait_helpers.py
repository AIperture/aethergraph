"""Dual-stage questions publish only their pre-registered public identity."""

from uuid import uuid4

import pytest

from aethergraph import NodeContext
from aethergraph.config.config import AppSettings
from aethergraph.core.graph.graph_fn import GraphFunction
from aethergraph.core.runtime.runtime_services import use_services
from aethergraph.core.tools.builtins.channel_tools import AskApproval, AskFiles, AskText, WaitText
from aethergraph.runner import run_async
from aethergraph.services.container.default_container import build_default_container
from aethergraph.services.continuations.continuation import Correlator
from aethergraph.services.integration.interactions import InteractionResolver


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_type", [AskText, AskApproval, AskFiles, WaitText])
@pytest.mark.parametrize("inline", [False, True])
async def test_dual_stage_question_is_registered_before_delivery(tmp_path, tool_type, inline):
    container = build_default_container(
        cfg=AppSettings(workspace=str(tmp_path), embed={"enabled": False}), root=str(tmp_path)
    )
    delivered = []

    class Adapter:
        capabilities = {"text", "input", "buttons"}

        async def send(self, event):
            resolved = await InteractionResolver(container.cont_store).resolve_exact(
                session_id="question-session",
                interaction_id=event.meta["interaction_id"],
                expected_kinds={"approval", "user_input", "user_files"},
            )
            delivered.append(resolved)
            if inline:
                return {"payload": {"text": "Ready"}}
            return {"correlator": Correlator("test", event.channel, "", "delivery")}

    container.channels.register_adapter("test", Adapter())

    async def graph(*, context: NodeContext):
        args = {} if tool_type is WaitText else {"prompt": "Reply"}
        wait = await tool_type().setup(context=context, channel="test:question", **args)
        stored = await container.cont_store.resolve_token(wait.token)
        assert stored.revision == 1
        assert stored.correlators == (
            Correlator("interaction", "public", message=stored.payload["_interaction_id"]),
        )
        if tool_type is WaitText:
            assert delivered == []
            assert wait.inline_payload is None
        else:
            assert delivered[0].continuation == stored
            assert wait.inline_payload == ({"text": "Ready"} if inline else None)
        return {"done": True}

    with use_services(container):
        try:
            result = await run_async(
                GraphFunction(
                    name=f"test.dual_question.{uuid4().hex}", fn=graph, inputs=[], outputs=["done"]
                ),
                {},
                session_id="question-session",
            )
            assert result == {"done": True}
        finally:
            await container.run_manager.close()
            await container.close_storage()
