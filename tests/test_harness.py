from __future__ import annotations

from pathlib import Path
import tempfile

import pytest

from aethergraph import NodeContext, graph_fn, graphify, tool
from aethergraph.config.context import set_current_settings
from aethergraph.config.loader import load_settings
from aethergraph.contracts.services.execution import CodeExecutionResult
from aethergraph.core.runtime.runtime_services import install_services
from aethergraph.server.app_factory import create_app
from aethergraph.services.harness import (
    HarnessAttachment,
    HarnessExportConfig,
    HarnessRunner,
    HarnessScenario,
    HarnessTarget,
    OperatorOverride,
    OperatorOverrideRegistry,
)


@graph_fn(name="harness.echo", inputs=["value"], outputs=["result"])
async def harness_echo(value: str) -> dict[str, str]:
    return {"result": value.upper()}


@graph_fn(
    name="harness.agent",
    inputs=["message", "attachments", "session_id", "user_meta"],
    outputs=["reply"],
    as_agent={
        "id": "harness_agent",
        "title": "Harness Agent",
        "short_description": "Harness test agent",
        "session_kind": "chat",
        "mode": "chat_v1",
        "memory_level": "session",
    },
)
async def harness_agent(
    message: str,
    attachments=None,
    session_id: str | None = None,
    user_meta=None,
    *,
    context: NodeContext,
) -> dict[str, str]:
    await context.memory().record_chat_user(message)
    if attachments:
        await context.memory().record(
            kind="tool_result",
            data={"attachments": attachments},
            text="attachments seen",
            tags=["chat", "attachment"],
        )
    await context.memory().record_chat_assistant(f"agent:{message}")
    return {"reply": f"agent:{message}"}


@tool(name="harness.ask_text_tool", outputs=["text"])
async def harness_ask_text_tool(*, context: NodeContext):
    text = await context.channel("ui:session").ask_text(prompt="Name?")
    return {"text": text}


@tool(name="harness.ask_files_tool", outputs=["count"])
async def harness_ask_files_tool(*, context: NodeContext):
    result = await context.channel("ui:session").ask_files(prompt="Upload", multiple=True)
    return {"count": len((result or {}).get("files", []))}


@graphify(name="harness.wait_graph", inputs=[], outputs=["text", "count"])
def harness_wait_graph():
    text = harness_ask_text_tool()
    count = harness_ask_files_tool(_after=text)
    return {"text": text.text, "count": count.count}


@graph_fn(name="harness.exec_graph", inputs=[], outputs=["stdout"])
async def harness_exec_graph(*, context: NodeContext) -> dict[str, str]:
    result = await context.execute("print('real')", timeout_s=5.0)
    return {"stdout": result.stdout.strip()}


@tool(name="local.send_user_input", outputs=["ok"])
async def local_send_user_input(*, context: NodeContext, user_input: str):
    await context.channel().send_text(f"You entered: {user_input}")
    return {"ok": True}


@graphify(name="local.channel.ask_text_graph", inputs=[], outputs=["user_input"])
def local_channel_ask_text_graph():
    user_input = harness_ask_text_tool()
    local_send_user_input(user_input=user_input.text, _after=user_input)
    return {"user_input": user_input.text}


@graph_fn(name="local.artifact.save_and_search", inputs=[], outputs=["artifact_uri", "found_count"])
async def local_artifact_save_and_search(*, context: NodeContext) -> dict[str, str | int]:
    arts = context.artifacts()
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as handle:
        handle.write("artifact harness content")
        temp_path = Path(handle.name)
    try:
        artifact = await arts.save_file(
            str(temp_path),
            kind="text",
            suggested_uri="./artifact-harness.txt",
            labels={"experiment": "artifact-harness"},
            tags=["artifact-harness", "example"],
            pin=True,
        )
    finally:
        temp_path.unlink(missing_ok=True)

    results = await arts.search(
        kind="text",
        labels={"experiment": "artifact-harness"},
        tags=["artifact-harness"],
        level="run",
    )
    return {"artifact_uri": artifact.uri, "found_count": len(results)}


@pytest.fixture()
def harness_container(tmp_path: Path):
    cfg = load_settings()
    set_current_settings(cfg)
    app = create_app(workspace=str(tmp_path), cfg=cfg, log_level="warning")
    install_services(app.state.container)
    container = app.state.container
    container.registry.register(
        nspace="graphfn", name="harness.echo", version="0.1.0", obj=harness_echo
    )
    container.registry.register(
        nspace="graph", name="harness.wait_graph", version="0.1.0", obj=harness_wait_graph.build()
    )
    container.registry.register(
        nspace="graphfn", name="harness.agent", version="0.1.0", obj=harness_agent
    )
    container.registry.register(
        nspace="graph",
        name="local.channel.ask_text_graph",
        version="0.1.0",
        obj=local_channel_ask_text_graph.build(),
    )
    container.registry.register(
        nspace="graphfn",
        name="local.artifact.save_and_search",
        version="0.1.0",
        obj=local_artifact_save_and_search,
    )
    container.registry.register(
        nspace="agent",
        name="harness_agent",
        version="0.1.0",
        obj={"id": "harness_agent"},
        meta={
            "id": "harness_agent",
            "title": "Harness Agent",
            "backing": {"type": "graphfn", "name": "harness.agent"},
            "run_visibility": "inline",
            "run_importance": "ephemeral",
        },
    )
    return container


@pytest.mark.asyncio
async def test_harness_runs_direct_graph_fn(harness_container, tmp_path: Path):
    runner = HarnessRunner(container=harness_container)
    scenario = HarnessScenario(
        id="direct-echo",
        target=HarnessTarget(target=harness_echo),
        inputs={"value": "hello"},
        expected_outputs={"result": "HELLO"},
        export=HarnessExportConfig(root_dir=str(tmp_path / "exports")),
    )
    result = await runner.run_scenario(scenario)
    assert result.status == "succeeded"
    assert result.outputs == {"result": "HELLO"}
    assert result.scores["exact_output"]["passed"] is True
    assert (tmp_path / "exports" / "direct-echo" / "manifest.json").exists()


@pytest.mark.asyncio
async def test_harness_runs_agent_with_session_semantics(harness_container):
    runner = HarnessRunner(container=harness_container)
    scenario = HarnessScenario(
        id="agent-run",
        target=HarnessTarget(agent_id="harness_agent"),
        message="hello agent",
        attachments=[HarnessAttachment(path=__file__)],
    )
    result = await runner.run_scenario(scenario)
    assert result.status == "succeeded"
    assert result.outputs == {"reply": "agent:hello agent"}
    assert any(
        event.get("payload", {}).get("type") == "user.message"
        for event in result.trace.channel_events
    )
    assert any(event.get("session_id") == result.session_id for event in result.trace.memory_events)


# @pytest.mark.asyncio
# async def test_harness_resolves_waits_and_exports_benchmark(harness_container, tmp_path: Path):
#     runner = HarnessRunner(container=harness_container)
#     wait_resolver = AttachmentResponder(
#         files=[HarnessAttachment(path=__file__, mimetype="text/x-python")],
#         text="Alice",
#         responses=[WaitResponse(kind="user_input", payload={"text": "Alice"})],
#     )
#     benchmark = HarnessBenchmark(
#         id="wait-benchmark",
#         scenarios=[
#             HarnessScenario(
#                 id="wait-scenario",
#                 target=HarnessTarget(graph_id="harness.wait_graph"),
#                 wait_resolver=wait_resolver,
#             )
#         ],
#         export=HarnessExportConfig(root_dir=str(tmp_path / "bench")),
#     )
#     result = await runner.run_benchmark(benchmark)
#     assert result.runs[0].status == "timeout"
#     assert result.runs[0].outputs is None
#     assert result.runs[0].waits == []
#     assert (tmp_path / "bench" / "wait-benchmark" / "runs.jsonl").exists()


@pytest.mark.asyncio
async def test_harness_operator_override_execution(harness_container):
    runner = HarnessRunner(container=harness_container)
    scenario = HarnessScenario(
        id="exec-override",
        target=HarnessTarget(target=harness_exec_graph),
        operator_overrides=OperatorOverrideRegistry(
            overrides=[
                OperatorOverride(
                    operator_type="execution",
                    operation="execute",
                    graph_id="harness.exec_graph",
                    mode="fixed_result",
                    result=CodeExecutionResult(stdout="fake\n", stderr="", exit_code=0),
                )
            ]
        ),
    )
    result = await runner.run_scenario(scenario)
    assert result.status == "succeeded"
    assert result.outputs == {"stdout": "fake"}


# @pytest.mark.asyncio
# async def test_harness_matches_local_channel_ask_text_script(harness_container):
#     runner = HarnessRunner(container=harness_container)
#     scenario = HarnessScenario(
#         id="local-channel-ask-text",
#         target=HarnessTarget(graph_id="local.channel.ask_text_graph"),
#         wait_resolver=ScriptedResponder(
#             responses=[WaitResponse(kind="user_input", payload={"text": "hello from harness"})]
#         ),
#     )
#     result = await runner.run_scenario(scenario)
#     assert result.status == "succeeded"
#     assert result.outputs == {"user_input": "hello from harness"}
#     assert len(result.waits) == 1


# @pytest.mark.asyncio
# async def test_harness_matches_local_artifact_save_search_script(harness_container):
#     runner = HarnessRunner(container=harness_container)
#     scenario = HarnessScenario(
#         id="local-artifact-save-search",
#         target=HarnessTarget(graph_id="local.artifact.save_and_search"),
#     )
#     result = await runner.run_scenario(scenario)
#     assert result.status == "succeeded"
#     assert result.outputs is not None
#     assert result.outputs["artifact_uri"].endswith("artifact-harness.txt")
#     assert result.outputs["found_count"] >= 1
