import json
from dataclasses import replace
from aethergraph.services.llm import ModelRequest, message_from_text
from aethergraph.services.llm.generic_client import GenericLLMClient
from aethergraph.services.llm.tool_calling import ToolDefinition
from aethergraph.services.llm.tool_discovery import ToolDiscoveryRequest


def test_loaded_declarations_share_provider_projection():
    from aethergraph.services.llm.adapters.chat import project_context_tools
    from aethergraph.services.llm.request_preparation import prepare_model_request

    tool = ToolDefinition(
        name="large_tool",
        description="x" * 12000,
        input_schema={"type": "object", "properties": {}},
        exposure="deferred",
    )
    discovery = ToolDiscoveryRequest(
        mode="native_client", search_schema={"type": "object", "properties": {}}
    )
    req = ModelRequest(
        messages=(message_from_text("user", "hello"),),
        tools=(tool,),
        native_tool_search=discovery,
        turn_id="t",
    )
    client = GenericLLMClient(provider="openai", model="gpt-5.6-luna", api_key="test")
    before = client.estimate(req)
    after = client.estimate(replace(req, active_tool_names=("large_tool",)))
    assert after.estimated_input_tokens > before.estimated_input_tokens + 2500
    _, tr = prepare_model_request(req)
    assert "large_tool" not in json.dumps(project_context_tools("openai_responses", tr))
    assert before.measurement_scope == "logical_context"


def test_effective_projection_is_counted_without_duplicate_transport_root():
    client = GenericLLMClient(provider="openai", model="gpt-5.6-luna", api_key="test")
    req = ModelRequest(
        messages=(message_from_text("user", "old " * 10000),),
        effective_messages=(message_from_text("user", "current"),),
    )
    estimate = client.estimate(req)
    assert estimate.estimated_input_tokens < 100
    assert estimate.measurement_scope == "engine_context_projection"
