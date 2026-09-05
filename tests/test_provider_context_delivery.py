"""Fresh context reaches actual serialized provider requests after cold restore."""
import asyncio
import json
from dataclasses import asdict

import pytest
from aethergraph.services.llm import ModelRequest, ToolCallOutput, ToolTransportCheckpoint
from aethergraph.services.llm.generic_client import GenericLLMClient
from aethergraph.services.llm import message_from_text
from test_llm_chat_contract import _FakeHttpClient, _native_tool_request


def _response(provider,index):
    name="lookup" if index<3 else "finish"
    args={"key":str(index)} if name=="lookup" else {}
    call_id=f"call_{index}"
    if provider=="openai":
        return {"id":f"resp_{index}","status":"completed","output":[{"type":"function_call","id":f"fc_{index}","call_id":call_id,"name":name,"arguments":json.dumps(args),"status":"completed"}]}
    if provider=="anthropic":
        return {"id":f"msg_{index}","stop_reason":"tool_use","content":[{"type":"tool_use","id":call_id,"name":name,"input":args}]}
    if provider=="google":
        return {"candidates":[{"index":0,"finishReason":"STOP","content":{"role":"model","parts":[{"functionCall":{"id":call_id,"name":name,"args":args},"thoughtSignature":f"opaque_{index}"}]}}]}
    return {"id":f"chat_{index}","choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":None,"tool_calls":[{"id":call_id,"type":"function","function":{"name":name,"arguments":json.dumps(args)}}]}}]}


@pytest.mark.asyncio
@pytest.mark.parametrize("provider",["openai","anthropic","google","openrouter"])
async def test_appended_context_and_exact_results_survive_three_decisions(provider):
    client=GenericLLMClient(provider=provider,model="test-model",api_key="test")
    http=_FakeHttpClient(_response(provider,1));client._client=http;client._bound_loop=asyncio.get_running_loop()
    tools=_native_tool_request(max_calls=1).tools
    messages=(message_from_text("user","ORIGINAL_REQUEST"),)
    first=await client.generate(ModelRequest(messages=messages,tools=tools,turn_id="turn"))
    assert first.continuation is not None
    # JSON round-trip models persisted transport, with a fresh client to rule out a warm cache fallback.
    checkpoint=ToolTransportCheckpoint(**json.loads(json.dumps(asdict(first.continuation))))
    client=GenericLLMClient(provider=provider,model="test-model",api_key="test")
    client._client=http;client._bound_loop=asyncio.get_running_loop()
    http.payload=_response(provider,2)
    messages=(*messages,message_from_text("user","UPDATED_GUIDANCE_REVISION_2"))
    second=await client.generate(ModelRequest(messages=messages,tools=tools,turn_id="turn",continuation=checkpoint,tool_outputs=(ToolCallOutput("call_1","EXACT_RESULT_ONE"),)))
    sent=json.dumps(http.last_json)
    assert "UPDATED_GUIDANCE_REVISION_2" in sent and "EXACT_RESULT_ONE" in sent
    http.payload=_response(provider,3)
    messages=(*messages,message_from_text("user","RETRIEVED_EVIDENCE_REVISION_3"))
    third=await client.generate(ModelRequest(messages=messages,tools=tools,turn_id="turn",continuation=second.continuation,tool_outputs=(ToolCallOutput("call_2","EXACT_RESULT_TWO"),)))
    assert third.calls[0].name=="finish"
    sent=json.dumps(http.last_json)
    assert sent.count("RETRIEVED_EVIDENCE_REVISION_3")==1
    assert sent.count("EXACT_RESULT_TWO")==1
    if provider!="openai":
        # Responses retains earlier turns behind previous_response_id; the other adapters replay them explicitly.
        for evidence in ("ORIGINAL_REQUEST","UPDATED_GUIDANCE_REVISION_2","EXACT_RESULT_ONE"):
            assert sent.count(evidence)==1, sent
    if provider=="google":
        assert "opaque_1" in sent and "opaque_2" in sent
