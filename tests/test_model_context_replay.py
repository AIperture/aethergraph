from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest
from test_llm_chat_contract import _FakeHttpClient

from aethergraph.services.llm import (
    LLMToolCallResponseError,
    ModelContextManagement,
    ModelRequest,
    message_from_text,
)
from aethergraph.services.llm.context_management import (
    ModelContextCheckpoint,
    model_context_message_digest,
)
from aethergraph.services.llm.generic_client import GenericLLMClient


def _checkpoint(**kwargs):
    return ModelContextCheckpoint(
        provider="openai",
        model="gpt-5.6-luna",
        protocol="responses.context_management.compact",
        source_message_count=0,
        source_message_digest=model_context_message_digest([]),
        payload={"compacted_input": [{"type": "compaction", "encrypted_content": "private"}]},
        **kwargs,
    )


def test_checkpoint_results_are_detached_and_identical_duplicates_are_idempotent():
    result = {"kind": "tool_output", "call_id": "call_1", "output": "done"}
    checkpoint = _checkpoint(pending_result_ids=("call_1",), replay_results=(result, dict(result)))
    result["output"] = "mutated"
    assert checkpoint.validated_replay_results(("call_1",))["call_1"]["output"] == "done"
    restored = ModelContextCheckpoint.from_dict(checkpoint.to_dict())
    assert restored == checkpoint
    assert "private" not in repr(checkpoint)
    assert "done" not in repr(checkpoint)
    assert replace(checkpoint, replay_results=()).payload == checkpoint.payload


@pytest.mark.parametrize(
    "results,code",
    [
        ((), "missing"),
        (({"kind": "tool_output", "call_id": "other", "output": "done"},), "conflict"),
        (
            (
                {"kind": "tool_output", "call_id": "call_1", "output": "a"},
                {"kind": "tool_output", "call_id": "call_1", "output": "b"},
            ),
            "conflict",
        ),
        (({"kind": "tool_output", "call_id": "call_1", "output": {}},), "invalid"),
        (
            (
                {
                    "kind": "discovery_result",
                    "provider_reference_id": "call_1",
                    "discovery_event_id": "event_1",
                    "status": "completed",
                    "tool_names": [],
                },
            ),
            "invalid",
        ),
    ],
)
def test_checkpoint_rejects_incomplete_or_invalid_replay(results, code):
    checkpoint = _checkpoint(pending_result_ids=("call_1",), replay_results=results)
    with pytest.raises(LLMToolCallResponseError) as error:
        checkpoint.validated_replay_results(("call_1",))
    assert error.value.code == f"model_context_replay_result_{code}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider,model", [("openai", "gpt-5.6-luna"), ("anthropic", "claude-sonnet-4-6")]
)
async def test_checkpoint_retains_only_latest_boundary_and_following_native_content(
    provider, model
):
    if provider == "openai":
        latest = {"type": "compaction", "id": "cmp_latest", "encrypted_content": "latest-private"}
        sibling = {
            "type": "message",
            "id": "msg_1",
            "content": [{"type": "output_text", "text": "done"}],
        }
        payload = {
            "id": "resp_1",
            "status": "completed",
            "output": [
                {"type": "compaction", "id": "cmp_old", "encrypted_content": "old-private"},
                latest,
                sibling,
            ],
        }
    else:
        latest = {"type": "compaction", "content": "latest summary"}
        sibling = {"type": "text", "text": "done"}
        payload = {
            "id": "msg_1",
            "stop_reason": "end_turn",
            "content": [{"type": "compaction", "content": "old summary"}, latest, sibling],
        }
    client = GenericLLMClient(provider=provider, model=model, api_key="test")
    client._client = _FakeHttpClient(payload)
    client._bound_loop = asyncio.get_running_loop()
    response = await client.generate(
        ModelRequest(
            messages=(message_from_text("user", "Start"),),
            context_management=ModelContextManagement(trigger_tokens=50000),
        )
    )
    checkpoint = response.context_checkpoint
    assert checkpoint is not None
    retained = checkpoint.payload[
        "compacted_input" if provider == "openai" else "assistant_content"
    ]
    assert retained == [latest, sibling]
