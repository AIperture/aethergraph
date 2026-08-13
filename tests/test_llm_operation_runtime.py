from __future__ import annotations

import inspect

from aethergraph.services.llm import generic_embed_client, image_runtime
from aethergraph.services.llm.operation_runtime import model_operation_dimensions


def test_non_chat_clients_delegate_lifecycle_ownership_once() -> None:
    embedding_source = inspect.getsource(generic_embed_client.GenericEmbeddingClient.embed_result)
    image_source = inspect.getsource(image_runtime._execute_image_generation)

    for source in (embedding_source, image_source):
        assert "execute_model_operation(" in source
        assert "_provider_retry.execute(" not in source
        assert "_operation_quota.reserve(" not in source
        assert "_operation_quota.reconcile(" not in source


def test_image_trace_projection_does_not_include_prompt_content() -> None:
    source = inspect.getsource(image_runtime._execute_image_generation)

    assert '"prompt_chars": len(invocation.prompt)' in source
    assert '"prompt": invocation.prompt' not in source


def test_common_operation_dimensions_prefer_explicit_non_sensitive_values() -> None:
    dimensions = model_operation_dimensions(
        profile_name="research",
        overrides={"run_id": "run-explicit", "user_id": "user-explicit"},
    )

    assert dimensions["run_id"] == "run-explicit"
    assert dimensions["user_id"] == "user-explicit"
    assert dimensions["profile_name"] == "research"
    assert "prompt" not in dimensions
    assert "texts" not in dimensions
    assert "images" not in dimensions
