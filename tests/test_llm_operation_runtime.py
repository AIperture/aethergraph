from __future__ import annotations

import inspect

import pytest

from aethergraph.services.llm import generic_client, generic_embed_client, image_runtime
from aethergraph.services.llm.operation_runtime import model_operation_dimensions
from aethergraph.services.llm.provider_transport import (
    LLMProviderRequestError,
    ProviderCallResult,
)
from aethergraph.services.llm.types import GeneratedImage, ImageGenerationResult


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


def test_image_result_validation_rejects_missing_normalized_payload() -> None:
    result = ProviderCallResult(ImageGenerationResult(images=[GeneratedImage()], usage={}))

    with pytest.raises(LLMProviderRequestError) as raised:
        image_runtime._validate_image_result(
            result,
            host=type("Host", (), {"provider": "openai"})(),
            model="gpt-image-test",
        )

    assert raised.value.code == "provider_response_malformed"
    assert raised.value.operation == "image"
    assert raised.value.retryable is False


@pytest.mark.parametrize(
    "image",
    (GeneratedImage(b64="aW1hZ2U="), GeneratedImage(url="https://example.test/i.png")),
)
def test_image_result_validation_accepts_canonical_payload_locator(image: GeneratedImage) -> None:
    image_runtime._validate_image_result(
        ProviderCallResult(ImageGenerationResult(images=[image], usage={})),
        host=type("Host", (), {"provider": "openai"})(),
        model="gpt-image-test",
    )


def test_common_operation_dimensions_prefer_explicit_non_sensitive_values() -> None:
    dimensions = model_operation_dimensions(
        profile_name="research",
        overrides={
            "tenant_id": "tenant-explicit",
            "project_id": "project-explicit",
            "run_id": "run-explicit",
            "user_id": "user-explicit",
            "scope_key": "workspace-explicit",
        },
    )

    assert dimensions["tenant_id"] == "tenant-explicit"
    assert dimensions["project_id"] == "project-explicit"
    assert dimensions["run_id"] == "run-explicit"
    assert dimensions["user_id"] == "user-explicit"
    assert dimensions["scope_key"] == "workspace-explicit"
    assert dimensions["profile_name"] == "research"
    assert "prompt" not in dimensions
    assert "texts" not in dimensions
    assert "images" not in dimensions


def test_chat_uses_the_same_common_dimension_projection() -> None:
    source = inspect.getsource(generic_client.GenericLLMClient._current_dimensions)

    assert "model_operation_dimensions(" in source
    assert "current_meter_context.get()" not in source
